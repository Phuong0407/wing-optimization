import ufl
from dolfinx import fem, io
from dolfinx.io import gmsh
from dolfinx.fem.petsc import NonlinearProblem
from types import SimpleNamespace
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import meshio
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from pathlib import Path
from functools import reduce
import basix
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.rank



VERBOSE = True
QUICK_TEST = True   # Set False for full optimization run

def vprint(*args, **kwargs):
    if VERBOSE and MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)


# LOCAL FRAME
def normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))

def local_frame_ufl(domain):
    t  = ufl.Jacobian(domain)
    t1 = ufl.as_vector([t[0, 0], t[1, 0], t[2, 0]])
    t2 = ufl.as_vector([t[0, 1], t[1, 1], t[2, 1]])
    e3 = normalize(ufl.cross(t1, t2))
    ey = ufl.as_vector([0, 1, 0])
    ez = ufl.as_vector([0, 0, 1])
    e1_trial = ufl.cross(ey, e3)
    norm_e1  = ufl.sqrt(ufl.dot(e1_trial, e1_trial))
    e1 = ufl.conditional(ufl.lt(norm_e1, 0.5), ez, normalize(e1_trial))
    e2 = normalize(ufl.cross(e3, e1))
    return e1, e2, e3

def local_frame(domain):
    FRAME = local_frame_ufl(domain)
    VT    = fem.functionspace(domain, ("DG", 0, (3,)))
    V0, _ = VT.sub(0).collapse()
    BASIS = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(3)]
    for i in range(3):
        e_exp = fem.Expression(FRAME[i], V0.element.interpolation_points)
        BASIS[i].interpolate(e_exp)
    return BASIS[0], BASIS[1], BASIS[2]





# SHELL KINEMATICS
def hstack(vecs):
    return ufl.as_matrix([[vi[i] for i in range(len(vi))] for vi in vecs]).T

def tangent_projection(e1, e2):
    return hstack([e1, e2])

def tangential_gradient(w, P):
    return ufl.dot(ufl.grad(w), P)

def membrane_strain(u, P):
    t_gu = ufl.dot(P.T, tangential_gradient(u, P))
    return ufl.sym(t_gu), t_gu

def bending_strain(theta, e3, P):
    beta = ufl.cross(e3, theta)
    return ufl.sym(ufl.dot(P.T, tangential_gradient(beta, P)))

def shear_strain(u, theta, e3, P):
    beta = ufl.cross(e3, theta)
    return tangential_gradient(ufl.dot(u, e3), P) - ufl.dot(P.T, beta)

def compute_drilling_strain(t_gu, theta, e3):
    return (t_gu[0, 1] - t_gu[1, 0]) / 2 + ufl.dot(theta, e3)

def shell_strains(u, theta, e1, e2, e3):
    P               = tangent_projection(e1, e2)
    eps, t_gu       = membrane_strain(u, P)
    kappa           = bending_strain(theta, e3, P)
    gamma           = shear_strain(u, theta, e3, P)
    drilling_strain = compute_drilling_strain(t_gu, theta, e3)
    return eps, kappa, gamma, drilling_strain


def isotropic_material(thickness, young, poisson, domain):
    # `thickness` can be a float (m) or an existing `fem.Constant` so callers
    # can update `.value` in-place during parametric studies / optimization.
    h = thickness if isinstance(thickness, fem.Constant) else fem.Constant(domain, float(thickness))
    E        = fem.Constant(domain, float(young))
    nu       = fem.Constant(domain, float(poisson))
    lmbda    = E * nu / (1 + nu) / (1 - 2 * nu)
    mu       = E / 2 / (1 + nu)
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)
    return SimpleNamespace(
            h=h,
            E=E,
            nu=nu,
            lmbda=lmbda,
            mu=mu,
            lmbda_ps=lmbda_ps,
            kind="isotropic",
        )



# MATERIAL DEFINITION
## VOIGT NOTATION
def to_voigt(e):
    return ufl.as_vector([e[0, 0], e[1, 1], 2.0 * e[0, 1]])

def from_voigt(v):
    return ufl.as_tensor([[v[0], v[2]], [v[2], v[1]]])




# SHELL STRESS RESULTANTS
def _plane_stress_iso(mat, e):
    tdim = e.ufl_shape[0]
    return mat.lmbda_ps * ufl.tr(e) * ufl.Identity(tdim) + 2 * mat.mu * e

def stress_resultants(mat, eps, kappa, gamma):
    if mat.kind == "isotropic":
        N = mat.h * _plane_stress_iso(mat, eps)
        M = mat.h**3 / 12.0 * _plane_stress_iso(mat, kappa)
        Q = mat.mu * mat.h * gamma

    elif mat.kind == "clt":
        eps_v   = to_voigt(eps)
        kappa_v = to_voigt(kappa)
        N_v = ufl.dot(mat.A_ufl, eps_v) + ufl.dot(mat.B_ufl, kappa_v)
        M_v = ufl.dot(mat.B_ufl, eps_v) + ufl.dot(mat.D_ufl, kappa_v)
        N   = from_voigt(N_v)
        M   = from_voigt(M_v)
        Q   = ufl.dot(mat.As_ufl, gamma)

    else:
        raise ValueError(f"Unknown material kind: {mat.kind!r}")
    return N, M, Q

def drilling_terms(mat, domain, drilling_strain):
    h_mesh = ufl.CellDiameter(domain)
    if mat.kind == "isotropic":
        G_eff = mat.mu # Assuming mat.mu is a UFL object or float
        H_val = mat.h  # Assuming mat.h is a UFL object or float
    else:
        G_eff = mat.G_eff # This is already a fem.Constant
        H_val = mat.H     # This is now also a fem.Constant
    stiffness = G_eff * H_val**3 / h_mesh**2
    stress    = stiffness * drilling_strain
    return stiffness, stress # stiffness is a UFL expression





# CFD LOAD
def map_traction(foamfile, femfile, outfile, skin_phys_tags, unit="mm", verbose=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        fm = meshio.read(str(foamfile))
        fp = fm.points
        ft = fm.point_data["traction"]
        foam_tri = fm.cells_dict.get("triangle", None)
    else:
        fp = None
        ft = None
        foam_tri = None

    fp = comm.bcast(fp, root=0)
    ft = comm.bcast(ft, root=0)
    foam_tri = comm.bcast(foam_tri, root=0)

    sm = meshio.read(str(femfile))

    all_triangles = sm.cells_dict.get("triangle", np.zeros((0, 3), dtype=np.int64))

    skin_triangles = []
    phys_data = None
    for key in ["gmsh:physical", "cell_tags"]:
        if key in sm.cell_data:
            phys_data = sm.cell_data[key]
            break

    if phys_data is not None:
        tri_blocks = [c.data for c in sm.cells if c.type == "triangle"]
        tag_blocks = [d for d, c in zip(phys_data, sm.cells) if c.type == "triangle"]
        for block_cells, block_tags in zip(tri_blocks, tag_blocks):
            mask = np.isin(block_tags, list(skin_phys_tags))
            skin_triangles.append(block_cells[mask])

        skin_triangles = np.concatenate(skin_triangles) if skin_triangles else all_triangles
    else:
        if verbose and rank == 0:
            vprint("[MAP] Warning: no physical tag data — using all triangles")
        skin_triangles = all_triangles

    skin_node_idx = np.unique(skin_triangles)
    sp_all = sm.points.astype(np.float64)
    if unit == "mm":
        sp_all *= 1e-3
    sp = sp_all[skin_node_idx]

    interp = RBFInterpolator(fp, ft, kernel="thin_plate_spline", neighbors=20)
    skin_traction = interp(sp)

    nodal_traction = np.zeros((len(sp_all), 3), dtype=np.float64)
    nodal_traction[skin_node_idx] = skin_traction

    trib_area = np.zeros(len(sp_all))
    for tri_nodes in skin_triangles:
        P0, P1, P2 = sp_all[tri_nodes[0]], sp_all[tri_nodes[1]], sp_all[tri_nodes[2]]
        area = 0.5 * np.linalg.norm(np.cross(P1 - P0, P2 - P0))
        trib_area[tri_nodes] += area / 3.0

    nodal_force = nodal_traction * trib_area[:, None]

    if verbose and rank == 0 and foam_tri is not None:
        foam_area = 0.5 * np.linalg.norm(
            np.cross(fp[foam_tri[:, 1]] - fp[foam_tri[:, 0]],
                     fp[foam_tri[:, 2]] - fp[foam_tri[:, 0]]),
            axis=1
        )
        foam_tract = ft[foam_tri].mean(axis=1)
        foam_force = (foam_tract * foam_area[:, None]).sum(axis=0)

        fem_force = nodal_force.sum(axis=0)
        err = np.linalg.norm(fem_force - foam_force) / np.linalg.norm(foam_force)

        vprint(f"[MAP] FOAM force [N] : {foam_force}")
        vprint(f"[MAP] FEM  force [N] : {fem_force}")
        vprint(f"[MAP] Force error    : {err*100:.2f}%")
        vprint(f"[MAP] Skin nodes     : {len(skin_node_idx)} / {len(sp_all)} total")

    if rank == 0:
        meshio.write(
            str(outfile),
            meshio.Mesh(
                points=sp_all,
                cells=[("triangle", all_triangles)],
                point_data={"traction": nodal_traction},
            ),
        )
        if verbose:
            vprint(f"[MAP] Saved =====> {outfile}")

    comm.barrier()
    return nodal_force


def load_traction_xdmf(xdmffile, domain, verbose=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        data  = meshio.read(str(xdmffile))
        pts   = data.points
        tract = data.point_data["traction"]
    else:
        pts = None
        tract = None

    pts   = comm.bcast(pts, root=0)
    tract = comm.bcast(tract, root=0)

    VT = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    f  = fem.Function(VT, name="traction")
    coords = VT.tabulate_dof_coordinates()

    if verbose and rank == 0:
        vprint(f"[LOAD] FEM  pts range : {domain.geometry.x.min(axis=0)} => {domain.geometry.x.max(axis=0)}")
        vprint(f"[LOAD] XDMF pts range : {pts.min(axis=0)} => {pts.max(axis=0)}")

    tree = cKDTree(pts)
    dist, idx = tree.query(coords, k=1)

    if verbose:
        dmax = comm.allreduce(dist.max(), op=MPI.MAX)
        if rank == 0:
            vprint(f"[LOAD] Max KDTree dist : {dmax:.4e} m (should be < 1E-3 m)")

    f.x.array[:] = tract[idx].reshape(-1)
    f.x.scatter_forward()
    return f


_E          = 210.e9
_NU         = 0.3
thickness   = 1E-3








MESHUNIT        = "mm"
INPUT_DIR       = Path("InputData")
OUTPUT_DIR      = Path("Result")
MESHFILE        = INPUT_DIR / "wingbox.msh"
FOAMFILE        = INPUT_DIR / "wing.vtp"
XDMFFILE        = INPUT_DIR / "FOAMData.xdmf"
MAPFILE         = INPUT_DIR / "MappedTraction.xdmf"

# IMPORT MESH
## PHYSICAL TAG FROM GMSH GEOMETRY FILE
TAG_UPPER     = 14
TAG_LOWER     = 15
TAG_RIB0      = 38
TAG_RIB750    = 39
TAG_RIB1500   = 40
TAG_RIB2250   = 41
TAG_RIB3000   = 42
TAG_MAINSPAR  = 43
TAG_TAILSPAR  = 44
TAG_SKIN      = [TAG_UPPER, TAG_LOWER]
TAG_RIBS      = [TAG_RIB0, TAG_RIB750, TAG_RIB1500, TAG_RIB2250, TAG_RIB3000]
TAG_COMPOSITE = TAG_SKIN + [TAG_MAINSPAR, TAG_TAILSPAR]
TAG_ALL       = TAG_COMPOSITE + TAG_RIBS

## LOAD MESH FROM GMSH FILE
MESH_IO     = gmsh.read_from_msh(MESHFILE, comm=MPI.COMM_WORLD, gdim=3)
DOMAIN      = MESH_IO.mesh
CELL_TAGS   = MESH_IO.cell_tags
FACET_TAGS  = MESH_IO.facet_tags
GDIM        = DOMAIN.geometry.dim
TDIM        = DOMAIN.topology.dim
FDIM        = TDIM - 1

if MESHUNIT == "mm":
    DOMAIN.geometry.x[:] *= 1E-3


E1, E2, E3 = local_frame(DOMAIN)

# FUNCTION SPACE
Ue          = basix.ufl.element("P",  DOMAIN.basix_cell(), 2, shape=(GDIM,))
Te          = basix.ufl.element("CR", DOMAIN.basix_cell(), 1, shape=(GDIM,))
V           = fem.functionspace(DOMAIN, basix.ufl.mixed_element([Ue, Te]))
v           = fem.Function(V)
u, theta    = ufl.split(v)
v_          = ufl.TestFunction(V)
u_, theta_  = ufl.split(v_)
dv          = ufl.TrialFunction(V)



# SHELL KINEMATICS
eps, kappa, gamma, drilling_strain = shell_strains(u, theta, E1, E2, E3)
eps_             = ufl.derivative(eps,           v, v_)
kappa_           = ufl.derivative(kappa,         v, v_)
gamma_           = ufl.derivative(gamma,         v, v_)
drilling_strain_ = ufl.replace(drilling_strain, {v: v_})

# BOUNDARY CONDITIONS: CLAMP ROOT

TAG_RIB0_CURVE = 45
ROOT = FACET_TAGS.find(TAG_RIB0_CURVE)

Vu, _ = V.sub(0).collapse()
dofs_u = fem.locate_dofs_topological((V.sub(0), Vu), FDIM, ROOT)
bc_u = fem.dirichletbc(fem.Function(Vu), dofs_u, V.sub(0))

Vt, _ = V.sub(1).collapse()
dofs_t = fem.locate_dofs_topological((V.sub(1), Vt), FDIM, ROOT)
bc_t = fem.dirichletbc(fem.Function(Vt), dofs_t, V.sub(1))

BCS = [bc_u, bc_t]

map_traction(XDMFFILE, MESHFILE, MAPFILE, TAG_SKIN, MESHUNIT, VERBOSE)
FTraction = load_traction_xdmf(MAPFILE, DOMAIN, VERBOSE)


# Initial fem.Constant for TOTAL thicknesses
t_skin_const = fem.Constant(DOMAIN, thickness)
t_spar_const = fem.Constant(DOMAIN, thickness)
t_rib_const  = fem.Constant(DOMAIN, thickness)

vprint("\n[MAT] SKIN (isotropic)")
MAT_SKIN = isotropic_material(t_skin_const, _E, _NU, DOMAIN)
MAT_SKIN._total_thickness_const = t_skin_const

vprint("\n[MAT] SPAR (isotropic)")
MAT_SPAR = isotropic_material(t_spar_const, _E, _NU, DOMAIN)
MAT_SPAR._total_thickness_const = t_spar_const

vprint("\n[MAT] RIB (isotropic)")
MAT_RIB = isotropic_material(t_rib_const, _E, _NU, DOMAIN)
MAT_RIB._total_thickness_const = t_rib_const

MATS = {
    14 : MAT_SKIN,
    15 : MAT_SKIN,
    38 : MAT_RIB,
    39 : MAT_RIB,
    40 : MAT_RIB,
    41 : MAT_RIB,
    42 : MAT_RIB,
    43 : MAT_SPAR,
    44 : MAT_SPAR,
}

dx = ufl.Measure("dx", domain=DOMAIN, subdomain_data=CELL_TAGS)

form_pieces = []
for tag, mat in MATS.items():
    N_t, M_t, Q_t = stress_resultants(mat, eps, kappa, gamma)
    _, drill_t     = drilling_terms(mat, DOMAIN, drilling_strain) # mat.H and mat.G_eff are now fem.Constant
    piece = (
        ufl.inner(N_t, eps_)
        + ufl.inner(M_t, kappa_)
        + ufl.inner(Q_t, gamma_)
        + drill_t * drilling_strain_
    ) * dx(tag)
    form_pieces.append(piece)

a_int       = reduce(lambda a, b: a + b, form_pieces)
L_ext       = sum(ufl.dot(FTraction, u_) * dx(tag) for tag in TAG_SKIN)
residual    = a_int - L_ext
tangent     = ufl.derivative(residual, v, dv)

problem = NonlinearProblem(
    F=residual,
    u=v,
    bcs=BCS,
    J=tangent,
    petsc_options_prefix="wing",
    petsc_options={
        "ksp_type"                  : "preonly",
        "pc_type"                   : "lu",
        "pc_factor_mat_solver_type" : "mumps",
        "snes_type"                 : "newtonls",
        "snes_rtol"                 : 1E-8,    # tighten to 1E-8 on HPC
        "snes_atol"                 : 1E-8,    # tighten to 1E-8 on HPC
        "snes_max_it"               : 25,      # increase to 25 on HPC
        "snes_monitor"              : None,
        "mat_mumps_icntl_14"        : 80,
        "mat_mumps_icntl_23"        : 4000,
    }
)

problem.solve()




# POST-PROCESSING

comm = MPI.COMM_WORLD

disp = v.sub(0).collapse()
disp.name = "Displacement"
rota = v.sub(1).collapse()
rota.name = "Rotation"

u_arr = disp.x.array.reshape(-1, 3)
t_arr = rota.x.array.reshape(-1, 3)

ux_max_local = np.abs(u_arr[:, 0]).max()
uy_max_local = np.abs(u_arr[:, 1]).max()
uz_max_local = np.abs(u_arr[:, 2]).max()
u_mag_local  = np.linalg.norm(u_arr, axis=1).max()

ux_max = comm.allreduce(ux_max_local, op=MPI.MAX)
uy_max = comm.allreduce(uy_max_local, op=MPI.MAX)
uz_max = comm.allreduce(uz_max_local, op=MPI.MAX)
u_max  = comm.allreduce(u_mag_local,  op=MPI.MAX)

tx_max_local = np.abs(t_arr[:, 0]).max()
ty_max_local = np.abs(t_arr[:, 1]).max()
tz_max_local = np.abs(t_arr[:, 2]).max()

tx_max = comm.allreduce(tx_max_local, op=MPI.MAX)
ty_max = comm.allreduce(ty_max_local, op=MPI.MAX)
tz_max = comm.allreduce(tz_max_local, op=MPI.MAX)

vprint("\n" + "="*60)
vprint("[POST] DISPLACEMENT & ROTATION")
vprint("="*60)
vprint(f"  |ux| max : {ux_max:.6e} m  ({ux_max*1e3:.3f} mm)")
vprint(f"  |uy| max : {uy_max:.6e} m  ({uy_max*1e3:.3f} mm)")
vprint(f"  |uz| max : {uz_max:.6e} m  ({uz_max*1e3:.3f} mm)")
vprint(f"  |u|  max : {u_max:.6e} m  ({u_max*1e3:.3f} mm)")
vprint()
vprint(f"  |thetax| max : {tx_max:.6e} rad")
vprint(f"  |thetay| max : {ty_max:.6e} rad")
vprint(f"  |thetaz| max : {tz_max:.6e} rad")
vprint()

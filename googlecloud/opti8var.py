import ufl
from dolfinx import fem, io
from dolfinx.io import gmsh
from dolfinx.fem.petsc import NonlinearProblem
from types import SimpleNamespace
import numpy as np
import meshio
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from pathlib import Path
from functools import reduce
import basix
import csv
from datetime import datetime
import nlopt
import numpy as np

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.rank



VERBOSE = True
QUICK_TEST = False

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





# MATERIAL DEFINITION
## VOIGT NOTATION
def to_voigt(e):
    return ufl.as_vector([e[0, 0], e[1, 1], 2.0 * e[0, 1]])

def from_voigt(v):
    return ufl.as_tensor([[v[0], v[2]], [v[2], v[1]]])

## CTL COMPOSITE MATERIAL
def _Q_ply(E1, E2, G12, nu12):
    nu21 = nu12 * E2 / E1
    d = 1 - nu12 * nu21
    return np.array([
        [ E1/d,      nu12*E2/d, 0   ],
        [ nu12*E2/d, E2/d,      0   ],
        [ 0,         0,         G12 ],
    ])

def _Qbar_ply(Q, angle):
    a = np.radians(angle)
    s = np.sin(a)
    c = np.cos(a)
    T = np.array([
        [ c**2,  s**2,   2*c*s       ],
        [ s**2,  c**2,  -2*c*s       ],
        [-c*s,   c*s,    c**2 - s**2 ],
    ])
    R = np.diag([1.0, 1.0, 2.0])
    Rinv = np.diag([1.0, 1.0, 0.5])
    return np.linalg.inv(T) @ Q @ R @ T @ Rinv

def compute_ABD(layup_angles, t_ply, E1, E2, G12, nu12):
    Q = _Q_ply(E1, E2, G12, nu12)
    H = t_ply * len(layup_angles)
    z = -H / 2.0
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for angle in layup_angles:
        Qb     = _Qbar_ply(Q, angle)
        z0, z1 = z, z + t_ply
        A += Qb * (z1 - z0)
        B += Qb * (z1**2 - z0**2) / 2.0
        D += Qb * (z1**3 - z0**3) / 3.0
        z  = z1
    return A, B, D, H

def clt_composite(layup_angles, t_ply, E1, E2, G12, nu12, G13=None, G23=None, kappa_s=5/6, label="CLT", verbose=False, domain=None):
    if G13 is None: G13 = G12
    if G23 is None: G23 = G12 * 0.5

    A_np, B_np, D_np, H = compute_ABD(layup_angles, t_ply, E1, E2, G12, nu12)

    max_B = np.abs(B_np).max()
    if verbose:
        vprint(f"[{label}] Layup   : {layup_angles}")
        vprint(f"[{label}] H       : {H*1E3:.2f} mm")
        vprint(f"[{label}] max|B|  : {max_B:.2e}  "
            f"{'SYMMETRIC' if max_B < 1E-6 * A_np.max() else 'NON-SYMMETRIC'}")
        vprint(f"[{label}] A11     : {A_np[0,0]/1E6:.2f} MPa·m")
        vprint(f"[{label}] D11     : {D_np[0,0]:.4f} N·m^2")

    As_np = kappa_s * H * np.array([
        [G13, 0.0],
        [0.0, G23],
    ])

    # EFFECTIVE IN-PLANE SHEAR FOR DRILLING STABILISATION
    G_eff        = float(A_np[2, 2]) / H
    H_const      = fem.Constant(domain, H)
    G_eff_const  = fem.Constant(domain, G_eff)
    A_ufl_const  = fem.Constant(domain, A_np)
    B_ufl_const  = fem.Constant(domain, B_np)
    D_ufl_const  = fem.Constant(domain, D_np)
    As_ufl_const = fem.Constant(domain, As_np)
    
    return SimpleNamespace(
        kind   = "clt",
        H      = H_const,
        t_ply  = t_ply,
        G_eff  = G_eff_const,
        A_ufl  = A_ufl_const,
        B_ufl  = B_ufl_const,
        D_ufl  = D_ufl_const,
        As_ufl = As_ufl_const,
        _layup_angles = layup_angles,
        _E1 = E1, _E2 = E2, _G12 = G12, _nu12 = nu12,
        _G13 = G13, _G23 = G23, _kappa_s = kappa_s,
    )

def build_laminate(layup, thickness_const, label):
    ply_thickness = thickness_const.value / len(layup)
    mat = clt_composite(
        layup,
        ply_thickness,
        CM_YNG1, CM_YNG2, CM_G12, CM_NU12,
        G13=CM_G12, G23=CM_G12 * 0.5, kappa_s=5/6,
        label=label, verbose=VERBOSE, domain=DOMAIN
    )
    mat._total_thickness_const = thickness_const
    return mat

def update_clt_material(mat, total_thickness, n_layers):
    t_ply = float(total_thickness) / n_layers
    A_np, B_np, D_np, H = compute_ABD(
        mat._layup_angles, t_ply, mat._E1, mat._E2, mat._G12, mat._nu12
    )
    As_np = mat._kappa_s * H * np.array([[mat._G13, 0.0], [0.0, mat._G23]])
    G_eff = float(A_np[2, 2]) / H

    mat._total_thickness_const.value = float(total_thickness)
    mat.A_ufl.value = A_np
    mat.B_ufl.value = B_np
    mat.D_ufl.value = D_np
    mat.As_ufl.value = As_np
    mat.G_eff.value = G_eff
    mat.H.value = H
    mat.t_ply = t_ply

def tsai_wu(sigma_mat, strengths):
    s1, s2, s6 = sigma_mat
    Xt = strengths["Xt"]
    Xc = strengths["Xc"]
    Yt = strengths["Yt"]
    Yc = strengths["Yc"]
    S = strengths["S"]
    F1 = 1/Xt - 1/Xc
    F2 = 1/Yt - 1/Yc
    F11 = 1/(Xt*Xc)
    F22 = 1/(Yt*Yc)
    F66 = 1/S**2
    F12 = -0.5 / np.sqrt(Xt * Xc * Yt * Yc)
    return (F1*s1 + F2*s2 + F11*s1**2 + F22*s2**2 + F66*s6**2 + 2*F12*s1*s2)

def hashin(sigma_mat, strengths):
    s1, s2, s6 = sigma_mat
    Xt, Xc     = strengths["Xt"], strengths["Xc"]
    Yt, Yc     = strengths["Yt"], strengths["Yc"]
    SL         = strengths["S"]
    ST         = strengths.get("ST", Yc / 2.0)
    out = {}
    if s1 >= 0:
        out["fiber_t"]  = (s1/Xt)**2 + (s6/SL)**2
    else:
        out["fiber_c"]  = (s1/Xc)**2
    if s2 >= 0:
        out["matrix_t"] = (s2/Yt)**2 + (s6/SL)**2
    else:
        out["matrix_c"] = ((s2/(2*ST))**2 + (Yc/(2*ST))**2 * (s2/Yc) + (s6/SL)**2)
    return out

def recover_and_evaluate_failure_cells(
    domain, v_sol, mat, cell_indices, strengths, criterion="tsai_wu", label="", basis=None
):
    assert mat.kind == "clt", "Failure recovery requires CLT material"
    if len(cell_indices) == 0:
        vprint(f"  [{label}] No cells — skipping failure analysis.")
        return 0.0, np.array([])

    u_h, theta_h = ufl.split(v_sol)
    if basis is None:
        e1, e2, e3 = local_frame(domain)
    else:
        e1, e2, e3 = basis
    P            = tangent_projection(e1, e2)
    eps_h, _     = membrane_strain(u_h, P)
    kappa_h      = bending_strain(theta_h, e3, P)

    DG0    = fem.functionspace(domain, ("DG", 0, (3,)))
    eps_fn = fem.Function(DG0)
    kap_fn = fem.Function(DG0)

    eps_fn.interpolate(fem.Expression(to_voigt(eps_h),   DG0.element.interpolation_points))
    kap_fn.interpolate(fem.Expression(to_voigt(kappa_h), DG0.element.interpolation_points))

    eps0_all  = eps_fn.x.array.reshape(-1, 3)
    kappa_all = kap_fn.x.array.reshape(-1, 3)
    eps0_vals  = eps0_all[cell_indices]
    kappa_vals = kappa_all[cell_indices]
    
    layup  = mat._layup_angles
    t_ply  = mat.t_ply
    Q_ply  = _Q_ply(mat._E1, mat._E2, mat._G12, mat._nu12)
    H      = float(mat.H.value)
    z      = -H / 2.0
    FI_all = []

    for angle in layup:
        z_mid      = z + t_ply / 2.0
        strain_k   = eps0_vals + z_mid * kappa_vals
        Qb_lam     = _Qbar_ply(Q_ply, angle)
        stress_lam = strain_k @ Qb_lam.T

        a    = np.radians(angle)
        c, s = np.cos(a), np.sin(a)
        T = np.array([
            [ c**2,  s**2,  2*c*s       ],
            [ s**2,  c**2, -2*c*s       ],
            [-c*s,   c*s,   c**2 - s**2 ],
        ])
        stress_mat = stress_lam @ T.T

        if criterion == "tsai_wu":
            FI_k = np.array([tsai_wu(s, strengths) for s in stress_mat])
        else:
            FI_k = np.array([max(hashin(s, strengths).values()) for s in stress_mat])

        FI_all.append(FI_k)
        vprint(f"  [{label}]  ply {angle:+4.0f} deg  max FI = {FI_k.max():.4f}")
        z += t_ply

    FI_all = np.array(FI_all)
    FI_max = FI_all.max()
    sf_str = f"{1/FI_max:.2f}" if FI_max > 0 else "inf"
    vprint(f"  [{label}]  Global max FI : {FI_max:.4f}  =>  SF = {sf_str}")
    return FI_max, FI_all


def _eval_failure_from_arrays(eps0_vals, kappa_vals, mat, strengths, criterion, label):
    layup  = mat._layup_angles
    t_ply  = mat.t_ply
    Q_ply  = _Q_ply(mat._E1, mat._E2, mat._G12, mat._nu12)
    H      = float(mat.H.value)
    z      = -H / 2.0
    FI_all = []
    for angle in layup:
        z_mid      = z + t_ply / 2.0
        strain_k   = eps0_vals + z_mid * kappa_vals
        Qb_lam     = _Qbar_ply(Q_ply, angle)
        stress_lam = strain_k @ Qb_lam.T
        a    = np.radians(angle)
        c, s = np.cos(a), np.sin(a)
        T = np.array([
            [ c**2,  s**2,  2*c*s       ],
            [ s**2,  c**2, -2*c*s       ],
            [-c*s,   c*s,   c**2 - s**2 ],
        ])
        stress_mat = stress_lam @ T.T
        if criterion == "tsai_wu":
            FI_k = np.array([tsai_wu(row, strengths) for row in stress_mat])
        else:
            FI_k = np.array([max(hashin(row, strengths).values()) for row in stress_mat])
        FI_all.append(FI_k)
        vprint(f"  [{label}]  ply {angle:+4.0f} deg  max FI = {FI_k.max():.4f}")
        z += t_ply
    FI_all = np.array(FI_all)
    FI_max = float(FI_all.max()) if FI_all.size > 0 else 0.0
    sf_str = f"{1/FI_max:.2f}" if FI_max > 0 else "inf"
    vprint(f"  [{label}]  Global max FI : {FI_max:.4f}  =>  SF = {sf_str}")
    return FI_max



def panel_buckling_index(eps_cells, mat, panel_width, buckling_k=4.0):
    t = float(mat.H.value)
    E_eff = float(mat.D_ufl.value[0, 0]) * 12 / t**3
    sigma_cr = buckling_k * (np.pi**2 * E_eff) / (12 * (1 - 0.3**2)) * (t / panel_width)**2
    eps_xx = eps_cells[:, 0]
    compressive = eps_xx[eps_xx < 0]
    if len(compressive) == 0:
        return 0.0
    sigma_max = float(mat.A_ufl.value[0, 0]) * np.abs(compressive).max() / t
    return sigma_max / sigma_cr



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
        G_eff = mat.mu
        H_val = mat.h
    else:
        G_eff = mat.G_eff
        H_val = mat.H
    stiffness = G_eff * H_val**3 / h_mesh**2
    stress    = stiffness * drilling_strain
    return stiffness, stress





# CFD LOAD
## OPENFOAM TRACTION PIPLINE
def map_traction(foamfile, femfile, outfile, skin_phys_tags, unit="mm", verbose=False):
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





# MATERIAL
## AS4/3501-6 CARBON EPOXY COMPOSITE
CM_YNG1 = 181.0E9
CM_YNG2 = 10.30E9
CM_G12  = 7.170E9
CM_NU12 = 0.28

# STRENGTH DATA FOR TSAI-WU CRITERION
CM_STRENGTH = {
    "Xt": 1500E6, "Xc":  900E6,
    "Yt":   50E6, "Yc":  200E6,
    "S":    70E6,
}




WRITE_OUTPUT    = True
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

if VERBOSE:
    vprint(f"\n[MESH] CELLS     : {DOMAIN.topology.index_map(TDIM).size_global}")
    vprint(f"[MESH] VERTICES  : {DOMAIN.topology.index_map(0).size_global}")
    for tag in TAG_ALL:
        local_n = len(CELL_TAGS.find(tag))
        global_n = comm.allreduce(local_n, op=MPI.SUM)
        vprint(f"[MESH] TAG {tag} : {global_n} cells")

E1, E2, E3 = local_frame(DOMAIN)


RIB_SPACING     = 0.75
SPAR_HEIGHT     = 0.40
RIB_PANEL       = 0.40
BUCKLING_K_SKIN = 4.0
BUCKLING_K_SPAR = 5.35
BUCKLING_K_RIB  = 6.97

# SKIP_BUCKLING_TAGS = {TAG_RIB0}



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

global_u = comm.allreduce(len(dofs_u[0]), op=MPI.SUM)
global_t = comm.allreduce(len(dofs_t[0]), op=MPI.SUM)
if comm.rank == 0:
    print(f"[BC] Root clamp: {global_u} displacement DOFs  | {global_t} rotation DOFs")




if not MAPFILE.exists():
    map_traction(XDMFFILE, MESHFILE, MAPFILE, TAG_SKIN, MESHUNIT, VERBOSE)
FTraction = load_traction_xdmf(MAPFILE, DOMAIN, VERBOSE)





SKIN_LAYUP  = [0,  0,  45, -45, -45,  45,  0, 0]
SPAR_LAYUP  = [0, 45, -45,   0,   0, -45, 45, 0]
RIB_LAYUP   = [0, 45, -45,  90,  90, -45, 45, 0]

t_skin_const    = fem.Constant(DOMAIN, 0.75e-3)
t_mspar_const   = fem.Constant(DOMAIN, 0.75e-3)
t_tspar_const   = fem.Constant(DOMAIN, 0.75e-3)
t_rib0_const    = fem.Constant(DOMAIN, 0.75e-3)
t_rib750_const  = fem.Constant(DOMAIN, 0.75e-3)
t_rib1500_const = fem.Constant(DOMAIN, 0.75e-3)
t_rib2250_const = fem.Constant(DOMAIN, 0.75e-3)
t_rib3000_const = fem.Constant(DOMAIN, 0.75e-3)

vprint("\n[MAT] SKIN")
MAT_SKIN        = build_laminate(SKIN_LAYUP, t_skin_const,    "SKIN")
vprint("\n[MAT] MAIN SPAR")
MAT_SPAR_MAIN   = build_laminate(SPAR_LAYUP, t_mspar_const,   "MAINSPAR")
vprint("\n[MAT] TAIL SPAR")
MAT_SPAR_TAIL   = build_laminate(SPAR_LAYUP, t_tspar_const,   "TAILSPAR")
vprint("\n[MAT] RIB0")
MAT_RIB0        = build_laminate(RIB_LAYUP,  t_rib0_const,    "RIB0")
vprint("\n[MAT] RIB750")
MAT_RIB750      = build_laminate(RIB_LAYUP,  t_rib750_const,  "RIB750")
vprint("\n[MAT] RIB1500")
MAT_RIB1500     = build_laminate(RIB_LAYUP,  t_rib1500_const, "RIB1500")
vprint("\n[MAT] RIB2250")
MAT_RIB2250     = build_laminate(RIB_LAYUP,  t_rib2250_const, "RIB2250")
vprint("\n[MAT] RIB3000")
MAT_RIB3000     = build_laminate(RIB_LAYUP,  t_rib3000_const, "RIB3000")


MATS = {
    14 : MAT_SKIN,
    15 : MAT_SKIN,
    38 : MAT_RIB0,
    39 : MAT_RIB750,
    40 : MAT_RIB1500,
    41 : MAT_RIB2250,
    42 : MAT_RIB3000,
    43 : MAT_SPAR_MAIN,
    44 : MAT_SPAR_TAIL,
}

dx = ufl.Measure("dx", domain=DOMAIN, subdomain_data=CELL_TAGS)

form_pieces = []
for tag, mat in MATS.items():
    N_t, M_t, Q_t = stress_resultants(mat, eps, kappa, gamma)
    _, drill_t     = drilling_terms(mat, DOMAIN, drilling_strain)
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
    petsc_options_prefix="wing_",
    petsc_options={
        "ksp_type"                  : "preonly",
        "pc_type"                   : "lu",
        "pc_factor_mat_solver_type" : "mumps",
        "snes_type"                 : "newtonls",
        "snes_rtol"                 : 1E-8,
        "snes_atol"                 : 1E-8,
        "snes_max_it"               : 25,
        "snes_monitor"              : "",
        "mat_mumps_icntl_14"        : 80,
        "mat_mumps_icntl_23"        : 4000,
    }
)


def compute_area(*tags):
    form = fem.form(sum(1 * dx(tag) for tag in tags))
    area_local = fem.assemble_scalar(form)
    area = comm.allreduce(area_local, op=MPI.SUM)
    return area

AREA_SKIN       = compute_area(TAG_UPPER, TAG_LOWER)
AREA_RIB0       = compute_area(TAG_RIB0)
AREA_RIB750     = compute_area(TAG_RIB750)
AREA_RIB1500    = compute_area(TAG_RIB1500)
AREA_RIB2250    = compute_area(TAG_RIB2250)
AREA_RIB3000    = compute_area(TAG_RIB3000)
AREA_SPAR_MAIN  = compute_area(TAG_MAINSPAR)
AREA_SPAR_TAIL  = compute_area(TAG_TAILSPAR)



DG0 = fem.functionspace(DOMAIN, ("DG", 0, (3,)))
eps_fn = fem.Function(DG0)
kap_fn = fem.Function(DG0)


K_MAP = {
    TAG_UPPER:    4.0,
    TAG_LOWER:    4.0,
    TAG_MAINSPAR: 5.35,
    TAG_TAILSPAR: 5.35,
    TAG_RIB750:   6.97,
    TAG_RIB1500:  6.97,
    TAG_RIB2250:  6.97,
    TAG_RIB3000:  6.97,
}




# POST-PROCESSING
def compute_tip_deflection(v, domain, tip_y=3.0, tol=0.05):
    disp   = v.sub(0).collapse()
    u_arr  = disp.x.array.reshape(-1, 3)

    Vu     = disp.function_space
    coords = Vu.tabulate_dof_coordinates()

    tip_mask = np.abs(coords[:, 1] - tip_y) < tol
    uz_tip   = np.abs(u_arr[tip_mask, 2]).max() if tip_mask.any() else 0.0
    return comm.allreduce(uz_tip, op=MPI.MAX)



def compute_relative_twist(v, domain, root_y=0.0, tip_y=3.0, tol=0.05):
    rota   = v.sub(1).collapse()
    t_arr  = rota.x.array.reshape(-1, 3)
    Vt     = rota.function_space
    coords = Vt.tabulate_dof_coordinates()

    root_mask = np.abs(coords[:, 1] - root_y) < tol
    tip_mask  = np.abs(coords[:, 1] - tip_y)  < tol

    if rank == 0:
        if tip_mask.any():
            vprint(f"[TWIST DEBUG] theta_x tip = {t_arr[tip_mask, 0].mean():.6f} rad")
            vprint(f"[TWIST DEBUG] theta_y tip = {t_arr[tip_mask, 1].mean():.6f} rad")
            vprint(f"[TWIST DEBUG] theta_z tip = {t_arr[tip_mask, 2].mean():.6f} rad")
        if root_mask.any():
            vprint(f"[TWIST DEBUG] theta_x root= {t_arr[root_mask, 0].mean():.6f} rad")
            vprint(f"[TWIST DEBUG] theta_y root= {t_arr[root_mask, 1].mean():.6f} rad")
            vprint(f"[TWIST DEBUG] theta_z root= {t_arr[root_mask, 2].mean():.6f} rad")

    if tip_mask.any():
        theta_tip_vec  = t_arr[tip_mask].mean(axis=0)
    else:
        theta_tip_vec  = np.zeros(3)

    if root_mask.any():
        theta_root_vec = t_arr[root_mask].mean(axis=0)
    else:
        theta_root_vec = np.zeros(3)

    theta_tip_vec  = comm.allreduce(theta_tip_vec,  op=MPI.SUM)
    theta_root_vec = comm.allreduce(theta_root_vec, op=MPI.SUM)

    n_tip  = comm.allreduce(int(tip_mask.sum()),  op=MPI.SUM)
    n_root = comm.allreduce(int(root_mask.sum()), op=MPI.SUM)

    theta_tip_vec  /= max(n_tip,  1)
    theta_root_vec /= max(n_root, 1)

    delta_theta = theta_tip_vec - theta_root_vec

    twist_y    = abs(delta_theta[1])
    twist_norm = np.linalg.norm(delta_theta)

    if rank == 0:
        vprint(f"[TWIST DEBUG] delta_theta = {np.degrees(delta_theta)} deg")
        vprint(f"[TWIST DEBUG] twist_y     = {np.degrees(twist_y):.4f} deg")
        vprint(f"[TWIST DEBUG] twist_norm  = {np.degrees(twist_norm):.4f} deg")

    return twist_norm

def compute_torsional_moment(domain, traction_fn, elastic_axis_x=0.3):
    x    = ufl.SpatialCoordinate(domain)
    t    = traction_fn
    r_x     = x[0] - elastic_axis_x
    M_tor   = t[2] * r_x

    form      = fem.form(M_tor * ufl.dx)
    M_local   = fem.assemble_scalar(form)
    M_total   = comm.allreduce(M_local, op=MPI.SUM)
    return abs(M_total)







# DESIGN LIMITS
DEFLECTION_LIMIT_M = 0.16
TWIST_LIMIT_RAD    = 0.01
FI_LIMIT           = 0.80
REF_MASS           = 101.1662



# PARETO LOGGER
LOG_FILE = OUTPUT_DIR / f"pareto_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

FIELDNAMES = [
    "iter",
    "t_skin_mm", "t_rib0_mm", "t_rib750_mm", "t_rib1500_mm",
    "t_rib2250_mm", "t_rib3000_mm", "t_mspar_mm", "t_tspar_mm",
    "mass_kg",
    "tip_deflection_mm",
    "relative_twist_deg",
    "fi_max",
    "obj_stiffness",
    "feasible",
]

_iter_count = 0

def init_log():
    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        vprint(f"[LOG] Pareto log => {LOG_FILE}")

def log_iteration(x, obj, fi, defl, twist, mass):
    global _iter_count
    if rank != 0:
        return
    _iter_count += 1
    feasible = int(fi <= FI_LIMIT and
                   defl <= DEFLECTION_LIMIT_M and
                   twist <= TWIST_LIMIT_RAD)
    row = {
        "iter"               : _iter_count,
        "t_skin_mm"          : round(x[0] * 1e3, 4),
        "t_rib0_mm"          : round(x[1] * 1e3, 4),
        "t_rib750_mm"        : round(x[2] * 1e3, 4),
        "t_rib1500_mm"       : round(x[3] * 1e3, 4),
        "t_rib2250_mm"       : round(x[4] * 1e3, 4),
        "t_rib3000_mm"       : round(x[5] * 1e3, 4),
        "t_mspar_mm"         : round(x[6] * 1e3, 4),
        "t_tspar_mm"         : round(x[7] * 1e3, 4),
        "mass_kg"            : round(mass,               4),
        "tip_deflection_mm"  : round(defl * 1e3,         4),
        "relative_twist_deg" : round(np.degrees(twist),  6),
        "fi_max"             : round(fi,                 6),
        "obj_stiffness"      : round(obj,                6),
        "feasible"           : feasible,
    }
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)




# SOLVE FUNCTION
def solve_wing(x):
    t_skin, t_rib0, t_rib750, t_rib1500, t_rib2250, t_rib3000, t_mspar, t_tspar = x

    update_clt_material(MAT_SKIN,      t_skin,    len(SKIN_LAYUP))
    update_clt_material(MAT_RIB0,      t_rib0,    len(RIB_LAYUP) )
    update_clt_material(MAT_RIB750,    t_rib750,  len(RIB_LAYUP) )
    update_clt_material(MAT_RIB1500,   t_rib1500, len(RIB_LAYUP) )
    update_clt_material(MAT_RIB2250,   t_rib2250, len(RIB_LAYUP) )
    update_clt_material(MAT_RIB3000,   t_rib3000, len(RIB_LAYUP) )
    update_clt_material(MAT_SPAR_MAIN, t_mspar,   len(SPAR_LAYUP))
    update_clt_material(MAT_SPAR_TAIL, t_tspar,   len(SPAR_LAYUP))

    problem.solve()

    tip_deflection  = compute_tip_deflection(v, DOMAIN, tip_y=3.0)
    relative_twist  = compute_relative_twist(v, DOMAIN, root_y=0.0, tip_y=3.0)

    u_h, theta_h = ufl.split(v)
    P_proj  = tangent_projection(E1, E2)
    eps_h,_ = membrane_strain(u_h,     P_proj)
    kappa_h = bending_strain(theta_h, E3, P_proj)

    eps_fn.interpolate(fem.Expression(to_voigt(eps_h),   DG0.element.interpolation_points))
    kap_fn.interpolate(fem.Expression(to_voigt(kappa_h), DG0.element.interpolation_points))

    eps0_all  = eps_fn.x.array.reshape(-1, 3)
    kappa_all = kap_fn.x.array.reshape(-1, 3)

    SKIP_BUCKLING = {TAG_RIB0}
    global_fi_max = 0.0

    for tag, mat in [
        (TAG_UPPER,    MAT_SKIN),
        (TAG_LOWER,    MAT_SKIN),
        (TAG_MAINSPAR, MAT_SPAR_MAIN),
        (TAG_TAILSPAR, MAT_SPAR_TAIL),
        (TAG_RIB0,     MAT_RIB0),
        (TAG_RIB750,   MAT_RIB750),
        (TAG_RIB1500,  MAT_RIB1500),
        (TAG_RIB2250,  MAT_RIB2250),
        (TAG_RIB3000,  MAT_RIB3000),
    ]:
        cells = CELL_TAGS.find(tag)
        if len(cells) > 0:
            eps_c = eps0_all[cells]
            kap_c = kappa_all[cells]
            fi_val = _eval_failure_from_arrays(
                eps_c, kap_c, mat, CM_STRENGTH,
                criterion="tsai_wu", label=f"TAG_{tag}"
            )
            panel_width  = (RIB_SPACING if tag in [TAG_UPPER, TAG_LOWER] else
                            SPAR_HEIGHT  if tag in [TAG_MAINSPAR, TAG_TAILSPAR] else
                            RIB_PANEL)
            buckling_val = (0.0 if tag in SKIP_BUCKLING else
                            panel_buckling_index(eps_c, mat, panel_width,
                                                 buckling_k=K_MAP.get(tag, 4.0)))
            combined = max(fi_val, buckling_val)
        else:
            fi_val = buckling_val = combined = 0.0

        global_tag    = comm.allreduce(combined,     op=MPI.MAX)
        tsaiwu_tag    = comm.allreduce(fi_val,       op=MPI.MAX)
        buckling_tag  = comm.allreduce(buckling_val, op=MPI.MAX)
        global_fi_max = max(global_fi_max, global_tag)

        if rank == 0:
            vprint(f"  [TAG_{tag}] TsaiWu={tsaiwu_tag:.4f}  "f"Buckling={buckling_tag:.4f}")

    rho = 1600.0
    mass_total = rho * (
        AREA_SKIN      * t_skin    +
        AREA_RIB0      * t_rib0    +
        AREA_RIB750    * t_rib750  +
        AREA_RIB1500   * t_rib1500 +
        AREA_RIB2250   * t_rib2250 +
        AREA_RIB3000   * t_rib3000 +
        AREA_SPAR_MAIN * t_mspar   +
        AREA_SPAR_TAIL * t_tspar
    )

    obj_mass = mass_total / REF_MASS

    if rank == 0:
        sf = 1 / global_fi_max if global_fi_max > 0 else float("inf")
        vprint("\n==============================")
        vprint(f"[POST] Tip deflection : {tip_deflection*1e3:.2f} mm  "
               f"(limit {DEFLECTION_LIMIT_M*1e3:.1f} mm)")
        vprint(f"[POST] Rel. twist     : {np.degrees(relative_twist):.4f} deg  "
               f"(limit {np.degrees(TWIST_LIMIT_RAD):.2f} deg)")
        vprint(f"[POST] Mass           : {mass_total:.4f} kg")
        vprint(f"[POST] FI max         : {global_fi_max:.4f}  SF={sf:.2f}")
        vprint(f"[POST] obj_mass       : {obj_mass:.4f}")
        vprint("==============================\n")

    return obj_mass, global_fi_max, tip_deflection, relative_twist, mass_total



# NLOPT WRAPPERS
_cache = {"x": None, "obj": None, "fi": None,
          "defl": None, "twist": None, "mass": None}

def _run_and_cache(x):
    if _cache["x"] is not None and np.allclose(x, _cache["x"], rtol=1e-12):
        return
    obj, fi, defl, twist, mass = solve_wing(x)
    _cache.update(x=np.copy(x), obj=obj,
                  fi=fi, defl=defl, twist=twist, mass=mass)
    log_iteration(x, obj, fi, defl, twist, mass)

def objective_mass(x, grad):
    _run_and_cache(x)
    return float(_cache["obj"])

def constraint_fi(x, grad):
    _run_and_cache(x)
    return float(_cache["fi"]) - FI_LIMIT

def constraint_deflection(x, grad):
    _run_and_cache(x)
    return float(_cache["defl"]) - DEFLECTION_LIMIT_M

def constraint_twist(x, grad):
    _run_and_cache(x)
    return float(_cache["twist"]) - TWIST_LIMIT_RAD



# NLOPT SETUP
N_VARS = 8
T_MAX  = 20.0e-3
T_INIT = 12.0e-3

T_MIN_SKIN = 1.0e-3
T_MIN_RIB  = 2.0e-3
T_MIN_SPAR = 1.0e-3

opt = nlopt.opt(nlopt.LN_COBYLA, N_VARS)
opt.set_lower_bounds([
    T_MIN_SKIN,
    T_MIN_RIB,
    T_MIN_RIB,
    T_MIN_RIB,
    T_MIN_RIB,
    T_MIN_RIB,
    T_MIN_SPAR,
    T_MIN_SPAR,
])
opt.set_upper_bounds([T_MAX] * N_VARS)
opt.set_xtol_rel(1e-5)
opt.set_ftol_rel(1e-5)
opt.set_maxeval(300)

opt.set_min_objective(objective_mass)


opt.add_inequality_constraint(constraint_fi,          1e-4)
opt.add_inequality_constraint(constraint_deflection,  1e-4)
opt.add_inequality_constraint(constraint_twist,       1e-5)

x0 = [T_INIT] * N_VARS





# RUN
init_log()

if QUICK_TEST:
    vprint("\n>>> QUICK TEST <<<")
    obj, fi, defl, twist, mass = solve_wing(x0)
    log_iteration(x0, obj, fi, defl, twist, mass)

else:
    vprint("\n>>> NLOPT COBYLA OPTIMISATION START <<<")
    vprint(f"    Objective : minimize  mass / REF_MASS")
    vprint(f"    s.t.      : FI         <= {FI_LIMIT}")
    vprint(f"                deflection <= {DEFLECTION_LIMIT_M*1e3:.0f} mm")
    vprint(f"                twist      <= {np.degrees(TWIST_LIMIT_RAD):.2f} deg")

    xopt = opt.optimize(x0)

    if rank == 0:
        vprint("\n========== OPTIMUM ==========")
        labels = ["Skin", "Rib0", "Rib750", "Rib1500",
                  "Rib2250", "Rib3000", "MainSpar", "TailSpar"]
        for lbl, t in zip(labels, xopt):
            vprint(f"  {lbl:>10s} : {t*1e3:.3f} mm")
        vprint(f"  Opt obj     : {opt.last_optimum_value():.6f}")
        vprint(f"  NLopt status: {opt.last_optimize_result()}")

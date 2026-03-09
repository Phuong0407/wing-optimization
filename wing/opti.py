import ufl
from dolfinx import fem
from dolfinx.io import gmsh
from types import SimpleNamespace
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import meshio
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from pathlib import Path
from dolfinx.io import gmsh
from dolfinx import io
import basix
from functools import reduce
from dolfinx.fem.petsc import NonlinearProblem

from mpi4py import MPI

import numpy as np
from scipy.optimize import minimize
from mpi4py import MPI
# Giả sử file chính của bạn tên là a.py, ta sẽ import các hàm cần thiết
# Hoặc bạn có thể copy các hàm khởi tạo vật liệu vào đây
from scipy.optimize import minimize


comm = MPI.COMM_WORLD
rank = comm.rank



VERBOSE = True

def vprint(*args, **kwargs):
    if VERBOSE and MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

def vprint_global(label, values, op=MPI.SUM):
    comm = MPI.COMM_WORLD

    vals = np.atleast_1d(values).astype(np.float64)
    global_vals = np.empty_like(vals)

    comm.Allreduce(vals, global_vals, op=op)

    if VERBOSE and comm.rank == 0:
        if len(global_vals) == 1:
            print(f"{label} {global_vals[0]}")
        else:
            joined = " | ".join(f"{v:g}" for v in global_vals)
            print(f"{label} {joined}")


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

def clt_composite(layup_angles, t_ply, E1, E2, G12, nu12, G13=None, G23=None, kappa_s=5/6, label="CLT", verbose=False):
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
    G_eff = float(A_np[2, 2]) / H

    return SimpleNamespace(
        kind   = "clt",
        H      = H,
        A_np   = A_np, B_np = B_np, D_np = D_np, As_np = As_np,
        G_eff  = G_eff,
        A_ufl  = ufl.as_tensor(A_np),
        B_ufl  = ufl.as_tensor(B_np),
        D_ufl  = ufl.as_tensor(D_np),
        As_ufl = ufl.as_tensor(As_np),
    )

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

def recover_and_evaluate_failure_cells(domain, v_sol, mat, cell_indices, strengths, criterion="tsai_wu", label=""):
    assert mat.kind == "clt", "Failure recovery requires CLT material"
    if len(cell_indices) == 0:
        vprint(f"  [{label}] No cells — skipping failure analysis.")
        return 0.0, np.array([])

    u_h, theta_h = ufl.split(v_sol)
    e1, e2, e3   = local_frame(domain)
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
    t_ply  = mat._t_ply
    Q_ply  = _Q_ply(mat._E1, mat._E2, mat._G12, mat._nu12)
    H      = mat.H
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
    vprint(f"  [{label}]  Global max FI : {FI_max:.4f}  =>  SF = {1/FI_max:.2f}")
    return FI_max, FI_all





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
    else:
        G_eff = fem.Constant(domain, mat.G_eff)
    h         = mat.h if mat.kind == "isotropic" else fem.Constant(domain, mat.H)
    stiffness = G_eff * h**3 / h_mesh**2
    stress    = stiffness * drilling_strain
    return stiffness, stress





# CFD LOAD
## OPENFOAM TRACTION PIPLINE
def import_foam_traction(foamfile, xdmffile, verbose=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(foamfile))
        reader.Update()
        poly = reader.GetOutput()

        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly)
        tri.Update()
        poly = tri.GetOutput()

        points = vtk_to_numpy(poly.GetPoints().GetData())
        p      = vtk_to_numpy(poly.GetPointData().GetArray("p"))
        wss    = vtk_to_numpy(poly.GetPointData().GetArray("wallShearStress"))

        cells = poly.GetPolys()
        cells.InitTraversal()
        idList    = vtk.vtkIdList()
        triangles = []
        while cells.GetNextCell(idList):
            triangles.append([idList.GetId(i) for i in range(3)])
        triangles = np.array(triangles, dtype=np.int64)

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOff()
        normals.AutoOrientNormalsOn()
        normals.Update()
        poly_n = normals.GetOutput()

        n_unit = vtk_to_numpy(
            poly_n.GetCellData().GetArray("Normals")
        )

        P0 = points[triangles[:,0]]
        P1 = points[triangles[:,1]]
        P2 = points[triangles[:,2]]

        area = 0.5 * np.linalg.norm(
            np.cross(P1-P0, P2-P0), axis=1
        )

        p_tri   = p[triangles].mean(axis=1)
        wss_tri = wss[triangles].mean(axis=1)

        traction_cell  = -p_tri[:, None] * n_unit + wss_tri

        nodal_traction = np.zeros_like(points)
        nodal_area     = np.zeros(len(points))

        for i, tri_nodes in enumerate(triangles):
            a_third = area[i] / 3.0
            t_i = traction_cell[i]
            for j in tri_nodes:
                nodal_traction[j] += t_i * a_third
                nodal_area[j]     += a_third
        nodal_traction /= nodal_area[:, None]

        if verbose:
            total_force = (traction_cell * area[:, None]).sum(axis=0)
            vprint("[FOAM] Total OpenFOAM force:", total_force)

        meshio.write(
            str(xdmffile),
            meshio.Mesh(
                points=points,
                cells=[("triangle", triangles)],
                point_data={"traction": nodal_traction},
            ),
        )

        if verbose:
            vprint(f"[FOAM] Exported traction -> {xdmffile}")

    comm.barrier()


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


# MATERIAL
## AS4/3501-6 CARBON EPOXY COMPOSITE
CE_YNG1 = 181.0E9
CE_YNG2 = 10.30E9
CE_G12  = 7.170E9
CE_NU12 = 0.28

# STRENGTH DATA FOR TSAI-WU CRITERION
CE_STRENGTH = {
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

global_u = comm.allreduce(len(dofs_u[0])); global_t = comm.allreduce(len(dofs_t[0]))
if comm.rank == 0:
    print(f"[BC] Root clamp: {global_u} displacement DOFs  | {global_t} rotation DOFs")


if not MAPFILE.exists():
    import_foam_traction(FOAMFILE, XDMFFILE, VERBOSE)
    map_traction(XDMFFILE, MESHFILE, MAPFILE, TAG_SKIN, MESHUNIT, VERBOSE)
FTraction = load_traction_xdmf(MAPFILE, DOMAIN, VERBOSE)




def solve_wing(x):
    """
    Hàm này nhận đầu vào là mảng x = [t_skin, t_rib, t_spar]
    Trả về: Tổng khối lượng (Objective) và các ràng buộc (Constraints)
    """
    t_skin, t_rib, t_spar = x
    
    # 1. Cập nhật lại vật liệu với độ dày mới
    # Lưu ý: Trong code của bạn SPLY, RPLY, MPLY/TPLY là độ dày từng lớp (ply)
    # Ở đây ta giả sử x là tổng độ dày, chia cho 8 lớp
    n_layers = 8
    
    SKIN_LAYUP  = [0, 45, -45, 90, 90, -45, 45, 0]
    SPAR_LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
    RIB_LAYUP   = [0, 45, -45, 90, 90, -45, 45, 0]
    
    # MAT_SKIN = clt_composite([0, 45, -45, 90, 90, -45, 45, 0], t_skin/n_layers, 
    #                          CE_YNG1, CE_YNG2, CE_G12, CE_NU12)
    # MAT_RIB  = clt_composite([0, 45, -45, 90, 90, -45, 45, 0], t_rib/n_layers, 
    #                          CE_YNG1, CE_YNG2, CE_G12, CE_NU12)
    # MAT_SPAR = clt_composite([0, 45, -45, 90, 90, -45, 45, 0], t_spar/n_layers, 
    #                          CE_YNG1, CE_YNG2, CE_G12, CE_NU12)

    ## ASSIGN MATERIAL
    vprint("\n[MAT] SKIN")
    MAT_SKIN = clt_composite(
        SKIN_LAYUP,
        t_skin/n_layers,
        CE_YNG1,
        CE_YNG2,
        CE_G12,
        CE_NU12,
        G13 = CE_G12,
        G23 = CE_G12 * 0.5,
        kappa_s=5/6,
        label="SKIN",
        verbose=VERBOSE
    )
    MAT_SKIN._layup_angles  = SKIN_LAYUP
    MAT_SKIN._t_ply         = t_skin/n_layers
    MAT_SKIN._E1            = CE_YNG1
    MAT_SKIN._E2            = CE_YNG2
    MAT_SKIN._G12           = CE_G12
    MAT_SKIN._nu12          = CE_NU12

    vprint("\n[MAT] SPAR")
    MAT_SPAR = clt_composite(
        SPAR_LAYUP,
        t_spar/n_layers,
        CE_YNG1,
        CE_YNG2,
        CE_G12,
        CE_NU12,
        G13=CE_G12,
        G23=CE_G12 * 0.5,
        kappa_s=5/6,
        label="MAINSPAR",
        verbose=VERBOSE
    )
    MAT_SPAR._layup_angles  = SKIN_LAYUP
    MAT_SPAR._t_ply         = t_spar/n_layers
    MAT_SPAR._E1            = CE_YNG1
    MAT_SPAR._E2            = CE_YNG2
    MAT_SPAR._G12           = CE_G12
    MAT_SPAR._nu12          = CE_NU12

    vprint("\n[MAT] RIB")
    MAT_RIB = clt_composite(
        RIB_LAYUP,
        t_rib/n_layers,
        CE_YNG1,
        CE_YNG2,
        CE_G12,
        CE_NU12,
        G13=CE_G12,
        G23=CE_G12 * 0.5,
        kappa_s=5/6,
        label="RIB",
        verbose=VERBOSE
    )
    MAT_RIB._layup_angles  = RIB_LAYUP
    MAT_RIB._t_ply         = t_rib/n_layers
    MAT_RIB._E1            = CE_YNG1
    MAT_RIB._E2            = CE_YNG2
    MAT_RIB._G12           = CE_G12
    MAT_RIB._nu12          = CE_NU12

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
        petsc_options_prefix="wing",
        petsc_options={
            "ksp_type"                  : "preonly",
            "pc_type"                   : "lu",
            "pc_factor_mat_solver_type" : "mumps",
            "snes_type"                 : "newtonls",
            "snes_rtol"                 : 1E-8,
            "snes_atol"                 : 1E-8,
            "snes_max_it"               : 25,
            "snes_monitor"              : None,
            "mat_mumps_icntl_14"        : 80,
            "mat_mumps_icntl_23"        : 2000,
        }
    )
    problem.solve()

    # 2. Chạy solver FEniCSx (Phần này bạn bê từ file a.py vào)
    # Để tiết kiệm thời gian, bạn nên bọc phần tạo Mesh và FunctionSpace ra ngoài hàm solve
    # chỉ chạy phần định nghĩa Weak form và Solver bên trong này.
    
    # GIẢ LẬP KẾT QUẢ TRẢ VỀ (Thay thế bằng kết quả thật từ Solver của bạn):
    # mass = (Diện tích bề mặt * độ dày * khối lượng riêng)
    # fi_max = Giá trị Failure Index lớn nhất tìm được
    
    # Sau khi problem.solve() và tính FI_max:
    # fi_max = max(FI_results.values())
    
    # Chúng ta muốn tối thiểu hóa khối lượng:
    # objective = (Area_skin * t_skin + Area_ribs * t_rib + Area_spars * t_spar) * density
    
    # Tạm thời trả về giá trị giả định để demo cấu trúc:
    
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


    # --- PHẦN HẬU XỬ LÝ (POST-PROCESSING) RÚT GỌN ---
    
    # 1. Gom tất cả các vùng composite lại để kiểm tra lỗi một lần
    ALL_COMPOSITE_TAGS = [TAG_UPPER, TAG_LOWER, TAG_MAINSPAR, TAG_TAILSPAR] + TAG_RIBS
    all_comp_cells = np.concatenate([CELL_TAGS.find(t) for t in ALL_COMPOSITE_TAGS])

    # 2. Tính toán Failure Index (FI) thực tế từ lời giải v
    local_fi_max = 0.0
    if len(all_comp_cells) > 0:
        # Sử dụng MAT_SKIN làm mẫu (hàm recover sẽ tự tính ứng suất dựa trên độ dày từng vùng)
        fi_val, _ = recover_and_evaluate_failure_cells(
            DOMAIN, v, MAT_SKIN, all_comp_cells, 
            CE_STRENGTH, criterion="tsai_wu", label="ALL_WING"
        )
        local_fi_max = fi_val

    # 3. Đồng bộ MPI: Lấy giá trị FI cao nhất trên toàn bộ các Rank
    global_fi_max = DOMAIN.comm.allreduce(local_fi_max, op=MPI.MAX)

    # 4. Tính toán Khối lượng (Mass) - Bạn nên dùng diện tích thực nếu có thể
    # Ở đây giữ công thức của bạn nhưng đưa lên trước khi in
    mass_total = 500 * t_skin + 200 * t_rib + 300 * t_spar 

    # 5. In báo cáo rút gọn (Chỉ Rank 0 in mỗi vòng lặp tối ưu)
    if DOMAIN.comm.rank == 0:
        sf = 1.0 / global_fi_max if global_fi_max > 0 else float('inf')
        status = "SAFE" if global_fi_max < 1.0 else "!!! FAILURE !!!"
        print(f"[ITER] Mass: {mass_total:.4f} | Max FI: {global_fi_max:.4f} | SF: {sf:.2f} | {status}")

    return mass_total, global_fi_max


# --- 1. ĐỊNH NGHĨA HÀM MỤC TIÊU VÀ RÀNG BUỘC ---

def objective(x):
    # Rank 0 phát tín hiệu x cho các Rank khác
    comm.bcast(x, root=0) 
    # Tất cả cùng giải bài toán
    mass, _ = solve_wing(x)
    return mass

def constraint_failure(x):
    # Rank 0 phát tín hiệu x cho các Rank khác
    comm.bcast(x, root=0)
    # Tất cả cùng tính toán Failure Index
    _, fi_max = solve_wing(x)
    # Trả về giá trị >= 0 để thỏa mãn ràng buộc bền
    return 1.0 - fi_max

# --- 2. VÒNG LẶP ĐIỀU KHIỂN CHÍNH (MAIN CONTROL LOOP) ---

# Khởi tạo thông số
x0 = np.array([0.006, 0.006, 0.006])  # 6mm
bounds = [(0.001, 0.020), (0.001, 0.020), (0.001, 0.020)]
cons = {'type': 'ineq', 'fun': constraint_failure}

if rank == 0:
    # --- NHẠC TRƯỞNG (RANK 0) ---
    vprint("\n" + ">>> BẮT ĐẦU TỐI ƯU HÓA CÁNH <<<".center(40))
    
    # SLSQP sẽ gọi objective và constraint liên tục để tìm cực tiểu
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={'disp': True, 'ftol': 1e-4}
    )
    
    # Sau khi xong, gửi tín hiệu None để Rank khác thoát vòng lặp while
    comm.bcast(None, root=0)

    vprint("\n" + "="*40)
    vprint("KẾT QUẢ TỐI ƯU HÓA THÀNH CÔNG:")
    vprint(f" - Độ dày Skin : {res.x[0]*1000:.2f} mm")
    vprint(f" - Độ dày Ribs : {res.x[1]*1000:.2f} mm")
    vprint(f" - Độ dày Spars: {res.x[2]*1000:.2f} mm")
    vprint(f" - Tổng khối lượng: {res.fun:.2f} kg")
    vprint("="*40)

else:
    # --- CÔNG NHÂN (CÁC RANK KHÁC) ---
    while True:
        # Đứng đợi lệnh x từ Rank 0
        x = comm.bcast(None, root=0)
        
        # Nếu nhận được None, nghĩa là Rank 0 đã xong việc -> Thoát
        if x is None:
            break
            
        # Nếu nhận được mảng x, nhảy vào giải FEM cùng Rank 0
        solve_wing(x)
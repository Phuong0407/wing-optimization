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
comm = MPI.COMM_WORLD

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

## ISOTROPIC MATERIAL
def isotropic_material(thickness, young, poisson, domain):
    h        = fem.Constant(domain, float(thickness))
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

def von_mises_iso_cells(domain, v_sol, mat, cell_indices, label=""):
    assert mat.kind == "isotropic"
    if len(cell_indices) == 0:
        vprint(f"  [{label}] No cells — skipping Von Mises analysis.")
        return 0.0

    u_h, theta_h = ufl.split(v_sol)
    e1, e2, e3 = local_frame(domain)
    P = tangent_projection(e1, e2)
    eps_h, _ = membrane_strain(u_h, P)
    kappa_h = bending_strain(theta_h, e3, P)

    DG0 = fem.functionspace(domain, ("DG", 0, (3,)))
    eps_fn = fem.Function(DG0)
    kap_fn = fem.Function(DG0)
    eps_fn.interpolate(fem.Expression(to_voigt(eps_h),   DG0.element.interpolation_points))
    kap_fn.interpolate(fem.Expression(to_voigt(kappa_h), DG0.element.interpolation_points))

    eps0_vals  = eps_fn.x.array.reshape(-1, 3)[cell_indices]
    kappa_vals = kap_fn.x.array.reshape(-1, 3)[cell_indices]

    E_val  = float(mat.E.value)
    nu_val = float(mat.nu.value)
    h_val  = float(mat.h.value)
    lps    = E_val * nu_val / (1 - nu_val**2)
    mu_val = E_val / (2 * (1 + nu_val))

    def constitutive(e_v):
        s11 = (lps + 2*mu_val) * e_v[:, 0] + lps * e_v[:, 1]
        s22 =  lps             * e_v[:, 0] + (lps + 2*mu_val) * e_v[:, 1]
        s12 =  mu_val          * e_v[:, 2]
        return s11, s22, s12

    e_mid = eps0_vals
    e_top = eps0_vals + (h_val / 2.0) * kappa_vals
    e_bot = eps0_vals - (h_val / 2.0) * kappa_vals

    vm_list = []
    for e_v, fiber in [(e_mid, "mid"), (e_top, "top"), (e_bot, "bot")]:
        s11, s22, s12 = constitutive(e_v)
        vm = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)
        vm_max = vm.max()
        vm_list.append(vm_max)
        vprint(f"  [{label}]  fiber={fiber}  max sigma_vm = {vm_max/1e6:.2f} MPa")

    return max(vm_list)


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

## ALUMINIUM 2024-T3
AL_YNG = 73.1E9
AL_NU  = 0.33
AL_YS  = 324E6

## PLY THICKNESS
SPLY = 0.75E-3
MPLY = 0.75E-3
TPLY = 0.75E-3
RPLY = 0.75E-3

## LAYUP CONFIGURATION
SKIN_LAYUP  = [0, 45, -45, 90, 90, -45, 45, 0]
MSPAR_LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
TSPAR_LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
RIB_LAYUP   = [0, 45, -45, 90, 90, -45, 45, 0]

## ASSIGN MATERIAL
vprint("\n[MAT] SKIN")
MAT_SKIN = clt_composite(
    SKIN_LAYUP,
    SPLY,
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
MAT_SKIN._t_ply         = SPLY
MAT_SKIN._E1            = CE_YNG1
MAT_SKIN._E2            = CE_YNG2
MAT_SKIN._G12           = CE_G12
MAT_SKIN._nu12          = CE_NU12

vprint("\n[MAT] MAIN SPAR")
MAT_MAINSPAR = clt_composite(
    MSPAR_LAYUP,
    MPLY,
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
MAT_MAINSPAR._layup_angles  = SKIN_LAYUP
MAT_MAINSPAR._t_ply         = SPLY
MAT_MAINSPAR._E1            = CE_YNG1
MAT_MAINSPAR._E2            = CE_YNG2
MAT_MAINSPAR._G12           = CE_G12
MAT_MAINSPAR._nu12          = CE_NU12

vprint("\n[MAT] TAIL SPAR")
MAT_TAILSPAR = clt_composite(
    TSPAR_LAYUP,
    TPLY,
    CE_YNG1,
    CE_YNG2,
    CE_G12,
    CE_NU12,
    G13=CE_G12,
    G23=CE_G12 * 0.5,
    kappa_s=5/6,
    label="TAILSPAR",
    verbose=VERBOSE
)
MAT_TAILSPAR._layup_angles  = MSPAR_LAYUP
MAT_TAILSPAR._t_ply         = SPLY
MAT_TAILSPAR._E1            = CE_YNG1
MAT_TAILSPAR._E2            = CE_YNG2
MAT_TAILSPAR._G12           = CE_G12
MAT_TAILSPAR._nu12          = CE_NU12

vprint("\n[MAT] RIB")
MAT_RIB = clt_composite(
    RIB_LAYUP,
    RPLY,
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
MAT_RIB._t_ply         = SPLY
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
    43 : MAT_MAINSPAR,
    44 : MAT_TAILSPAR,
}










VERBOSE         = True
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



## COMPUTE LOCAL FRAME
E1, E2, E3 = local_frame(DOMAIN)

if WRITE_OUTPUT:
    RESULTS_FOLDER = Path("Result")
    RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
    with io.VTKFile(MPI.COMM_WORLD, RESULTS_FOLDER / "LocalFrame.pvd", "w") as VTKFILE:
        VTKFILE.write_mesh(DOMAIN)
        VTKFILE.write_function(E1, 0.0)
        VTKFILE.write_function(E2, 0.0)
        VTKFILE.write_function(E3, 0.0)



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




# AERODYNAMIC TRACTION
# import_foam_traction(FOAMFILE, XDMFFILE, VERBOSE)
map_traction(XDMFFILE, MESHFILE, MAPFILE, TAG_SKIN, MESHUNIT, VERBOSE)
FTraction = load_traction_xdmf(MAPFILE, DOMAIN, VERBOSE)





# WEAK FORMULATION
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


# SOLVER
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


## TSAI-WU FAILURE FOR COMPOSITE REGIONS
vprint("="*60)
vprint("[POST] COMPOSITE FAILURE (TSAI-WU)")
vprint("="*60)

FI_results = {}

for tag, mat, lbl in [
    (TAG_SKIN,     MAT_SKIN,     "SKIN"),
    (TAG_MAINSPAR, MAT_MAINSPAR, "MAINSPAR"),
    (TAG_TAILSPAR, MAT_TAILSPAR, "TAILSPAR"),
    (TAG_RIBS,     MAT_RIB,      "RIBS"),
]:
    cells = (
        np.concatenate([CELL_TAGS.find(t) for t in tag])
        if isinstance(tag, list)
        else CELL_TAGS.find(tag)
    )

    vprint(f"\n  {lbl}  ({len(cells)} cells)")
    FI_max, _ = recover_and_evaluate_failure_cells(
        DOMAIN, v, mat, cells,
        CE_STRENGTH,
        criterion="tsai_wu",
        label=lbl
    )
    FI_results[lbl] = FI_max


## SUMMARY TABLE
vprint("\n" + "="*60)
vprint("[POST] SUMMARY")
vprint("="*60)

vprint(f"  Max displacement magnitude : {u_max*1e3:.3f} mm\n")

vprint("  Composite regions — Tsai-Wu:")
for lbl, fi in FI_results.items():
    sf   = 1/fi if fi > 0 else float("inf")
    flag = "  *** FAILURE ***" if fi >= 1.0 else ""
    vprint(f"    {lbl:<12}  FI = {fi:.4f}   SF = {sf:.2f}{flag}")
vprint()


## EXPORT RESULTS
if WRITE_OUTPUT:
    RESULT_FOLDER = Path("Result")
    RESULT_FOLDER.mkdir(exist_ok=True, parents=True)

    Vout = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))

    disp_out = fem.Function(Vout)
    disp_out.interpolate(disp)
    disp_out.name = "Displacement"

    rota_out = fem.Function(Vout)
    rota_out.interpolate(rota)
    rota_out.name = "Rotation"

    with io.XDMFFile(comm, RESULT_FOLDER / "results.xdmf", "w") as xdmf:
        xdmf.write_mesh(DOMAIN)
        xdmf.write_function(disp_out)
        xdmf.write_function(rota_out)

    vprint(f"[EXPORT] Results -> {RESULT_FOLDER / 'results.xdmf'}")

    DG0 = fem.functionspace(DOMAIN, ("DG", 0))
    mat_field = fem.Function(DG0, name="MaterialTag")

    for tag in TAG_ALL:
        cells = CELL_TAGS.find(tag)
        mat_field.x.array[cells] = float(tag)

    mat_field.x.scatter_forward()

    with io.XDMFFile(comm, RESULT_FOLDER / "material_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(DOMAIN)
        xdmf.write_function(mat_field)

    vprint(f"[EXPORT] Material tags -> {RESULT_FOLDER / 'material_tags.xdmf'}")

problem.solver.destroy()

import ufl
from dolfinx import fem
from mpi4py import MPI
from dolfinx.io import gmsh
import vtk
import meshio
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
from dolfinx import mesh, fem, io, plot
from scipy.interpolate import RBFInterpolator
from types import SimpleNamespace
from functools import reduce
import basix
import dolfinx.fem.petsc
from dolfinx.fem.petsc import NonlinearProblem
from pathlib import Path
import sys


# LOCAL FRAME
def normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))

def _local_frame_ufl(domain):
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

def local_frame(domain, gdim):
    FRAME = _local_frame_ufl(domain)
    VT    = fem.functionspace(domain, ("DG", 0, (gdim,)))
    V0, _ = VT.sub(0).collapse()
    BASIS = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(gdim)]
    for i in range(gdim):
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


# ISOTROPIC MATERIAL
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

# CLT COMPOSITE MATERIAL
def _Q_ply(E1, E2, G12, nu12):
    nu21 = nu12 * E2 / E1
    d    = 1 - nu12 * nu21
    return np.array([
        [ E1 / d,         nu12 * E2 / d, 0   ],
        [ nu12 * E2 / d,  E2 / d,        0   ],
        [ 0,              0,             G12 ],
    ])

def _Qbar_ply(Q, angle_deg):
    a    = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    T = np.array([
        [ c**2,  s**2,   2*c*s       ],
        [ s**2,  c**2,  -2*c*s       ],
        [-c*s,   c*s,    c**2 - s**2 ],
    ])
    R    = np.diag([1.0, 1.0, 2.0])
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

def clt_material(layup_angles, t_ply, E1, E2, G12, nu12,
                 G13=None, G23=None, kappa_s=5/6, label="CLT"):
    if G13 is None: G13 = G12
    if G23 is None: G23 = G12 * 0.5

    A_np, B_np, D_np, H = compute_ABD(layup_angles, t_ply, E1, E2, G12, nu12)

    max_B = np.abs(B_np).max()
    print(f"[{label}] Layup   : {layup_angles}")
    print(f"[{label}] H       : {H*1e3:.2f} mm")
    print(f"[{label}] max|B|  : {max_B:.2e}  "
          f"{'SYMMETRIC' if max_B < 1e-6 * A_np.max() else 'NON-SYMMETRIC'}")
    print(f"[{label}] A11     : {A_np[0,0]/1e6:.2f} MPa·m")
    print(f"[{label}] D11     : {D_np[0,0]:.4f} N·m^2")

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


# VOIGT NOTATION
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
        G_eff = mat.mu
    else:
        G_eff = fem.Constant(domain, mat.G_eff)
    h         = mat.h if mat.kind == "isotropic" else fem.Constant(domain, mat.H)
    stiffness = G_eff * h**3 / h_mesh**2
    stress    = stiffness * drilling_strain
    return stiffness, stress



# OPENFOAM TRACTION PIPELINE
def import_foam_traction(foamfile, xdmffile, verbose=False):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(foamfile)
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
    triangles = np.array(triangles)

    P0 = points[triangles[:, 0]]
    P1 = points[triangles[:, 1]]
    P2 = points[triangles[:, 2]]
    v1 = P1 - P0
    v2 = P2 - P0
    n  = np.cross(v1, v2)

    area   = 0.5 * np.linalg.norm(n, axis=1)
    n_unit = n / np.linalg.norm(n, axis=1)[:, None]

    p_tri   = p[triangles].mean(axis=1)
    wss_tri = wss[triangles].mean(axis=1)

    traction_cell  = -p_tri[:, None] * n_unit + wss_tri

    nodal_traction = np.zeros_like(points)
    nodal_area     = np.zeros(len(points))

    for i, tri in enumerate(triangles):
        for j in tri:
            nodal_traction[j] += traction_cell[i] * area[i] / 3.0
            nodal_area[j]     += area[i] / 3.0
    nodal_traction /= nodal_area[:, None]

    if verbose:
        total_force = (traction_cell * area[:, None]).sum(axis=0)
        print("[FOAM] Total OpenFOAM force:", total_force)

    meshio.write(xdmffile, meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        point_data={"traction": nodal_traction},
    ))
    print(f"[FOAM] Exported traction -> {xdmffile}")

# def map_traction(foamfile, femfile, outfile):
#     fm  = meshio.read(foamfile)
#     ft  = fm.point_data["traction"]
#     fp  = fm.points

#     sm  = meshio.read(femfile)
#     sp  = sm.points * 1e-3
#     st  = sm.cells_dict["triangle"]

#     interp         = RBFInterpolator(fp, ft, kernel="thin_plate_spline", neighbors=20)
#     nodal_traction = interp(sp)

#     trib_area = np.zeros(len(sp))
#     for tri in st:
#         P0, P1, P2 = sp[tri[0]], sp[tri[1]], sp[tri[2]]
#         area = 0.5 * np.linalg.norm(np.cross(P1 - P0, P2 - P0))
#         for j in tri:
#             trib_area[j] += area / 3.0
#     nodal_force = nodal_traction * trib_area[:, np.newaxis]

#     foam_tri  = fm.cells_dict["triangle"]
#     foam_area = 0.5 * np.linalg.norm(
#         np.cross(fp[foam_tri[:, 1]] - fp[foam_tri[:, 0]],
#                  fp[foam_tri[:, 2]] - fp[foam_tri[:, 0]]), axis=1)
#     foam_tract = ft[foam_tri].mean(axis=1)
#     foam_force = (foam_tract * foam_area[:, np.newaxis]).sum(axis=0)
#     fem_force  = nodal_force.sum(axis=0)
#     err        = np.linalg.norm(fem_force - foam_force) / np.linalg.norm(foam_force)
#     print(f"[MAP] FOAM force [N] : {foam_force}")
#     print(f"[MAP] FEM  force [N] : {fem_force}")
#     print(f"[MAP] Force error    : {err*100:.2f}%")

#     meshio.write(outfile, meshio.Mesh(
#         points=sp,
#         cells=[("triangle", st)],
#         point_data={"traction": nodal_traction},
#     ))
#     print(f"[MAP] Saved =====> {outfile}")
#     return nodal_force


# # BEFORE
# def map_traction(foamfile, femfile, outfile):
#     fm  = meshio.read(foamfile)
#     ft  = fm.point_data["traction"]
#     fp  = fm.points

#     sm  = meshio.read(femfile)
#     sp  = sm.points * 1e-3
#     st  = sm.cells_dict["triangle"]

#     interp         = RBFInterpolator(fp, ft, kernel="thin_plate_spline", neighbors=20)
#     nodal_traction = interp(sp)
#     ...


# AFTER
def map_traction(foamfile, femfile, outfile, skin_phys_tags=(14, 15)):
    """
    Map OpenFOAM traction to the skin nodes only.
    Rib and spar nodes are excluded — they are interior structural
    elements with no direct aerodynamic traction, and their coplanar
    geometry causes the RBF polynomial matrix to be singular.

    skin_phys_tags: physical surface tags for upper and lower skin
                    (14, 15) matching wing.geo / model_import.py
    """
    fm  = meshio.read(foamfile)
    ft  = fm.point_data["traction"]
    fp  = fm.points

    sm  = meshio.read(femfile)

    # ── Collect skin triangle connectivity and skin node indices ──────
    # meshio stores physical tag per cell in sm.cell_data["gmsh:physical"]
    skin_triangles = []
    all_triangles  = sm.cells_dict.get("triangle", np.zeros((0, 3), dtype=int))

    # Find which cells belong to skin physical groups
    phys_data = None
    for key in ["gmsh:physical", "cell_tags"]:
        if key in sm.cell_data:
            phys_data = sm.cell_data[key]
            break

    if phys_data is not None:
        # cell_data is a list (one array per cell block)
        for block_cells, block_tags in zip(
            [c.data for c in sm.cells if c.type == "triangle"],
            [d for d, c in zip(phys_data, sm.cells) if c.type == "triangle"]
        ):
            mask = np.isin(block_tags, list(skin_phys_tags))
            skin_triangles.append(block_cells[mask])
        skin_triangles = np.concatenate(skin_triangles) if skin_triangles \
                         else all_triangles
    else:
        # Fallback: no tag data — use all triangles (skin-only mesh)
        print("[MAP] Warning: no physical tag data in mesh — "
              "using all triangles")
        skin_triangles = all_triangles

    # Unique skin node indices and their coordinates
    skin_node_idx       = np.unique(skin_triangles)
    sp_all              = sm.points * 1e-3          # all mesh points in metres
    sp                  = sp_all[skin_node_idx]     # skin points only

    # ── RBF interpolation (FOAM -> skin nodes) ────────────────────────
    interp         = RBFInterpolator(fp, ft, kernel="thin_plate_spline",
                                     neighbors=20)
    skin_traction  = interp(sp)

    # ── Expand back to full mesh size (zero for non-skin nodes) ───────
    nodal_traction             = np.zeros((len(sp_all), 3))
    nodal_traction[skin_node_idx] = skin_traction

    # ── Force balance check ───────────────────────────────────────────
    trib_area = np.zeros(len(sp_all))
    for tri in skin_triangles:
        P0, P1, P2 = sp_all[tri[0]], sp_all[tri[1]], sp_all[tri[2]]
        area = 0.5 * np.linalg.norm(np.cross(P1 - P0, P2 - P0))
        for j in tri:
            trib_area[j] += area / 3.0
    nodal_force = nodal_traction * trib_area[:, np.newaxis]

    foam_tri  = fm.cells_dict["triangle"]
    foam_area = 0.5 * np.linalg.norm(
        np.cross(fp[foam_tri[:, 1]] - fp[foam_tri[:, 0]],
                 fp[foam_tri[:, 2]] - fp[foam_tri[:, 0]]), axis=1)
    foam_tract = ft[foam_tri].mean(axis=1)
    foam_force = (foam_tract * foam_area[:, np.newaxis]).sum(axis=0)
    fem_force  = nodal_force.sum(axis=0)
    err        = np.linalg.norm(fem_force - foam_force) / np.linalg.norm(foam_force)
    print(f"[MAP] FOAM force [N] : {foam_force}")
    print(f"[MAP] FEM  force [N] : {fem_force}")
    print(f"[MAP] Force error    : {err*100:.2f}%")
    print(f"[MAP] Skin nodes     : {len(skin_node_idx)} / {len(sp_all)} total")

    meshio.write(outfile, meshio.Mesh(
        points=sp_all,
        cells=[("triangle", all_triangles)],
        point_data={"traction": nodal_traction},
    ))
    print(f"[MAP] Saved =====> {outfile}")
    return nodal_force




def load_traction_xdmf(xdmffile, domain, gdim):
    data   = meshio.read(xdmffile)
    pts    = data.points
    tract  = data.point_data["traction"]

    VT     = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    f      = fem.Function(VT, name="traction")
    coords = VT.tabulate_dof_coordinates()

    print(f"[LOAD] FEM  pts range : {domain.geometry.x.min(axis=0)} -> {domain.geometry.x.max(axis=0)}")
    print(f"[LOAD] XDMF pts range : {pts.min(axis=0)} -> {pts.max(axis=0)}")

    tree = cKDTree(pts)
    dist, idx = tree.query(coords, k=1)
    print(f"[LOAD] Max KDTree dist : {dist.max():.4e} m (should be < 1e-3 m)")

    f.x.array[:] = tract[idx].flatten()
    f.x.scatter_forward()
    return f


# POST-PROCESSING
# FAILURE ANALYSIS
def tsai_wu(sigma_mat, strengths):
    s1, s2, s6 = sigma_mat
    Xt, Xc     = strengths["Xt"], strengths["Xc"]
    Yt, Yc     = strengths["Yt"], strengths["Yc"]
    S          = strengths["S"]
    F1  =  1/Xt - 1/Xc
    F2  =  1/Yt - 1/Yc
    F11 =  1/(Xt*Xc)
    F22 =  1/(Yt*Yc)
    F66 =  1/S**2
    F12 = -0.5 / np.sqrt(Xt * Xc * Yt * Yc)
    return (F1*s1 + F2*s2
            + F11*s1**2 + F22*s2**2
            + F66*s6**2 + 2*F12*s1*s2)

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
        out["matrix_c"] = ((s2/(2*ST))**2
                           + (Yc/(2*ST))**2 * (s2/Yc)
                           + (s6/SL)**2)
    return out


# CLT FAILURE RECOVERY — RESTRICTED TO A SUBSET OF CELLS
def recover_and_evaluate_failure_cells(domain, v_sol, mat, cell_indices, strengths, criterion="tsai_wu", label=""):
    assert mat.kind == "clt", "Failure recovery requires CLT material"
    if len(cell_indices) == 0:
        print(f"  [{label}] No cells — skipping failure analysis.")
        return 0.0, np.array([])

    u_h, theta_h = ufl.split(v_sol)
    e1, e2, e3   = local_frame(domain, domain.geometry.dim)
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
        print(f"  [{label}]  ply {angle:+4.0f} deg  max FI = {FI_k.max():.4f}")
        z += t_ply

    FI_all = np.array(FI_all)
    FI_max = FI_all.max()
    print(f"  [{label}]  Global max FI : {FI_max:.4f}  =>  SF = {1/FI_max:.2f}")
    return FI_max, FI_all


# VON MISES AT TOP/BOTTOM FIBER - ALUMINIUM RIBS
def von_mises_iso_cells(domain, v_sol, mat, cell_indices, label=""):
    assert mat.kind == "isotropic"
    if len(cell_indices) == 0:
        print(f"  [{label}] No cells — skipping Von Mises analysis.")
        return 0.0

    u_h, theta_h = ufl.split(v_sol)
    e1, e2, e3   = local_frame(domain, domain.geometry.dim)
    P            = tangent_projection(e1, e2)
    eps_h, _     = membrane_strain(u_h, P)
    kappa_h      = bending_strain(theta_h, e3, P)

    DG0    = fem.functionspace(domain, ("DG", 0, (3,)))
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
        print(f"  [{label}]  fiber={fiber}  max sigma_vm = {vm_max/1e6:.2f} MPa")

    return max(vm_list)





# CONFIGURATION
MESHFILE = "wing_new.msh"
FOAMFILE = "../wing.vtp"
XDMFFILE = "FOAMData.xdmf"
MAPFILE  = "MappedTraction.xdmf"

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

TAG_ROOT      = 52

_E1, _E2, _G12, _nu12 = 181e9, 10.3e9, 7.17e9, 0.28
_t_ply = 0.75e-3

# SPARS CARRY MORE BENDING -> COULD USE A STIFFER 0-DOMINATED LAYUP.
_LAYUP_SKIN = [0, 45, -45, 90, 90, -45, 45, 0]   # 8 plies, H = 6 mm
_LAYUP_SPAR = [0, 45, -45, 90, 90, -45, 45, 0]   # SAME — CHANGE IF REQUIRED


# COMPOSITE STRENGTH DATA (FOR TSAI-WU)
STRENGTHS_CLT = {
    "Xt": 1500e6, "Xc":  900e6,
    "Yt":   50e6, "Yc":  200e6,
    "S":    70e6,
}

# ALUMINIUM 2024-T3 RIB PROPERTIES
_E_ALU   = 73.1e9
_NU_ALU  = 0.33
_T_RIB   = 3.0e-3
SY_ALU   = 324e6





# MESH
MESH_IO    = gmsh.read_from_msh(MESHFILE, comm=MPI.COMM_WORLD, gdim=3)
DOMAIN     = MESH_IO.mesh
DOMAIN.geometry.x[:] *= 1e-3
CELL_TAGS  = MESH_IO.cell_tags
FACET_TAGS = MESH_IO.facet_tags
GDIM       = DOMAIN.geometry.dim
TDIM       = DOMAIN.topology.dim
FDIM       = TDIM - 1

import numpy as np
all_tags = np.unique(FACET_TAGS.values)
print(f"[BC] Available facet tags: {all_tags}")
print(f"[BC] Facets at TAG_ROOT={TAG_ROOT}: {len(FACET_TAGS.find(TAG_ROOT))}")

# sys.exit()


print(f"\n[MESH] Cells     : {DOMAIN.topology.index_map(TDIM).size_global}")
print(f"[MESH] Vertices  : {DOMAIN.topology.index_map(0).size_global}")
for tag in TAG_ALL:
    n = len(CELL_TAGS.find(tag))
    print(f"[MESH]   tag {tag} : {n} cells")


# LOCAL FRAME
E1, E2, E3 = local_frame(DOMAIN, GDIM)

RESULTS_FOLDER = Path("LocalFrame")
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
with io.VTKFile(MPI.COMM_WORLD, RESULTS_FOLDER / "LocalFrame.pvd", "w") as vtk_f:
    vtk_f.write_mesh(DOMAIN)
    vtk_f.write_function(E1, 0.0)
    vtk_f.write_function(E2, 0.0)
    vtk_f.write_function(E3, 0.0)


# FUNCTION SPACE
Ue       = basix.ufl.element("P",  DOMAIN.basix_cell(), 1, shape=(GDIM,))
Te       = basix.ufl.element("CR", DOMAIN.basix_cell(), 1, shape=(GDIM,))
V        = fem.functionspace(DOMAIN, basix.ufl.mixed_element([Ue, Te]))
v        = fem.Function(V)
u, theta = ufl.split(v)
v_       = ufl.TestFunction(V)
u_, theta_ = ufl.split(v_)
dv       = ufl.TrialFunction(V)



# SHELL KINEMATICS  (global — shared strain fields across all materials)
eps, kappa, gamma, drilling_strain = shell_strains(u, theta, E1, E2, E3)
eps_             = ufl.derivative(eps,           v, v_)
kappa_           = ufl.derivative(kappa,         v, v_)
gamma_           = ufl.derivative(gamma,         v, v_)
drilling_strain_ = ufl.replace(drilling_strain, {v: v_})


#



# # MATS = {
# #     TAG_SKIN     : MAT_SKIN,
# #     2            : MAT_RIB,       # rib0
# #     3            : MAT_RIB,       # rib750
# #     4            : MAT_RIB,       # rib1500
# #     5            : MAT_RIB,       # rib2250
# #     6            : MAT_RIB,       # rib3000
# #     TAG_MAINSPAR : MAT_MAINSPAR,
# #     TAG_TAILSPAR : MAT_TAILSPAR,
# # }

# MATS = {
#     14 : MAT_SKIN,      # upper
#     15 : MAT_SKIN,      # lower
#     38 : MAT_RIB,       # rib0
#     39 : MAT_RIB,       # rib750
#     40 : MAT_RIB,       # rib1500
#     41 : MAT_RIB,       # rib2250
#     42 : MAT_RIB,       # rib3000
#     43 : MAT_MAINSPAR,
#     44 : MAT_TAILSPAR,
# }


# _E_ISO  = 70e9        # giống Cast3M test (70 GPa)
# _NU_ISO = 0.3
# _T_ISO  = 6.0e-3      # 6 mm

# print("\n[MAT] ── Isotropic benchmark material ───────────────")
# print(f"[MAT]  E = {_E_ISO/1e9:.1f} GPa  nu = {_NU_ISO}  t = {_T_ISO*1e3:.1f} mm")

# MAT_ALL = isotropic_material(_T_ISO, _E_ISO, _NU_ISO, DOMAIN)

# # Apply same material to ALL surface tags
# MATS = {tag: MAT_ALL for tag in TAG_ALL}


# # BOUNDARY CONDITIONS - CLAMP ROOT
# def at_root(x):
#     return np.isclose(x[1], 0.0, atol=1e-3)

# ROOT_FACETS = FACET_TAGS.find(TAG_ROOT)
# print("Number of root facets:", len(ROOT_FACETS))

# root_dofs_u = fem.locate_dofs_topological(V.sub(0), FDIM, ROOT_FACETS)
# print("Displacement DOFs:", len(root_dofs_u))

# root_dofs_theta = fem.locate_dofs_topological(V.sub(1), FDIM, ROOT_FACETS)
# print("Rotation DOFs:", len(root_dofs_theta))

# Vu, _       = V.sub(0).collapse()
# root_dofs_u = fem.locate_dofs_topological((V.sub(0), Vu), FDIM, ROOT_FACETS)
# uD          = fem.Function(Vu);      uD.x.array[:] = 0.0

# Vt, _       = V.sub(1).collapse()
# root_dofs_t = fem.locate_dofs_topological((V.sub(1), Vt), FDIM, ROOT_FACETS)
# thetaD      = fem.Function(Vt);  thetaD.x.array[:] = 0.0

# BCS = [
#     fem.dirichletbc(uD,     root_dofs_u, V.sub(0)),
#     fem.dirichletbc(thetaD, root_dofs_t, V.sub(1)),
# ]


# print(f"\n[BC] Root clamped: {len(root_dofs_u)} displacement DOFs  |  "
    #   f"{len(root_dofs_t)} rotation DOFs")

# BOUNDARY CONDITIONS - CLAMP ROOT
def at_root(x):
    return np.isclose(x[1], 0.0, atol=1e-8)

Vu, _       = V.sub(0).collapse()
root_dofs_u = fem.locate_dofs_geometrical((V.sub(0), Vu), at_root)
uD          = fem.Function(Vu);  uD.x.array[:] = 0.0

Vt, _       = V.sub(1).collapse()
root_dofs_t = fem.locate_dofs_geometrical((V.sub(1), Vt), at_root)
thetaD      = fem.Function(Vt);  thetaD.x.array[:] = 0.0

BCS = [
    fem.dirichletbc(uD,     root_dofs_u, V.sub(0)),
    fem.dirichletbc(thetaD, root_dofs_t, V.sub(1)),
]

print(f"[BC] Root clamped: {len(root_dofs_u[0])} displacement DOFs  |  {len(root_dofs_t[0])} rotation DOFs")

# print(f"\n[BC] Root clamped: {len(root_dofs_u)} displacement DOFs  |  "
    #   f"{len(root_dofs_t)} rotation DOFs")

# print(f"[BC] Root clamped: {len(root_dofs_u[0])} displacement DOFs  |  {len(root_dofs_t[0])} rotation DOFs")


# AERODYNAMIC TRACTION
import_foam_traction(FOAMFILE, XDMFFILE, verbose=True)
# map_traction(XDMFFILE, MESHFILE, MAPFILE)
map_traction(XDMFFILE, MESHFILE, MAPFILE, skin_phys_tags=(14, 15))
FTraction = load_traction_xdmf(MAPFILE, DOMAIN, GDIM)





# WEAK FORMULATION  — multi-material subdomain integral
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

a_int    = reduce(lambda a, b: a + b, form_pieces)

# Aero traction on skin only
# L_ext    = ufl.dot(FTraction, u_) * dx(TAG_SKIN)
L_ext = sum(ufl.dot(FTraction, u_) * dx(tag) for tag in TAG_SKIN)

residual = a_int - L_ext
tangent  = ufl.derivative(residual, v, dv)


# # SOLVE
# problem = NonlinearProblem(
#     F=residual, u=v, bcs=BCS, J=tangent,
#     petsc_options_prefix="wing",
#     # petsc_options={
#     #     "ksp_type"                  : "preonly",
#     #     "pc_type"                   : "lu",
#     #     "pc_factor_mat_solver_type" : "mumps",
#     #     "snes_type"                 : "newtonls",
#     #     "snes_rtol"                 : 1e-8,
#     #     "snes_atol"                 : 1e-8,
#     #     "snes_max_it"               : 25,
#     #     "snes_monitor"              : None,
#     # },
#     # petsc_options={
#     #     "snes_type": "newtonls",
#     #     "snes_rtol": 1e-8,
#     #     "snes_atol": 1e-8,
#     #     "snes_max_it": 25,
#     #     "snes_monitor": None,
#     #     "ksp_monitor": None,
#     #     "ksp_converged_reason": None,
#     #     "ksp_type": "gmres",
#     #     "ksp_rtol": 1e-8,
#     #     "pc_type": "hypre",
#     #     "pc_hypre_type": "boomeramg",
#     # }
#     petsc_options={
#         "snes_type": "newtonls",
#         "snes_monitor": None,
#         "snes_converged_reason": None,

#         "ksp_type": "gmres",
#         "ksp_rtol": 1e-5,
#         "ksp_max_it": 500,
#         "ksp_monitor": None,
#         "ksp_converged_reason": None,

#         "pc_type": "hypre",
#         "pc_hypre_type": "boomeramg",
#     }
# )
# problem.solve()

# converged = problem.solver.getConvergedReason()
# n_iter    = problem.solver.getIterationNumber()
# print(f"\n[SNES] converged reason : {converged}")
# print(f"[SNES] iterations       : {n_iter}")
# assert converged > 0, f"Solver did not converge (reason {converged})"

# from dolfinx.fem.petsc import NonlinearProblem, LinearProblem

# problem = LinearProblem(
#     a_int, L_ext, bcs=BCS,
#     u=v,
#     petsc_options_prefix="wing",
#     petsc_options={
#         "ksp_type"                  : "preonly",
#         "pc_type"                   : "lu",
#         "pc_factor_mat_solver_type" : "mumps",
#     },
# )
# problem.solve()
# print("[SOLVE] Done")


# from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
# from petsc4py import PETSc

# a_bilinear = ufl.derivative(a_int, v, dv)   # bilinear form đúng

# a_compiled = fem.form(a_bilinear)
# L_compiled = fem.form(L_ext)

# A = assemble_matrix(a_compiled, bcs=BCS)
# A.assemble()

# b = assemble_vector(L_compiled)
# apply_lifting(b, [a_compiled], [BCS])
# b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# set_bc(b, BCS)

# ksp = PETSc.KSP().create(DOMAIN.comm)
# ksp.setOperators(A)
# # ksp.setType("preonly")
# # ksp.getPC().setType("lu")
# # ksp.getPC().setFactorSolverType("mumps")
# # ksp.solve(b, v.vector)
# # ksp.setType("gmres")
# # pc = ksp.getPC()
# # pc.setType("bjacobi")
# # pc.setFactorSolverType("ilu")
# # ksp.setTolerances(rtol=1e-5)

# ksp.setType("gmres")
# ksp.getPC().setType("hypre")
# ksp.getPC().setHYPREType("boomeramg")
# # ksp.setTolerances(rtol=1e-4)
# ksp.setTolerances(rtol=1E-4, atol=0.0, divtol=1e6, max_it=10000)
# ksp.setMonitor(lambda ksp, its, rnorm:
#                print(f"[KSP] it={its:4d}  |r|={rnorm:.3e}"))

# r0 = None
# def mon(ksp, its, rnorm):
#     global r0
#     if its == 0:
#         r0 = rnorm
#     print(f"[KSP] it={its:4d}  |r|={rnorm:.3e}  rel={rnorm/r0:.3e}")
# ksp.setMonitor(mon)

# ksp.solve(b, v.x.petsc_vec)
# v.x.scatter_forward()

# print("[SOLVE] Done")
# ksp.destroy()
# A.destroy()
# b.destroy()



# ─────────────────────────────────────────────
# SOLVER (FAST NONLINEAR FRAMEWORK)
# ─────────────────────────────────────────────

from dolfinx.fem.petsc import NonlinearProblem

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
        "snes_rtol"                 : 1e-8,
        "snes_atol"                 : 1e-8,
        "snes_max_it"               : 25,
        "snes_monitor"              : None,
        # MUMPS memory options
        "mat_mumps_icntl_14"        : 80,   # % memory increase (default 20)
        "mat_mumps_icntl_23"        : 4000, # max MB per process
    }
)

problem.solve()







# ── Sau khi solve ─────────────────────────────────────────────────
disp = v.sub(0).collapse()
rota = v.sub(1).collapse()

u_arr = disp.x.array.reshape(-1, 3)
t_arr = rota.x.array.reshape(-1, 3)

comm = MPI.COMM_WORLD
max_u = comm.allreduce(np.max(np.abs(u_arr), axis=0), op=MPI.MAX)
max_t = comm.allreduce(np.max(np.abs(t_arr), axis=0), op=MPI.MAX)

# ── Global rotation theta → local RX RY RZ ────────────────────────
# theta trong FEniCSx là global vector (3 components)
# Cast3M RX RY RZ là global rotations → so sánh trực tiếp

CAST3M_CFD = {
    "Ux" : 3.92755e-3,
    "Uy" : 3.25625e-3,
    "Uz" : 1.06327e-1,
    "RX" : 8.41811e-2,
    "RY" : 3.51164e-2,
    "RZ" : 5.23593e-1,   # drilling — expect mismatch
}

FENICS = {
    "Ux" : max_u[0],
    "Uy" : max_u[1],
    "Uz" : max_u[2],
    "RX" : max_t[0],
    "RY" : max_t[1],
    "RZ" : max_t[2],
}

# Scale FEniCSx Steel → Al nếu đang dùng E=210GPa
E_current = 210e9   # đổi thành 70e9 nếu đang dùng Al
E_cast3m  = 70e9
scale = E_current / E_cast3m

print("\n" + "="*65)
print("VALIDATION vs Cast3M — CFD load, Al 70GPa, t=6mm")
print("="*65)
print(f"  (FEniCSx scaled × {scale:.1f} vì đang dùng E={E_current/1e9:.0f}GPa)")
print(f"{'DOF':<6} {'Cast3M':>12} {'FEniCSx×scale':>14} {'Error%':>9}  Note")
print("-"*65)

for key in ["Ux", "Uy", "Uz", "RX", "RY", "RZ"]:
    ref  = CAST3M_CFD[key]
    val  = FENICS[key] * scale
    err  = abs(val - ref) / abs(ref) * 100

    if key == "RZ":
        flag = "⚠️  drilling"
        note = "(DKT vs RM — expected mismatch)"
    elif err < 5:
        flag = "✅"
        note = ""
    elif err < 15:
        flag = "⚠️"
        note = ""
    else:
        flag = "❌"
        note = ""

    print(f"{key:<6} {ref:>12.5e} {val:>14.5e} {err:>8.1f}%  {flag} {note}")

print("="*65)
print(f"\nSummary:")
print(f"  Uz error : {abs(FENICS['Uz']*scale - CAST3M_CFD['Uz'])/CAST3M_CFD['Uz']*100:.2f}%")
print(f"  Ux error : {abs(FENICS['Ux']*scale - CAST3M_CFD['Ux'])/CAST3M_CFD['Ux']*100:.2f}%")
print(f"  Uy error : {abs(FENICS['Uy']*scale - CAST3M_CFD['Uy'])/CAST3M_CFD['Uy']*100:.2f}%")











# # Solver info
# converged = problem.solver.getConvergedReason()
# n_iter    = problem.solver.getIterationNumber()

# print(f"\n[KSP] converged reason : {converged}")
# print(f"[KSP] iterations       : {n_iter}")

# assert converged > 0, "Linear solve did not converge"


# # POST-PROCESSING
# disp = v.sub(0).collapse();  disp.name = "Displacement"
# rota = v.sub(1).collapse();  rota.name = "Rotation"

# # Global max displacement
# vdim_u     = disp.function_space.element.value_shape[0]
# disp_arr   = disp.x.array.reshape(-1, vdim_u)
# disp_mag   = np.linalg.norm(disp_arr, axis=1)
# local_max  = disp_mag.max()
# global_max = MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)
# print(f"\n[POST] Max displacement : {global_max:.6e} m  ({global_max*1e3:.3f} mm)")

# # TSAI-WU FAILURE FOR COMPOSITE REGIONS
# print("\n[POST] ── Composite failure (Tsai-Wu) ────────────────────────────")
# FI_results = {}
# for tag, mat, lbl in [
#     (TAG_SKIN,     MAT_SKIN,     "SKIN"),
#     (TAG_MAINSPAR, MAT_MAINSPAR, "MAINSPAR"),
#     (TAG_TAILSPAR, MAT_TAILSPAR, "TAILSPAR"),
# ]:
#     # cells = CELL_TAGS.find(tag)
#     if isinstance(tag, list):
#         cells = np.concatenate([CELL_TAGS.find(t) for t in tag])
#     else:
#         cells = CELL_TAGS.find(tag)
#     print(f"\n  {lbl}  ({len(cells)} cells)")
#     FI_max, _ = recover_and_evaluate_failure_cells(
#         DOMAIN, v, mat, cells, STRENGTHS_CLT,
#         criterion="tsai_wu", label=lbl,
#     )
#     FI_results[lbl] = FI_max

# # ── Von Mises — aluminium ribs ────────────────────────────────────────
# print("\n[POST] ── Rib Von Mises stress ───────────────────────────────────")
# VM_results = {}
# rib_names  = {38: "RIB0", 39: "RIB750", 40: "RIB1500", 41: "RIB2250", 42: "RIB3000"}
# for tag, lbl in rib_names.items():
#     cells = CELL_TAGS.find(tag)
#     print(f"\n  {lbl}  ({len(cells)} cells)")
#     vm_max = von_mises_iso_cells(DOMAIN, v, MAT_RIB, cells, label=lbl)
#     VM_results[lbl] = vm_max

# # ── Summary table ─────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("[POST] SUMMARY")
# print("="*60)
# print(f"  Max displacement : {global_max*1e3:.3f} mm")
# print()
# print("  Composite regions — Tsai-Wu FI:")
# for lbl, fi in FI_results.items():
#     sf = 1/fi if fi > 0 else float("inf")
#     flag = "  *** FAILURE ***" if fi >= 1.0 else ""
#     print(f"    {lbl:<12}  FI = {fi:.4f}   SF = {sf:.2f}{flag}")
# print()
# print("  Aluminium ribs — Von Mises:")
# for lbl, vm in VM_results.items():
#     ratio = vm / SY_ALU
#     flag  = "  *** YIELDING ***" if ratio >= 1.0 else ""
#     print(f"    {lbl:<12}  σ_vm = {vm/1e6:7.2f} MPa   σ_vm/σ_y = {ratio:.3f}{flag}")


# ═══════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════
RESULT_FOLDER = Path("Result")
RESULT_FOLDER.mkdir(exist_ok=True, parents=True)

Vout     = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))
disp_out = fem.Function(Vout);  disp_out.interpolate(disp);  disp_out.name = "Displacement"
rota_out = fem.Function(Vout);  rota_out.interpolate(rota);  rota_out.name = "Rotation"

with io.XDMFFile(MPI.COMM_WORLD, RESULT_FOLDER / "results.xdmf", "w") as xdmf:
    xdmf.write_mesh(DOMAIN)
    xdmf.write_function(disp_out)
    xdmf.write_function(rota_out)

print(f"\n[EXPORT] Results -> {RESULT_FOLDER / 'results.xdmf'}")

# ── Export material tag field for ParaView verification ───────────────
DG0_scalar = fem.functionspace(DOMAIN, ("DG", 0))
mat_field  = fem.Function(DG0_scalar, name="MaterialTag")
for tag in TAG_ALL:
    cells = CELL_TAGS.find(tag)
    mat_field.x.array[cells] = float(tag)
mat_field.x.scatter_forward()

with io.XDMFFile(MPI.COMM_WORLD, RESULT_FOLDER / "material_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(DOMAIN)
    xdmf.write_function(mat_field)

print(f"[EXPORT] Material tags -> {RESULT_FOLDER / 'material_tags.xdmf'}")
print("         (open in ParaView to verify skin/rib/spar assignment)")

# problem.solver.destroy()
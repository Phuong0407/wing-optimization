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
    Q = _Q_ply(E1, E2, G12, nu12) # Q is a numpy array, not UFL
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
    G_eff = float(A_np[2, 2]) / H
    
    # Create fem.Constant objects for the tensors and H, G_eff
    # This allows their values to be updated without recompiling the UFL forms
    H_const = fem.Constant(domain, H)
    G_eff_const = fem.Constant(domain, G_eff)
    A_ufl_const = fem.Constant(domain, A_np)
    B_ufl_const = fem.Constant(domain, B_np)
    D_ufl_const = fem.Constant(domain, D_np)
    As_ufl_const = fem.Constant(domain, As_np)
    
    return SimpleNamespace(
        kind   = "clt",
        H      = H_const, # Total thickness (fem.Constant)
        t_ply  = t_ply,   # Ply thickness (float)
        G_eff  = G_eff_const, # fem.Constant
        A_ufl  = A_ufl_const, # fem.Constant
        B_ufl  = B_ufl_const, # fem.Constant
        D_ufl  = D_ufl_const, # fem.Constant
        As_ufl = As_ufl_const, # fem.Constant
        # Store initial material properties for recomputing ABD
        _layup_angles = layup_angles,
        _E1 = E1, _E2 = E2, _G12 = G12, _nu12 = nu12,
        _G13 = G13, _G23 = G23, _kappa_s = kappa_s,
    )


def update_clt_material(mat, total_thickness, n_layers):
    """Update CLT constants in-place without rebuilding UFL forms."""
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
    
    layup  = mat._layup_angles # These are the constant layup angles
    t_ply  = mat.t_ply         # This is the current ply thickness (float)
    Q_ply  = _Q_ply(mat._E1, mat._E2, mat._G12, mat._nu12) # These are constant material properties
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
    """Pure-numpy failure evaluation from pre-interpolated strain arrays."""
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



def panel_buckling_index(eps0, mat, panel_width, k=BUCKLING_K):
    """
    Evaluate compression panel buckling index.

    eps0 : membrane strain array (n_cells,3)
    mat  : CLT laminate object
    panel_width : panel width (m)
    """

    # laminate bending stiffness
    D11 = mat.D_np[0, 0]

    # laminate thickness
    t = mat.H

    # critical buckling stress
    sigma_cr = k * np.pi**2 * D11 / (panel_width**2 * t)

    # lamina stiffness
    Q = _Q_ply(mat._E1, mat._E2, mat._G12, mat._nu12)

    # stress recovery
    stress = eps0 @ Q.T

    sigma_x = stress[:, 0]

    # only compression causes buckling
    sigma_comp = np.minimum(sigma_x, 0.0)

    # buckling index
    BI = np.abs(sigma_comp) / sigma_cr

    return BI.max()    



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

# Global definitions for n_layers and layups
n_layers = 8 # Assuming this is fixed
# Skin: 50% 0° (spanwise bending) + 50% ±45° (shear) — symmetric [0/0/45/-45]s
SKIN_LAYUP  = [0,  0,  45, -45, -45, 45,  0,  0]
# Spar web: ±45° dominant for shear, 0° outer plies for buckling — symmetric [0/45/-45/45]s
SPAR_LAYUP  = [0, 45, -45, 45, 45, -45, 45,  0]
# Rib: quasi-isotropic (FI very low, in-plane shear panel) — symmetric [0/45/-45/90]s
RIB_LAYUP   = [0, 45, -45, 90, 90, -45, 45,  0]

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



# ------------------------------------------------------------
# PANEL GEOMETRY (m)
# ------------------------------------------------------------

RIB_SPACING = 0.75      # khoảng cách giữa ribs
SPAR_HEIGHT = 0.40      # chiều cao spar web
RIB_PANEL   = 0.40      # panel rib

BUCKLING_K  = 4.0       # simply supported plate




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

global_u = comm.allreduce(len(dofs_u[0]), op=MPI.SUM); global_t = comm.allreduce(len(dofs_t[0]), op=MPI.SUM)
if comm.rank == 0:
    print(f"[BC] Root clamp: {global_u} displacement DOFs  | {global_t} rotation DOFs")


if not MAPFILE.exists():
    import_foam_traction(FOAMFILE, XDMFFILE, VERBOSE)
    map_traction(XDMFFILE, MESHFILE, MAPFILE, TAG_SKIN, MESHUNIT, VERBOSE)
FTraction = load_traction_xdmf(MAPFILE, DOMAIN, VERBOSE)


# Initial fem.Constant for TOTAL thicknesses
t_skin_const = fem.Constant(DOMAIN, 0.75e-3)
t_spar_const = fem.Constant(DOMAIN, 0.75e-3)
t_rib_const  = fem.Constant(DOMAIN, 0.75e-3)

vprint("\n[MAT] SKIN")
MAT_SKIN = clt_composite(
    SKIN_LAYUP,
    t_skin_const.value / n_layers, # Initial ply thickness
    CE_YNG1, CE_YNG2, CE_G12, CE_NU12,
    G13 = CE_G12, G23 = CE_G12 * 0.5, kappa_s=5/6,
    label="SKIN", verbose=VERBOSE, domain=DOMAIN
)
# Store the fem.Constant for the *total* thickness, so we can update its .value.
MAT_SKIN._total_thickness_const = t_skin_const

vprint("\n[MAT] SPAR")
MAT_SPAR = clt_composite(
    SPAR_LAYUP,
    t_spar_const.value / n_layers, # Initial ply thickness
    CE_YNG1, CE_YNG2, CE_G12, CE_NU12,
    G13=CE_G12, G23=CE_G12 * 0.5, kappa_s=5/6,
    label="MAINSPAR", verbose=VERBOSE, domain=DOMAIN
)
MAT_SPAR._total_thickness_const = t_spar_const

vprint("\n[MAT] RIB")
MAT_RIB = clt_composite(
    RIB_LAYUP,
    t_rib_const.value / n_layers, # Initial ply thickness
    CE_YNG1, CE_YNG2, CE_G12, CE_NU12,
    G13=CE_G12, G23=CE_G12 * 0.5, kappa_s=5/6,
    label="RIB", verbose=VERBOSE, domain=DOMAIN
)
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
    petsc_options_prefix="wing_",
    petsc_options={
        "ksp_type"                  : "preonly",
        "pc_type"                   : "lu",
        "pc_factor_mat_solver_type" : "mumps",
        "snes_type"                 : "newtonls",
        "snes_rtol"                 : 1E-8,    # tighten to 1E-8 on HPC
        "snes_atol"                 : 1E-8,    # tighten to 1E-8 on HPC
        "snes_max_it"               : 25,      # increase to 25 on HPC
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

AREA_SKIN = compute_area(TAG_UPPER, TAG_LOWER)
AREA_RIB = compute_area(*TAG_RIBS)
AREA_SPAR = compute_area(TAG_MAINSPAR, TAG_TAILSPAR)


DG0 = fem.functionspace(DOMAIN, ("DG", 0, (3,)))
eps_fn = fem.Function(DG0)
kap_fn = fem.Function(DG0)


def solve_wing(x):

    t_skin_total, t_rib_total, t_spar_total = x

    update_clt_material(MAT_SKIN, t_skin_total, n_layers)
    update_clt_material(MAT_RIB,  t_rib_total,  n_layers)
    update_clt_material(MAT_SPAR, t_spar_total, n_layers)

    # FEM solve
    problem.solve()

    disp = v.sub(0).collapse()
    rota = v.sub(1).collapse()

    u_arr = disp.x.array.reshape(-1, 3)
    t_arr = rota.x.array.reshape(-1, 3)

    ux_max = comm.allreduce(np.abs(u_arr[:,0]).max(), op=MPI.MAX)
    uy_max = comm.allreduce(np.abs(u_arr[:,1]).max(), op=MPI.MAX)
    uz_max = comm.allreduce(np.abs(u_arr[:,2]).max(), op=MPI.MAX)

    u_max = comm.allreduce(
        np.linalg.norm(u_arr, axis=1).max(),
        op=MPI.MAX
    )

    tx_max = comm.allreduce(np.abs(t_arr[:,0]).max(), op=MPI.MAX)
    ty_max = comm.allreduce(np.abs(t_arr[:,1]).max(), op=MPI.MAX)
    tz_max = comm.allreduce(np.abs(t_arr[:,2]).max(), op=MPI.MAX)

    if VERBOSE and rank == 0:
        print("\n==============================")
        print("[POST] DISPLACEMENT")
        print("==============================")

        print(f"|ux|max {ux_max:.6e}")
        print(f"|uy|max {uy_max:.6e}")
        print(f"|uz|max {uz_max:.6e}")
        print(f"|u|max  {u_max:.6e}")
        print()
        print(f"  |thetax| max : {tx_max:.6e} rad")
        print(f"  |thetay| max : {ty_max:.6e} rad")
        print(f"  |thetaz| max : {tz_max:.6e} rad")
        print()

    # ---- strain recovery ----

    u_h, theta_h = ufl.split(v)

    P_proj = tangent_projection(E1, E2)

    eps_h, _ = membrane_strain(u_h, P_proj)
    kappa_h  = bending_strain(theta_h, E3, P_proj)

    eps_fn.interpolate(
        fem.Expression(
            to_voigt(eps_h),
            DG0.element.interpolation_points
        )
    )

    kap_fn.interpolate(
        fem.Expression(
            to_voigt(kappa_h),
            DG0.element.interpolation_points
        )
    )

    eps0_all  = eps_fn.x.array.reshape(-1,3)
    kappa_all = kap_fn.x.array.reshape(-1,3)

    # ---- failure evaluation ----
    local_fi_max = 0.0

    for tag, mat in [
        (TAG_UPPER, MAT_SKIN),
        (TAG_LOWER, MAT_SKIN),
        (TAG_MAINSPAR, MAT_SPAR),
        (TAG_TAILSPAR, MAT_SPAR),
        *[(t, MAT_RIB) for t in TAG_RIBS]
    ]:

        cells = CELL_TAGS.find(tag)

        if len(cells) == 0:
            continue

        eps_cells = eps0_all[cells]
        kap_cells = kappa_all[cells]

        # ------------------------------------
        # TSAI-WU FAILURE
        # ------------------------------------

        fi_val = _eval_failure_from_arrays(
            eps_cells,
            kap_cells,
            mat,
            CE_STRENGTH,
            criterion="tsai_wu"
        )

        # ------------------------------------
        # PANEL WIDTH SELECTION
        # ------------------------------------

        if tag in [TAG_UPPER, TAG_LOWER]:
            panel_width = RIB_SPACING

        elif tag in [TAG_MAINSPAR, TAG_TAILSPAR]:
            panel_width = SPAR_HEIGHT

        else:
            panel_width = RIB_PANEL

        # ------------------------------------
        # BUCKLING CHECK
        # ------------------------------------

        buckling_val = panel_buckling_index(
            eps_cells,
            mat,
            panel_width
        )

        # ------------------------------------
        # COMBINED FAILURE INDEX
        # ------------------------------------

        combined = max(fi_val, buckling_val)

        local_fi_max = max(local_fi_max, combined)

        if VERBOSE and rank == 0:
            print(
                f"[TAG_{tag}] "
                f"TsaiWu={fi_val:.4f} "
                f"Buckling={buckling_val:.4f}"
            )

    global_fi_max = comm.allreduce(local_fi_max, op=MPI.MAX)

    # ---- mass model ----

    rho = 1600.0

    mass_total = rho * (
        AREA_SKIN * t_skin_total +
        AREA_RIB  * t_rib_total +
        AREA_SPAR * t_spar_total
    )

    if rank == 0:

        sf = 1/global_fi_max if global_fi_max>0 else float("inf")

        status = "SAFE" if global_fi_max < 1 else "FAIL"

        print(
            f"[ITER] "
            f"Mass={mass_total:.4f} "
            f"FI={global_fi_max:.4f} "
            f"SF={sf:.2f} "
            f"{status}"
        )

    return mass_total, global_fi_max



import nlopt
import numpy as np

_last_fi = 0.0

def combined_func(x):
    global _last_x, _last_mass, _last_fi

    if _last_x is not None and np.allclose(x, _last_x):
        return _last_mass, _last_fi

    mass, fi = solve_wing(x)

    _last_x = np.copy(x)
    _last_mass = mass
    _last_fi = fi

    return mass, fi

def objective_mma(x, grad):
    comm.bcast({"cmd": "eval", "x": x}, root=0)
    
    mass, fi = combined_func(x)
    
    global _last_fi
    _last_fi = fi
    
    if grad.size > 0:
        pass
    return mass

def constraint_mma(x, grad):
    if grad.size > 0:
        pass
    return _last_fi - 1.0


opt = nlopt.opt(nlopt.LN_COBYLA, 3) 
opt.set_lower_bounds([0.001, 0.001, 0.001])
opt.set_upper_bounds([0.020, 0.020, 0.020])
opt.set_xtol_rel(1e-4)  # tighten back to 1e-4 
opt.set_maxeval(50)      # increase to 50 on HPC

opt.set_min_objective(objective_mma)
opt.add_inequality_constraint(constraint_mma, 1e-6)

if QUICK_TEST:
    vprint("\n>>> QUICK TEST: single solve at initial point <<<")
    mass, fi = solve_wing([0.006, 0.006, 0.006])
    vprint(f"[TEST] mass={mass:.4f}  FI={fi:.4f}")
else:
    if rank == 0:
        vprint("\n>>> NLOPT OPTIMIZATION START (MMA/COBYLA) <<<")
        try:
            xopt = opt.optimize([0.006, 0.006, 0.006])
            vprint(f"\nKẾT QUẢ: Skin={xopt[0]*1000:.2f}mm, Rib={xopt[1]*1000:.2f}mm, Spar={xopt[2]*1000:.2f}mm")
        finally:
            comm.bcast({"cmd": "stop"}, root=0)
    else:
        while True:
            msg = comm.bcast(None, root=0)
            if msg["cmd"] == "stop": break
            if msg["cmd"] == "eval":
                combined_func(msg["x"])


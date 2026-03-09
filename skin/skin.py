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
import basix
import dolfinx.fem.petsc
from dolfinx.fem.petsc import NonlinearProblem
from pathlib import Path



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
    FRAME  = _local_frame_ufl(domain)
    VT     = fem.functionspace(domain, ("DG", 0, (gdim,)))
    V0, _  = VT.sub(0).collapse()
    BASIS  = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(gdim)]
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
    P                 = tangent_projection(e1, e2)
    eps, t_gu         = membrane_strain(u, P)
    kappa             = bending_strain(theta, e3, P)
    gamma             = shear_strain(u, theta, e3, P)
    drilling_strain   = compute_drilling_strain(t_gu, theta, e3)
    return eps, kappa, gamma, drilling_strain



# ISOTROPIC MATERIAL
def isotropic_material(thickness, young, poisson, domain):
    h        = fem.Constant(domain, float(thickness))
    E        = fem.Constant(domain, float(young))
    nu       = fem.Constant(domain, float(poisson))
    lmbda    = E * nu / (1 + nu) / (1 - 2 * nu)
    mu       = E / 2 / (1 + nu)
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)
    return SimpleNamespace(h=h, E=E, nu=nu, lmbda=lmbda, mu=mu, lmbda_ps=lmbda_ps, kind="isotropic")

# CLT COMPOSITE MATERIAL
def _Q_ply(E1, E2, G12, nu12):
    nu21 = nu12 * E2 / E1
    d    = 1 - nu12 * nu21
    return np.array([
        [ E1 / d,        nu12 * E2 / d,  0   ],
        [ nu12 * E2 / d, E2 / d,         0   ],
        [ 0,             0,              G12 ],
    ])


def _Qbar_ply(Q, angle_deg):
    a    = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    T = np.array([
        [ c**2,   s**2,   2*c*s          ],
        [ s**2,   c**2,  -2*c*s          ],
        [-c*s,    c*s,    c**2 - s**2    ],
    ])
    R    = np.diag([1.0, 1.0, 2.0])
    Rinv = np.diag([1.0, 1.0, 0.5])
    return np.linalg.inv(T) @ Q @ R @ T @ Rinv

def compute_ABD(layup_angles, t_ply, E1, E2, G12, nu12):
    Q   = _Q_ply(E1, E2, G12, nu12)
    H   = t_ply * len(layup_angles)
    z   = -H / 2.0
    A   = np.zeros((3, 3))
    B   = np.zeros((3, 3))
    D   = np.zeros((3, 3))
    for angle in layup_angles:
        Qb = _Qbar_ply(Q, angle)
        z0, z1 = z, z + t_ply
        A += Qb * (z1 - z0)
        B += Qb * (z1**2 - z0**2) / 2.0
        D += Qb * (z1**3 - z0**3) / 3.0
        z  = z1
    return A, B, D, H

def clt_material(layup_angles, t_ply,E1, E2, G12, nu12,G13=None, G23=None, kappa_s=5/6):
    if G13 is None: G13 = G12
    if G23 is None: G23 = G12 * 0.5

    A_np, B_np, D_np, H = compute_ABD(layup_angles, t_ply, E1, E2, G12, nu12)

    max_B = np.abs(B_np).max()
    print(f"[CLT] Layup  : {layup_angles}")
    print(f"[CLT] H      : {H*1e3:.2f} mm")
    print(f"[CLT] max|B| : {max_B:.2e}  {'SYMMETRIC' if max_B < 1e-6 * A_np.max() else 'NON-SYMMETRIC'}")
    print(f"[CLT] A11    : {A_np[0,0]/1e6:.2f} MPa·m")
    print(f"[CLT] D11    : {D_np[0,0]:.4f} N·m²")

    As_np = kappa_s * H * np.array([
        [ G13, 0.0 ],
        [ 0.0, G23 ],
    ])

    # EFFECTIVE IN-PLANE SHEAR FOR DRILLING STABILISATION
    G_eff = float(A_np[2, 2]) / H

    return SimpleNamespace(
        kind   = "clt",
        H      = H,
        A_np   = A_np,
        B_np   = B_np,
        D_np   = D_np,
        As_np  = As_np,
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
    return ufl.as_tensor([
        [v[0], v[2]],
        [v[2], v[1]],
    ])


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

    h = mat.h if mat.kind == "isotropic" else fem.Constant(domain, mat.H)

    stiffness = G_eff * h**3 / h_mesh**2
    stress    = stiffness * drilling_strain
    return stiffness, stress



# IMPORT AND MAP OPENFOAM DATA
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
    idList = vtk.vtkIdList()
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

    area = 0.5 * np.linalg.norm(n, axis=1)
    n_unit = n / np.linalg.norm(n, axis=1)[:, None]

    p_tri   = p[triangles].mean(axis=1)
    wss_tri = wss[triangles].mean(axis=1)

    traction_cell = -p_tri[:, None] * n_unit + wss_tri

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

    print(f"[FOAM] Exported traction to the file {xdmffile}")

def map_traction(foamfile, femfile, outfile):
    fm  = meshio.read(foamfile)
    fp  = fm.points
    ft  = fm.points[fm.cells_dict["triangle"]] if "triangle" not in fm.cells_dict else fp
    ft  = fm.point_data["traction"]
    fp  = fm.points

    sm  = meshio.read(femfile)
    sp  = sm.points * 1e-3
    st  = sm.cells_dict["triangle"]

    interp         = RBFInterpolator(fp, ft, kernel="thin_plate_spline", neighbors=20)
    nodal_traction = interp(sp)

    trib_area = np.zeros(len(sp))
    for tri in st:
        P0, P1, P2 = sp[tri[0]], sp[tri[1]], sp[tri[2]]
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

    meshio.write(outfile, meshio.Mesh(
        points=sp,
        cells=[("triangle", st)],
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

    print(f"[LOAD] FEM pts range : {domain.geometry.x.min(axis=0)} =====> {domain.geometry.x.max(axis=0)}")
    print(f"[LOAD] XDMF pts range : {pts.min(axis=0)} =====> {pts.max(axis=0)}")

    tree = cKDTree(pts)
    dist, idx = tree.query(coords, k=1)
    print(f"[LOAD] Max KDTree dist : {dist.max():.4e} m   (should be < 1e-3 m)")

    f.x.array[:] = tract[idx].flatten()
    f.x.scatter_forward()
    return f





# FAILURE CRITERIA (post-processing)
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
        out["fiber_t"] = (s1/Xt)**2 + (s6/SL)**2
    else:
        out["fiber_c"] = (s1/Xc)**2

    if s2 >= 0:
        out["matrix_t"] = (s2/Yt)**2 + (s6/SL)**2
    else:
        out["matrix_c"] = ((s2/(2*ST))**2 + (Yc/(2*ST))**2 * (s2/Yc) + (s6/SL)**2)
    return out


def recover_and_evaluate_failure(domain, v_sol, mat, strengths, criterion="tsai_wu"):
    assert mat.kind == "clt", "Failure recovery requires CLT material"

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

    eps0_vals  = eps_fn.x.array.reshape(-1, 3)
    kappa_vals = kap_fn.x.array.reshape(-1, 3)

    layup   = mat._layup_angles
    t_ply   = mat._t_ply
    Q_ply   = _Q_ply(mat._E1, mat._E2, mat._G12, mat._nu12)

    H   = mat.H
    z   = -H / 2.0
    FI_all = []

    for angle in layup:
        z_mid    = z + t_ply / 2.0
        strain_k = eps0_vals + z_mid * kappa_vals

        Qb_lam    = _Qbar_ply(Q_ply, angle)
        stress_lam = strain_k @ Qb_lam.T

        a = np.radians(angle)
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
        print(f"  ply {angle:+4.0f} deg  max FI = {FI_k.max():.4f}")
        z += t_ply

    FI_all = np.array(FI_all)
    FI_max = FI_all.max()
    print(f"\n[FAIL] Global max FI : {FI_max:.4f}  =====>  SF = {1/FI_max:.2f}")
    return FI_max, FI_all






MESHFILE = "skin.msh"
FOAMFILE = "../wing.vtp"
XDMFFILE = "FOAMData.xdmf"
MAPFILE  = "MappedTraction.xdmf"

# LOAD MESH
MESH = gmsh.read_from_msh(MESHFILE, comm=MPI.COMM_WORLD, gdim=3)
DOMAIN = MESH.mesh
DOMAIN.geometry.x[:] *= 1E-3
CELL_TAGS = MESH.cell_tags
FACET_TAGS = MESH.facet_tags
GDIM = DOMAIN.geometry.dim
TDIM = DOMAIN.topology.dim
FDIM = TDIM - 1


# LOCAL FRAME
E1, E2, E3 = local_frame(DOMAIN, GDIM)

RESULTS_FOLDER = Path("LocalFrame")
RESULTS_FOLDER .mkdir(exist_ok=True, parents=True)
with io.VTKFile(MPI.COMM_WORLD, RESULTS_FOLDER / "LocalFrame.pvd", "w") as vtk_f:
    vtk_f.write_mesh(DOMAIN)
    vtk_f.write_function(E1, 0.0)
    vtk_f.write_function(E2, 0.0)
    vtk_f.write_function(E3, 0.0)

# FUNCTION SPACE
Ue          = basix.ufl.element("P",  DOMAIN.basix_cell(), 2, shape=(GDIM,))
Te          = basix.ufl.element("CR", DOMAIN.basix_cell(), 1, shape=(GDIM,))
V           = fem.functionspace(DOMAIN, basix.ufl.mixed_element([Ue, Te]))
v           = fem.Function(V)
u, theta    = ufl.split(v)
v_          = ufl.TestFunction(V)
u_, theta_  = ufl.split(v_)
dv          = ufl.TrialFunction(V)

# SHELL KINEMATIC
eps, kappa, gamma, drilling_strain = shell_strains(u, theta, E1, E2, E3)
eps_             = ufl.derivative(eps,            v, v_)
kappa_           = ufl.derivative(kappa,          v, v_)
gamma_           = ufl.derivative(gamma,          v, v_)
drilling_strain_ = ufl.replace(drilling_strain,  {v: v_})

# MATERIAL
_E1, _E2, _G12, _nu12 = 181E9, 10.3E9, 7.17E9, 0.28
_t_ply  = 6E-3/8
_layup  = [0, 45, -45, 90, 90, -45, 45, 0]

MAT = clt_material(
    layup_angles = _layup,
    t_ply        = _t_ply,
    E1  = _E1,  E2  = _E2,  G12 = _G12,  nu12 = _nu12,
    G13 = _G12, G23 = _G12 * 0.5,
    kappa_s = 5/6,
)

MAT._layup_angles = _layup
MAT._t_ply  = _t_ply
MAT._E1     = _E1;  MAT._E2  = _E2
MAT._G12    = _G12; MAT._nu12 = _nu12

N, M, Q = stress_resultants(MAT, eps, kappa, gamma)
_, drilling_stress = drilling_terms(MAT, DOMAIN, drilling_strain)

ROOT_FACETS = FACET_TAGS.find(11)

Vu, _         = V.sub(0).collapse()
root_dofs_u   = fem.locate_dofs_topological((V.sub(0), Vu), FDIM, ROOT_FACETS)
uD            = fem.Function(Vu);  uD.x.array[:] = 0.0

Vt, _         = V.sub(1).collapse()
root_dofs_t   = fem.locate_dofs_topological((V.sub(1), Vt), FDIM, ROOT_FACETS)
thetaD        = fem.Function(Vt);  thetaD.x.array[:] = 0.0

BCS = [
    fem.dirichletbc(uD,     root_dofs_u, V.sub(0)),
    fem.dirichletbc(thetaD, root_dofs_t, V.sub(1)),
]


# LOAD AND MAP TRACTION FROM OPENFOAM
# import_foam_traction(FOAMFILE, XDMFFILE, verbose=True)
map_traction(XDMFFILE, MESHFILE, MAPFILE)
FTraction = load_traction_xdmf(MAPFILE, DOMAIN, GDIM)
FTraction = - FTraction



# WEAK FORMULATION
dx = ufl.Measure("dx", DOMAIN)

a_int = (
    ufl.inner(N, eps_)
    + ufl.inner(M, kappa_)
    + ufl.inner(Q, gamma_)
    + drilling_stress * drilling_strain_
) * dx

L_ext    = ufl.dot(FTraction, u_) * dx
residual = a_int - L_ext
tangent  = ufl.derivative(residual, v, dv)



# SOLVE
problem = NonlinearProblem(
    F=residual, u=v, bcs=BCS, J=tangent,
    petsc_options_prefix="wing",
    petsc_options={
        "ksp_type"                   : "preonly",
        "pc_type"                    : "lu",
        "pc_factor_mat_solver_type"  : "mumps",
        "snes_type"                  : "newtonls",
        "snes_rtol"                  : 1e-8,
        "snes_atol"                  : 1e-8,
        "snes_max_it"                : 25,
        "snes_monitor"               : None,
    },
)
problem.solve()

converged = problem.solver.getConvergedReason()
n_iter    = problem.solver.getIterationNumber()
print(f"[SNES] converged reason : {converged}")
print(f"[SNES] iterations       : {n_iter}")
assert converged > 0, f"Solver did not converge (reason {converged})"

# POST-PROCESSING
# --- Displacement components ---
disp = v.sub(0).collapse()
vdim_u = disp.function_space.element.value_shape[0]
disp_arr = disp.x.array.reshape(-1, vdim_u)

ux_local = np.max(np.abs(disp_arr[:, 0]))
uy_local = np.max(np.abs(disp_arr[:, 1]))
uz_local = np.max(np.abs(disp_arr[:, 2]))
umag_local = np.max(np.linalg.norm(disp_arr, axis=1))

ux_max = MPI.COMM_WORLD.allreduce(ux_local, op=MPI.MAX)
uy_max = MPI.COMM_WORLD.allreduce(uy_local, op=MPI.MAX)
uz_max = MPI.COMM_WORLD.allreduce(uz_local, op=MPI.MAX)
umag_max = MPI.COMM_WORLD.allreduce(umag_local, op=MPI.MAX)

# --- Rotation components ---
rota = v.sub(1).collapse()
vdim_r = rota.function_space.element.value_shape[0]
rota_arr = rota.x.array.reshape(-1, vdim_r)

rx_local = np.max(np.abs(rota_arr[:, 0]))
ry_local = np.max(np.abs(rota_arr[:, 1]))
rz_local = np.max(np.abs(rota_arr[:, 2]))
rmag_local = np.max(np.linalg.norm(rota_arr, axis=1))

rx_max = MPI.COMM_WORLD.allreduce(rx_local, op=MPI.MAX)
ry_max = MPI.COMM_WORLD.allreduce(ry_local, op=MPI.MAX)
rz_max = MPI.COMM_WORLD.allreduce(rz_local, op=MPI.MAX)
rmag_max = MPI.COMM_WORLD.allreduce(rmag_local, op=MPI.MAX)

if MPI.COMM_WORLD.rank == 0:
    print("\n[POST] DISPLACEMENT")
    print("==============================")
    print(f"|ux|max {ux_max:.6e} m")
    print(f"|uy|max {uy_max:.6e} m")
    print(f"|uz|max {uz_max:.6e} m")
    print(f"|u|max  {umag_max:.6e} m")

    print("\n[POST] ROTATION")
    print("==============================")
    print(f"|rx|max {rx_max:.6e} rad")
    print(f"|ry|max {ry_max:.6e} rad")
    print(f"|rz|max {rz_max:.6e} rad")
    print(f"|r|max  {rmag_max:.6e} rad")
if MAT.kind == "clt":
    STRENGTHS = {
        "Xt": 1500e6, "Xc": 900e6,
        "Yt":   50e6, "Yc": 200e6,
        "S":    70e6,
    }
    print("\n[POST] Tsai-Wu failure indices per ply:")
    FI_max, FI_all = recover_and_evaluate_failure(
        DOMAIN, v, MAT, STRENGTHS, criterion="tsai_wu"
    )

# EXPORT SOLUTION 
RESULT_FOLDER = Path("Result")
RESULT_FOLDER.mkdir(exist_ok=True, parents=True)

Vout      = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))
disp_out  = fem.Function(Vout); disp_out.interpolate(disp); disp_out.name = "Displacement"
rota_out  = fem.Function(Vout); rota_out.interpolate(rota);  rota_out.name = "Rotation"

with io.XDMFFile(MPI.COMM_WORLD, RESULT_FOLDER / "results.xdmf", "w") as xdmf:
    xdmf.write_mesh(DOMAIN)
    xdmf.write_function(disp_out)
    xdmf.write_function(rota_out)

print(f"[EXPORT] Results written to {RESULT_FOLDER / 'results.xdmf'}")

# CLEANUP
problem.solver.destroy()

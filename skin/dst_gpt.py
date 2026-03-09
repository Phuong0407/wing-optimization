import ufl
import basix
from types import SimpleNamespace
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

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


# ════════════════════════════════════════════════════════════════════════
# LOCAL FRAME (symbolic only — no DG interpolation)
# ════════════════════════════════════════════════════════════════════════

def normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))

def local_frame(domain):
    J  = ufl.Jacobian(domain)
    t1 = ufl.as_vector([J[0, 0], J[1, 0], J[2, 0]])
    t2 = ufl.as_vector([J[0, 1], J[1, 1], J[2, 1]])
    e3 = normalize(ufl.cross(t1, t2))

    ey = ufl.as_vector([0, 1, 0])
    ez = ufl.as_vector([0, 0, 1])
    e1_trial = ufl.cross(ey, e3)
    e1 = ufl.conditional(
        ufl.lt(ufl.sqrt(ufl.dot(e1_trial, e1_trial)), 0.5),
        ez,
        normalize(e1_trial),
    )
    e2 = normalize(ufl.cross(e3, e1))
    return e1, e2, e3


# ════════════════════════════════════════════════════════════════════════
# FUNCTION SPACE — P1/P1
# ════════════════════════════════════════════════════════════════════════

def build_space(domain):
    cell = domain.basix_cell()
    gdim = domain.geometry.dim

    Ue = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
    Te = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))

    V  = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Te]))
    v  = fem.Function(V)
    dv = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)

    ndof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"[DST] DOFs: {ndof}")
    return V, v, dv, v_


# ════════════════════════════════════════════════════════════════════════
# KINEMATICS
# ════════════════════════════════════════════════════════════════════════

def hstack(vecs):
    return ufl.as_matrix([[vi[i] for i in range(len(vi))] for vi in vecs]).T

def shell_strains(u, theta, e1, e2, e3):
    P = hstack([e1, e2])

    def tgrad(w):
        return ufl.dot(ufl.grad(w), P)

    t_gu = ufl.dot(P.T, tgrad(u))
    eps  = ufl.sym(t_gu)

    beta  = ufl.cross(e3, theta)
    kappa = ufl.sym(ufl.dot(P.T, tgrad(beta)))

    gamma = tgrad(ufl.dot(u, e3)) - ufl.dot(P.T, beta)

    return eps, kappa, gamma

def compute_drilling(u, theta, e1, e2, e3):
    P    = hstack([e1, e2])
    tgu  = ufl.dot(P.T, ufl.dot(ufl.grad(u), P))   # 2×2 tangential grad
    skew = (tgu[0, 1] - tgu[1, 0]) / 2.0            # skew-symmetric part
    return skew + ufl.dot(theta, e3)                 # true drilling strain


# ════════════════════════════════════════════════════════════════════════
# MATERIAL — isotropic only (CLT can be re-added later)
# ════════════════════════════════════════════════════════════════════════

def isotropic_material(domain, E, nu, thickness):
    h  = fem.Constant(domain, float(thickness))
    E  = fem.Constant(domain, float(E))
    nu = fem.Constant(domain, float(nu))

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu    = E / (2 * (1 + nu))
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)

    return SimpleNamespace(
        h=h, E=E, nu=nu,
        lmbda_ps=lmbda_ps, mu=mu
    )


def plane_stress(mat, e):
    I = ufl.Identity(2)
    return mat.lmbda_ps * ufl.tr(e) * I + 2 * mat.mu * e


# ════════════════════════════════════════════════════════════════════════
# WEAK FORM — MITC3 / DST
# ════════════════════════════════════════════════════════════════════════
def build_forms(domain, V, mat, f_ext=None):

    v  = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)

    u, theta   = ufl.split(v)
    u_, theta_ = ufl.split(v_)

    e1, e2, e3 = local_frame(domain)

    eps,  kappa,  gamma  = shell_strains(u,  theta,  e1, e2, e3)
    eps_, kappa_, gamma_ = shell_strains(u_, theta_, e1, e2, e3)

    N = mat.h * plane_stress(mat, eps)
    M = mat.h**3 / 12.0 * plane_stress(mat, kappa)
    Q = mat.mu * mat.h * gamma

    dx_mb = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
    dx_s  = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 1})
    
    # h_mesh = ufl.CellDiameter(domain)
    # alpha  = mat.mu * mat.h**3 / h_mesh**2

    # drill_u  = compute_drilling(u,  theta,  e1, e2, e3)
    # drill_u_ = compute_drilling(u_, theta_, e1, e2, e3)


    # a = (
    #     ufl.inner(N, eps_) +
    #     ufl.inner(M, kappa_)
    # ) * dx_mb \
    #   + ufl.inner(Q, gamma_) * dx_s \
    #   + alpha * drill_u * drill_u_ * dx_mb

    P      = hstack([e1, e2])
    def drilling(uu, tt):
        tgu  = ufl.dot(P.T, ufl.dot(ufl.grad(uu), P))
        return (tgu[0, 1] - tgu[1, 0]) / 2.0 + ufl.dot(tt, e3)

    h_mesh = ufl.CellDiameter(domain)
    alpha  = mat.mu * mat.h**3 / h_mesh**2

    a = (
          ufl.inner(N, eps_)
        + ufl.inner(M, kappa_)
        + alpha * drilling(u, theta) * drilling(u_, theta_)   # ← fixed
    ) * dx_mb \
      + ufl.inner(Q, gamma_) * dx_s

    if f_ext is not None:
        L = ufl.dot(f_ext, u_) * dx_mb
    else:
        L = 0 * u_[0] * dx_mb

    return a, L, eps, kappa, gamma, N, M, Q


# ════════════════════════════════════════════════════════════════════════
# SOLVER
# ════════════════════════════════════════════════════════════════════════
def solve(a, L, bcs):

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="dst",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    return problem.solve()



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


MESHFILE = "skin1.msh"
FOAMFILE = "../wing.vtp"
XDMFFILE = "FOAMData1.xdmf"
MAPFILE  = "MappedTraction1.xdmf"

# LOAD MESH
MESH = gmsh.read_from_msh(MESHFILE, comm=MPI.COMM_WORLD, gdim=3)
DOMAIN = MESH.mesh
DOMAIN.geometry.x[:] *= 1E-3
CELL_TAGS = MESH.cell_tags
FACET_TAGS = MESH.facet_tags
GDIM = DOMAIN.geometry.dim
TDIM = DOMAIN.topology.dim
FDIM = TDIM - 1

# E1, E2, E3 = local_frame(DOMAIN)

# RESULTS_FOLDER = Path("LocalFrame")
# RESULTS_FOLDER .mkdir(exist_ok=True, parents=True)
# with io.VTKFile(MPI.COMM_WORLD, RESULTS_FOLDER / "LocalFrame_.pvd", "w") as vtk_f:
#     vtk_f.write_mesh(DOMAIN)
#     vtk_f.write_function(E1, 0.0)
#     vtk_f.write_function(E2, 0.0)
#     vtk_f.write_function(E3, 0.0)


V, v, dv, v_ = build_space(DOMAIN)

E  = 210E9
nu = 0.3
t  = 3e-3

MAT = isotropic_material(DOMAIN, E, nu, t)

Vu, _ = V.sub(0).collapse()
Vt, _ = V.sub(1).collapse()

ROOT_FACETS = FACET_TAGS.find(11)
root_dofs_u = fem.locate_dofs_topological((V.sub(0), Vu), FDIM, ROOT_FACETS)
root_dofs_t = fem.locate_dofs_topological((V.sub(1), Vt), FDIM, ROOT_FACETS)

uD = fem.Function(Vu); uD.x.array[:] = 0.0
tD = fem.Function(Vt); tD.x.array[:] = 0.0

BCS = [
    fem.dirichletbc(uD, root_dofs_u, V.sub(0)),
    fem.dirichletbc(tD, root_dofs_t, V.sub(1)),
]


import_foam_traction(FOAMFILE, XDMFFILE, verbose=True)
map_traction(XDMFFILE, MESHFILE, MAPFILE)
FTraction = load_traction_xdmf(MAPFILE, DOMAIN, GDIM)

a, L, eps, kappa, gamma, N, M, Q = build_forms(DOMAIN, V, MAT, f_ext=FTraction)

# v = solve(DOMAIN, V, a, L, BCS)
v = solve(a, L, BCS)



from dolfinx import fem
from mpi4py import MPI
import numpy as np
import ufl
comm = MPI.COMM_WORLD

# # v_sol là nghiệm Function (mixed)
# u_h, theta_h = ufl.split(v)   # IMPORTANT: split của Function -> coefficient, không phải Trial/Test
# # Local frame (symbolic OK)
# e1, e2, e3 = local_frame(DOMAIN)


# # Strains from the SOLUTION (not test-side!)
# eps_h, kappa_h, gamma_h = shell_strains(u_h, theta_h, e1, e2, e3)

# # Stress resultants from SOLUTION
# N_h = MAT.h * plane_stress(MAT, eps_h)
# M_h = MAT.h**3 / 12.0 * plane_stress(MAT, kappa_h)
# Q_h = MAT.mu * MAT.h * gamma_h

# dx_mb = ufl.Measure("dx", domain=DOMAIN, metadata={"quadrature_degree": 2})
# dx_s  = ufl.Measure("dx", domain=DOMAIN, metadata={"quadrature_degree": 1})

# # Strain energy densities (use 0.5!)
# E_mb_form = fem.form(0.5 * (ufl.inner(N_h, eps_h) + ufl.inner(M_h, kappa_h)) * dx_mb)
# E_sh_form = fem.form(0.5 * (ufl.inner(Q_h, gamma_h)) * dx_s)

# E_mb = fem.assemble_scalar(E_mb_form)
# E_sh = fem.assemble_scalar(E_sh_form)

# # Drilling penalty energy (nếu bạn có penalty cho theta_z)
# h = ufl.CellDiameter(DOMAIN)
# k_dr = MAT.mu * MAT.h**3 / h**2          # nếu bạn dùng thêm hệ số alpha thì nhân vào đây
# E_dr_form = fem.form(0.5 * k_dr * theta_h[2]**2 * dx_mb)
# E_dr = fem.assemble_scalar(E_dr_form)

# comm = MPI.COMM_WORLD
# E_mb = comm.allreduce(E_mb, op=MPI.SUM)
# E_sh = comm.allreduce(E_sh, op=MPI.SUM)
# E_dr = comm.allreduce(E_dr, op=MPI.SUM)

# print("\n=== ENERGY CHECK ===")
# print(f"E_mb = {E_mb:.6e}")
# print(f"E_sh = {E_sh:.6e}")
# print(f"E_dr = {E_dr:.6e}")
# print(f"E_dr / (E_mb+E_sh) = {E_dr/(E_mb+E_sh):.3e}")

# ── Correct energy check for MITC3 ─────────────────────────────────
# Use the SAME integration rules as the solver, not full quadrature
dx_mb = ufl.Measure("dx", domain=DOMAIN, metadata={"quadrature_degree": 2})
dx_s  = ufl.Measure("dx", domain=DOMAIN, metadata={"quadrature_degree": 1})  # ← solver rule

u_h, theta_h = v.sub(0).collapse(), v.sub(1).collapse()
e1, e2, e3   = local_frame(DOMAIN)

# Re-split from the Function (not collapsed sub)
u_split, theta_split = ufl.split(v)
eps_h, kappa_h, gamma_h = shell_strains(u_split, theta_split, e1, e2, e3)

N_h = MAT.h         * plane_stress(MAT, eps_h)
M_h = MAT.h**3/12.0 * plane_stress(MAT, kappa_h)
Q_h = MAT.mu * MAT.h * gamma_h

# Solver-consistent energies
E_mb_form   = fem.form(0.5 * ufl.inner(N_h, eps_h)   * dx_mb  # membrane
                     + 0.5 * ufl.inner(M_h, kappa_h)  * dx_mb) # bending
E_sh_form   = fem.form(0.5 * ufl.inner(Q_h, gamma_h) * dx_s)  # ← MITC3 rule

# External work  W = ∫ f·u dΩ
W_ext_form  = fem.form(ufl.dot(FTraction, u_split) * dx_mb)

E_mb  = comm.allreduce(fem.assemble_scalar(E_mb_form),  op=MPI.SUM)
E_sh  = comm.allreduce(fem.assemble_scalar(E_sh_form),  op=MPI.SUM)
W_ext = comm.allreduce(fem.assemble_scalar(W_ext_form), op=MPI.SUM)

E_tot = E_mb + E_sh
ratio = 2 * E_tot / W_ext  # must be ≈ 1.0

print(f"E_mb  = {E_mb:.4e} J")
print(f"E_sh  = {E_sh:.4e} J   (MITC3-integrated — not a true energy)")
print(f"E_tot = {E_tot:.4e} J")
print(f"W_ext = {W_ext:.4e} J")
print(f"2*E_int / W_ext = {ratio:.6f}   (target: 1.000)")











disp = v.sub(0).collapse()
rota = v.sub(1).collapse()
disp.name = "Displacement"
rota.name = "Rotation"

vdim_u   = disp.function_space.element.value_shape[0]
disp_arr = disp.x.array.reshape(-1, vdim_u)
disp_mag = np.linalg.norm(disp_arr, axis=1)
local_max  = disp_mag.max()
global_max = MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)
print(f"\n[POST] Max displacement magnitude : {global_max:.6e} m")

u_sol = v.sub(0).collapse()
theta_sol = v.sub(1).collapse()

import numpy as np
from mpi4py import MPI

u_arr = u_sol.x.array.reshape(-1, 3)
t_arr = theta_sol.x.array.reshape(-1, 3)

comm = MPI.COMM_WORLD

max_u = comm.allreduce(np.max(np.abs(u_arr), axis=0), op=MPI.MAX)
max_t = comm.allreduce(np.max(np.abs(t_arr), axis=0), op=MPI.MAX)

print("\n=== MAX DOFs ===")
print(f"Ux max : {max_u[0]:.6e}")
print(f"Uy max : {max_u[1]:.6e}")
print(f"Uz max : {max_u[2]:.6e}")
print(f"θx max : {max_t[0]:.6e}")
print(f"θy max : {max_t[1]:.6e}")
print(f"θz max : {max_t[2]:.6e}")


# if MAT.kind == "clt":
#     STRENGTHS = {
#         "Xt": 1500e6, "Xc": 900e6,
#         "Yt":   50e6, "Yc": 200e6,
#         "S":    70e6,
#     }
#     print("\n[POST] Tsai-Wu failure indices per ply:")
#     FI_max, FI_all = recover_and_evaluate_failure(
#         DOMAIN, v, MAT, STRENGTHS, criterion="tsai_wu"
#     )

# EXPORT SOLUTION 
RESULT_FOLDER = Path("Result")
RESULT_FOLDER.mkdir(exist_ok=True, parents=True)

Vout      = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))
disp_out  = fem.Function(Vout); disp_out.interpolate(disp); disp_out.name = "Displacement"
rota_out  = fem.Function(Vout); rota_out.interpolate(rota);  rota_out.name = "Rotation"

with io.XDMFFile(MPI.COMM_WORLD, RESULT_FOLDER / "results_.xdmf", "w") as xdmf:
    xdmf.write_mesh(DOMAIN)
    xdmf.write_function(disp_out)
    xdmf.write_function(rota_out)

print(f"[EXPORT] Results written to {RESULT_FOLDER / 'results_.xdmf'}")
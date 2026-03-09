# WING lumped nodal force (same CSV as Cast3M)
import LocalFrame
import MeshImport
from ShellKinematics import shell_strains
from IsotropicShell import IsotropicShell
from dolfinx import fem, io
import basix
import ufl
import numpy as np
from mpi4py import MPI
import dolfinx.fem.petsc
from petsc4py import PETSc
from scipy.spatial import cKDTree
from pathlib import Path

# ── MESH ─────────────────────────────────────────────────────────────────────
MESHFILE = "wing.msh"
DOMAIN, CELL_TAGS, FACET_TAGS = MeshImport.load_mesh(MESHFILE)
GDIM = DOMAIN.geometry.dim
TDIM = DOMAIN.topology.dim
FDIM = TDIM - 1

# ── LOCAL FRAME ───────────────────────────────────────────────────────────────
E1, E2, E3 = LocalFrame.local_frame(DOMAIN, GDIM)

# ── FUNCTION SPACE ────────────────────────────────────────────────────────────
Ue       = basix.ufl.element("P",  DOMAIN.basix_cell(), 1, shape=(GDIM,))
Te       = basix.ufl.element("CR", DOMAIN.basix_cell(), 1, shape=(GDIM,))
V        = fem.functionspace(DOMAIN, basix.ufl.mixed_element([Ue, Te]))
v        = fem.Function(V)
u, theta = ufl.split(v)
v_       = ufl.TestFunction(V)
u_, theta_ = ufl.split(v_)
dv       = ufl.TrialFunction(V)




# # ── SHELL KINEMATICS ──────────────────────────────────────────────────────────
# eps, kappa, gamma, eps_d = shell_strains(u=u, theta=theta, e1=E1, e2=E2, e3=E3)
# eps_   = ufl.derivative(eps,   v, v_)
# kappa_ = ufl.derivative(kappa, v, v_)
# gamma_ = ufl.derivative(gamma, v, v_)
# eps_d_ = ufl.replace(eps_d, {v: v_})

# # ── MATERIAL ──────────────────────────────────────────────────────────────────
# thickness = 1e-3
# Young     = 210e9
# Poisson   = 0.3
# material  = IsotropicShell(DOMAIN, thickness=thickness, E=Young, nu=Poisson)
# N = material.membrane_stress(eps)
# M = material.bending_stress(kappa)
# Q = material.shear_stress(gamma)

# alpha_drilling     = fem.Constant(DOMAIN, 1e-4)
# drilling_stiffness = alpha_drilling * Young * thickness
# drilling_stress    = drilling_stiffness * eps_d




# ── SHELL KINEMATICS ──────────────────────────────────────────────────────────
# Test function strains (v_)
du_, dtheta_ = ufl.split(v_)
eps_, kappa_, gamma_, eps_d_ = shell_strains(u=du_, theta=dtheta_, e1=E1, e2=E2, e3=E3)

# Trial function strains (dv)
du, dtheta = ufl.split(dv)
eps, kappa, gamma, eps_d = shell_strains(u=du, theta=dtheta, e1=E1, e2=E2, e3=E3)

# ── MATERIAL ──────────────────────────────────────────────────────────────────
thickness = 1e-3
Young     = 210e9
Poisson   = 0.3
material  = IsotropicShell(DOMAIN, thickness=thickness, E=Young, nu=Poisson)
N = material.membrane_stress(eps)
M = material.bending_stress(kappa)
Q = material.shear_stress(gamma)
alpha_drilling     = fem.Constant(DOMAIN, 1.0)
drilling_stiffness = alpha_drilling * Young * thickness
drilling_stress    = drilling_stiffness * eps_d







# ── BOUNDARY CONDITIONS ───────────────────────────────────────────────────────
ROOT_FACETS = FACET_TAGS.find(11)

Vu, _        = V.sub(0).collapse()
root_dofs_u  = fem.locate_dofs_topological((V.sub(0), Vu), FDIM, ROOT_FACETS)
uD           = fem.Function(Vu)
uD.x.array[:] = 0.0

Vtheta, _       = V.sub(1).collapse()
root_dofs_theta = fem.locate_dofs_topological((V.sub(1), Vtheta), FDIM, ROOT_FACETS)
thetaD          = fem.Function(Vtheta)
thetaD.x.array[:] = 0.0

BC_DISP = fem.dirichletbc(uD,     root_dofs_u,     V.sub(0))
BC_ROTA = fem.dirichletbc(thetaD, root_dofs_theta, V.sub(1))
BCS     = [BC_DISP, BC_ROTA]

# ── LOAD: read the same CSV as Cast3M ────────────────────────────────────────
data       = np.loadtxt("CASTEMForce.csv", delimiter=";", skiprows=1)
CSV_POINTS = data[:, 0:3]   # [m]
CSV_FORCES = data[:, 3:6]   # [N]
print("Total force from CSV [N]:", CSV_FORCES.sum(axis=0))
print("Reference              : [184, -4294, 17846]")

# Map CSV nodes → FEniCS displacement DOFs by coordinates
# P2 elements have midpoint DOFs — only vertex DOFs match the CSV exactly
Vu_load, u_dofs_in_V = V.sub(0).collapse()
DOF_COORDS = Vu_load.tabulate_dof_coordinates()  # all DOFs including midpoints

TREE      = cKDTree(CSV_POINTS)
DIST, IDX = TREE.query(DOF_COORDS)

# Identify which DOFs are vertices (distance ≈ 0) vs midpoints (distance > 0)
VERTEX_MASK = DIST < 1e-6
MIDPT_MASK  = ~VERTEX_MASK

# print(f"Vertex DOFs matched : {VERTEX_MASK.sum()}")
# print(f"Midpoint DOFs (P2)  : {MIDPT_MASK.sum()}")
# print(f"Max distance at vertices [m]: {DIST[VERTEX_MASK].max():.2e}")
# print(f"Min distance at midpoints [m]: {DIST[MIDPT_MASK].min():.2e}")

# assert VERTEX_MASK.any(), "No DOFs matched CSV — check mesh units!"
# assert DIST[VERTEX_MASK].max() < 1e-6, "Vertex match failed!"

# # Each DOF coordinate corresponds to one scalar DOF in the vector function space
# # Vu_load is a vector space of shape (GDIM,), so DOF_COORDS has shape (N_dofs*GDIM/GDIM, 3)
# # but x.array has shape (N_dofs*GDIM,) — need to assign component by component

# b_load = fem.Function(Vu_load)
# b_load.x.array[:] = 0.0

# vertex_indices = np.where(VERTEX_MASK)[0]   # indices into DOF_COORDS
# matched_forces = CSV_FORCES[IDX[VERTEX_MASK]]  # shape (N_vertices, 3)

# # DOF_COORDS has one row per node, but x.array is flat with GDIM entries per node
# for comp in range(GDIM):
#     b_load.x.array[vertex_indices * GDIM + comp] = matched_forces[:, comp]

# b_load.x.scatter_forward()
# print("Total FEniCS force [N]:", 
#       b_load.x.array.reshape(-1, GDIM).sum(axis=0))







print(f"Vertex DOFs matched : {VERTEX_MASK.sum()}")
print(f"Midpoint DOFs (P2)  : {MIDPT_MASK.sum()}")
print(f"Max distance at vertices [m]: {DIST[VERTEX_MASK].max():.2e}")
if MIDPT_MASK.sum() > 0:
    print(f"Min distance at midpoints [m]: {DIST[MIDPT_MASK].min():.2e}")
else:
    print(f"No midpoint DOFs (P1 elements) ✅")

# P1: all DOFs are vertices, assign directly
b_load = fem.Function(Vu_load)
b_load.x.array[:] = 0.0
vertex_indices = np.where(VERTEX_MASK)[0]
matched_forces = CSV_FORCES[IDX[VERTEX_MASK]]
for comp in range(GDIM):
    b_load.x.array[vertex_indices * GDIM + comp] = matched_forces[:, comp]
b_load.x.scatter_forward()
print("Total FEniCS force [N]:",
      b_load.x.array.reshape(-1, GDIM).sum(axis=0))




# ── WEAK FORM (internal virtual work only — no L_ext in UFL) ─────────────────
# a_int = (
#     ufl.inner(N, eps_)
#   + ufl.inner(M, kappa_)
#   + ufl.inner(Q, gamma_)
#   + drilling_stress * eps_d_
# ) * ufl.dx

# tangent = ufl.derivative(a_int, v, dv)



# ── WEAK FORM: proper bilinear form ──────────────────────────────────────────
a_form = fem.form((
    ufl.inner(N, eps_)
  + ufl.inner(M, kappa_)
  + ufl.inner(Q, gamma_)
  + drilling_stress * eps_d_
) * ufl.dx)












# # ── ASSEMBLE ──────────────────────────────────────────────────────────────────
# a_form = fem.form(tangent)
# A      = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=BCS)
# A.assemble()

# # Create RHS vector with the correct size from the bilinear form
# b = A.createVecRight()
# b.zeroEntries()

# # Get the global DOF indices of the displacement subspace in the mixed space
# u_local_size = Vu.dofmap.index_map.size_local * Vu.dofmap.index_map_bs
# u_dofs = V.sub(0).collapse()[1]   # mapping from Vu DOFs → V DOFs

# b_array = b.getArray(readonly=False)
# b_array[u_dofs] = b_load.x.array[:u_local_size]
# b.setArray(b_array)

# # Apply BCs
# dolfinx.fem.petsc.apply_lifting(b, [a_form], bcs=[BCS])
# b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# dolfinx.fem.petsc.set_bc(b, BCS)

# # ── ASSEMBLE ──────────────────────────────────────────────────────────────────
# a_form = fem.form(tangent)

A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=BCS)
A.assemble()









b = A.createVecRight()
b.zeroEntries()

# Inject nodal forces into displacement block
b_array = b.getArray(readonly=False)
b_array[u_dofs_in_V] = b_load.x.array
b.setArray(b_array)

# All Dirichlet BCs are zero — no lifting needed, just zero out BC DOFs
dolfinx.fem.petsc.set_bc(b, BCS)

# ── SOLVE ─────────────────────────────────────────────────────────────────────
ksp = PETSc.KSP().create(DOMAIN.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.solve(b, v.x.petsc_vec)
v.x.petsc_vec.ghostUpdate(
    addv=PETSc.InsertMode.INSERT,
    mode=PETSc.ScatterMode.FORWARD
)
print(f"KSP converged reason: {ksp.getConvergedReason()}")
assert ksp.getConvergedReason() > 0, "Solver did not converge!"


# # ── STIFFNESS DIAGNOSTIC ──────────────────────────────────────────────────────
# # Check total integrated area of the mesh
# from dolfinx.fem.petsc import assemble_matrix
# one = fem.Constant(DOMAIN, 1.0)
# area_form = fem.form(one * ufl.dx)
# total_area = fem.assemble_scalar(area_form)
# print(f"Total mesh area [m²]: {total_area:.6f}")

# # Expected: wing surface area from CAD
# # If this is 2× the expected area, the mesh has doubled faces (top+bottom)


# # ── SOLVE ─────────────────────────────────────────────────────────────────────
# ksp = PETSc.KSP().create(DOMAIN.comm)
# ksp.setOperators(A)
# ksp.setType("preonly")
# ksp.getPC().setType("lu")
# ksp.getPC().setFactorSolverType("mumps")
# ksp.solve(b, v.x.petsc_vec)
# v.x.petsc_vec.ghostUpdate(
#     addv=PETSc.InsertMode.INSERT,
#     mode=PETSc.ScatterMode.FORWARD
# )
# print(f"KSP converged reason: {ksp.getConvergedReason()}")
# assert ksp.getConvergedReason() > 0, "Solver did not converge!"

# ── LOCAL ROTATIONS (comparable to Cast3M RX, RY, RZ) ────────────────────────
Vscalar    = fem.functionspace(DOMAIN, ("DG", 0))
theta_fn   = v.sub(1)
theta_loc1 = LocalFrame.project_scalar(ufl.dot(theta_fn, E2), Vscalar, DOMAIN)
theta_loc2 = LocalFrame.project_scalar(ufl.dot(theta_fn, E1), Vscalar, DOMAIN)
theta_loc3 = LocalFrame.project_scalar(ufl.dot(theta_fn, E3), Vscalar, DOMAIN)
print(f"Max theta_local_1 (≈ RX) = {np.max(np.abs(theta_loc1.x.array)):.6e} rad")
print(f"Max theta_local_2 (≈ RY) = {np.max(np.abs(theta_loc2.x.array)):.6e} rad")
print(f"Max theta_local_3 (≈ RZ) = {np.max(np.abs(theta_loc3.x.array)):.6e} rad")

# ── POST-PROCESSING ───────────────────────────────────────────────────────────
disp = v.sub(0).collapse()
rota = v.sub(1).collapse()
disp.name = "Displacement"
rota.name = "Rotation"

disp_arr = disp.x.array.reshape(-1, GDIM)
rota_arr = rota.x.array.reshape(-1, GDIM)
print("disp.name =", disp.name)
print("rota.name =", rota.name)
for i in range(GDIM):
    print(f"Max u[{i}]     = {np.max(np.abs(disp_arr[:, i])):.6e} m")
for i in range(GDIM):
    print(f"Max theta[{i}] = {np.max(np.abs(rota_arr[:, i])):.6e} rad")

# ── EXPORT ────────────────────────────────────────────────────────────────────
# RESULT_FOLDER = Path("Results")
# RESULT_FOLDER.mkdir(exist_ok=True, parents=True)

# Vout = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))

# disp_out = fem.Function(Vout, name="Displacement")
# disp_out.interpolate(disp)

# rota_out = fem.Function(Vout, name="Rotation")
# rota_out.interpolate(rota)

# with io.XDMFFile(MPI.COMM_WORLD, RESULT_FOLDER / "displacement.xdmf", "w") as xdmf:
#     xdmf.write_mesh(DOMAIN)
#     xdmf.write_function(disp_out)
#     xdmf.write_function(rota_out)





# # Check total force actually injected into the system
# print("Sum of b_load [N]:", b_load.x.array.reshape(-1, GDIM).sum(axis=0))
# print("CSV total force [N]:", CSV_FORCES.sum(axis=0))
# print("Ratio:", b_load.x.array.reshape(-1, GDIM).sum(axis=0) / CSV_FORCES.sum(axis=0))

# print("Number of CSV nodes:", len(CSV_POINTS))
# print("Number of FEniCS vertex DOFs matched:", VERTEX_MASK.sum())
# print("Number of unique u_dofs_in_V:", len(np.unique(u_dofs_in_V)))
# print("Number of total u_dofs_in_V:", len(u_dofs_in_V))



# print("eps ufl_shape:", eps.ufl_shape)
# print("kappa ufl_shape:", kappa.ufl_shape)
# print("gamma ufl_shape:", gamma.ufl_shape)





# ── ENERGY BALANCE DIAGNOSTIC ─────────────────────────────────────────────────
# After solve, compute each energy contribution separately

u_sol, theta_sol = ufl.split(v)
eps_sol, kappa_sol, gamma_sol, eps_d_sol = shell_strains(
    u=u_sol, theta=theta_sol, e1=E1, e2=E2, e3=E3)
N_sol = material.membrane_stress(eps_sol)
M_sol = material.bending_stress(kappa_sol)
Q_sol = material.shear_stress(gamma_sol)
drilling_stress_sol = drilling_stiffness * eps_d_sol

E_membrane = fem.assemble_scalar(fem.form(ufl.inner(N_sol, eps_sol) * ufl.dx))
E_bending  = fem.assemble_scalar(fem.form(ufl.inner(M_sol, kappa_sol) * ufl.dx))
E_shear    = fem.assemble_scalar(fem.form(ufl.inner(Q_sol, gamma_sol) * ufl.dx))
E_drilling = fem.assemble_scalar(fem.form(drilling_stress_sol * eps_d_sol * ufl.dx))
E_internal = E_membrane + E_bending + E_shear + E_drilling  # ← add this
Fu         = np.dot(b.getArray(), v.x.petsc_vec.getArray())  # ← add this

print(f"E_membrane : {E_membrane:.6e} J")
print(f"E_bending  : {E_bending:.6e} J")
print(f"E_shear    : {E_shear:.6e} J")
print(f"E_drilling : {E_drilling:.6e} J")
print(f"E_internal : {E_internal:.6e} J")
print(f"F·u        : {Fu:.6e} J")
print(f"Ratio F·u / (2*E_internal) : {Fu / (2*E_internal):.6f}")  # should be 1.0





# True F·u: use the original unmodified force vector
b_true = A.createVecRight()
b_true.zeroEntries()
b_array = b_true.getArray(readonly=False)
b_array[u_dofs_in_V] = b_load.x.array
b_true.setArray(b_array)
# Do NOT apply set_bc to b_true

Fu_true = b_true.dot(v.x.petsc_vec)
print(f"True F·u = {Fu_true:.6e} J")
print(f"2 * E_internal = {2*E_internal:.6e} J")
print(f"Ratio = {Fu_true / (2*E_internal):.6f}")  # should be 1.0










import numpy as np
from dolfinx import fem
from mpi4py import MPI

# Create a standalone displacement space matching your Ue (P1 vector)
Vu_pure = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))

# Locate clamped dofs in this pure displacement space
root_facets = FACET_TAGS.find(11)
clamped_u_pure = fem.locate_dofs_topological(Vu_pure, FDIM, root_facets)

print("Pure displacement space:")
print("  total scalar dofs:", Vu_pure.dofmap.index_map.size_local * Vu_pure.dofmap.index_map_bs)
print("  clamped scalar dofs:", len(clamped_u_pure))

# DOF coordinates (one row per scalar dof for vector spaces in dolfinx)
dof_coords = Vu_pure.tabulate_dof_coordinates().reshape(-1, GDIM)

clamped_coords = dof_coords[clamped_u_pure]

print("Clamped bbox xmin:", clamped_coords.min(axis=0))
print("Clamped bbox xmax:", clamped_coords.max(axis=0))

x = clamped_coords[:, 0]
print("x-plane stats [m]:")
print("min |x|:", np.min(np.abs(x)))
print("max |x|:", np.max(np.abs(x)))
print("mean|x|:", np.mean(np.abs(x)))










# ── CLEANUP ───────────────────────────────────────────────────────────────────
ksp.destroy()
A.destroy()
b.destroy()


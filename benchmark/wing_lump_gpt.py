import numpy as np
from mpi4py import MPI
from pathlib import Path
from scipy.spatial import cKDTree

import ufl
import basix
from dolfinx import fem
import dolfinx.fem.petsc
from petsc4py import PETSc

import LocalFrame
import MeshImport
from ShellKinematics import shell_strains
from IsotropicShell import IsotropicShell


# =============================================================================
# 0) Inputs
# =============================================================================
MESHFILE = "wing.msh"
CSVFILE  = "CASTEMForce.csv"

THICKNESS = 1e-3
YOUNG     = 210e9
NU        = 0.3

ROOT_X    = 0.0
ROOT_TOL  = 1e-3   # Cast3M plane tolerance


# =============================================================================
# 1) Mesh
# =============================================================================
DOMAIN, CELL_TAGS, FACET_TAGS = MeshImport.load_mesh(MESHFILE)
GDIM = DOMAIN.geometry.dim
TDIM = DOMAIN.topology.dim
FDIM = TDIM - 1

# Local frame
E1, E2, E3 = LocalFrame.local_frame(DOMAIN, GDIM)

DOMAIN.topology.create_connectivity(0, DOMAIN.topology.dim)

# =============================================================================
# 2) Function spaces (match what you used)
#    u: P1 vector, theta: CR1 vector
# =============================================================================
Ue = basix.ufl.element("P",  DOMAIN.basix_cell(), 1, shape=(GDIM,))
Te = basix.ufl.element("CR", DOMAIN.basix_cell(), 1, shape=(GDIM,))
V  = fem.functionspace(DOMAIN, basix.ufl.mixed_element([Ue, Te]))

v        = fem.Function(V, name="state")
u, theta = ufl.split(v)

v_         = ufl.TestFunction(V)
u_, theta_ = ufl.split(v_)

dv       = ufl.TrialFunction(V)
du, dth  = ufl.split(dv)


# =============================================================================
# 3) Shell strains for bilinear form (trial vs test explicitly)
# =============================================================================
# Test-function strains
eps_t, kappa_t, gamma_t, epsd_t = shell_strains(u=u_, theta=theta_, e1=E1, e2=E2, e3=E3)
# Trial-function strains
eps_u, kappa_u, gamma_u, epsd_u = shell_strains(u=du, theta=dth, e1=E1, e2=E2, e3=E3)

material = IsotropicShell(DOMAIN, thickness=THICKNESS, E=YOUNG, nu=NU)

N_u = material.membrane_stress(eps_u)
M_u = material.bending_stress(kappa_u)
Q_u = material.shear_stress(gamma_u)

# Drilling stabilization (keep, but make it easy to tune)
alpha_drilling     = fem.Constant(DOMAIN, 1.0)
drilling_stiffness = alpha_drilling * YOUNG * THICKNESS
drill_u            = drilling_stiffness * epsd_u

a = (
    ufl.inner(N_u, eps_t)
  + ufl.inner(M_u, kappa_t)
  + ufl.inner(Q_u, gamma_t)
  + drill_u * epsd_t
) * ufl.dx

a_form = fem.form(a)


# =============================================================================
# 4) Cast3M-equivalent root clamp: plane x = 0 +/- 1e-3
#    IMPORTANT: do NOT use Physical Curve("root",11)
# =============================================================================
# def root_plane(x):
#     return np.isclose(x[0], ROOT_X, atol=ROOT_TOL)

# # Collapse subspaces once
# Vu, u_to_V = V.sub(0).collapse()
# Vt, t_to_V = V.sub(1).collapse()

# root_dofs_u = fem.locate_dofs_geometrical(Vu, root_plane)
# root_dofs_t = fem.locate_dofs_geometrical(Vt, root_plane)

# uD = fem.Function(Vu)
# uD.x.array[:] = 0.0

# tD = fem.Function(Vt)
# tD.x.array[:] = 0.0

# bc_u = fem.dirichletbc(uD, root_dofs_u, V.sub(0))
# bc_t = fem.dirichletbc(tD, root_dofs_t, V.sub(1))
# BCS  = [bc_u, bc_t]

# # --- diagnostic: what did we clamp? (bbox in physical coordinates)
# dof_coords_u = Vu.tabulate_dof_coordinates().reshape(-1, GDIM)
# clamped_u_coords = dof_coords_u[root_dofs_u]
# if DOMAIN.comm.rank == 0:
#     print("Root clamp (plane-based) diagnostics:")
#     print("  clamped scalar dofs (u):", len(root_dofs_u))
#     if len(root_dofs_u) > 0:
#         print("  bbox xmin:", clamped_u_coords.min(axis=0))
#         print("  bbox xmax:", clamped_u_coords.max(axis=0))
#         xabs = np.abs(clamped_u_coords[:, 0])
#         print("  x stats: min|x|", xabs.min(), "max|x|", xabs.max(), "mean|x|", xabs.mean())
#     print()


# Collapse subspaces once
Vu, _ = V.sub(0).collapse()
Vt, _ = V.sub(1).collapse()

def root_plane(x):
    return np.isclose(x[0], ROOT_X, atol=ROOT_TOL)

# Locate DOFs (returns numpy array already for geometrical locator)
root_dofs_u = fem.locate_dofs_geometrical((V.sub(0), Vu), root_plane)
root_dofs_t = fem.locate_dofs_geometrical((V.sub(1), Vt), root_plane)

# Ensure correct dtype
root_dofs_u = np.array(root_dofs_u, dtype=np.int32)
root_dofs_t = np.array(root_dofs_t, dtype=np.int32)

# Zero functions on collapsed spaces
uD = fem.Function(Vu)
uD.x.array[:] = 0.0

tD = fem.Function(Vt)
tD.x.array[:] = 0.0

# IMPORTANT: pass subspace explicitly here
bc_u = fem.dirichletbc(uD, root_dofs_u, V.sub(0))
bc_t = fem.dirichletbc(tD, root_dofs_t, V.sub(1))

BCS = [bc_u, bc_t]










# =============================================================================
# 5) Load: read same CSV as Cast3M and map to displacement DOFs (component-wise)
# =============================================================================
data       = np.loadtxt(CSVFILE, delimiter=";", skiprows=1)
CSV_POINTS = data[:, 0:3]   # [m]
CSV_FORCES = data[:, 3:6]   # [N]

if DOMAIN.comm.rank == 0:
    print("Total force from CSV [N]:", CSV_FORCES.sum(axis=0))
    print("Reference                : [184, -4294, 17846]")
    print()

# Map CSV points -> mesh vertex coordinates (geometric)
# We build a vertex-based force table first, then scatter into DOFs.
Xvert = DOMAIN.geometry.x
treeV = cKDTree(CSV_POINTS)
distV, idxV = treeV.query(Xvert, k=1)

# Use a tight tolerance; if this fails, your CSV coords and mesh coords do not match.
tol_match = 1e-10
if DOMAIN.comm.rank == 0:
    print("CSV↔mesh vertex match:")
    print("  max nearest distance [m]:", distV.max())
    print("  mean nearest distance [m]:", distV.mean())
    print()

if distV.max() > 1e-6:
    if DOMAIN.comm.rank == 0:
        print("WARNING: CSV points do not coincide with mesh vertices (max dist > 1e-6).")
        print("         Your 'lumped nodal forces' are not on the same node set.")
        print("         You should export CSV directly from this mesh, or reduce tolerance carefully.")
        print()

# Vertex forces (size = nVertices x 3)
F_vertex = CSV_FORCES[idxV]

# Now populate b_load on Vu (vector P1 space) robustly component-wise
b_load = fem.Function(Vu)
b_load.x.array[:] = 0.0

# For each component i, locate dofs for that component and set values at matching vertices.
# This avoids assuming any special dof ordering.
# --- Build missing connectivity for any later topo queries (safe anyway)
DOMAIN.topology.create_connectivity(0, DOMAIN.topology.dim)

# b_load on collapsed displacement space Vu (vector P1)
b_load = fem.Function(Vu)
b_load.x.array[:] = 0.0

# DOF coordinates for Vu (for vector spaces, dolfinx repeats coords per component)
dof_coords = Vu.tabulate_dof_coordinates().reshape(-1, GDIM)

# Map each dof coordinate to nearest CSV point
tree = cKDTree(CSV_POINTS)
dist, idx = tree.query(dof_coords, k=1)

# Sanity: should be ~1e-14 like your vertex check
if DOMAIN.comm.rank == 0:
    print("DOF↔CSV match:")
    print("  max dist:", dist.max())
    print("  mean dist:", dist.mean())

# Assign the correct component.
# In dolfinx, for a vector space with block size = GDIM,
# dofs are typically ordered with repeating coordinates; the component is dof_index % GDIM.
for j in range(GDIM):
    comp_mask = (np.arange(len(dist)) % GDIM) == j
    b_load.x.array[comp_mask] = CSV_FORCES[idx[comp_mask], j]

b_load.x.scatter_forward()

if DOMAIN.comm.rank == 0:
    print("Total b_load (DOF-sum, diagnostic only):",
          b_load.x.array.reshape(-1, GDIM).sum(axis=0))
b_load.x.scatter_forward()

if DOMAIN.comm.rank == 0:
    # This sum is on DOFs, so not directly "total force" unless 1 dof per vertex per comp.
    # Still a useful sanity signal.
    print("Loaded b_load stats:")
    arr = b_load.x.array
    print("  min/max:", arr.min(), arr.max())
    print()


# =============================================================================
# 6) Assemble matrix and RHS; inject forces into the mixed system
# =============================================================================
A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=BCS)
A.assemble()

b = A.createVecRight()
b.zeroEntries()

# Inject nodal forces into the displacement block using the collapse map u_to_V:
b_arr = b.getArray(readonly=False)
b_arr[u_to_V] = b_load.x.array
b.setArray(b_arr)

# Apply zero Dirichlet conditions (no lifting needed for zero BCs)
dolfinx.fem.petsc.set_bc(b, BCS)


# =============================================================================
# 7) Solve (linear)
# =============================================================================
ksp = PETSc.KSP().create(DOMAIN.comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

ksp.solve(b, v.x.petsc_vec)
v.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

if DOMAIN.comm.rank == 0:
    print("KSP converged reason:", ksp.getConvergedReason())
assert ksp.getConvergedReason() > 0, "Solver did not converge!"


# =============================================================================
# 8) Post-process comparable outputs
# =============================================================================
disp = v.sub(0).collapse()
rota = v.sub(1).collapse()

disp_arr = disp.x.array.reshape(-1, GDIM)
rota_arr = rota.x.array.reshape(-1, GDIM)

if DOMAIN.comm.rank == 0:
    for i in range(GDIM):
        print(f"Max u[{i}]     = {np.max(np.abs(disp_arr[:, i])):.6e} m")
    for i in range(GDIM):
        print(f"Max theta[{i}] = {np.max(np.abs(rota_arr[:, i])):.6e} rad")

# Local rotations (as you did)
Vscalar    = fem.functionspace(DOMAIN, ("DG", 0))
theta_fn   = v.sub(1)
theta_loc1 = LocalFrame.project_scalar(ufl.dot(theta_fn, E2), Vscalar, DOMAIN)
theta_loc2 = LocalFrame.project_scalar(ufl.dot(theta_fn, E1), Vscalar, DOMAIN)
theta_loc3 = LocalFrame.project_scalar(ufl.dot(theta_fn, E3), Vscalar, DOMAIN)

if DOMAIN.comm.rank == 0:
    print(f"Max theta_local_1 (≈ RX) = {np.max(np.abs(theta_loc1.x.array)):.6e} rad")
    print(f"Max theta_local_2 (≈ RY) = {np.max(np.abs(theta_loc2.x.array)):.6e} rad")
    print(f"Max theta_local_3 (≈ RZ) = {np.max(np.abs(theta_loc3.x.array)):.6e} rad")


# =============================================================================
# 9) Energy/work diagnostic (expect ratio 0.5 for linear system)
# =============================================================================
u_sol, theta_sol = ufl.split(v)
eps_sol, kappa_sol, gamma_sol, epsd_sol = shell_strains(u=u_sol, theta=theta_sol, e1=E1, e2=E2, e3=E3)

N_sol = material.membrane_stress(eps_sol)
M_sol = material.bending_stress(kappa_sol)
Q_sol = material.shear_stress(gamma_sol)
drill_sol = drilling_stiffness * epsd_sol

E_mem = fem.assemble_scalar(fem.form(ufl.inner(N_sol, eps_sol) * ufl.dx))
E_ben = fem.assemble_scalar(fem.form(ufl.inner(M_sol, kappa_sol) * ufl.dx))
E_shr = fem.assemble_scalar(fem.form(ufl.inner(Q_sol, gamma_sol) * ufl.dx))
E_drl = fem.assemble_scalar(fem.form(drill_sol * epsd_sol * ufl.dx))

E_int = E_mem + E_ben + E_shr + E_drl

# Use the *unmodified* RHS to compute true work
b_true = A.createVecRight()
b_true.zeroEntries()
b_true_arr = b_true.getArray(readonly=False)
b_true_arr[u_to_V] = b_load.x.array
b_true.setArray(b_true_arr)

Fu_true = b_true.dot(v.x.petsc_vec)

if DOMAIN.comm.rank == 0:
    print(f"E_membrane : {E_mem:.6e} J")
    print(f"E_bending  : {E_ben:.6e} J")
    print(f"E_shear    : {E_shr:.6e} J")
    print(f"E_drilling : {E_drl:.6e} J")
    print(f"E_internal : {E_int:.6e} J")
    print(f"True F·u   : {Fu_true:.6e} J")
    print(f"Ratio F·u / (2*E_internal): {Fu_true/(2*E_int):.6f}")


# =============================================================================
# 10) Export
# =============================================================================
outdir = Path("Results_planeBC")
outdir.mkdir(exist_ok=True, parents=True)

disp.name = "Displacement"
rota.name = "Rotation"

with dolfinx.io.VTKFile(DOMAIN.comm, outdir / "results.pvd", "w") as vtk:
    vtk.write_mesh(DOMAIN)
    vtk.write_function(disp, 0.0)
    vtk.write_function(rota, 0.0)


# =============================================================================
# 11) Cleanup
# =============================================================================
ksp.destroy()
A.destroy()
b.destroy()
b_true.destroy()

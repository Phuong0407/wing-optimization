import LocalFrame
import MeshImport
from ShellKinematics import shell_strains
from IsotropicShell import IsotropicShell
from FOAMImport import import_foam_traction, is_xdmffile_valie, load_traction_from_xdmf
from dolfinx import mesh, fem, io, plot
import basix
import ufl

import os
import numpy as np
from mpi4py import MPI
import dolfinx.fem.petsc
from dolfinx.fem.petsc import NonlinearProblem
from pathlib import Path





# MESH IMPORT
MESHFILE = "wing.msh"
DOMAIN, CELL_TAGS, FACET_TAGS = MeshImport.load_mesh(MESHFILE)
GDIM = DOMAIN.geometry.dim
TDIM = DOMAIN.topology.dim
FDIM = TDIM - 1

# LOCAL FRAME
E1, E2, E3 = LocalFrame.local_frame(DOMAIN,GDIM)

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
eps, kappa, gamma, eps_d = shell_strains(u=u, theta=theta, e1=E1, e2=E2, e3=E3)

eps_    = ufl.derivative(eps, v, v_)
kappa_  = ufl.derivative(kappa, v, v_)
gamma_  = ufl.derivative(gamma, v, v_)
eps_d_  = ufl.replace(eps_d, {v: v_})

thickness   = 6E-3
Young       = 210E9
Poisson     = 0.3
material    = IsotropicShell(DOMAIN, thickness=thickness, E=Young, nu=Poisson)

N = material.membrane_stress(eps)
M = material.bending_stress(kappa)
Q = material.shear_stress(gamma)

# h_mesh = ufl.CellDiameter(DOMAIN)
alpha_drilling = fem.Constant(DOMAIN, 1.0)
# alpha_drilling = fem.Constant(DOMAIN, 1e-4)
drilling_stiffness = alpha_drilling * Young * thickness
drilling_stress = drilling_stiffness * eps_d

GMSH_ROOT_ID = 11
ROOT_FACETS = FACET_TAGS.find(GMSH_ROOT_ID)

Vu, _ = V.sub(0).collapse()
root_dofs_u = fem.locate_dofs_topological((V.sub(0), Vu),FDIM,ROOT_FACETS)
uD = fem.Function(Vu)
uD.x.array[:] = 0.0

Vtheta, _ = V.sub(1).collapse()
root_dofs_theta = fem.locate_dofs_topological((V.sub(1), Vtheta),FDIM,ROOT_FACETS)
thetaD = fem.Function(Vtheta)
thetaD.x.array[:] = 0.0

BC_DISP = fem.dirichletbc(uD, root_dofs_u, V.sub(0))
BC_ROTA = fem.dirichletbc(thetaD, root_dofs_theta, V.sub(1))
BCS = [BC_DISP, BC_ROTA]

FOAMFILE = "wing.vtp"
XDMFFILE = "FOAMData.xdmf"
# if is_xdmffile_valie(XDMFFILE):
#   print(f"{XDMFFILE} already exists, skipping import.")
# else:
import_foam_traction(FOAMFILE,XDMFFILE)
FTraction = load_traction_from_xdmf(XDMFFILE, DOMAIN, GDIM)



# WEAK FORMULATION
## distance in functional space
dx = ufl.Measure("dx", DOMAIN)
ds = ufl.Measure("ds", DOMAIN)
## internal virtual work
a_int = (
  ufl.inner(N, eps_)
  + ufl.inner(M, kappa_)
  + ufl.inner(Q, gamma_)
  + drilling_stress * eps_d_
) * dx
## external virtual work
L_ext       = ufl.dot(FTraction, u_) * dx
residual    = a_int - L_ext
tangent     = ufl.derivative(residual, v, dv)

# SOLVE
## solver declaration
problem = NonlinearProblem(
  F=residual,
  u=v,
  bcs=BCS,
  J=tangent,
  petsc_options_prefix="wing",
  petsc_options={
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_type": "newtonls",
    "snes_rtol": 1e-8,
    "snes_atol": 1e-8,
    "snes_max_it": 25,
    "snes_monitor": None,
  }
)
## solution of problem
problem.solve()



# =========================
# ENERGY DEBUG (MPI-safe)
# =========================
from dolfinx import fem
import ufl
import numpy as np
from mpi4py import MPI

dx = ufl.Measure("dx", domain=DOMAIN)

# IMPORTANT:
# Use energy densities: 1/2 * (stress resultant : corresponding strain)
# (valid for linear elastic constitutive laws N(eps), M(kappa), Q(gamma))

# Internal energy contributions
U_mem_form   = fem.form(0.5 * ufl.inner(N, eps) * dx)
U_bend_form  = fem.form(0.5 * ufl.inner(M, kappa) * dx)
U_shear_form = fem.form(0.5 * ufl.inner(Q, gamma) * dx)

# Drilling: eps_d is scalar-like in your formulation
# drilling_stress = drilling_stiffness * eps_d
U_drill_form = fem.form(0.5 * (drilling_stress * eps_d) * dx)

# Assemble local energies (scalar on each rank)
U_mem_local   = fem.assemble_scalar(U_mem_form)
U_bend_local  = fem.assemble_scalar(U_bend_form)
U_shear_local = fem.assemble_scalar(U_shear_form)
U_drill_local = fem.assemble_scalar(U_drill_form)

# Reduce to global energies
comm = MPI.COMM_WORLD
U_mem   = comm.allreduce(U_mem_local,   op=MPI.SUM)
U_bend  = comm.allreduce(U_bend_local,  op=MPI.SUM)
U_shear = comm.allreduce(U_shear_local, op=MPI.SUM)
U_drill = comm.allreduce(U_drill_local, op=MPI.SUM)

U_total = U_mem + U_bend + U_shear + U_drill
eps0 = 1e-30  # avoid division by zero

if comm.rank == 0:
    print("\n=== ENERGY DECOMPOSITION (strain energy) ===")
    print(f"U_membrane = {U_mem:.6e}  J")
    print(f"U_bending  = {U_bend:.6e}  J")
    print(f"U_shear    = {U_shear:.6e}  J")
    print(f"U_drilling = {U_drill:.6e}  J")
    print(f"U_total    = {U_total:.6e}  J")

    print("\n--- Energy fractions ---")
    print(f"membrane fraction = {U_mem  /(U_total+eps0):.4%}")
    print(f"bending  fraction = {U_bend /(U_total+eps0):.4%}")
    print(f"shear    fraction = {U_shear/(U_total+eps0):.4%}")
    print(f"drilling fraction = {U_drill/(U_total+eps0):.4%}")

    print("\n--- Quick interpretation ---")
    print("• Thin-shell (bending-dominated) typically: shear fraction → very small.")
    print("• If shear fraction stays significant: bending–shear coupling is present.")
    print("• If drilling fraction is large: your alpha_drilling may be too strong (artificial stiffening).")








converged = problem.solver.getConvergedReason()
n_iter    = problem.solver.getIterationNumber()
print(f"SNES converged reason : {converged}")
print(f"SNES iterations       : {n_iter}")
assert converged > 0, f"Solver did not converge! Reason: {converged}"

# ── Local rotation components (comparable to Cast3M RX, RY, RZ) ─────────────
Vscalar = fem.functionspace(DOMAIN, ("DG", 0))

theta_fn = v.sub(1)

theta_loc1 = LocalFrame.project_scalar(ufl.dot(theta_fn, E2), Vscalar, DOMAIN)
theta_loc2 = LocalFrame.project_scalar(ufl.dot(theta_fn, E1), Vscalar, DOMAIN)
theta_loc3 = LocalFrame.project_scalar(ufl.dot(theta_fn, E3), Vscalar, DOMAIN)

print(f"Max theta_local_1 (≈ RX) = {np.max(np.abs(theta_loc1.x.array)):.6e} rad")
print(f"Max theta_local_2 (≈ RY) = {np.max(np.abs(theta_loc2.x.array)):.6e} rad")
print(f"Max theta_local_3 (≈ RZ) = {np.max(np.abs(theta_loc3.x.array)):.6e} rad")


# POST-PROCESSING
disp = v.sub(0).collapse()
rota = v.sub(1).collapse()
disp.name = "Displacement"
rota.name = "Rotation"

## maximum displacement field
vdim_u = disp.function_space.element.value_shape[0]
vdim_t = rota.function_space.element.value_shape[0]

disp_arr = disp.x.array.reshape(-1, vdim_u)
rota_arr = rota.x.array.reshape(-1, vdim_t)

disp_max = np.max(np.abs(disp_arr), axis=0)
rota_max = np.max(np.abs(rota_arr), axis=0)

print("disp.name =", disp.name)
print("rota.name =", rota.name)

for i in range(vdim_u):
  print(f"Max u[{i}] = {disp_max[i]:.6e} m")
for i in range(vdim_t):
  print(f"Max theta[{i}] = {rota_max[i]:.6e} rad")

## export result
RESULT_FOLDER = Path("Results")
RESULT_FOLDER.mkdir(exist_ok=True, parents=True)

Vout = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))
disp_out = fem.Function(Vout)
disp_out.interpolate(disp)

Vtheta_out = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))
rota_out = fem.Function(Vtheta_out)
rota_out.interpolate(rota)

disp_out.name = "Displacement"
rota_out.name = "Rotation"

with io.XDMFFile(MPI.COMM_WORLD, RESULT_FOLDER / "displacement.xdmf", "w") as xdmf:
  xdmf.write_mesh(DOMAIN)
  xdmf.write_function(disp_out)
  xdmf.write_function(rota_out)


# Collapse once
u_out = v.sub(0).collapse()
u_out.name = "Displacement"

theta_out = v.sub(1).collapse()
theta_out.name = "Rotation"

# Write in XDMF (cleaner than PVD)
#with io.XDMFFile(MPI.COMM_WORLD, results_folder / "results.xdmf", "w") as xdmf:
#    xdmf.write_mesh(domain)
#    xdmf.write_function(u_out)
#    xdmf.write_function(theta_out)

# ─── Compute max displacement magnitude (MPI-safe) ──────────────────────────

u_arr = u_out.x.array.reshape(-1, DOMAIN.geometry.dim)
local_max = np.max(np.linalg.norm(u_arr, axis=1))
global_max = MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)

#if MPI.COMM_WORLD.rank == 0:
#    print(f"Max displacement magnitude: {global_max:.6e} m")

if MPI.COMM_WORLD.rank == 0:
    print(f"Max displacement magnitude: {global_max:.6e} m")

    # Tách từng component
    u_arr = u_out.x.array.reshape(-1, 3)
    
    ux_max = MPI.COMM_WORLD.allreduce(np.max(np.abs(u_arr[:, 0])), op=MPI.MAX)
    uy_max = MPI.COMM_WORLD.allreduce(np.max(np.abs(u_arr[:, 1])), op=MPI.MAX)
    uz_max = MPI.COMM_WORLD.allreduce(np.max(np.abs(u_arr[:, 2])), op=MPI.MAX)
    
    print(f"Max |Ux| = {ux_max:.6e} m")
    print(f"Max |Uy| = {uy_max:.6e} m")
    print(f"Max |Uz| = {uz_max:.6e} m")

# cleanup PETSc object
problem.solver.destroy()

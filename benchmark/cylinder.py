import LocalFrame
import MeshImport
from ShellKinematics import shell_strains
from IsotropicShell import IsotropicShell
from FOAMImport import import_constant_pressure_traction, load_traction_from_xdmf
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
MESHFILE = "cylinder.msh"
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

thickness   = 0.05
Young       = 210E9
Poisson     = 0.3
material    = IsotropicShell(DOMAIN, thickness=thickness, E=Young, nu=Poisson)

N = material.membrane_stress(eps)
M = material.bending_stress(kappa)
Q = material.shear_stress(gamma)

# h_mesh = ufl.CellDiameter(DOMAIN)
# alpha_drilling = fem.Constant(DOMAIN, 1.0)
alpha_drilling = fem.Constant(DOMAIN, 1e-4)
drilling_stiffness = alpha_drilling * Young * thickness
drilling_stress = drilling_stiffness * eps_d

GMSH_ROOT_ID = 13
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

XDMFFILE = "cylinder.xdmf"
P_INTERNAL = 101325

import_constant_pressure_traction(P_INTERNAL, MESHFILE, XDMFFILE)
FTraction = load_traction_from_xdmf(XDMFFILE, DOMAIN, GDIM)



# DEBUG — add this right after loading
trac_arr = FTraction.x.array
print(f"FTraction range: {trac_arr.min():.3e} → {trac_arr.max():.3e}")
print(f"NaN count: {np.isnan(trac_arr).sum()}, Inf count: {np.isinf(trac_arr).sum()}")




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

# cleanup PETSc object
problem.solver.destroy()




print(f"Root facets found : {len(ROOT_FACETS)}")
print(f"Root dofs u       : {len(root_dofs_u[0])}")
print(f"Root dofs theta   : {len(root_dofs_theta[0])}")


# Dùng coords từ function space, không phải mesh geometry
coords = disp.function_space.tabulate_dof_coordinates()

disp_arr = disp.x.array.reshape(-1, 3)
R_vec = coords[:, :2]
R_mag = np.linalg.norm(R_vec, axis=1, keepdims=True)
R_hat = R_vec / np.where(R_mag < 1e-15, 1.0, R_mag)

u_xy = disp_arr[:, :2]
u_radial = np.sum(u_xy * R_hat, axis=1)

print(f"Max radial displacement : {u_radial.max():.6e} m")
print(f"Min radial displacement : {u_radial.min():.6e} m")
print(f"Analytical u_r          : {101325*1**2/(210e9*1e-3):.6e} m")
print(f"Ratio FEM/analytical    : {u_radial.max()/(101325*1/(210e9*1e-3)):.2f}")
print(f"Max axial u[2]          : {np.max(np.abs(disp_arr[:,2])):.6e} m")
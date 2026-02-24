import numpy as np
from pathlib import Path
from mpi4py import MPI
from dolfinx import mesh, fem, io, plot
from dolfinx.io import gmsh
from ufl import Jacobian, replace
import ufl
import dolfinx.fem.petsc
import basix
import meshio

import matplotlib.pyplot as plt
import sys
from scipy.spatial import cKDTree



# FEMFILENAME = "../data/CAD/FEMMesh.msh"
MESHFILENAME = "../data/CAD/wing.msh"

mesh_data = gmsh.read_from_msh(MESHFILENAME,MPI.COMM_WORLD,gdim=3)

domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags
physical_groups = mesh_data.physical_groups

domain.geometry.x[:] *= 1e-3

gdim = domain.geometry.dim
tdim = domain.topology.dim

print(f"Geometrical dimension = {gdim}")
print(f"Topological dimension = {tdim}")


# =============================================================================
# MATERIAL PARAMETERS (Composite & Fiber Orientation)
# =============================================================================
thick = fem.Constant(domain, 6E-3)

# 1. Base Ply Properties (Example: T300/Epoxy Carbon Fiber)
# Change these to your exact material specs
E1 = 140e9   # Longitudinal modulus (Pa)
E2 = 10e9    # Transverse modulus (Pa)
G12 = 5e9    # In-plane shear modulus (Pa)
nu12 = 0.3   # Major Poisson's ratio
nu21 = nu12 * E2 / E1

# Transverse shear moduli (approximations for composite)
G13 = G12
G23 = G12 * 0.8 

# 2. Define ETA (Fiber Orientation Parameter)
# eta = 0.6 means 60% of fibers are at 0 degrees.
# The remaining 40% is split equally among 90, +45, and -45 degrees.


# Read eta from command line if provided, otherwise default to 0.6
if len(sys.argv) > 1:
    eta = float(sys.argv[1])
else:
    eta = 0.6

# ... (rest of your composite material block from previous message: E1, E2, fraction_0, get_Q_bar, etc.)

fraction_0   = eta
fraction_90  = 0.1
fraction_45  = (0.9 - eta) / 2
fraction_m45 = (0.9 - eta) / 2

def get_Q_bar(theta_deg):
    """Calculates the transformed 3x3 plane stress stiffness matrix for a given angle."""
    c = np.cos(np.radians(theta_deg))
    s = np.sin(np.radians(theta_deg))
    
    Q11 = E1 / (1 - nu12 * nu21)
    Q22 = E2 / (1 - nu12 * nu21)
    Q12 = nu12 * Q22
    Q66 = G12
    
    T11 = Q11*c**4 + 2*(Q12 + 2*Q66)*(s**2)*(c**2) + Q22*s**4
    T12 = (Q11 + Q22 - 4*Q66)*(s**2)*(c**2) + Q12*(s**4 + c**4)
    T22 = Q11*s**4 + 2*(Q12 + 2*Q66)*(s**2)*(c**2) + Q22*c**4
    T16 = (Q11 - Q12 - 2*Q66)*s*(c**3) + (Q12 - Q22 + 2*Q66)*(s**3)*c
    T26 = (Q11 - Q12 - 2*Q66)*(s**3)*c + (Q12 - Q22 + 2*Q66)*s*(c**3)
    T66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*(s**2)*(c**2) + Q66*(s**4 + c**4)
    
    return np.array([[T11, T12, T16],
                     [T12, T22, T26],
                     [T16, T26, T66]])

# 3. Calculate Homogenized In-Plane Stiffness (C_hom)
C_hom_np = (fraction_0 * get_Q_bar(90) + 
            fraction_90 * get_Q_bar(0) + 
            fraction_45 * get_Q_bar(45) + 
            fraction_m45 * get_Q_bar(-45))
C_hom = ufl.as_tensor(C_hom_np)
#because of the rotation we used 0degrees correspond to chordwise direction and 90degress correspond to spanwise  

# 4. Calculate Homogenized Transverse Shear Stiffness (C_shear)
C_shear_np = fraction_0 * np.array([[G13, 0], [0, G23]]) + \
             fraction_90 * np.array([[G23, 0], [0, G13]]) + \
             (fraction_45 + fraction_m45) * np.array([[(G13+G23)/2, 0], [0, (G13+G23)/2]])
C_shear = ufl.as_tensor(C_shear_np)
# =============================================================================

def normalize(v):
  return v / ufl.sqrt(ufl.dot(v, v))

def local_frame(mesh): # Compute local orthonormal frame (e1, e2, e3) at each point based on the Jacobian
  t = ufl.Jacobian(mesh)
  if gdim == 2:
    t1 = ufl.as_vector([t[0, 0], t[1, 0], 0])
    t2 = ufl.as_vector([t[0, 1], t[1, 1], 0])
  else:
    t1 = ufl.as_vector([t[0, 0], t[1, 0], t[2, 0]])
    t2 = ufl.as_vector([t[0, 1], t[1, 1], t[2, 1]])
  e3 = normalize(ufl.cross(t1, t2))
  ey = ufl.as_vector([0, 1, 0])
  ez = ufl.as_vector([0, 0, 1])
  e1_trial = ufl.cross(ey, e3)
  norm_e1 = ufl.sqrt(ufl.dot(e1_trial, e1_trial))
  e1 = ufl.conditional(ufl.lt(norm_e1, 0.5), ez, normalize(e1_trial))
  e2 = normalize(ufl.cross(e3, e1))
  return e1, e2, e3

VT = fem.functionspace(domain, ("DG", 0, (gdim,)))
V0, _ = VT.sub(0).collapse()

frame = local_frame(domain)
basis_vectors = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(gdim)]

for i in range(gdim):
  e_exp = fem.Expression(frame[i], V0.element.interpolation_points)
  basis_vectors[i].interpolate(e_exp)

e1, e2, e3 = basis_vectors

results_folder = Path("LocalFrame")
results_folder.mkdir(exist_ok=True, parents=True)
with dolfinx.io.VTKFile(MPI.COMM_WORLD, results_folder / "local_frame.pvd", "w") as vtk:
  vtk.write_mesh(domain)
  vtk.write_function(e1, 0.0)
  vtk.write_function(e2, 0.0)
  vtk.write_function(e3, 0.0)


Ue = basix.ufl.element("P", domain.basix_cell(), 2, shape=(gdim,))  # displacement finite element
Te = basix.ufl.element("CR", domain.basix_cell(), 1, shape=(gdim,)) # rotation finite element
V = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Te]))

v = fem.Function(V)
u, theta = ufl.split(v)

v_ = ufl.TestFunction(V)
u_, theta_ = ufl.split(v_)
dv = ufl.TrialFunction(V)


def vstack(vectors):
  """Stack a list of vectors vertically."""
  return ufl.as_matrix([[v[i] for i in range(len(v))] for v in vectors])

def hstack(vectors):
  """Stack a list of vectors horizontally."""
  return vstack(vectors).T

# In-plane projection
P_plane = hstack([e1, e2])

def t_grad(u):
  """Tangential gradient operator"""
  g = ufl.grad(u)
  return ufl.dot(g, P_plane)


t_gu = ufl.dot(P_plane.T, t_grad(u))
eps = ufl.sym(t_gu)
beta = ufl.cross(e3, theta)
kappa = ufl.sym(ufl.dot(P_plane.T, t_grad(beta)))
gamma = t_grad(ufl.dot(u, e3)) - ufl.dot(P_plane.T, beta)

eps_ = ufl.derivative(eps, v, v_)
kappa_ = ufl.derivative(kappa, v, v_)
gamma_ = ufl.derivative(gamma, v, v_)


# =============================================================================
# KINEMATICS & CONSTITUTIVE LAWS (Composite)
# =============================================================================
# Helper functions to map 2x2 local tensors to/from 3x1 Voigt vectors
def to_voigt(e):
    # Maps [[e11, e12], [e21, e22]] -> [e11, e22, 2*e12]
    return ufl.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

def from_voigt(v):
    # Maps [v1, v2, v3] -> [[v1, v3/2], [v3/2, v2]]
    return ufl.as_tensor([[v[0], v[2]/2], 
                          [v[2]/2, v[1]]])

# Membrane forces (N)
N_voigt = thick * ufl.dot(C_hom, to_voigt(eps))
N = from_voigt(N_voigt)

# Bending moments (M)
M_voigt = (thick**3 / 12.0) * ufl.dot(C_hom, to_voigt(kappa))
M = from_voigt(M_voigt)

# Transverse shear forces (Q) with Shear Correction Factor
kappa_shear = 5.0 / 6.0 
Q = kappa_shear * thick * ufl.dot(C_shear, gamma)
# =============================================================================



drilling_strain = (t_gu[0, 1] - t_gu[1, 0]) / 2 + ufl.dot(theta, e3)
drilling_strain_ = ufl.replace(drilling_strain, {v: v_})

h_mesh = ufl.CellDiameter(domain)

G_in_plane = fem.Constant(domain, C_hom_np[2, 2])
drilling_stiffness = G_in_plane * thick**3 / h_mesh**2


drilling_stress = drilling_stiffness * drilling_strain



tdim = domain.topology.dim
fdim = tdim - 1
root_facets = facet_tags.find(11)

Vu, _ = V.sub(0).collapse()
root_dofs_u = fem.locate_dofs_topological((V.sub(0), Vu),fdim,root_facets)
uD = fem.Function(Vu)
uD.x.array[:] = 0.0

Vtheta, _ = V.sub(1).collapse()
root_dofs_theta = fem.locate_dofs_topological((V.sub(1), Vtheta),fdim,root_facets)
thetaD = fem.Function(Vtheta)
thetaD.x.array[:] = 0.0

bc_u = fem.dirichletbc(uD, root_dofs_u, V.sub(0))
bc_theta = fem.dirichletbc(thetaD, root_dofs_theta, V.sub(1))

bcs = [bc_u, bc_theta]



# Load traction vector from FOAM data
foam = meshio.read("../data/FOAM/ExtractedFOAMData.xdmf")
foam_pts   = foam.points
foam_tract = foam.point_data["traction"]

# Function space P1 vector for traction
VT_traction = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
f_traction  = fem.Function(VT_traction, name="traction")

# Point coordinates of dolfinx can be different from meshio ordering
# We have to interpolate
# Map by coordinates: FOAM mesh → dolfinx DOFs
dof_coords = VT_traction.tabulate_dof_coordinates()
tree = cKDTree(foam_pts)
distances, idx = tree.query(dof_coords, k=1)
f_traction.x.array[:] = foam_tract[idx].flatten()
f_traction.x.scatter_forward()

dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain)

# Internal virtual work
a_int = (
    ufl.inner(N, eps_)
  + ufl.inner(M, kappa_)
  + ufl.inner(Q, gamma_)
  + drilling_stress * drilling_strain_
) * dx

# External virtual work: traction across the entire surface
L_ext = ufl.dot(f_traction, u_) * dx

residual = a_int - L_ext

tangent = ufl.derivative(residual, v, dv)

# ─── 4. Solve ────────────────────────────────────────────────────────────────
from dolfinx.fem.petsc import NonlinearProblem

# ─── Nonlinear Problem (API  0.10.0) ──────────────────────────────────────
problem = NonlinearProblem(
    F=residual,
    u=v,
    bcs=bcs,
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

# ─── Solve ────────────────────────────────────────────────────────────────────
problem.solve()

converged = problem.solver.getConvergedReason()
n_iter    = problem.solver.getIterationNumber()
print(f"SNES converged reason : {converged}")
print(f"SNES iterations       : {n_iter}")
assert converged > 0, f"Solver did not converge! Reason: {converged}"

# ─── Extract solution ─────────────────────────────────────────────────────────
u_sol     = v.sub(0).collapse()
theta_sol = v.sub(1).collapse()

print(f"Max displacement : {np.max(np.abs(u_sol.x.array)):.6e} m")
print(f"Max rotation     : {np.max(np.abs(theta_sol.x.array)):.6e} rad")

# ─── Export ───────────────────────────────────────────────────────────────────
results_folder = Path("Results")
results_folder.mkdir(exist_ok=True, parents=True)

with dolfinx.io.VTKFile(MPI.COMM_WORLD, results_folder / "displacement.pvd", "w") as vtk:
    vtk.write_mesh(domain)
    vtk.write_function(u_sol, 0.0)
    vtk.write_function(theta_sol, 0.0)

# Cleanup PETSc objects
problem.solver.destroy()

# ─── 5. Output Results ───────────────────────────────────────────────────────
u_sol = v.sub(0).collapse()
theta_sol = v.sub(1).collapse()

results_folder = Path("Results")
results_folder.mkdir(exist_ok=True, parents=True)

with dolfinx.io.VTKFile(
    MPI.COMM_WORLD,
    results_folder / "displacement.pvd",
    "w"
) as vtk:
    vtk.write_mesh(domain)
    vtk.write_function(u_sol, 0.0)
    vtk.write_function(theta_sol, 0.0)

# Compute max displacement magnitude (MPI-safe)
u_arr = u_sol.x.array.reshape(-1, domain.geometry.dim)
local_max = np.max(np.linalg.norm(u_arr, axis=1))
global_max = MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)

print(f"Max displacement magnitude: {global_max:.6e} m")

# DEBUG DISPLACEMENT FIELD
import meshio
from scipy.spatial import cKDTree

foam = meshio.read("../data/FOAM/ExtractedFOAMData.xdmf")
foam_pts   = foam.points
foam_tract = foam.point_data["traction"]

print(f"FOAM pts range   : {foam_pts.min(axis=0)} -> {foam_pts.max(axis=0)}")
print(f"Traction range   : {foam_tract.min():.4e} -> {foam_tract.max():.4e} Pa")
print(f"Traction mean mag: {np.linalg.norm(foam_tract, axis=1).mean():.4e} Pa")

# After imported into dolfinx
print(f"f_traction array range: {f_traction.x.array.min():.4e} -> {f_traction.x.array.max():.4e}")

print(f"FEM pts range: {domain.geometry.x.min(axis=0)} -> {domain.geometry.x.max(axis=0)}")

# ─── 5. Visualization ───────────────────────────────────────────────────────
u_out = v.sub(0).collapse()
u_out.name = "Displacement"

theta_out = v.sub(1).collapse()
theta_out.name = "Rotation"

with io.VTKFile(MPI.COMM_WORLD, results_folder / "results.pvd", "w") as vtk:
    vtk.write_function(u_out, 0.0)
    vtk.write_function(theta_out, 0.0)




import meshio
import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import fem, io
from dolfinx.fem import Expression, Function, functionspace, form
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, locate_entities
from scipy.interpolate import NearestNDInterpolator
from dolfinx.mesh import compute_midpoints
import ufl
from pathlib import Path





def map_traction(y_rib, node_coord, FOAMFILE):
  FOAM = meshio.read(FOAMFILE)
  FOAMPOINT = FOAM.points
  FOAMTRACTION = FOAM.point_data["traction"]

  y_coor = FOAMPOINT[:,1]
  y_span = y_coor.max() - y_coor.min()
  y_band = y_span  * 0.15
  mask = np.abs(FOAMPOINT[:,1] - y_rib) < y_band

  interp_x = NearestNDInterpolator(FOAMPOINT[mask],FOAMTRACTION[mask,0])
  interp_y = NearestNDInterpolator(FOAMPOINT[mask],FOAMTRACTION[mask,1])
  interp_z = NearestNDInterpolator(FOAMPOINT[mask],FOAMTRACTION[mask,2])
  TRACTIONX = interp_x(node_coord)
  TRACTIONY = interp_y(node_coord)
  TRACTIONZ = interp_z(node_coord)

  return TRACTIONX, TRACTIONY, TRACTIONZ



def SIMP(rho, PENAL, Emin, E0):
  return Emin + rho**PENAL * (E0 - Emin)

def SIMP_DEV(rho, PENAL, Emin, E0):
  return PENAL * rho**(PENAL - 1) * (E0 - Emin)

def strain(u):
  return ufl.sym(ufl.grad(u))

def stress(u, rho, PENAL, Emin, E0, NU, GDIM):
  E = SIMP(rho, PENAL, Emin, E0)
  LMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
  MU = E / (2 * (1 + NU))
  LMBDA_PS = 2 * LMBDA * MU / (LMBDA + 2 * MU)
  return LMBDA_PS * ufl.tr(strain(u)) * ufl.Identity(GDIM) + 2 * MU * strain(u)

def OC_update(rho_old, dc, volfrac, move):
  l1, l2 = 1e-9, 1e9
  while (l2 - l1) / (l1 + l2) > 1e-6:
    lmid    = 0.5 * (l1 + l2)
    rho_new = np.clip(
      rho_old * np.sqrt(np.maximum(-dc / lmid, 0)),
      np.maximum(rho_old - move, 1e-3),
      np.minimum(rho_old + move, 1.0),
    )
    if rho_new.mean() > volfrac:
      l1 = lmid
    else:
      l2 = lmid
  return rho_new









VOLFRAC   = 0.4
P_SIMP    = 3
LAME_NU   = 0.3
E0        = 210E9
FILTER_R  = 0.015
MOVE      = 0.2
MAX_ITER  = 150
TOL       = 1E-3


RIBINDEX = 2
RIBFILE = f"rib{RIBINDEX}.xdmf"
FOAMFILE = "FOAMData.xdmf"

with XDMFFile(MPI.COMM_WORLD, RIBFILE, "r") as f:
  DOMAIN = f.read_mesh(name="Grid")

GDIM = DOMAIN.geometry.dim
TDIM = DOMAIN.topology.dim
NUM_CELLS = DOMAIN.topology.index_map(TDIM).size_local
MIDPOINTS = compute_midpoints(DOMAIN, TDIM, np.arange(NUM_CELLS))


CG1 = functionspace(DOMAIN, ("CG", 1, (GDIM,)))
DG0 = functionspace(DOMAIN, ("DG", 0))


V   = fem.functionspace(DOMAIN, ("CG", 1, (GDIM,)))
DG0 = fem.functionspace(DOMAIN, ("DG", 0))
V1  = fem.functionspace(DOMAIN, ("CG", 1))

NODE_COORD  = V1.tabulate_dof_coordinates()
Y_RIB       = float(np.mean(NODE_COORD[:, 1]))

TRACTIONX, TRACTIONY, TRACTIONZ = map_traction(Y_RIB, NODE_COORD, FOAMFILE)

TRACTION = fem.Function(V, name="traction")
TRACTION.x.array[0::GDIM] = TRACTIONX
TRACTION.x.array[1::GDIM] = TRACTIONY
TRACTION.x.array[2::GDIM] = TRACTIONZ
TRACTION.x.scatter_forward()

rho = fem.Function(DG0, name="density")
rho.x.array[:] = VOLFRAC




# BOUNDARY CONDITION
BOUNDARY_POINT = DOMAIN.geometry.x
CHORD = BOUNDARY_POINT[:, 0].max() - BOUNDARY_POINT[:, 0].min()
XMIN  = BOUNDARY_POINT[:, 0].min()

def spar_boundary(x):
  FRONT = np.abs(x[0] - (XMIN + 0.25 * CHORD)) < 0.02 * CHORD
  REAR  = np.abs(x[0] - (XMIN + 0.75 * CHORD)) < 0.02 * CHORD
  return FRONT | REAR

SPAR_DOFS = fem.locate_dofs_geometrical(V, spar_boundary)
u_zero    = fem.Function(V)
bcs       = [fem.dirichletbc(u_zero, SPAR_DOFS)]



# VARIATIONAL FORM
u   = ufl.TrialFunction(V)
v_  = ufl.TestFunction(V)

a_form = ufl.inner(stress(u, rho), strain(v_)) * ufl.dx
L_form = ufl.dot(TRACTION, v_) * ufl.ds




results_dir = Path("topopt")
results_dir.mkdir(parents=True, exist_ok=True)

with io.VTKFile(MPI.COMM_WORLD, str(results_dir / f"rib_{RIBINDEX}_res.pvd"), "w") as vtk:
  vtk.write_mesh(DOMAIN)

  for it in range(MAX_ITER):
    prob = LinearProblem(a_form, L_form, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
    uh = prob.solve()

    compliance = fem.assemble_scalar(fem.form(ufl.dot(TRACTION, uh) * ufl.ds))

    dE      = SIMP_DEV(rho)
    dc_form = fem.form(-dE * ufl.inner(strain(uh), strain(uh)) * ufl.dx)
    dc_vec  = fem.assemble_vector(dc_form)

    rho_old        = rho.x.array.copy()
    rho.x.array[:] = OC_update(rho_old, dc_vec.array, VOLFRAC)

    change = np.max(np.abs(rho.x.array - rho_old))
    print(f"Iter {it:3d}  C={compliance:.4e}  "
          f"vol={rho.x.array.mean():.3f}  change={change:.5f}")

    if it % 10 == 0:
      vtk.write_function(rho, float(it))

    if change < TOL:
      print(f"Converged at iteration {it}")
      break

    vtk.write_function(rho, float(it))

with XDMFFile(MPI.COMM_WORLD, str(results_dir / f"rib_{RIBINDEX}_rho_final.xdmf"), "w") as FILE:
  FILE.write_mesh(DOMAIN)
  FILE.write_function(rho)

print(f"Done -> {results_dir}")
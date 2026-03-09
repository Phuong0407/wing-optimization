import ufl
from dolfinx import fem
from mpi4py import MPI
from dolfinx.io import gmsh
import os
import vtk
import meshio
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
from dolfinx import mesh, fem, io, plot
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RBFInterpolator

from types import SimpleNamespace
import ufl
import basix

import os
import numpy as np
from mpi4py import MPI
import dolfinx.fem.petsc
from dolfinx.fem.petsc import NonlinearProblem
from pathlib import Path





#================= ISOTROPIC MATERIAL =================#
thickness   = 2E-3
young       = 210E9
poisson     = 0.3
def isotropic_material(thickness, young, poisson, DOMAIN):
  THICKNESS = fem.Constant(DOMAIN, thickness)
  YOUNG     = fem.Constant(DOMAIN, young)
  NU        = fem.Constant(DOMAIN, poisson)
  LMBDA     = YOUNG * NU / (1 + NU) / (1 - 2 * NU)
  MU        = YOUNG / 2 / (1 + NU)
  LMBDA_PS  = 2 * LMBDA * MU / (LMBDA + 2 * MU)

  return SimpleNamespace(
    h=THICKNESS,
    E=YOUNG,
    NU=NU,
    LMBDA=LMBDA,
    MU=MU,
    LMBDA_PS=LMBDA_PS,
  )
#================= ISOTROPIC MATERIAL =================#

#================= CTL COMPOSITE MATERIAL =================#
thickness   = 2E-3
young       = 210E9
poisson     = 0.3
def isotropic_material(thickness, young, poisson, DOMAIN):
  THICKNESS = fem.Constant(DOMAIN, thickness)
  YOUNG     = fem.Constant(DOMAIN, young)
  NU        = fem.Constant(DOMAIN, poisson)
  LMBDA     = YOUNG * NU / (1 + NU) / (1 - 2 * NU)
  MU        = YOUNG / 2 / (1 + NU)
  LMBDA_PS  = 2 * LMBDA * MU / (LMBDA + 2 * MU)

  return SimpleNamespace(
    h=THICKNESS,
    E=YOUNG,
    NU=NU,
    LMBDA=LMBDA,
    MU=MU,
    LMBDA_PS=LMBDA_PS,
  )
#================= CTL COMPOSITE MATERIAL =================#





#================= MESH LOAD =================#
def load_mesh(filename, gdim=3):
  MESH = gmsh.read_from_msh(
    filename,
    comm=MPI.COMM_WORLD,
    gdim=gdim
  )
  DOMAIN = MESH.mesh
  DOMAIN.geometry.x[:] *= 1E-3
  return DOMAIN, MESH.cell_tags, MESH.facet_tags
#================= END MESH LOAD =================#



#================= LOCAL FRAME =================#
def normalize(v):
  return v / ufl.sqrt(ufl.dot(v, v))

def _local_frame(domain):
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

def local_frame(domain,gdim):
  FRAME = _local_frame(domain)
  VT = fem.functionspace(domain, ("DG", 0, (gdim,)))
  V0, _ = VT.sub(0).collapse()
  BASIS_VECTORS = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(gdim)]
  for i in range(gdim):
    e_exp = fem.Expression(FRAME[i], V0.element.interpolation_points)
    BASIS_VECTORS[i].interpolate(e_exp)
  E1, E2, E3 = BASIS_VECTORS
  return E1, E2, E3
#================= END LOCAL FRAME =================#



#================= SHELL KINEMATICS =================#
def hstack(vecs):
  return ufl.as_matrix([[vi[i] for i in range(len(vi))] for vi in vecs]).T

def tangent_projection(e1,e2):
  return hstack([e1, e2])

def tangential_gradient(w, P_plane):
  return ufl.dot(ufl.grad(w), P_plane)

def membrane_strain(u, P_plane):
  t_gu = ufl.dot(P_plane.T, tangential_gradient(u, P_plane))
  return ufl.sym(t_gu), t_gu

def bending_strain(theta, e3, P_plane):
  beta = ufl.cross(e3, theta)
  return ufl.sym(ufl.dot(P_plane.T, tangential_gradient(beta, P_plane)))

def shear_strain(u, theta, e3, P_plane):
  beta = ufl.cross(e3, theta)
  return (tangential_gradient(ufl.dot(u, e3), P_plane)- ufl.dot(P_plane.T, beta))

def compute_drilling_strain(t_gu, theta, e3):
  return (t_gu[0, 1] - t_gu[1, 0]) / 2 + ufl.dot(theta, e3)

def shell_strains(u, theta, e1, e2, e3):
  P_plane           = tangent_projection(e1, e2)
  eps, t_gu         = membrane_strain(u, P_plane)
  kappa             = bending_strain(theta, e3, P_plane)
  gamma             = shear_strain(u, theta, e3, P_plane)
  drilling_strain   = compute_drilling_strain(t_gu, theta, e3)
  return eps, kappa, gamma, drilling_strain
#================= END SHELL KINEMATICS =================#



#================= OPENFOAM LOAD =================#
def import_foam_traction(FOAMFILE, XDMFFILE=None, VERBOSE=False):
  FOAMFILE = "wing.vtp"
  FOAMREADER = vtk.vtkXMLPolyDataReader()
  FOAMREADER.SetFileName(FOAMFILE)
  FOAMREADER.Update()
  poly = FOAMREADER.GetOutput()

  TRIANGULATION = vtk.vtkTriangleFilter()
  TRIANGULATION.SetInputData(poly)
  TRIANGULATION.Update()
  poly = TRIANGULATION.GetOutput()

  NORMAL_FILTER = vtk.vtkPolyDataNormals()
  NORMAL_FILTER.SetInputData(poly)
  NORMAL_FILTER.ComputePointNormalsOn()
  NORMAL_FILTER.ComputeCellNormalsOff()
  NORMAL_FILTER.AutoOrientNormalsOn()
  NORMAL_FILTER.AutoOrientNormalsOff()
  NORMAL_FILTER.ConsistencyOn()
  NORMAL_FILTER.SplittingOff()
  NORMAL_FILTER.Update()
  POLY = NORMAL_FILTER.GetOutput()

  POINTS  = vtk_to_numpy(POLY.GetPoints().GetData())
  P       = vtk_to_numpy(POLY.GetPointData().GetArray("p"))
  WSS     = vtk_to_numpy(POLY.GetPointData().GetArray("wallShearStress"))
  NORMALS = vtk_to_numpy(POLY.GetPointData().GetArray("Normals"))
  NORMALS  = -NORMALS

  if VERBOSE:
    print(f"Points  : {POINTS.shape}")
    print(f"p       : {P.shape}")
    print(f"wss     : {WSS.shape}")
    print(f"normals : {NORMALS.shape}")

  TRACTION     = -P[:,np.newaxis] * NORMALS + WSS
  TRACTIONMAG  = np.linalg.norm(TRACTION, axis=1)

  if VERBOSE:
    print(f"p range            : {P.min():.2f} -> {P.max():.2f} Pa")
    print(f"traction magnitude : {TRACTIONMAG.min():.2f} -> {TRACTIONMAG.max():.2f} Pa")

  CELLS = POLY.GetPolys()
  CELLS.InitTraversal()
  IDLIST = vtk.vtkIdList()

  TRIANGLES = []
  while CELLS.GetNextCell(IDLIST):
    TRIANGLES.append([IDLIST.GetId(i) for i in range(3)])
  TRIANGLES = np.array(TRIANGLES)

  print("Points:", POINTS.shape)
  print("Triangles:", TRIANGLES.shape)

  mesh = meshio.Mesh(
    points=POINTS,
    cells=[("triangle", TRIANGLES)],
    point_data={
      "p"       : P,
      "normals" : NORMALS,
      "wss"     : WSS,
      "traction": TRACTION,
    }
  )

  meshio.write(XDMFFILE, mesh)
  print(f"Export the wing-patch traction from FOAM to the file {XDMFFILE}")

def map_traction(foamfile, meshfile, outfile):
  FOAMMESH     = meshio.read(foamfile)
  FOAMPOINT    = FOAMMESH.points
  FOAMTRIANGLE = FOAMMESH.cells_dict["triangle"]
  FOAMTRACTION = FOAMMESH.point_data["traction"]

  FEMMESH       = meshio.read(meshfile)
  FEMPOINT      = FEMMESH.points * 1e-3
  FEMTRIANGLE   = FEMMESH.cells_dict["triangle"]

  interp = RBFInterpolator(
    FOAMPOINT,
    FOAMTRACTION,
    kernel='thin_plate_spline',
    neighbors=20
  )
  NODAL_TRACTION = interp(FEMPOINT)
  
  # FOAM_CENTROIDS = FOAMPOINT[FOAMTRIANGLE].mean(axis=1)
  # FOAM_TRI_TRACT = FOAMTRACTION[FOAMTRIANGLE].mean(axis=1)
  # FOAM_TRI_AREA  = 0.5 * np.linalg.norm(np.cross(
  #   FOAMPOINT[FOAMTRIANGLE[:,1]] - FOAMPOINT[FOAMTRIANGLE[:,0]],
  #   FOAMPOINT[FOAMTRIANGLE[:,2]] - FOAMPOINT[FOAMTRIANGLE[:,0]]), axis=1)

  # FEM_CENTROIDS = FEMPOINT[FEMTRIANGLE].mean(axis=1)
  # FEM_TRI_AREA  = 0.5 * np.linalg.norm(np.cross(
  #   FEMPOINT[FEMTRIANGLE[:,1]] - FEMPOINT[FEMTRIANGLE[:,0]],
  #   FEMPOINT[FEMTRIANGLE[:,2]] - FEMPOINT[FEMTRIANGLE[:,0]]), axis=1)

  # TREE = cKDTree(FEM_CENTROIDS)
  # _, FEM_TRI_ID = TREE.query(FOAM_CENTROIDS)

  # FEM_TRI_FORCE = np.zeros((len(FEMTRIANGLE), 3))
  # for i, fid in enumerate(FEM_TRI_ID):
  #   FEM_TRI_FORCE[fid] += FOAM_TRI_TRACT[i] * FOAM_TRI_AREA[i]

  # FEM_TRI_TRACTION = FEM_TRI_FORCE / (FEM_TRI_AREA[:,np.newaxis] + 1e-12)

  # FOAM_FORCE = np.sum(FOAM_TRI_TRACT * FOAM_TRI_AREA[:,np.newaxis], axis=0)
  # FEM_FORCE  = np.sum(FEM_TRI_FORCE, axis=0)
  
  # print("FOAM force [N]:", FOAM_FORCE)
  # print("FEM  force [N]:", FEM_FORCE)
  # print("Error:", np.linalg.norm(FEM_FORCE-FOAM_FORCE)/np.linalg.norm(FOAM_FORCE))

  # NUM_POINT   = len(FEMPOINT)
  # NODAL_FORCE = np.zeros((NUM_POINT, 3))
  # for i, TRI in enumerate(FEMTRIANGLE):
  #   for j in TRI:
  #     NODAL_FORCE[j] += FEM_TRI_FORCE[i] / 3.0

  # TRIBUTARY_AREA = np.zeros(NUM_POINT)
  # for i, TRI in enumerate(FEMTRIANGLE):
  #     for j in TRI:
  #         TRIBUTARY_AREA[j] += FEM_TRI_AREA[i] / 3.0

  # NODAL_TRACTION = NODAL_FORCE / (TRIBUTARY_AREA[:,np.newaxis] + 1e-12)

  # ── Compute tributary area per node ───────────────────────────────
  TRIBUTARY_AREA = np.zeros(len(FEMPOINT))
  for TRI in FEMTRIANGLE:
    P0, P1, P2 = FEMPOINT[TRI[0]], FEMPOINT[TRI[1]], FEMPOINT[TRI[2]]
    AREA = 0.5 * np.linalg.norm(np.cross(P1 - P0, P2 - P0))
    for j in TRI:
      TRIBUTARY_AREA[j] += AREA / 3.0

  NODAL_FORCE = NODAL_TRACTION * TRIBUTARY_AREA[:, np.newaxis]

  FOAM_TRI_AREA = 0.5 * np.linalg.norm(
    np.cross(FOAMPOINT[FOAMTRIANGLE[:, 1]] - FOAMPOINT[FOAMTRIANGLE[:, 0]],
    FOAMPOINT[FOAMTRIANGLE[:, 2]] - FOAMPOINT[FOAMTRIANGLE[:, 0]]), axis=1)
  FOAM_TRI_TRACT = FOAMTRACTION[FOAMTRIANGLE].mean(axis=1)
  FOAM_FORCE = np.sum(FOAM_TRI_TRACT * FOAM_TRI_AREA[:, np.newaxis], axis=0)

  FEM_FORCE  = NODAL_FORCE.sum(axis=0)
  error      = np.linalg.norm(FEM_FORCE - FOAM_FORCE) / np.linalg.norm(FOAM_FORCE)

  print(f"FOAM force [N] : {FOAM_FORCE}")
  print(f"FEM  force [N] : {FEM_FORCE}")
  print(f"Force error    : {error:.4f} ({error*100:.2f}%)")

  MappedMesh = meshio.Mesh(
    points=FEMPOINT,
    cells=[("triangle", FEMTRIANGLE)],
    point_data={"traction": NODAL_TRACTION}
  )
  meshio.write(outfile, MappedMesh)
  print("Saved:", outfile)
  return NODAL_FORCE

def load_traction_xdmf(XDMFFILE, DOMAIN, GDIM,):
  FOAM = meshio.read(XDMFFILE)
  FOAMPoint = FOAM.points
  FOAMTraction = FOAM.point_data["traction"]
  
  VTTraction = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))
  FTraction  = fem.Function(VTTraction, name="traction")
  
  DOFCOORDS = VTTraction.tabulate_dof_coordinates()
  TREE = cKDTree(FOAMPoint)
  _, ID = TREE.query(DOFCOORDS, k=1)
  FTraction.x.array[:] = FOAMTraction[ID].flatten()
  FTraction.x.scatter_forward()
  return FTraction
#================= END OPENFOAM LOAD =================#



#================= SHELL STRESS =================#
def _plane_stress_(material, e):
  TDIM = e.ufl_shape[0]
  return (material.LMBDA_PS * ufl.tr(e) * ufl.Identity(TDIM) + 2 * material.MU * e)

def membrane_stress(material, eps):
  return material.h * _plane_stress_(material, eps)

def bending_stress(material, kappa):
  return material.h**3 / 12 * _plane_stress_(material, kappa)

def shear_stress(material, gamma):
  return material.MU * material.h * gamma
#================= SHELL STRESS =================#





MESHFILE = "wing.msh"
FOAMFILE = "wing.vtp"
XDMFFILE = "FOAMData.xdmf"

DOMAIN, CELL_TAGS, FACET_TAGS = load_mesh(MESHFILE)
GDIM = DOMAIN.geometry.dim
TDIM = DOMAIN.topology.dim
FDIM = TDIM - 1

E1, E2, E3 = local_frame(DOMAIN,GDIM)

Ue          = basix.ufl.element("P",  DOMAIN.basix_cell(), 2, shape=(GDIM,))
Te          = basix.ufl.element("CR", DOMAIN.basix_cell(), 1, shape=(GDIM,))
V           = fem.functionspace(DOMAIN, basix.ufl.mixed_element([Ue, Te]))
v           = fem.Function(V)
u, theta    = ufl.split(v)
v_          = ufl.TestFunction(V)
u_, theta_  = ufl.split(v_)
dv          = ufl.TrialFunction(V)

eps, kappa, gamma, drilling_strain = shell_strains(u=u, theta=theta, e1=E1, e2=E2, e3=E3)
eps_                = ufl.derivative(eps, v, v_)
kappa_              = ufl.derivative(kappa, v, v_)
gamma_              = ufl.derivative(gamma, v, v_)
drilling_strain_    = ufl.replace(drilling_strain, {v: v_})

MATERIAL = isotropic_material(thickness, young, poisson, DOMAIN)

N = membrane_stress(MATERIAL, eps)
M = bending_stress(MATERIAL, kappa)
Q = shear_stress(MATERIAL, gamma)

h_mesh = ufl.CellDiameter(DOMAIN)
drilling_stiffness = MATERIAL.E * MATERIAL.h**3 / h_mesh**2
drilling_stress = drilling_stiffness * drilling_strain

ROOT_FACETS = FACET_TAGS.find(11)

Vu, _ = V.sub(0).collapse()
root_dofs_u = fem.locate_dofs_topological((V.sub(0), Vu), FDIM, ROOT_FACETS)
uD = fem.Function(Vu)
uD.x.array[:] = 0.0

Vtheta, _ = V.sub(1).collapse()
root_dofs_theta = fem.locate_dofs_topological((V.sub(1), Vtheta), FDIM, ROOT_FACETS)
thetaD = fem.Function(Vtheta)
thetaD.x.array[:] = 0.0

BC_DISP = fem.dirichletbc(uD, root_dofs_u, V.sub(0))
BC_ROTA = fem.dirichletbc(thetaD, root_dofs_theta, V.sub(1))
BCS = [BC_DISP, BC_ROTA]



import_foam_traction(FOAMFILE,XDMFFILE)
FTraction = load_traction_xdmf(XDMFFILE, DOMAIN, GDIM)





# WEAK FORMULATION
## distance in functional space
dx = ufl.Measure("dx", DOMAIN)
ds = ufl.Measure("ds", DOMAIN)
## internal virtual work
a_int = (
  ufl.inner(N, eps_)
  + ufl.inner(M, kappa_)
  + ufl.inner(Q, gamma_)
  + drilling_stress * drilling_strain_
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

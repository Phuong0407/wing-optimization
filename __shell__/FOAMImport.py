import os
import vtk
import meshio
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
from dolfinx import fem

VERBOSE = False



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
  NORMAL_FILTER.ConsistencyOn()
  NORMAL_FILTER.SplittingOff()
  NORMAL_FILTER.Update()
  POLY = NORMAL_FILTER.GetOutput()

  POINTS  = vtk_to_numpy(POLY.GetPoints().GetData())
  P       = vtk_to_numpy(POLY.GetPointData().GetArray("p"))
  WSS     = vtk_to_numpy(POLY.GetPointData().GetArray("wallShearStress"))
  NORMALS = vtk_to_numpy(POLY.GetPointData().GetArray("Normals"))

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
        "traction"     : TRACTION,
    }
  )

  meshio.write(XDMFFILE, mesh)
  print(f"Export the wing-patch traction from FOAM to the file {XDMFFILE}")



def is_xdmffile_valie(path):
  if not os.path.exists(path):
    return False
  try:
    mesh = meshio.read(path)
  except Exception:
    return False
  if "triangle" not in [c.type for c in mesh.cells]:
    return False
  if "traction" not in mesh.point_data:
    return False
  traction = mesh.point_data["traction"]
  if traction.ndim != 2 or traction.shape[1] != 3:
    return False
  return True

def load_traction_from_xdmf(XDMFFILE, DOMAIN, GDIM,):
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

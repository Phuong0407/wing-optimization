import vtk
import numpy as np
import meshio
from vtk.util.numpy_support import vtk_to_numpy

# Reading OpenFOAM data in the form of VTK file
# Provide the data file with extension .vtp
FOAMFILE = "wing.vtp"
FOAMReader = vtk.vtkXMLPolyDataReader()
FOAMReader.SetFileName(FOAMFILE)
FOAMReader.Update()
poly = FOAMReader.GetOutput()

# The mesh from OpenFOAM is done using SnappyHexMesh
# so it contains few to no triangles.
# Therefore, a triangulation is needed
triangulation = vtk.vtkTriangleFilter()
triangulation.SetInputData(poly)
triangulation.Update()
poly = triangulation.GetOutput()

# The number of 
# Therefore, the dataset is taken by normal filter to match the number
normal_filter = vtk.vtkPolyDataNormals()
normal_filter.SetInputData(poly)
normal_filter.ComputePointNormalsOn()
normal_filter.ComputeCellNormalsOff()
normal_filter.AutoOrientNormalsOn()
normal_filter.ConsistencyOn()
normal_filter.SplittingOff()
normal_filter.Update()
poly = normal_filter.GetOutput()

points  = vtk_to_numpy(poly.GetPoints().GetData())
p       = vtk_to_numpy(poly.GetPointData().GetArray("p"))
wss     = vtk_to_numpy(poly.GetPointData().GetArray("wallShearStress"))
normals = vtk_to_numpy(poly.GetPointData().GetArray("Normals"))

print(f"Points  : {points.shape}")
print(f"p       : {p.shape}")
print(f"wss     : {wss.shape}")
print(f"normals : {normals.shape}")

traction     = -p[:,np.newaxis] * normals + wss
traction_mag = np.linalg.norm(traction,axis=1)

print(f"p range            : {p.min():.2f} -> {p.max():.2f} Pa")
print(f"traction magnitude : {traction_mag.min():.2f} -> {traction_mag.max():.2f} Pa")

cells = poly.GetPolys()
cells.InitTraversal()
idList = vtk.vtkIdList()

triangles = []
while cells.GetNextCell(idList):
  triangles.append([idList.GetId(i) for i in range(3)])
triangles = np.array(triangles)

print("Points:", points.shape)
print("Triangles:", triangles.shape)

mesh = meshio.Mesh(
  points=points,
  cells=[("triangle",triangles)],
  point_data={
    "p"            : p,
    "normals"      : normals,
    "traction"     : traction,
    "traction_mag" : traction_mag,
    "wss"          : wss,
  }
)

meshio.write("ExtractedFOAMData.xdmf", mesh)
print("Export the wing-patch data from FOAM to the file ExtractedFOAMData.xdmf")

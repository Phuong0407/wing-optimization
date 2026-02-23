import vtk
import numpy as np
import meshio
from vtk.util.numpy_support import vtk_to_numpy

data_reader = vtk.vtkXMLPolyDataReader()
data_reader.SetFileName("wing.vtp")
data_reader.Update()
poly = data_reader.GetOutput()

triangulate = vtk.vtkTriangleFilter()
triangulate.SetInputData(poly)
triangulate.Update()
poly = triangulate.GetOutput()

normal_filter = vtk.vtkPolyDataNormals()
normal_filter.SetInputData(poly)
normal_filter.ComputePointNormalsOn()
normal_filter.ComputeCellNormalsOff()
normal_filter.AutoOrientNormalsOn()
normal_filter.ConsistencyOn()
normal_filter.SplittingOff()
normal_filter.Update()
poly = normal_filter.GetOutput()

points      = vtk_to_numpy(poly.GetPoints().GetData())
p           = vtk_to_numpy(poly.GetPointData().GetArray("p"))
shearstress = vtk_to_numpy(poly.GetPointData().GetArray("wallShearStress"))
normals     = vtk_to_numpy(poly.GetPointData().GetArray("Normals"))

print(f"Points  : {points.shape}")
print(f"p       : {p.shape}")
print(f"wss     : {shearstress.shape}")
print(f"normals : {normals.shape}")

p_inf = 54019.55
p_gauge = p - p_inf

traction     = -p_gauge[:, np.newaxis] * normals + shearstress
traction_mag = np.linalg.norm(traction, axis=1) 

print(f"p range      : {p.min():.2f} → {p.max():.2f} Pa")
print(f"traction mag : {traction_mag.min():.2f} → {traction_mag.max():.2f} Pa")

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
  cells=[("triangle", triangles)],
  point_data={
    "p_gauge"     : p_gauge,
    "normals"     : normals,
    "traction"    : traction,
    "traction_mag": traction_mag,
    "shearstress" : shearstress,
  }
)
meshio.write("wing_surface.xdmf", mesh)

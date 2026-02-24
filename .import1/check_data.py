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

points = vtk_to_numpy(poly.GetPoints().GetData())
p = vtk_to_numpy(poly.GetPointData().GetArray("p"))

cells = poly.GetPolys()
cells.InitTraversal()
idList = vtk.vtkIdList()

triangles = []
while cells.GetNextCell(idList):
  triangles.append([idList.GetId(i) for i in range(3)])

triangles = np.array(triangles)

print("Points:", points.shape)
print("Triangles:", triangles.shape)

# --- compute face force directly from geometry ---
F_faces = []
for tri in triangles:
    x0, x1, x2 = points[tri]
    nA = 0.5 * np.cross(x1 - x0, x2 - x0)   # area vector
    p_tri = p[tri].mean()                  # average point pressure
    f_tri = -p_tri * nA
    F_faces.append(f_tri)

F_faces = np.array(F_faces)

print("Total force from VTP:", F_faces.sum(axis=0))

f_nodes = np.zeros_like(points)

for tri, f in zip(triangles, F_faces):
    for i in tri:
        f_nodes[i] += f / 3.0

print("Total nodal force:", f_nodes.sum(axis=0))

mesh = meshio.Mesh(
    points=points,
    cells=[("triangle", triangles)],
    cell_data={"f_face": [f_nodes]}
)

meshio.write("wing_data.xdmf", mesh)
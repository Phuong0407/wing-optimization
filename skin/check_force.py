import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

FOAMFILE = "../wing.vtp"

print("\n=== READ FOAM VTP ===")

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(FOAMFILE)
reader.Update()
poly = reader.GetOutput()

# Triangulate (in case)
tri = vtk.vtkTriangleFilter()
tri.SetInputData(poly)
tri.Update()
poly = tri.GetOutput()

points = vtk_to_numpy(poly.GetPoints().GetData())
p      = vtk_to_numpy(poly.GetPointData().GetArray("p"))
wss    = vtk_to_numpy(poly.GetPointData().GetArray("wallShearStress"))

cells = poly.GetPolys()
cells.InitTraversal()
idList = vtk.vtkIdList()

triangles = []
while cells.GetNextCell(idList):
    triangles.append([idList.GetId(i) for i in range(3)])
triangles = np.array(triangles)

print(f"Points    : {points.shape}")
print(f"Triangles : {triangles.shape}")

# ------------------------------------------------------------
# Compute geometric normals from triangle orientation
# ------------------------------------------------------------
areas     = []
normals   = []
centroids = []

for tri in triangles:
    P0, P1, P2 = points[tri[0]], points[tri[1]], points[tri[2]]
    v1 = P1 - P0
    v2 = P2 - P0
    n  = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(n)
    if area > 0:
        n_unit = n / np.linalg.norm(n)
    else:
        n_unit = np.zeros(3)
    areas.append(area)
    normals.append(n_unit)
    centroids.append((P0 + P1 + P2) / 3)

areas     = np.array(areas)
normals   = np.array(normals)
centroids = np.array(centroids)

print("\n=== BASIC INFO ===")
print("Total area:", areas.sum())
print("Mean normal:", normals.mean(axis=0))

# ------------------------------------------------------------
# Pressure force (triangle-averaged)
# ------------------------------------------------------------
p_tri = p[triangles].mean(axis=1)
wss_tri = wss[triangles].mean(axis=1)

F_pressure = (-p_tri[:,None] * normals * areas[:,None]).sum(axis=0)
F_shear    = ( wss_tri * areas[:,None]).sum(axis=0)
F_total    = F_pressure + F_shear

print("\n=== FORCE INTEGRATION ===")
print("Pressure force :", F_pressure)
print("Shear force    :", F_shear)
print("Total force    :", F_total)

# ------------------------------------------------------------
# Sanity: pressure-only magnitude check
# ------------------------------------------------------------
print("\nPressure magnitude range:", p.min(), "→", p.max())
print("Shear magnitude range   :", 
      np.linalg.norm(wss,axis=1).min(), 
      "→",
      np.linalg.norm(wss,axis=1).max())

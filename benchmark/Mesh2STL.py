import meshio
import numpy as np

MESH  = meshio.read("wing.msh")
POINT = MESH.points * 1e-3

mesh_out = meshio.Mesh(
  points=POINT,
  cells=[("triangle", MESH.cells_dict["triangle"])]
)

print("==================== EXPORT MESH TO STL ====================")
bbox = np.ptp(POINT, axis=0)
print("bbox =", bbox)
print("number of points =", len(POINT))
meshio.write("wing.stl", mesh_out, file_format="stl")
print("write mesh file to stl file: DONE")
print("==================== EXPORT MESH TO STL ====================")
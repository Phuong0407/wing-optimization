import meshio
import numpy as np

FEMMESHFILE = "../meshdata/wing.msh"
mesh = meshio.read(FEMMESHFILE)

cells  = mesh.cells_dict["triangle"]
points = mesh.points * 1E-3

mesh_out = meshio.Mesh(
  points=points,
  cells=[("triangle", cells)]
)

meshio.write("FEMMesh.xdmf", mesh_out)
print("Saved: FEMMesh.xdmf")
print(f"Points   : {points.shape}")
print(f"Triangles: {cells.shape}")

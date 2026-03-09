import meshio
import numpy as np

m = meshio.read("2ndwing.msh")
points = m.points * 1E-3

triangle6_cells = m.cells_dict["triangle6"]

mesh_out = meshio.Mesh(
  points=points,
  cells=[("triangle6", triangle6_cells)]
)

meshio.write("2ndwing.xdmf", mesh_out)
print(f"Points   : {points.shape}")
print(f"Triangle6: {triangle6_cells.shape}")
print("Saved: 2ndwing.xdmf")

import meshio

msh = meshio.read("wing.msh")

# Lọc chỉ lấy triangle cells
import numpy as np
cells    = msh.cells_dict["triangle"]
points   = msh.points * 1e-3

mesh_out = meshio.Mesh(
    points=points,
    cells=[("triangle", cells)]
)

meshio.write("wing_skin_fenics.xdmf", mesh_out)
print("Saved: wing_skin_fenics.xdmf")
print(f"Points   : {points.shape}")
print(f"Triangles: {cells.shape}")

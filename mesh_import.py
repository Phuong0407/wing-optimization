import meshio
import numpy as np

msh = meshio.read("wingTRI.msh")

msh.points[:, :3] *= 1e-3

triangle_cells = msh.cells_dict["triangle"]
triangle_data  = msh.cell_data_dict["gmsh:physical"]["triangle"]

triangle_mesh = meshio.Mesh(
    points=msh.points,
    cells=[("triangle", triangle_cells)],
    cell_data={"gmsh:physical": [triangle_data]},
)

meshio.write("wingTRI.xdmf", triangle_mesh)
import meshio
import numpy as np

msh = meshio.read("wing.msh")

points = msh.points * 1e-3

triangle_cells = msh.cells_dict["triangle"]
line_cells = msh.cells_dict["line"]

triangle_data = msh.cell_data_dict["gmsh:physical"]["triangle"]
line_data = msh.cell_data_dict["gmsh:physical"]["line"]

mesh_tri = meshio.Mesh(
  points=points,
  cells=[("triangle", triangle_cells)],
  cell_data={"name_to_read": [triangle_data]}
)

mesh_line = meshio.Mesh(
  points=points,
  cells=[("line", line_cells)],
  cell_data={"name_to_read": [line_data]}
)

meshio.write("mesh.xdmf", mesh_tri)
meshio.write("mesh_facet.xdmf", mesh_line)

cells = msh.cells_dict["triangle"]
cell_data = {
  "gmsh:physical": [
    msh.cell_data_dict["gmsh:physical"]["triangle"]
  ]
}

mesh_out = meshio.Mesh(
  points=points, cells=[("triangle", cells)],
  cell_data=cell_data
)
meshio.write("wing.xdmf", mesh_out)

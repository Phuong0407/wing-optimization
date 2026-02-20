import meshio

msh = meshio.read("cylinder_shell.msh")
tri = msh.get_cells_type("triangle")

mesh = meshio.Mesh(points=msh.points,cells=[("triangle", tri)],)

meshio.write("cylinder.xdmf", mesh)

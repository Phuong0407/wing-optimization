import meshio
m = meshio.read("rib_2_final.vtk")
meshio.write("rib_2_final.msh", m, file_format="gmsh22")

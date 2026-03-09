import meshio
m = meshio.read("2ndwing.msh")
print(m.cells_dict.keys())

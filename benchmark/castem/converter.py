import meshio

msh = meshio.read("wingCAST3M.msh")

points = msh.points
triangles = msh.cells_dict.get("triangle", [])

with open("wingCAST3M.dgibi", "w") as f:
    f.write("DEBUT;\n\n")

    # Write points
    for i, (x, y, z) in enumerate(points):
        f.write(f"P{i+1} = POINT {x} {y} {z} ;\n")

    f.write("\n")

    # Write elements
    for i, tri in enumerate(triangles):
        n1, n2, n3 = tri + 1
        f.write(f"E{i+1} = TRI3 P{n1} P{n2} P{n3} ;\n")

    f.write("\n")

    # Merge elements into mesh
    f.write("MAIL = E1")
    for i in range(2, len(triangles)+1):
        f.write(f" ET E{i}")
    f.write(" ;\n")

    f.write("\nFIN;\n")

print("Generated wingCAST3M.dgibi")
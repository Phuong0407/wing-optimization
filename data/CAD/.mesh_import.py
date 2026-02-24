import meshio
import numpy as np

msh = meshio.read("wing.msh")
pts = msh.points * 1e-3
tri = msh.cells_dict["triangle"]
tags = msh.cell_data_dict["gmsh:physical"]["triangle"]

fixed_tri = []
for idx, (i, j, k) in enumerate(tri):
    x0, x1, x2 = pts[i], pts[j], pts[k]
    nA = np.cross(x1 - x0, x2 - x0)
    
    if tags[idx] == 14:  # upper: normal +z
        if nA[2] < 0:
            fixed_tri.append([k, j, i])
        else:
            fixed_tri.append([i, j, k])
    else:  # tag 15, lower: normal -z
        if nA[2] > 0:
            fixed_tri.append([k, j, i])
        else:
            fixed_tri.append([i, j, k])

fixed_tri = np.array(fixed_tri)

normals = []
for idx, (i, j, k) in enumerate(fixed_tri):
    nA = np.cross(pts[j]-pts[i], pts[k]-pts[i])
    normals.append(nA / np.linalg.norm(nA))
normals = np.array(normals)

upper_mask = tags == 14
lower_mask = tags == 15
print("Upper: % +z:", (normals[upper_mask, 2] > 0).mean() * 100)
print("Lower: % -z:", (normals[lower_mask, 2] < 0).mean() * 100)

meshio.write("wingTRI_fixed.xdmf", meshio.Mesh(
    points=pts,
    cells=[("triangle", fixed_tri)],
    cell_data={"gmsh:physical": [tags]}
))
print("Saved wingTRI_fixed.xdmf")

# Export upper surface riêng
upper_idx = np.where(tags == 14)[0]
upper_tri_data = fixed_tri[upper_idx]
upper_tags_data = tags[upper_idx]

meshio.write("wingTRI_upper.xdmf", meshio.Mesh(
    points=pts,
    cells=[("triangle", upper_tri_data)],
    cell_data={"gmsh:physical": [upper_tags_data]}
))

# Export lower surface riêng
lower_idx = np.where(tags == 15)[0]
lower_tri_data = fixed_tri[lower_idx]
lower_tags_data = tags[lower_idx]

meshio.write("wingTRI_lower.xdmf", meshio.Mesh(
    points=pts,
    cells=[("triangle", lower_tri_data)],
    cell_data={"gmsh:physical": [lower_tags_data]}
))

print(f"Upper: {len(upper_tri_data)} triangles → wingTRI_upper.xdmf")
print(f"Lower: {len(lower_tri_data)} triangles → wingTRI_lower.xdmf")

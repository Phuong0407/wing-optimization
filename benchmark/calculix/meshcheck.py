# In ra mesh info de so sanh
import meshio
import numpy as np

fem_mesh  = meshio.read("FEMMesh.xdmf")
foam_mesh = meshio.read("MappedTraction.xdmf")

print(f"FEMMesh points    : {fem_mesh.points.shape}")
print(f"MappedTract points: {foam_mesh.points.shape}")

# Check element type
for k in fem_mesh.cells_dict:
    print(f"FEMMesh cells     : {k} -> {fem_mesh.cells_dict[k].shape}")
for k in foam_mesh.cells_dict:
    print(f"MappedTract cells : {k} -> {foam_mesh.cells_dict[k].shape}")

import meshio
m = meshio.read("../wing.msh")
print(m.cells_dict.keys())

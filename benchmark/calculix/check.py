import meshio
import numpy as np

mesh = meshio.read("MappedTraction.xdmf")
points    = mesh.points
triangles = mesh.cells_dict["triangle"]

# Tinh edge length trung binh
edge_lengths = []
for conn in triangles:
    p0, p1, p2 = points[conn[0]], points[conn[1]], points[conn[2]]
    edge_lengths.append(np.linalg.norm(p1 - p0))
    edge_lengths.append(np.linalg.norm(p2 - p1))
    edge_lengths.append(np.linalg.norm(p0 - p2))

edge_lengths = np.array(edge_lengths)
print(f"Mean edge length : {edge_lengths.mean():.4f}")
print(f"Min edge length  : {edge_lengths.min():.4f}")
print(f"Max edge length  : {edge_lengths.max():.4f}")
print(f"Current THICK    : 1.0 mm")
print(f"Ratio THICK/mean : {1.0/edge_lengths.mean():.4f}")

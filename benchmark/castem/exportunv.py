import meshio
import numpy as np

m = meshio.read("../../data/CAD/wing.msh")
points = m.points * 1e-3  # mm -> m, convert ngay tại đây

mesh_out = meshio.Mesh(
    points=points,
    cells=[("triangle", m.cells_dict["triangle"])]
)
meshio.write("wing.stl", mesh_out, file_format="stl")
print("Done")

# Verify
print(f"X: {points[:,0].min():.6f} -> {points[:,0].max():.6f} m")
print(f"Y: {points[:,1].min():.6f} -> {points[:,1].max():.6f} m")
print(f"Z: {points[:,2].min():.6f} -> {points[:,2].max():.6f} m")
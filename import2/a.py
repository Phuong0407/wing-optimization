import numpy as np
import pyvista as pv

FOAMFile = "wing.vtp"
FOAMData = pv.read(FOAMFile)

FOAMData = FOAMData.compute_normals(
    point_normals=True,
    cell_normals=False,
    auto_orient_normals=True,
    consistent_normals=True
)

points      = np.array(FOAMData.points)
pressure    = np.array(FOAMData.point_data["p"])
normals     = np.array(FOAMData.point_data["Normals"])
shearstress = np.array(FOAMData.point_data["wallShearStress"])

print(f"points shape  : {points.shape}")
print(f"pressure range: {pressure.min():.2f} → {pressure.max():.2f}")
print(f"normals shape : {normals.shape}")

traction = -pressure[:, np.newaxis] * normals + shearstress

print("=== Data Summary ===")
print(f"N points        : {len(points)}")
print(f"WSS magnitude   : min={np.linalg.norm(shearstress,axis=1).min():.4f}, max={np.linalg.norm(shearstress,axis=1).max():.4f}")

FOAMData.compute_cell_sizes()
areas_cell = np.array(FOAMData.cell_data["Area"])

from scipy.spatial import cKDTree
cell_centers = np.array(FOAMData.cell_centers().points)
tree = cKDTree(cell_centers)
_, idx = tree.query(points)
areas_point = areas_cell[idx]

F_total = np.sum(traction * areas_point[:, np.newaxis], axis=0)

print(f"\nTotal Force on wing:")
print(f"  Fx (drag) = {F_total[0]:.4f} N")
print(f"  Fy (span) = {F_total[1]:.4f} N")
print(f"  Fz (lift) = {F_total[2]:.4f} N")

np.save("of_points.npy",   points)
np.save("of_pressure.npy", pressure)
np.save("of_normals.npy",  normals)
np.save("of_traction.npy", traction)
np.save("of_areas.npy",    areas_point)
print("\nSaved: of_points.npy, of_pressure.npy, of_normals.npy, of_traction.npy")
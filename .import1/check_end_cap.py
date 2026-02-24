import numpy as np
import meshio

msh     = meshio.read("wing_surface.xdmf")
pts     = msh.points
tris    = msh.cells_dict["triangle"]
normals = msh.point_data["normals"]
traction = msh.point_data["traction"]

v0    = pts[tris[:,0]]
v1    = pts[tris[:,1]]  
v2    = pts[tris[:,2]]
areas = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)

# Tìm tip: y_max
y_max = pts[:,1].max()
y_min = pts[:,1].min()
print(f"Wing span: y = {y_min:.4f} → {y_max:.4f} m")

# Identify tip cap triangles (gần y_max)
tri_centers_y = (pts[tris[:,0],1] + pts[tris[:,1],1] + pts[tris[:,2],1]) / 3.0
tip_mask = tri_centers_y > (y_max - 0.05)  # threshold 5cm

print(f"Tip cap triangles: {tip_mask.sum()}")
print(f"Tip cap area: {areas[tip_mask].sum():.6f} m²")

# Tính lực từ tip cap
t_center = (traction[tris[:,0]] + traction[tris[:,1]] + traction[tris[:,2]]) / 3.0
F_tip = np.sum(t_center[tip_mask] * areas[tip_mask, np.newaxis], axis=0)
print(f"\nForce từ tip cap:")
print(f"  Fx_tip = {F_tip[0]:.4f} N")
print(f"  Fy_tip = {F_tip[1]:.4f} N")
print(f"  Fz_tip = {F_tip[2]:.4f} N")

# Nếu Fy_tip ≈ Fy_error thì confirm
Fy_error = -4433.4000 - (-294.9657)
print(f"\nFy error = {Fy_error:.4f} N")
print(f"Fy_tip   = {F_tip[1]:.4f} N")
print(f"Match?   = {abs(F_tip[1] - Fy_error) < 100}")

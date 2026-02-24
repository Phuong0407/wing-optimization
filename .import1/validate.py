import numpy as np
import meshio

# ── Kết quả từ OpenFOAM (nhập tay) ──────────────────────────────
F_of = np.array([1.133076e+03, -4.433400e+03, 1.782476e+04])  # thay bằng giá trị thực từ OpenFOAM
print(f"=== OpenFOAM Reference ===")
print(f"Fx = {F_of[0]:.4f} N")
print(f"Fy = {F_of[1]:.4f} N")
print(f"Fz = {F_of[2]:.4f} N")
print(f"|F| = {np.linalg.norm(F_of):.4f} N")

# ── Load structural mesh + mapped traction ───────────────────────
msh             = meshio.read("wing_final.xdmf")
pts             = msh.points
tris            = msh.cells_dict["triangle"]
traction        = msh.point_data["traction"]

# ── Integrate traction sur le mesh ──────────────────────────────
v0 = pts[tris[:, 0]]
v1 = pts[tris[:, 1]]
v2 = pts[tris[:, 2]]
areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

t_center = (traction[tris[:,0]] + traction[tris[:,1]] + traction[tris[:,2]]) / 3.0
F_mapped = np.sum(t_center * areas[:, np.newaxis], axis=0)

print(f"\n=== Mapped Traction Force ===")
print(f"Fx = {F_mapped[0]:.4f} N")
print(f"Fy = {F_mapped[1]:.4f} N")
print(f"Fz = {F_mapped[2]:.4f} N")
print(f"|F| = {np.linalg.norm(F_mapped):.4f} N")

# ── Validation ───────────────────────────────────────────────────
print(f"\n=== Validation ===")
print(f"{'':6} {'OpenFOAM':>12} {'Mapped':>12} {'Error %':>10}")
print(f"{'-'*42}")
for i, label in enumerate(["Fx", "Fy", "Fz"]):
    err = abs(F_mapped[i] - F_of[i]) / (abs(F_of[i]) + 1e-10) * 100
    print(f"{label:6} {F_of[i]:12.4f} {F_mapped[i]:12.4f} {err:9.2f}%")

err_total = abs(np.linalg.norm(F_mapped) - np.linalg.norm(F_of)) / np.linalg.norm(F_of) * 100
print(f"{'|F|':6} {np.linalg.norm(F_of):12.4f} {np.linalg.norm(F_mapped):12.4f} {err_total:9.2f}%")

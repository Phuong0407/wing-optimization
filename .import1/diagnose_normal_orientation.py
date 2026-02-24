import numpy as np
import meshio
import pyvista as pv

# Load data
msh      = meshio.read("wing_surface.xdmf")   # OpenFOAM mesh
pts      = msh.points
tris     = msh.cells_dict["triangle"]
traction = msh.point_data["traction"]
normals  = msh.point_data["normals"]
p        = msh.point_data["p"]

# ── Check 1: Normal orientation ──────────────────────────────────
# Normals phải hướng ra ngoài (away from wing interior)
# Centroid của wing
centroid = pts.mean(axis=0)
print(f"Wing centroid: {centroid}")

# Vector từ centroid đến mỗi point
outward = pts - centroid

# Dot product: nếu > 0 thì normal hướng ra ngoài (correct)
dot = np.sum(normals * outward, axis=1)
pct_correct = (dot > 0).mean() * 100
print(f"Normals pointing outward: {pct_correct:.1f}%")
print(f"Normals pointing inward : {100-pct_correct:.1f}%")

# ── Check 2: Traction components ────────────────────────────────
print(f"\nTraction x: {traction[:,0].min():.2f} → {traction[:,0].max():.2f}")
print(f"Traction y: {traction[:,1].min():.2f} → {traction[:,1].max():.2f}")
print(f"Traction z: {traction[:,2].min():.2f} → {traction[:,2].max():.2f}")

# ── Check 3: Pressure sign ───────────────────────────────────────
# OpenFOAM compressible: p là absolute pressure hay gauge?
print(f"\nPressure: {p.min():.2f} → {p.max():.2f} Pa")
print(f"Pressure mean: {p.mean():.2f} Pa")
# Nếu mean >> 0 thì là absolute pressure → cần trừ p_inf

import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from mpi4py import MPI
from dolfinx import io
from dolfinx.fem import functionspace

with io.XDMFFile(MPI.COMM_WORLD, "rib_0.xdmf", "r") as f:
    domain = f.read_mesh(name="Grid")

V1          = functionspace(domain, ("CG", 1))
node_coords = V1.tabulate_dof_coordinates()
y_rib       = float(np.mean(node_coords[:, 1]))

foam        = meshio.read("FOAMData.xdmf")
foam_pts    = foam.points
foam_p      = foam.point_data["p"]
foam_n      = foam.point_data["normals"]   # ← dùng normals!
y_span      = foam_pts[:,1].max() - foam_pts[:,1].min()
band        = np.abs(foam_pts[:,1] - y_rib) < 0.15 * y_span

# ── PHÂN TÁCH BẰNG NORMAL Z COMPONENT ────────────────────────────────
# Upper surface (suction): normal hướng +z (ra ngoài phía trên)
# Lower surface (pressure): normal hướng -z (ra ngoài phía dưới)
nz          = foam_n[:, 2]

print("Normal Z range:", nz[band].min(), "→", nz[band].max())
print("Normal Z mean :", nz[band].mean())

# Split dựa theo normal direction
foam_upper  = band & (nz >  0.1)   # normal pointing up = upper surface
foam_lower  = band & (nz < -0.1)   # normal pointing down = lower surface
foam_edge   = band & (np.abs(nz) <= 0.1)  # LE/TE edge region

print(f"\nUpper (nz> 0.1): {foam_upper.sum()} pts")
print(f"Lower (nz<-0.1): {foam_lower.sum()} pts")
print(f"Edge  (|nz|≤0.1): {foam_edge.sum()} pts")

print(f"\nPressure UPPER : {foam_p[foam_upper].min():.1f} → "
      f"{foam_p[foam_upper].max():.1f}  mean={foam_p[foam_upper].mean():.1f}")
print(f"Pressure LOWER : {foam_p[foam_lower].min():.1f} → "
      f"{foam_p[foam_lower].max():.1f}  mean={foam_p[foam_lower].mean():.1f}")

# Interpolate lên rib nodes
p_upper = NearestNDInterpolator(
    foam_pts[foam_upper], foam_p[foam_upper])(node_coords)
p_lower = NearestNDInterpolator(
    foam_pts[foam_lower], foam_p[foam_lower])(node_coords)

# Net pressure → lift direction
dp = p_lower - p_upper
print(f"\nDP = p_lower - p_upper:")
print(f"  mean : {dp.mean():.1f} Pa")
print(f"  range: {dp.min():.1f} → {dp.max():.1f} Pa")
print(f"  sign : {(dp>0).mean()*100:.1f}% positive")

# Visualize FOAM surface colored by normal direction
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot 1: normal z component
sc = axes[0].scatter(foam_pts[band,0], foam_pts[band,2],
                     c=nz[band], cmap='RdBu_r', vmin=-1, vmax=1, s=2)
plt.colorbar(sc, ax=axes[0], label='Normal Z')
axes[0].set_title('Surface normals Z component\n(blue=upper, red=lower)')
axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('z [m]')
axes[0].set_aspect('equal')

# Plot 2: pressure colored by surface side
colors = np.where(foam_upper[band], 'steelblue',
         np.where(foam_lower[band], 'crimson', 'gray'))
for c, lbl in [('steelblue','Upper'), ('crimson','Lower'), ('gray','Edge')]:
    m = (colors==c)
    if m.sum()>0:
        axes[1].scatter(foam_pts[band][m,0], foam_pts[band][m,2],
                        c=c, s=3, label=f'{lbl} ({m.sum()})')
axes[1].set_title('Surface identification via normals')
axes[1].set_xlabel('x [m]'); axes[1].set_ylabel('z [m]')
axes[1].set_aspect('equal')
axes[1].legend(markerscale=3)

# Plot 3: DP on rib
sc = axes[2].scatter(node_coords[:,0]*1e3, node_coords[:,2]*1e3,
                     c=dp, cmap='RdBu_r', s=3)
plt.colorbar(sc, ax=axes[2], label='ΔP [Pa]')
axes[2].set_title(f'ΔP = p_lower - p_upper\nmean={dp.mean():.0f} Pa')
axes[2].set_xlabel('x [mm]'); axes[2].set_ylabel('z [mm]')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig("dp_normals.png", dpi=150, bbox_inches='tight')
print("\nSaved: dp_normals.png")
plt.show()

# Lưu DP để dùng trong optimization
np.save("dp_rib0.npy", dp)
print(f"Saved: dp_rib0.npy")
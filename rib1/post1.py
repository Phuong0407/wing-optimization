# clean_topology.py — làm sạch kết quả bằng morphological filter
import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpi4py import MPI
from dolfinx import io, fem
from dolfinx.fem import form, functionspace
from dolfinx.mesh import compute_midpoints
from scipy.spatial import KDTree
import ufl
import meshio

with io.XDMFFile(MPI.COMM_WORLD, "topopt/rib_0_rho_final.xdmf", "r") as f:
    domain = f.read_mesh(name="Grid")

tdim      = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
midpoints = compute_midpoints(domain, tdim, np.arange(num_cells))
pts       = domain.geometry.x
cells     = np.array(domain.geometry.dofmap).reshape(-1, 3)
centroids = pts[cells].mean(axis=1)

with h5py.File("topopt/rib_0_rho_final.h5","r") as f:
    density = f["Function/density/0"][:].flatten()

z_min    = pts[:,2].min()
z_max    = pts[:,2].max()
FLANGE_T = 0.05*(z_max-z_min)
passive  = ((midpoints[:,2] < z_min+FLANGE_T) |
            (midpoints[:,2] > z_max-FLANGE_T))
free     = ~passive

# ── SPATIAL SMOOTHING trên density ───────────────────────────────────
# Gaussian-like weighted average trên neighbors
R_smooth = 0.030   # 30mm smoothing radius
tree     = KDTree(centroids)
dist_mat = tree.sparse_distance_matrix(tree, R_smooth).tocsr()
weights  = dist_mat.copy()
weights.data = np.maximum(R_smooth - weights.data, 0)
w_sum    = np.array(weights.sum(1)).flatten()
density_smooth = (weights @ density) / (w_sum + 1e-30)

# ── THỬ NHIỀU THRESHOLD ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

for idx, (thresh, use_smooth) in enumerate([
    (0.3,  False), (0.4,  False), (0.5,  False), (0.6,  False),
    (0.3,  True),  (0.4,  True),  (0.5,  True),  (0.6,  True),
]):
    row = idx // 4
    col = idx % 4
    ax  = axes[row, col]

    d    = density_smooth if use_smooth else density
    tag  = "smoothed" if use_smooth else "raw"
    solid = d > thresh

    ax.set_facecolor('#E8E8E8')
    ax.scatter(centroids[solid & free,   0]*1e3,
               centroids[solid & free,   2]*1e3,
               c='steelblue', s=3, label='Solid')
    ax.scatter(centroids[solid & passive,0]*1e3,
               centroids[solid & passive,2]*1e3,
               c='crimson', s=3, label='Flange')
    ax.set_title(f'ρ>{thresh} [{tag}]\nsolid={solid.mean()*100:.1f}%')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_aspect('equal')

plt.suptitle('Topology at different thresholds — raw vs smoothed',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("topology_thresholds.png", dpi=150, bbox_inches='tight')
print("Saved: topology_thresholds.png")
plt.show()

# ── EXPORT best version ───────────────────────────────────────────────
# Dùng smoothed + threshold 0.5
best_solid = density_smooth > 0.5
meshio.write("rib_0_solid_clean.stl",
    meshio.Mesh(points=pts,
                cells=[("triangle", cells[best_solid])]))
print(f"Saved: rib_0_solid_clean.stl")
print(f"Solid cells: {best_solid.sum()}/{num_cells} = {best_solid.mean()*100:.1f}%")

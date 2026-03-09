import numpy as np
import meshio
import h5py
import triangle as tr
from skimage import measure
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator
from matplotlib.path import Path as MplPath
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────
RIB_INDEX  = 2
RHO_XDMF   = f"topopt/rib_{RIB_INDEX}_rho_final.xdmf"
OUTPUT_DIR = Path("topopt")
THRESHOLD  = 0.5
GRID_RES   = 400
SMOOTH_SIG = 6.0
N_BOUNDARY = 120
MAX_AREA   = 5e-6

# ── LOAD ──────────────────────────────────────────────────────────────
print("Loading...")
with h5py.File(Path(RHO_XDMF).with_suffix(".h5"), "r") as f:
    leaves = []
    f.visit(lambda name: leaves.append(name) if isinstance(f[name], h5py.Dataset) else None)
    pts     = f[next(k for k in leaves if "geometry" in k.lower())][()]
    tris    = f[next(k for k in leaves if "topology" in k.lower())][()]
    rho_raw = f[next(k for k in leaves if "density"  in k.lower())][()]

if tris.shape[1] == 4:
    tris = tris[:, 1:]

rho_vals = rho_raw.squeeze()
n_pts, n_tris = len(pts), len(tris)

# DG0 → point data
if len(rho_vals) == n_tris:
    rho_pt = np.zeros(n_pts)
    count  = np.zeros(n_pts)
    for i, tri in enumerate(tris):
        for node in tri:
            rho_pt[node] += rho_vals[i]
            count[node]  += 1
    rho_vals = rho_pt / np.maximum(count, 1)

print(f"  {n_pts} pts, {n_tris} tris, rho=[{rho_vals.min():.3f},{rho_vals.max():.3f}]")

y_rib = float(np.mean(pts[:, 1]))
x_2d  = pts[:, 0]
z_2d  = pts[:, 2]

# ── RASTERIZE ─────────────────────────────────────────────────────────
print("Rasterizing...")
margin = 0.02 * max(x_2d.max() - x_2d.min(), z_2d.max() - z_2d.min())
x_min, x_max = x_2d.min() - margin, x_2d.max() + margin
z_min, z_max = z_2d.min() - margin, z_2d.max() + margin

xi, zi = np.linspace(x_min, x_max, GRID_RES), np.linspace(z_min, z_max, GRID_RES)
Xi, Zi = np.meshgrid(xi, zi)

rho_grid = np.clip(
    RBFInterpolator(np.column_stack([x_2d, z_2d]), rho_vals, kernel="linear", epsilon=1)
    (np.column_stack([Xi.ravel(), Zi.ravel()])).reshape(GRID_RES, GRID_RES), 0, 1
)
rho_smooth = gaussian_filter(rho_grid, sigma=SMOOTH_SIG)

# ── EXTRACT CONTOURS ──────────────────────────────────────────────────
print("Extracting contours...")
all_contours = measure.find_contours(rho_smooth, THRESHOLD)
if len(all_contours) == 0:
    raise RuntimeError("No contour found.")
print(f"  Total contours: {len(all_contours)}")

def to_phys(c):
    return np.column_stack([
        np.interp(c[:, 1], [0, GRID_RES-1], [x_min, x_max]),
        np.interp(c[:, 0], [0, GRID_RES-1], [z_min, z_max])
    ])

# Outer = largest bounding box → airfoil boundary
bbox_areas = [(c[:, 0].max()-c[:, 0].min()) * (c[:, 1].max()-c[:, 1].min())
              for c in all_contours]
outer_idx  = int(np.argmax(bbox_areas))
outer_xz   = to_phys(all_contours[outer_idx])
print(f"  Outer contour: index={outer_idx}, {len(outer_xz)} pts, "
      f"bbox={bbox_areas[outer_idx]:.4f}")

# Inner holes = contours inside airfoil
airfoil_path = MplPath(outer_xz)
holes = []
for i, c in enumerate(all_contours):
    if i == outer_idx:
        continue
    c_phys = to_phys(c)
    if airfoil_path.contains_points(c_phys.mean(axis=0, keepdims=True))[0] and len(c) > 10:
        holes.append(c_phys)
print(f"  Inner holes: {len(holes)}")

# ── DOWNSAMPLE ────────────────────────────────────────────────────────
def downsample_arc(pts_2d, n):
    diffs = np.diff(pts_2d, axis=0, append=pts_2d[:1])
    arc   = np.cumsum(np.linalg.norm(diffs, axis=1))
    arc   = np.insert(arc, 0, 0)[:-1]
    idx   = np.clip(np.searchsorted(arc, np.linspace(0, arc[-1], n, endpoint=False)),
                    0, len(pts_2d) - 1)
    return pts_2d[idx]

boundary_xz = downsample_arc(outer_xz, N_BOUNDARY)

# ── MESH ──────────────────────────────────────────────────────────────
print("Meshing...")
all_verts = list(boundary_xz)
all_segs  = [[j, (j+1) % N_BOUNDARY] for j in range(N_BOUNDARY)]
hole_pts  = []

for hole in holes:
    n_h    = max(20, len(hole) // 4)
    h_ds   = downsample_arc(hole, n_h)
    offset = len(all_verts)
    all_verts.extend(h_ds)
    for j in range(n_h):
        all_segs.append([offset + j, offset + (j+1) % n_h])
    hole_pts.append(h_ds.mean(axis=0).tolist())

mesh_input = {
    "vertices": np.array(all_verts),
    "segments": np.array(all_segs, dtype=int),
}
if hole_pts:
    mesh_input["holes"] = np.array(hole_pts)

mesh_out = tr.triangulate(mesh_input, f"pqa{MAX_AREA}q30")
pts_2d   = mesh_out["vertices"]
tris_f   = mesh_out["triangles"].astype(int)
print(f"  Final mesh: {len(pts_2d)} pts, {len(tris_f)} tris")

# ── RESTORE 3D + EXPORT ───────────────────────────────────────────────
pts_3d       = np.zeros((len(pts_2d), 3))
pts_3d[:, 0] = pts_2d[:, 0]
pts_3d[:, 1] = y_rib
pts_3d[:, 2] = pts_2d[:, 1]

final_mesh = meshio.Mesh(points=pts_3d, cells=[("triangle", tris_f)])
meshio.write(str(OUTPUT_DIR / f"rib_{RIB_INDEX}_final.xdmf"), final_mesh)
meshio.write(str(OUTPUT_DIR / f"rib_{RIB_INDEX}_final.vtk"),  final_mesh)

boundary_3d       = np.zeros((len(boundary_xz), 3))
boundary_3d[:, 0] = boundary_xz[:, 0]
boundary_3d[:, 1] = y_rib
boundary_3d[:, 2] = boundary_xz[:, 1]
np.save(str(OUTPUT_DIR / f"rib_{RIB_INDEX}_final_boundary.npy"), boundary_3d)

print(f"Done -> {OUTPUT_DIR}/rib_{RIB_INDEX}_final.xdmf")
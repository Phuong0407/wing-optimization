import numpy as np

displacements = []
node_ids = []
reading = False

with open("wing_calculix.dat", "r") as f:
    for line in f:
        if "displacements (vx,vy,vz)" in line:
            reading = True
            continue
        if reading:
            parts = line.split()
            if len(parts) == 4:
                try:
                    node_id = int(parts[0])
                    ux = float(parts[1])
                    uy = float(parts[2])
                    uz = float(parts[3])
                    displacements.append([ux, uy, uz])
                    node_ids.append(node_id)
                except ValueError:
                    reading = False

displacements = np.array(displacements)
node_ids      = np.array(node_ids)
mag           = np.linalg.norm(displacements, axis=1)

print(f"Nodes read               : {len(displacements)}")
print(f"Max displacement (mag)   : {mag.max():.6e} m")
print(f"Max Ux                   : {np.abs(displacements[:,0]).max():.6e} m")
print(f"Max Uy                   : {np.abs(displacements[:,1]).max():.6e} m")
print(f"Max Uz                   : {np.abs(displacements[:,2]).max():.6e} m")
print(f"Node with max disp       : {node_ids[np.argmax(mag)]}")

# Export cho benchmark
np.save("calculix_displacements.npy", displacements)
np.save("calculix_node_ids.npy", node_ids)
print("\nSaved: calculix_displacements.npy")













"""
PostProcess_Rotations.py
========================
Tính Rx, Ry, Rz từ gradient chuyển vị
Đọc từ wing_calculix.dat + wing_calculix.inp
"""
import numpy as np
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────
# BƯỚC 1: Đọc tọa độ nút từ .inp
# ─────────────────────────────────────────────
def read_nodes_from_inp(inp_file):
    coords = {}
    reading = False
    with open(inp_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith("*NODE"):
                reading = True
                continue
            if reading:
                if line.startswith("*"):
                    reading = False
                    continue
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        nid = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        coords[nid] = np.array([x, y, z])
                    except ValueError:
                        continue
    return coords

# ─────────────────────────────────────────────
# BƯỚC 2: Đọc chuyển vị từ .dat
# ─────────────────────────────────────────────
def read_displacements_from_dat(dat_file):
    displacements = []
    node_ids = []
    reading = False
    with open(dat_file, "r") as f:
        for line in f:
            if "displacements (vx,vy,vz)" in line:
                reading = True
                continue
            if reading:
                parts = line.split()
                if len(parts) == 4:
                    try:
                        node_ids.append(int(parts[0]))
                        displacements.append([float(parts[1]),
                                              float(parts[2]),
                                              float(parts[3])])
                    except ValueError:
                        reading = False
    return np.array(node_ids), np.array(displacements)

# ─────────────────────────────────────────────
# BƯỚC 3: Tính rotation từ gradient (least squares)
# ─────────────────────────────────────────────
def compute_rotations(node_ids, displacements, coords, k_neighbors=8):
    """
    Tại mỗi nút, fit gradient tensor J = dU/dx bằng least squares
    từ các nút lân cận, rồi lấy phần antisymmetric:
        Rx = 0.5*(dUz/dy - dUy/dz)
        Ry = 0.5*(dUx/dz - dUz/dx)
        Rz = 0.5*(dUy/dx - dUx/dy)
    """
    coord_array = np.array([coords[n] for n in node_ids])
    tree = cKDTree(coord_array)
    rotations = np.zeros((len(node_ids), 3))

    for i in range(len(node_ids)):
        k = min(k_neighbors + 1, len(node_ids))
        dists, idxs = tree.query(coord_array[i], k=k)
        idxs = idxs[1:]  # bỏ chính nó

        dx = coord_array[idxs] - coord_array[i]   # (k, 3)
        du = displacements[idxs] - displacements[i]  # (k, 3)

        # Loại bỏ nút quá xa (outlier)
        d_max = dists[1:].mean() * 3.0
        mask = dists[1:] < d_max
        if mask.sum() < 3:
            continue

        # Least squares: dx @ J = du  →  J shape (3, 3)
        J, _, _, _ = np.linalg.lstsq(dx[mask], du[mask], rcond=None)

        # Antisymmetric part = rotation tensor
        rotations[i, 0] = 0.5 * (J[2, 1] - J[1, 2])  # Rx
        rotations[i, 1] = 0.5 * (J[0, 2] - J[2, 0])  # Ry
        rotations[i, 2] = 0.5 * (J[1, 0] - J[0, 1])  # Rz

    return rotations

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    INP_FILE = "wing_calculix.inp"
    DAT_FILE = "wing_calculix.dat"

    print("Reading nodes from .inp ...")
    coords = read_nodes_from_inp(INP_FILE)
    print(f"  Nodes loaded: {len(coords)}")

    print("Reading displacements from .dat ...")
    node_ids, displacements = read_displacements_from_dat(DAT_FILE)
    print(f"  Nodes read  : {len(node_ids)}")

    print("Computing rotations ...")
    rotations = compute_rotations(node_ids, displacements, coords, k_neighbors=8)

    # ── Report ──
    mag_u = np.linalg.norm(displacements, axis=1)
    mag_r = np.linalg.norm(rotations, axis=1)

    print(f"\n{'='*45}")
    print(f"  Max |U|  : {mag_u.max():.6e} m")
    print(f"  Max Ux   : {np.abs(displacements[:,0]).max():.6e} m")
    print(f"  Max Uy   : {np.abs(displacements[:,1]).max():.6e} m")
    print(f"  Max Uz   : {np.abs(displacements[:,2]).max():.6e} m")
    print(f"{'─'*45}")
    print(f"  Max |R|  : {mag_r.max():.6e} rad")
    print(f"  Max Rx   : {np.abs(rotations[:,0]).max():.6e} rad")
    print(f"  Max Ry   : {np.abs(rotations[:,1]).max():.6e} rad")
    print(f"  Max Rz   : {np.abs(rotations[:,2]).max():.6e} rad")
    print(f"{'='*45}")

    # ── Save ──
    np.save("calculix_displacements.npy", displacements)
    np.save("calculix_rotations.npy", rotations)
    np.save("calculix_node_ids.npy", node_ids)
    print("\nSaved: calculix_displacements.npy")
    print("       calculix_rotations.npy")
    print("       calculix_node_ids.npy")

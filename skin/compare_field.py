from dolfinx import io, fem
from mpi4py import MPI
import h5py
import numpy as np
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

comm = MPI.COMM_WORLD

# --- Load meshes ---
with io.XDMFFile(comm, "Result/results.xdmf", "r") as xdmf:
    mesh_ref = xdmf.read_mesh(name="mesh")
with io.XDMFFile(comm, "Result/results_.xdmf", "r") as xdmf:
    mesh_dst = xdmf.read_mesh(name="mesh")

# --- Load displacement nodal data (P1-nodal output) ---
with h5py.File("Result/results.h5", "r") as f:
    u_ref_data = f["/Function/Displacement/0"][:]
with h5py.File("Result/results_.h5", "r") as f:
    u_dst_data = f["/Function/Displacement/0"][:]

V_ref = fem.functionspace(mesh_ref, ("Lagrange", 1, (3,)))
u_ref = fem.Function(V_ref)
u_ref.x.array[:] = u_ref_data.reshape(-1)

V_dst = fem.functionspace(mesh_dst, ("Lagrange", 1, (3,)))
u_dst = fem.Function(V_dst)
u_dst.x.array[:] = u_dst_data.reshape(-1)

# --- Sample points: midpoints of each dst cell ---
tdim = mesh_dst.topology.dim
mesh_dst.topology.create_connectivity(tdim, 0)
conn = mesh_dst.topology.connectivity(tdim, 0)
cells_v = conn.array.reshape(-1, conn.num_links(0))  # usually (ncell,3) for triangles
coords = mesh_dst.geometry.x
midpoints = coords[cells_v].mean(axis=1)  # (ncell, 3)

# --- Locate those points in the reference mesh ---
tree = bb_tree(mesh_ref, mesh_ref.topology.dim)
candidates = compute_collisions_points(tree, midpoints)
cells_ref = compute_colliding_cells(mesh_ref, candidates, midpoints)

u_ref_eval = np.zeros((len(midpoints), 3))
u_dst_eval = np.zeros((len(midpoints), 3))
valid = np.zeros(len(midpoints), dtype=bool)

for i, pt in enumerate(midpoints):
    ref_cells_i = cells_ref.links(i)
    if len(ref_cells_i) == 0:
        continue
    # Evaluate ref in the found reference cell
    u_ref_eval[i] = u_ref.eval(pt, int(ref_cells_i[0]))
    # Evaluate dst in its own cell i (THIS is the key fix)
    u_dst_eval[i] = u_dst.eval(pt, i)
    valid[i] = True

n_valid = int(comm.allreduce(valid.sum(), op=MPI.SUM))
n_total = int(comm.allreduce(len(valid), op=MPI.SUM))
if comm.rank == 0:
    print(f"[COMPARE] valid points: {n_valid}/{n_total}  (missing {(n_total-n_valid)/n_total:.2%})")

# --- Compute error on valid points only ---
u_ref_v = u_ref_eval[valid]
u_dst_v = u_dst_eval[valid]
diff = u_dst_v - u_ref_v

L2_diff = np.sqrt(np.sum(np.linalg.norm(diff, axis=1)**2))
L2_ref  = np.sqrt(np.sum(np.linalg.norm(u_ref_v, axis=1)**2))

# Avoid divide-by-zero nonsense
if L2_ref < 1e-14 and comm.rank == 0:
    print("[COMPARE] Reference norm ~ 0, cannot form relative L2.")
elif comm.rank == 0:
    print("Relative L2 error:", L2_diff / L2_ref)
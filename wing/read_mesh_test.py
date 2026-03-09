import ufl
from dolfinx import fem, io, mesh
from dolfinx.io import gmsh
from mpi4py import MPI
from pathlib import Path
import numpy as np
import sys

INPUT_DIR       = Path("InputData")
MESHFILE        = INPUT_DIR / "wing_new.msh"

if not MESHFILE.exists():
    MESHFILE = Path("wing.msh")

if not MESHFILE.exists():
    print(f"Error: Mesh file {MESHFILE} not found.")
    sys.exit(1)

MESH_IO     = gmsh.read_from_msh(str(MESHFILE), comm=MPI.COMM_WORLD, gdim=3)
DOMAIN      = MESH_IO.mesh
CELL_TAGS   = MESH_IO.cell_tags
FACET_TAGS  = MESH_IO.facet_tags
GDIM        = DOMAIN.geometry.dim
TDIM        = DOMAIN.topology.dim
FDIM        = TDIM - 1

print("="*60)
print(f"MESH INFO: {MESHFILE}")
print("="*60)

print(f"[MESH] Topology Dim (TDIM) : {TDIM}")
print(f"[MESH] Geometry Dim (GDIM) : {GDIM}")
print(f"[MESH] Global Cells        : {DOMAIN.topology.index_map(TDIM).size_global}")
print(f"[MESH] Global Vertices     : {DOMAIN.topology.index_map(0).size_global}")

# Bounding Box
coords = DOMAIN.geometry.x
min_coords = np.min(coords, axis=0)
max_coords = np.max(coords, axis=0)
print("-" * 40)
print(f"[GEO] Bounding Box:")
print(f"  X: {min_coords[0]:.4f} -> {max_coords[0]:.4f} (L = {max_coords[0]-min_coords[0]:.4f})")
print(f"  Y: {min_coords[1]:.4f} -> {max_coords[1]:.4f} (L = {max_coords[1]-min_coords[1]:.4f})")
print(f"  Z: {min_coords[2]:.4f} -> {max_coords[2]:.4f} (L = {max_coords[2]-min_coords[2]:.4f})")

# Cell Tags
print("-" * 40)
print("[TAGS] Cell Tags (Physical Groups - Volumes/Surfaces):")
if CELL_TAGS:
    unique_cell_tags = np.unique(CELL_TAGS.values)
    for tag in unique_cell_tags:
        count = len(CELL_TAGS.find(tag))
        print(f"  Tag {tag:<4}: {count} cells")
else:
    print("  No cell tags found.")

# Facet Tags
print("-" * 40)
print("[TAGS] Facet Tags (Physical Groups - Boundaries):")
if FACET_TAGS:
    unique_facet_tags = np.unique(FACET_TAGS.values)
    for tag in unique_facet_tags:
        count = len(FACET_TAGS.find(tag))
        print(f"  Tag {tag:<4}: {count} facets")
else:
    print("  No facet tags found.")

print("="*60)

# -----------------------------------------------------------------------------
# NEW: Export Tags for Visualization (ParaView)
# -----------------------------------------------------------------------------
print("[VISUALIZATION] Exporting tags to XDMF...")

def export_individual_tags(domain, meshtags, dim, label_prefix):
    if not meshtags:
        return
    unique_ids = np.unique(meshtags.values)
    for tag in unique_ids:
        # Extract indices for this specific tag
        indices = meshtags.find(tag)
        # Create a new MeshTags object containing only this tag
        values = np.full(len(indices), tag, dtype=np.int32)
        mt_single = mesh.meshtags(domain, dim, indices, values)
        
        filename = f"{label_prefix}_{tag}.xdmf"
        with io.XDMFFile(domain.comm, filename, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_meshtags(mt_single, domain.geometry)
        print(f"  -> {filename} ({len(indices)} entities)")

# 1. Export Cell Tags
export_individual_tags(DOMAIN, CELL_TAGS, TDIM, "debug_cell_tag")

# 2. Export Facet Tags
export_individual_tags(DOMAIN, FACET_TAGS, FDIM, "debug_facet_tag")

# 3. Debug: Check for Root (Tag 52) at Y=0
print("-" * 40)
print("[DEBUG] Checking for Root geometry at Y=0...")
y_coords = DOMAIN.geometry.x[:, 1]
nodes_at_root = np.where(np.isclose(y_coords, 0.0, atol=1e-3))[0]
print(f"  Nodes found at Y=0 : {len(nodes_at_root)}")
if len(nodes_at_root) > 0 and (FACET_TAGS is None or 52 not in FACET_TAGS.values):
    print("  WARNING: Geometry exists at Y=0, but Tag 52 is missing in the mesh file.")
print("="*60)

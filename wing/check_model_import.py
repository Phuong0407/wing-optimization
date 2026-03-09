import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh
from dolfinx import fem

MESHFILE = "wing_connected.msh"

print("\n=== IMPORT MESH ===")
mesh_data = gmsh.read_from_msh(
    MESHFILE, MPI.COMM_WORLD, gdim=3
)

domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

# -------------------------------------------------
# 1) Basic dimensions
# -------------------------------------------------
tdim = domain.topology.dim
gdim = domain.geometry.dim

print(f"Topology dim : {tdim}")
print(f"Geometry dim : {gdim}")

if tdim == 2:
    print("→ Mesh is 2D surface (shell-type mesh)")
elif tdim == 3:
    print("→ Mesh is 3D volume (solid mesh)")
else:
    print("→ Unexpected topology dimension")

# -------------------------------------------------
# 2) Bounding box (unit sanity check)
# -------------------------------------------------
coords = domain.geometry.x
xmin = coords.min(axis=0)
xmax = coords.max(axis=0)

print("\n=== BOUNDING BOX ===")
print(f"x-range : {xmin[0]:.3f} → {xmax[0]:.3f}")
print(f"y-range : {xmin[1]:.3f} → {xmax[1]:.3f}")
print(f"z-range : {xmin[2]:.3f} → {xmax[2]:.3f}")

span = xmax[1] - xmin[1]
print(f"Span in y-direction: {span:.3f}")

if span > 100:
    print("→ Units likely mm")
elif span < 10:
    print("→ Units likely meters")
else:
    print("→ Units ambiguous")

# -------------------------------------------------
# 3) Check material tags (cell_tags)
# -------------------------------------------------
print("\n=== CELL TAGS (MATERIAL REGIONS) ===")

if cell_tags is None:
    print("✗ No cell_tags found!")
else:
    unique_tags = np.unique(cell_tags.values)
    print("Material tags found:", unique_tags)

    for tag in unique_tags:
        cells = cell_tags.find(tag)
        print(f"  Tag {tag}: {len(cells)} cells")

# -------------------------------------------------
# 4) Check facet tags (BC)
# -------------------------------------------------
print("\n=== FACET TAGS (BOUNDARY REGIONS) ===")

if facet_tags is None:
    print("✗ No facet_tags found!")
else:
    unique_facets = np.unique(facet_tags.values)
    print("Facet tags found:", unique_facets)

    for tag in unique_facets:
        facets = facet_tags.find(tag)
        print(f"  Tag {tag}: {len(facets)} facets")

# -------------------------------------------------
# 5) DG0 material test (sanity)
# -------------------------------------------------
print("\n=== DG0 FIELD TEST ===")

if cell_tags is not None:
    V0 = fem.functionspace(domain, ("DG", 0))
    test_field = fem.Function(V0)

    for tag in np.unique(cell_tags.values):
        cells = cell_tags.find(tag)
        test_field.x.array[cells] = tag

    test_field.x.scatter_forward()

    print("DG0 assignment successful.")
    print("Field min:", test_field.x.array.min())
    print("Field max:", test_field.x.array.max())

print("\n=== CHECK COMPLETE ===")

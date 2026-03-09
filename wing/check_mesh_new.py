"""
check_mesh.py
─────────────
Verify that wing_full.msh is a conforming mesh.

KEY CONCEPT — Physical Group Tag vs Geometric Entity Tag
─────────────────────────────────────────────────────────
These are TWO DIFFERENT namespaces in Gmsh:

  Physical Group tag  (what you define in .geo):
      Physical Surface("upper", 14) = {2};
      → 14 is the PHYSICAL GROUP identifier
      → 2  is the GEOMETRIC ENTITY tag (what OCC assigned)

  The bridge:
      geo_entities = gmsh.model.getEntitiesForPhysicalGroup(2, 14)
      → returns [2]  (the actual surface OCC tag)

  WRONG:  gmsh.model.mesh.getElements(2, tag=14)
          → Gmsh looks for geometric surface 14 → does not exist → crash

  CORRECT: for gt in gmsh.model.getEntitiesForPhysicalGroup(2, 14):
               gmsh.model.mesh.getElements(2, gt)

Usage
─────
  python check_mesh.py
"""

import gmsh
import numpy as np
from collections import defaultdict

MESHFILE = "wing_full.msh"

# Physical group tags — defined in wing.geo
SURF_PHYS = {
    14: "upper",
    15: "lower",
    38: "rib0",
    39: "rib750",
    40: "rib1500",
    41: "rib2250",
    42: "rib3000",
    43: "mainspar",
    44: "tailspar",
}

# Physical curve tags for rib perimeters — from wing.geo
RIB_PERIMETER_CURVES = {
    38: 45,   # ribrootc
    39: 46,   # rib750c
    40: 47,   # rib1500c
    41: 48,   # rib2250c
    42: 49,   # ribtailc
}

SKIN_PHYS = [14, 15]
TAG_ROOT  = 52

gmsh.initialize()
gmsh.open(MESHFILE)
gmsh.model.mesh.renumberNodes()

print("=" * 60)
print(f"Mesh file : {MESHFILE}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def geo_tags(dim, phys_tag):
    """Geometric entity tags belonging to a physical group."""
    try:
        return list(gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag))
    except Exception:
        return []

# def get_nodes_of_phys(dim, phys_tag):
#     """All nodes in a physical group — returns (node_tags, Nx3 coords)."""
#     all_tags, all_coords = [], []
#     for gt in geo_tags(dim, phys_tag):
#         try:
#             # ntags, coords, _ = gmsh.model.mesh.getNodes(dim, gt)
#             ntags, coords, _ = gmsh.model.mesh.getNodes(dim, gt, includeBoundary=True)
#             if len(ntags):
#                 all_tags.append(ntags)
#                 all_coords.append(coords.reshape(-1, 3))
#         except Exception:
#             pass
#     if not all_tags:
#         return np.array([], dtype=int), np.zeros((0, 3))
#     return np.concatenate(all_tags), np.concatenate(all_coords)


def get_nodes_of_phys(dim, phys_tag):
    all_tags, all_coords = [], []

    for gt in geo_tags(dim, phys_tag):

        if dim == 2:
            # includeBoundary=True is safe for surfaces:
            # returns interior nodes + edge nodes + corner nodes, no warnings
            try:
                ntags, coords, _ = gmsh.model.mesh.getNodes(
                    dim, gt, includeBoundary=True
                )
                if len(ntags):
                    all_tags.append(ntags)
                    all_coords.append(coords.reshape(-1, 3))
            except Exception:
                pass

        elif dim == 1:
            # includeBoundary=True on curves causes "Failed to compute
            # parameters" warnings and unreliable results for dim=0 vertices.
            # Fix: query the curve nodes WITHOUT boundary, then collect the
            # dim=0 endpoint vertices explicitly via getBoundary.

            # Step A — interior curve nodes only
            try:
                ntags, coords, _ = gmsh.model.mesh.getNodes(
                    dim, gt, includeBoundary=False
                )
                if len(ntags):
                    all_tags.append(ntags)
                    all_coords.append(coords.reshape(-1, 3))
            except Exception:
                pass

            # Step B — dim=0 endpoint vertices of this curve
            try:
                bounds = gmsh.model.getBoundary(
                    [(1, gt)], oriented=False, combined=False
                )
                for (vdim, vtag) in bounds:
                    if vdim != 0:
                        continue
                    vtags, vcoords, _ = gmsh.model.mesh.getNodes(0, vtag)
                    if len(vtags):
                        all_tags.append(vtags)
                        all_coords.append(vcoords.reshape(-1, 3))
            except Exception:
                pass

    if not all_tags:
        return np.array([], dtype=int), np.zeros((0, 3))

    # Deduplicate — corner vertices are shared by multiple curves
    tags   = np.concatenate(all_tags)
    coords = np.concatenate(all_coords)
    _, unique_idx = np.unique(tags, return_index=True)
    return tags[unique_idx], coords[unique_idx]


# ─────────────────────────────────────────────────────────────────────
# CHECK 1 — Physical groups exist and have elements
# ─────────────────────────────────────────────────────────────────────
print("\nCHECK 1 — Physical groups")
print("-" * 40)

all_ok = True

for phys_tag, name in SURF_PHYS.items():
    entities = geo_tags(2, phys_tag)
    if not entities:
        print(f"  FAIL  phys={phys_tag:2d}  {name:<12}  physical group not found")
        all_ok = False
        continue

    n_elem = 0
    for gt in entities:
        try:
            etypes, etags, _ = gmsh.model.mesh.getElements(2, gt)
            n_elem += sum(len(t) for t in etags)
        except Exception:
            pass

    status = "ok  " if n_elem > 0 else "FAIL"
    print(f"  {status}  phys={phys_tag:2d}  {name:<12}  "
          f"{n_elem:5d} elements   geo entities={entities}")
    if n_elem == 0:
        all_ok = False

# Root BC curve check
phys_curves = gmsh.model.getPhysicalGroups(1)
root_found  = any(t == TAG_ROOT for (_, t) in phys_curves)
if root_found:
    root_entities = geo_tags(1, TAG_ROOT)
    root_nodes    = set()
    for c in root_entities:
        try:
            ntags, _, _ = gmsh.model.mesh.getNodes(1, c)
            root_nodes.update(ntags.tolist())
        except Exception:
            pass
    print(f"\n  ok    phys curve 'root' tag={TAG_ROOT}  "
          f"{len(root_entities)} curves  {len(root_nodes)} nodes")
    print(f"        geo curve entities : {root_entities}")
else:
    print(f"\n  FAIL  Physical Curve 'root' tag={TAG_ROOT} not found")
    all_ok = False


# ─────────────────────────────────────────────────────────────────────
# CHECK 2 — No duplicate nodes at rib-skin interfaces
# ─────────────────────────────────────────────────────────────────────
print("\nCHECK 2 — Duplicate nodes at rib-skin interfaces")
print("-" * 40)

# Collect all skin nodes
skin_coords = {}
for phys_tag in SKIN_PHYS:
    ntags, coords = get_nodes_of_phys(2, phys_tag)
    for nt, xyz in zip(ntags, coords):
        skin_coords[int(nt)] = xyz

print(f"  Skin total nodes : {len(skin_coords)}")

# Spatial lookup: rounded xyz -> node_tag
coord_to_node = {}
for nt, xyz in skin_coords.items():
    key = tuple(np.round(xyz, decimals=6))
    if key in coord_to_node:
        print(f"  FAIL  Duplicate skin node at {np.round(xyz, 4)}  "
              f"tags {coord_to_node[key]} and {nt}")
        all_ok = False
    coord_to_node[key] = nt

# Check each rib perimeter
for rib_phys, curve_phys in RIB_PERIMETER_CURVES.items():
    rib_name = SURF_PHYS[rib_phys]
    ntags, coords = get_nodes_of_phys(1, curve_phys)

    if len(ntags) == 0:
        print(f"  WARN  {rib_name:<12}  0 nodes on perimeter "
              f"(phys curve {curve_phys}) — check .geo tag")
        continue

    n_shared = n_mismatch = n_missing = 0
    for nt, xyz in zip(ntags, coords):
        key = tuple(np.round(xyz, decimals=6))
        if key not in coord_to_node:
            n_missing += 1
        elif coord_to_node[key] != int(nt):
            n_mismatch += 1
        else:
            n_shared += 1

    if n_missing > 0 or n_mismatch > 0:
        print(f"  FAIL  {rib_name:<12}  "
              f"shared={n_shared}  missing={n_missing}  "
              f"diff_tag_same_coord={n_mismatch}  <- NOT conforming")
        all_ok = False
    else:
        print(f"  ok    {rib_name:<12}  "
              f"{n_shared} boundary nodes shared with skin (same tag)")


# ─────────────────────────────────────────────────────────────────────
# CHECK 3 — Interface edges shared by exactly 2 elements
# ─────────────────────────────────────────────────────────────────────
print("\nCHECK 3 — Interface edges shared by 2 elements")
print("-" * 40)

# Build edge -> [(phys_tag, elem_tag)] across the whole mesh
edge_to_elems = defaultdict(list)

for phys_tag in SURF_PHYS:
    for gt in geo_tags(2, phys_tag):
        try:
            etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(2, gt)
        except Exception:
            continue
        for etags, enodes in zip(etags_list, enodes_list):
            if len(etags) == 0:
                continue
            n_per  = len(enodes) // len(etags)
            enodes = enodes.reshape(-1, n_per)
            for etag, en in zip(etags, enodes):
                corners = [int(en[0]), int(en[1]), int(en[2])]
                for i in range(3):
                    edge = tuple(sorted([corners[i], corners[(i+1) % 3]]))
                    edge_to_elems[edge].append((phys_tag, int(etag)))

# Check rib perimeters
for rib_phys, curve_phys in RIB_PERIMETER_CURVES.items():
    rib_name         = SURF_PHYS[rib_phys]
    n_shared_edges   = 0
    n_boundary_edges = 0

    for gt in geo_tags(1, curve_phys):
        try:
            etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(1, gt)
        except Exception:
            continue
        for etags, enodes in zip(etags_list, enodes_list):
            if len(etags) == 0:
                continue
            n_per  = len(enodes) // len(etags)
            enodes = enodes.reshape(-1, n_per)
            for en in enodes:
                edge = tuple(sorted([int(en[0]), int(en[-1])]))
                if len(edge_to_elems.get(edge, [])) >= 2:
                    n_shared_edges += 1
                else:
                    n_boundary_edges += 1

    if n_shared_edges == 0 and n_boundary_edges == 0:
        print(f"  WARN  {rib_name:<12}  no edges found (phys curve {curve_phys})")
        continue

    if n_boundary_edges > 0:
        print(f"  FAIL  {rib_name:<12}  "
              f"shared={n_shared_edges}  unshared={n_boundary_edges}  "
              f"<- NOT conforming")
        all_ok = False
    else:
        print(f"  ok    {rib_name:<12}  "
              f"all {n_shared_edges} interface edges shared by 2 elements")


# ─────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_ok:
    print("RESULT :  ALL CHECKS PASSED")
    print("  Mesh is conforming — no Nitsche method needed.")
    print("  Tag 52 root BC is correctly defined.")
    print("  Safe to run wing.py.")
else:
    print("RESULT :  SOME CHECKS FAILED")
    print("  If CHECK 1 fails: physical group tags in SURF_PHYS do not")
    print("    match the .geo file — update the dict at top of this script.")
    print("  If CHECK 2 fails: Merge did not produce shared nodes.")
    print("  If CHECK 3 fails: interface edges are topologically disconnected.")
print("=" * 60)

gmsh.finalize()
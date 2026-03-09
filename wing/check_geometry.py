# # import gmsh
# # import sys

# # FILES = [
# #     "skin.stp", "rib0.stp", "rib750.stp", "rib1500.stp",
# #     "rib2250.stp", "rib3000.stp", "spar_main.stp", "spar_tail.stp",
# # ]

# # gmsh.initialize()
# # gmsh.model.add("wing_check")

# # # Import
# # for f in FILES:
# #     try:
# #         gmsh.model.occ.importShapes(f)
# #         print(f"  ✓  {f}")
# #     except Exception as e:
# #         print(f"  ✗  {f} → {e}")
# #         gmsh.finalize()
# #         sys.exit(1)

# # gmsh.model.occ.synchronize()

# # # Bounding box
# # print("\n=== Bounding box ===")
# # for (dim, tag) in gmsh.model.getEntities(2):
# #     xmn,ymn,zmn,xmx,ymx,zmx = gmsh.model.getBoundingBox(dim, tag)
# #     print(f"  surf {tag:3d}: x=[{xmn:.0f},{xmx:.0f}]  "
# #           f"y=[{ymn:.0f},{ymx:.0f}]  z=[{zmn:.0f},{zmx:.0f}]")

# # # Kiểm tra overlap giữa các components
# # print("\n=== Kiểm tra giao nhau ===")
# # surfs = gmsh.model.getEntities(2)
# # ok = True
# # for i in range(len(surfs)):
# #     for j in range(i+1, len(surfs)):
# #         b1 = gmsh.model.getBoundingBox(*surfs[i])
# #         b2 = gmsh.model.getBoundingBox(*surfs[j])
# #         # Kiểm tra bounding box overlap theo Y
# #         overlap_y = b1[1] < b2[4] and b2[1] < b1[4]
# #         overlap_x = b1[0] < b2[3] and b2[0] < b1[3]
# #         if overlap_x and overlap_y:
# #             print(f"  ✓  surf {surfs[i][1]} ↔ surf {surfs[j][1]} giao nhau → sẽ kết nối được")
# #             ok = True

# # if ok:
# #     print("\n  ✓  Geometry hợp lệ — chạy build_mesh.py để tạo mesh")

# # gmsh.finalize()


# # import gmsh
# # import sys
# # from collections import defaultdict

# # FILES = [
# #     "skin.stp", "rib0.stp", "rib750.stp", "rib1500.stp",
# #     "rib2250.stp", "rib3000.stp", "spar_main.stp", "spar_tail.stp",
# # ]

# # gmsh.initialize()
# # gmsh.model.add("wing_fragment_check")

# # # -------------------------
# # # 1) Import geometry
# # # -------------------------
# # for f in FILES:
# #     try:
# #         gmsh.model.occ.importShapes(f)
# #         print(f"✓ Imported {f}")
# #     except Exception as e:
# #         print(f"✗ Failed {f}: {e}")
# #         gmsh.finalize()
# #         sys.exit(1)

# # gmsh.model.occ.synchronize()

# # # -------------------------
# # # 2) Remove duplicates (important)
# # # -------------------------
# # gmsh.model.occ.healShapes()
# # gmsh.model.occ.synchronize()

# # gmsh.model.occ.removeAllDuplicates()
# # gmsh.model.occ.synchronize()

# # # -------------------------
# # # 3) Fragment all surfaces
# # # -------------------------
# # surfs_before = gmsh.model.getEntities(2)
# # gmsh.model.occ.fragment(surfs_before, [])
# # gmsh.model.occ.synchronize()

# # print("\nFragment done.")

# # # -------------------------
# # # 4) Build edge → surface adjacency map
# # # -------------------------
# # edge_to_surfaces = defaultdict(list)

# # edges = gmsh.model.getEntities(1)

# # for dim, edge_tag in edges:
# #     up, down = gmsh.model.getAdjacencies(dim, edge_tag)
# #     # down = adjacent surfaces
# #     for s in down:
# #         edge_to_surfaces[edge_tag].append(s)

# # # -------------------------
# # # 5) Count shared edges
# # # -------------------------
# # shared_edges = []
# # for edge, attached_surfs in edge_to_surfaces.items():
# #     if len(attached_surfs) >= 2:
# #         shared_edges.append((edge, attached_surfs))

# # print(f"\nTotal edges: {len(edges)}")
# # print(f"Shared edges (2+ surfaces): {len(shared_edges)}")

# # if len(shared_edges) == 0:
# #     print("\n❌ No shared edges detected — geometry NOT connected.")
# # else:
# #     print("\n✓ Shared edges detected — geometry is topologically connected.")
# #     print("\nSample shared edges:")
# #     for e, s in shared_edges[:10]:
# #         print(f"  Edge {e} shared by surfaces {s}")

# # # -------------------------
# # # 6) Check connected components at surface level
# # # -------------------------
# # print("\nChecking surface connectivity graph...")

# # surface_graph = defaultdict(set)

# # for edge, surfs in shared_edges:
# #     for i in range(len(surfs)):
# #         for j in range(i+1, len(surfs)):
# #             surface_graph[surfs[i]].add(surfs[j])
# #             surface_graph[surfs[j]].add(surfs[i])

# # # DFS to count connected components
# # visited = set()
# # components = 0

# # for _, s in gmsh.model.getEntities(2):
# #     if s not in visited:
# #         stack = [s]
# #         while stack:
# #             cur = stack.pop()
# #             if cur not in visited:
# #                 visited.add(cur)
# #                 stack.extend(surface_graph[cur])
# #         components += 1

# # print(f"\nSurface connected components: {components}")

# # if components == 1:
# #     print("✓ Geometry forms ONE connected shell structure.")
# # else:
# #     print("❌ Geometry split into multiple disconnected parts.")

# # gmsh.finalize()


# """
# check_geometry.py
# ─────────────────
# Diagnostic script — run this BEFORE model_import.py to verify that
# all .stp components from CATIA are geometrically sound and will form
# one connected shell structure after fragment.

# Checks performed
# ────────────────
# 1. Import every .stp file and report bounding boxes.
# 2. Remove duplicate entities (safe, no topology change for clean CATIA export).
# 3. Fragment all surfaces so shared boundaries are made explicit.
# 4. Verify shared edges exist between components (correct use of getAdjacencies).
# 5. DFS connectivity — confirm the wing forms exactly ONE connected shell.
# 6. Report which component pairs are NOT connected (useful for debugging).
# """

# import gmsh
# import sys
# from collections import defaultdict

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIGURATION
# # ─────────────────────────────────────────────────────────────────────────────
# FILES = [
#     ("skin",      "skin.stp"),
#     ("rib0",      "rib0.stp"),
#     ("rib750",    "rib750.stp"),
#     ("rib1500",   "rib1500.stp"),
#     ("rib2250",   "rib2250.stp"),
#     ("rib3000",   "rib3000.stp"),
#     ("mainspar",  "spar_main.stp"),
#     ("tailspar",  "spar_tail.stp"),
# ]

# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 1 — Import
# # ─────────────────────────────────────────────────────────────────────────────
# gmsh.initialize()
# gmsh.model.add("wing_check")

# print("\n" + "="*60)
# print("STEP 1 — Import .stp files")
# print("="*60)

# name_to_tags = {}   # component name → set of surface tags (before fragment)

# for name, filepath in FILES:
#     tags_before = {tag for (_, tag) in gmsh.model.getEntities(2)}
#     try:
#         gmsh.model.occ.importShapes(filepath)
#         gmsh.model.occ.synchronize()
#     except Exception as e:
#         print(f"  ✗  {filepath} → {e}")
#         gmsh.finalize()
#         sys.exit(1)
#     tags_after   = {tag for (_, tag) in gmsh.model.getEntities(2)}
#     new_tags     = tags_after - tags_before
#     name_to_tags[name] = new_tags

#     # Print bounding box for each imported surface
#     for t in sorted(new_tags):
#         xmn, ymn, zmn, xmx, ymx, zmx = gmsh.model.getBoundingBox(2, t)
#         print(f"  ✓  {name:10s}  surf={t:3d}  "
#               f"x=[{xmn:6.1f},{xmx:6.1f}]  "
#               f"y=[{ymn:6.1f},{ymx:6.1f}]  "
#               f"z=[{zmn:6.1f},{zmx:6.1f}]")

# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 2 — Remove duplicates (safe for CATIA export, no healShapes)
# #
# # NOTE: healShapes() is intentionally NOT called here.
# #       For clean CATIA .stp exports it is unnecessary and can silently
# #       merge or split surfaces, corrupting subsequent tag tracking.
# #       removeAllDuplicates() is sufficient and safe.
# # ─────────────────────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("STEP 2 — Remove duplicate entities")
# print("="*60)

# surfs_before_dedup = len(gmsh.model.getEntities(2))
# edges_before_dedup = len(gmsh.model.getEntities(1))

# gmsh.model.occ.removeAllDuplicates()
# gmsh.model.occ.synchronize()

# surfs_after_dedup = len(gmsh.model.getEntities(2))
# edges_after_dedup = len(gmsh.model.getEntities(1))

# print(f"  Surfaces : {surfs_before_dedup} → {surfs_after_dedup}  "
#       f"(removed {surfs_before_dedup - surfs_after_dedup})")
# print(f"  Edges    : {edges_before_dedup} → {edges_after_dedup}  "
#       f"(removed {edges_before_dedup - edges_after_dedup})")

# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 3 — Fragment
# #
# # fragment() makes the topology explicit: wherever two surfaces share a
# # boundary geometrically, a single shared edge is created so that the
# # resulting mesh will be conforming (shared nodes, no duplicates).
# # ─────────────────────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("STEP 3 — Fragment all surfaces")
# print("="*60)

# surfs_pre_frag = gmsh.model.getEntities(2)
# edges_pre_frag = gmsh.model.getEntities(1)
# n_surfs_pre    = len(surfs_pre_frag)
# n_edges_pre    = len(edges_pre_frag)

# result, mapping = gmsh.model.occ.fragment(surfs_pre_frag, [])
# gmsh.model.occ.synchronize()

# surfs_post_frag = gmsh.model.getEntities(2)
# edges_post_frag = gmsh.model.getEntities(1)
# n_surfs_post    = len(surfs_post_frag)
# n_edges_post    = len(edges_post_frag)

# print(f"  Surfaces : {n_surfs_pre} → {n_surfs_post}")
# print(f"  Edges    : {n_edges_pre} → {n_edges_post}")

# # Update name_to_tags through the fragment mapping
# old_to_new = {}
# for i, (dim, old_tag) in enumerate(surfs_pre_frag):
#     old_to_new[old_tag] = [t for (d, t) in mapping[i] if d == 2]

# for name in name_to_tags:
#     new = set()
#     for old_tag in name_to_tags[name]:
#         new.update(old_to_new.get(old_tag, [old_tag]))
#     name_to_tags[name] = new

# print()
# for name, tags in name_to_tags.items():
#     print(f"  {name:10s} → {sorted(tags)}")

# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 4 — Build edge → surface adjacency map
# #
# # FIX: gmsh.model.getAdjacencies(dim=1, tag) returns (upward, downward).
# #      For an edge (dim=1):
# #        upward   = adjacent surfaces (dim 2)  ← what we want
# #        downward = adjacent vertices (dim 0)  ← NOT what we want
# #      Previous code used `down` which gave vertex tags instead of surfaces.
# # ─────────────────────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("STEP 4 — Shared edge analysis")
# print("="*60)

# edge_to_surfaces = defaultdict(list)

# for (dim, edge_tag) in gmsh.model.getEntities(1):
#     upward, downward = gmsh.model.getAdjacencies(dim, edge_tag)
#     # upward = adjacent surfaces (dim 2) — correct direction
#     for surf_tag in upward:
#         edge_to_surfaces[edge_tag].append(surf_tag)

# shared_edges = [
#     (edge, surfs)
#     for edge, surfs in edge_to_surfaces.items()
#     if len(surfs) >= 2
# ]

# total_edges = len(gmsh.model.getEntities(1))
# print(f"  Total edges       : {total_edges}")
# print(f"  Shared edges (≥2) : {len(shared_edges)}")

# if not shared_edges:
#     print("\n  ❌ No shared edges — surfaces are NOT topologically connected.")
#     print("     Likely cause: surfaces do not touch within OCC tolerance.")
#     print("     Check that .stp files share exact boundary curves from CATIA.")
#     gmsh.finalize()
#     sys.exit(1)
# else:
#     print("\n  ✓  Shared edges found — at least some surfaces are connected.")
#     print("\n  Sample shared edges (first 10):")
#     for e, s in shared_edges[:10]:
#         print(f"    Edge {e:4d}  ↔  surfaces {s}")

# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 5 — DFS connected-components check
# # ─────────────────────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("STEP 5 — Surface connectivity (DFS)")
# print("="*60)

# # Build undirected surface adjacency graph from shared edges
# surface_graph = defaultdict(set)
# for edge, surfs in shared_edges:
#     for i in range(len(surfs)):
#         for j in range(i + 1, len(surfs)):
#             surface_graph[surfs[i]].add(surfs[j])
#             surface_graph[surfs[j]].add(surfs[i])

# all_surf_tags = [tag for (_, tag) in gmsh.model.getEntities(2)]
# visited       = set()
# components    = []

# for s in all_surf_tags:
#     if s not in visited:
#         component = []
#         stack = [s]
#         while stack:
#             cur = stack.pop()
#             if cur not in visited:
#                 visited.add(cur)
#                 component.append(cur)
#                 stack.extend(surface_graph[cur] - visited)
#         components.append(set(component))

# print(f"\n  Surface connected components : {len(components)}")

# if len(components) == 1:
#     print("  ✓  Geometry forms ONE connected shell — ready for meshing.\n")
# else:
#     print(f"  ❌  Geometry is split into {len(components)} disconnected parts.\n")

#     # Identify which named components are isolated
#     print("  Component breakdown:")
#     for i, comp in enumerate(components):
#         members = []
#         for name, tags in name_to_tags.items():
#             if tags & comp:
#                 members.append(name)
#         print(f"    Component {i+1} ({len(comp)} surfaces): {', '.join(members)}")

#     print("\n  Suggestion: verify in CATIA that the listed components share")
#     print("  exact boundary curves with the skin or neighbouring ribs/spars.")

# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 6 — Inter-component shared edge count (connectivity matrix)
# # ─────────────────────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("STEP 6 — Inter-component connectivity matrix")
# print("="*60)

# names = [n for (n, _) in FILES]
# conn  = defaultdict(int)   # (nameA, nameB) → shared edge count

# for edge, surfs in shared_edges:
#     comp_of = {}
#     for name, tags in name_to_tags.items():
#         for s in surfs:
#             if s in tags:
#                 comp_of[s] = name

#     surf_names = list(comp_of.values())
#     for i in range(len(surf_names)):
#         for j in range(i + 1, len(surf_names)):
#             a, b = sorted([surf_names[i], surf_names[j]])
#             if a != b:
#                 conn[(a, b)] += 1

# if conn:
#     print(f"\n  {'Component A':<12}  {'Component B':<12}  Shared edges")
#     print(f"  {'-'*12}  {'-'*12}  {'-'*12}")
#     for (a, b), count in sorted(conn.items()):
#         print(f"  {a:<12}  {b:<12}  {count}")
# else:
#     print("  ⚠  No inter-component connections detected.")

# print()
# gmsh.finalize()

"""
check_geometry.py
─────────────────
Diagnostic script — run this BEFORE model_import.py to verify that
all .stp components from CATIA are geometrically sound and will form
one conforming, connected shell structure after fragment().

Topology assumption (CATIA origin)
────────────────────────────────────
  • skin.stp     → upper + lower outer surface (lofted from airfoil curves)
  • rib*.stp     → filled planar cross-sections; their perimeter lies exactly
                   on the skin surface (same CATIA edge entity)
  • spar_*.stp   → spanwise vertical surfaces; long edges shared with skin

Because CATIA already exports geometrically coincident boundaries as the
SAME topological entity, fragment() is sufficient to create shared edges.
removeAllDuplicates() is NOT called because it renumbers surface tags
without returning a mapping, silently emptying tag_map for renamed surfaces,
which causes them to disappear from the mesh.

Checks performed
────────────────
  1. Import every .stp and print bounding boxes.
  2. Set OCC tolerance so fragment finds near-coincident CATIA boundaries.
  3. fragment() — make shared edges explicit.
  4. Build edge→surface adjacency using getAdjacencies UPWARD direction.
  5. DFS connectivity — confirm one connected shell.
  6. Inter-component connectivity matrix — shows which pairs share edges.
"""

import gmsh
import sys
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
FILES = [
    ("skin",      "skin.stp"),
    ("rib0",      "rib0.stp"),
    ("rib750",    "rib750.stp"),
    ("rib1500",   "rib1500.stp"),
    ("rib2250",   "rib2250.stp"),
    ("rib3000",   "rib3000.stp"),
    ("mainspar",  "spar_main.stp"),
    ("tailspar",  "spar_tail.stp"),
]

# OCC tolerance in mm — should be smaller than the tightest gap between
# components but larger than CATIA export floating-point noise (~1e-6 mm).
OCC_TOL = 1e-2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Import
# ─────────────────────────────────────────────────────────────────────────────
gmsh.initialize()
gmsh.model.add("wing_check")

# Set OCC tolerance BEFORE any import so all subsequent operations use it
gmsh.option.setNumber("Geometry.Tolerance",        OCC_TOL)
gmsh.option.setNumber("Geometry.ToleranceBoolean", OCC_TOL)

print("\n" + "="*60)
print("STEP 1 — Import .stp files")
print("="*60)

name_to_tags = {}   # component name → set of surface tags (before fragment)

for name, filepath in FILES:
    tags_before = {tag for (_, tag) in gmsh.model.getEntities(2)}
    try:
        gmsh.model.occ.importShapes(filepath)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"  x  {filepath} -> {e}")
        gmsh.finalize()
        sys.exit(1)

    tags_after = {tag for (_, tag) in gmsh.model.getEntities(2)}
    new_tags   = tags_after - tags_before
    name_to_tags[name] = new_tags

    for t in sorted(new_tags):
        xmn, ymn, zmn, xmx, ymx, zmx = gmsh.model.getBoundingBox(2, t)
        print(f"  ok {name:10s}  surf={t:3d}  "
              f"x=[{xmn:7.1f},{xmx:7.1f}]  "
              f"y=[{ymn:7.1f},{ymx:7.1f}]  "
              f"z=[{zmn:7.1f},{zmx:7.1f}]")

total_surfs = len(gmsh.model.getEntities(2))
total_edges = len(gmsh.model.getEntities(1))
print(f"\n  Total surfaces after import : {total_surfs}")
print(f"  Total edges   after import : {total_edges}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Fragment
#
# fragment() detects geometrically coincident boundaries and creates
# explicit shared topological edges, giving a conforming mesh without
# Nitsche coupling.
#
# WHY NO removeAllDuplicates() HERE:
#   removeAllDuplicates() renumbers surface tags without returning a mapping.
#   Any tag that gets renumbered disappears from name_to_tags via set
#   intersection, causing those surfaces to get no Physical Group and
#   therefore not appear in the exported mesh.
#   fragment() handles coincident entities internally and DOES return a
#   mapping, so tag tracking remains correct.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Fragment (enforce conforming topology)")
print("="*60)

surfs_pre_frag = gmsh.model.getEntities(2)
edges_pre_frag = gmsh.model.getEntities(1)

result, mapping = gmsh.model.occ.fragment(surfs_pre_frag, [])
gmsh.model.occ.synchronize()

surfs_post_frag = gmsh.model.getEntities(2)
edges_post_frag = gmsh.model.getEntities(1)

print(f"  Surfaces : {len(surfs_pre_frag)} -> {len(surfs_post_frag)}")
print(f"  Edges    : {len(edges_pre_frag)} -> {len(edges_post_frag)}"
      f"  (delta = {len(edges_post_frag) - len(edges_pre_frag):+d})")

# Update name_to_tags using the fragment mapping (the only safe way)
old_to_new = {}
for i, (dim, old_tag) in enumerate(surfs_pre_frag):
    old_to_new[old_tag] = [t for (d, t) in mapping[i] if d == 2]

for name in name_to_tags:
    updated = set()
    for old_tag in name_to_tags[name]:
        updated.update(old_to_new.get(old_tag, [old_tag]))
    name_to_tags[name] = updated

print()
for name, tags in name_to_tags.items():
    n = len(tags)
    print(f"  {name:10s} -> {n} surface(s)  {sorted(tags)}")

# Warn if any component lost all surfaces through fragment
lost = [n for n, tags in name_to_tags.items() if not tags]
if lost:
    print(f"\n  WARNING: these components have no surfaces after fragment: {lost}")
    print("  This usually means their original surface tags were not in")
    print("  surfs_pre_frag — check import order and synchronize() calls.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Shared edge analysis
#
# CORRECT use of getAdjacencies:
#   gmsh.model.getAdjacencies(dim=1, tag) -> (upward, downward)
#   upward   = entities of dim+1 = surfaces (dim 2)  <- what we want
#   downward = entities of dim-1 = vertices (dim 0)  <- NOT what we want
#
# PREVIOUS BUG: code used `down` (vertices) instead of `up` (surfaces),
# so edge_to_surfaces was mapping edge tags to vertex tags — the shared
# edge count and DFS were entirely wrong.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Shared edge analysis")
print("="*60)

edge_to_surfaces = defaultdict(list)
for (dim, edge_tag) in gmsh.model.getEntities(1):
    upward, _downward = gmsh.model.getAdjacencies(dim, edge_tag)
    # upward = adjacent surfaces (dim 2) — correct direction
    for surf_tag in upward:
        edge_to_surfaces[edge_tag].append(surf_tag)

shared_edges = [(e, s) for e, s in edge_to_surfaces.items() if len(s) >= 2]
total_edges_now = len(gmsh.model.getEntities(1))

print(f"  Total edges          : {total_edges_now}")
print(f"  Shared edges (>= 2)  : {len(shared_edges)}")

if not shared_edges:
    print("\n  NO shared edges found.")
    print("  Possible causes:")
    print("  - CATIA rib/spar boundaries do not coincide with skin edges.")
    print("  - OCC_TOL too small — try increasing OCC_TOL in this script.")
    print("  - .stp files use different units (check bounding boxes above).")
    gmsh.finalize()
    sys.exit(1)
else:
    print("\n  Shared edges detected.")
    print("\n  Sample (first 10):")
    for e, s in shared_edges[:10]:
        print(f"    Edge {e:4d}  <->  surfaces {s}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — DFS connectivity
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — DFS connectivity")
print("="*60)

surface_graph = defaultdict(set)
for edge, surfs in shared_edges:
    for i in range(len(surfs)):
        for j in range(i + 1, len(surfs)):
            surface_graph[surfs[i]].add(surfs[j])
            surface_graph[surfs[j]].add(surfs[i])

all_surf_tags = [tag for (_, tag) in gmsh.model.getEntities(2)]
visited       = set()
components    = []

for s in all_surf_tags:
    if s not in visited:
        component = []
        stack = [s]
        while stack:
            cur = stack.pop()
            if cur not in visited:
                visited.add(cur)
                component.append(cur)
                stack.extend(surface_graph[cur] - visited)
        components.append(set(component))

print(f"\n  Connected components : {len(components)}")

if len(components) == 1:
    print("  Geometry forms ONE connected shell — ready for meshing.\n")
else:
    print(f"\n  {len(components)} disconnected parts:\n")
    for i, comp in enumerate(components):
        members = [n for n, tags in name_to_tags.items() if tags & comp]
        print(f"    Component {i+1}  ({len(comp)} surfaces):  {', '.join(members)}")
    print("\n  -> Check that the listed components share exact boundary")
    print("     curves with the skin or adjacent ribs/spars in CATIA.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Inter-component connectivity matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 — Inter-component connectivity matrix")
print("="*60)

surf_to_name = {}
for name, tags in name_to_tags.items():
    for t in tags:
        surf_to_name[t] = name

conn = defaultdict(int)
for edge, surfs in shared_edges:
    names_on_edge = list({surf_to_name.get(s, "?") for s in surfs})
    for i in range(len(names_on_edge)):
        for j in range(i + 1, len(names_on_edge)):
            a, b = sorted([names_on_edge[i], names_on_edge[j]])
            if a != b:
                conn[(a, b)] += 1

if conn:
    col = 14
    print(f"\n  {'Component A':<{col}}  {'Component B':<{col}}  Shared edges")
    print(f"  {'-'*col}  {'-'*col}  {'-'*12}")
    for (a, b), count in sorted(conn.items()):
        print(f"  {a:<{col}}  {b:<{col}}  {count}")
else:
    print("  No inter-component connections found.")

print()
gmsh.finalize()
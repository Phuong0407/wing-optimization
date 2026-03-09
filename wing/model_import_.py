# import gmsh
# import sys

# # ─────────────────────────────────────────────
# # Thứ tự import QUAN TRỌNG — không đổi
# # ─────────────────────────────────────────────
# FILES = [
#     ("skin",     "skin.stp"),
#     ("rib0",     "rib0.stp"),
#     ("rib750",   "rib750.stp"),
#     ("rib1500",  "rib1500.stp"),
#     ("rib2250",  "rib2250.stp"),
#     ("rib3000",  "rib3000.stp"),
#     ("mainspar", "spar_main.stp"),
#     ("tailspar", "spar_tail.stp"),
# ]

# # Material tags — dùng trong FEniCSx với cell_tags.find(tag)
# TAGS = {
#     "skin"     : 1,
#     "rib0"     : 2,
#     "rib750"   : 3,
#     "rib1500"  : 4,
#     "rib2250"  : 5,
#     "rib3000"  : 6,
#     "mainspar" : 7,
#     "tailspar" : 8,
# }

# # BC tags (curves)
# TAG_ROOT = 11
# TAG_TIP  = 12

# MESH_SIZE  = 10.0
# OUTPUT_MSH = "wing_connected.msh"
# TOL        = 5.0

# # ─────────────────────────────────────────────
# gmsh.initialize()
# gmsh.model.add("wing")

# # ─────────────────────────────────────────────
# # BƯỚC 1: Import — track tags trước fragment
# # ─────────────────────────────────────────────
# print("\n=== BƯỚC 1: Import ===")
# tag_map = {}   # name → set of surface tags (trước fragment)

# for name, filepath in FILES:
#     before = {tag for (_, tag) in gmsh.model.getEntities(2)}
#     try:
#         gmsh.model.occ.importShapes(filepath)
#         gmsh.model.occ.synchronize()
#     except Exception as e:
#         print(f"  ✗  {filepath} → {e}")
#         gmsh.finalize()
#         sys.exit(1)
#     after    = {tag for (_, tag) in gmsh.model.getEntities(2)}
#     new_tags = after - before
#     tag_map[name] = new_tags
#     print(f"  ✓  {name:10s} ({filepath}) → tags {sorted(new_tags)}")

# # ─────────────────────────────────────────────
# # BƯỚC 2: Fragment + cập nhật tag_map
# # ─────────────────────────────────────────────
# print("\n=== BƯỚC 2: Fragment ===")

# all_surfs_before = gmsh.model.getEntities(dim=2)
# edges_before     = gmsh.model.getEntities(dim=1)

# result, mapping = gmsh.model.occ.fragment(all_surfs_before, [])
# gmsh.model.occ.synchronize()

# # mapping[i] → list of new (dim,tag) cho all_surfs_before[i]
# old_to_new = {}
# for i, (dim, old_tag) in enumerate(all_surfs_before):
#     old_to_new[old_tag] = [t for (d, t) in mapping[i] if d == 2]

# # Cập nhật tag_map
# for name in tag_map:
#     new = set()
#     for old_tag in tag_map[name]:
#         new.update(old_to_new.get(old_tag, [old_tag]))
#     tag_map[name] = new

# edges_after = gmsh.model.getEntities(dim=1)
# surfs_after = gmsh.model.getEntities(dim=2)
# delta = len(edges_after) - len(edges_before)

# print(f"  Surfaces : {len(all_surfs_before)} → {len(surfs_after)}")
# print(f"  Edges    : {len(edges_before)} → {len(edges_after)}")
# print(f"  {'✓' if delta > 0 else '⚠'}  {delta} shared edges mới")

# for name, tags in tag_map.items():
#     print(f"  {name:10s} → {sorted(tags)}")

# # ─────────────────────────────────────────────
# # BƯỚC 3: Physical Groups — surfaces (vật liệu)
# # ─────────────────────────────────────────────
# print("\n=== BƯỚC 3: Physical Groups (surfaces) ===")

# for name, mat_tag in TAGS.items():
#     tags = sorted(tag_map[name])
#     if tags:
#         gmsh.model.addPhysicalGroup(2, tags, tag=mat_tag, name=name)
#         print(f"  ✓  {name:10s} tag={mat_tag} → surfaces {tags}")
#     else:
#         print(f"  ⚠  {name:10s} → không tìm thấy surface nào")

# # ─────────────────────────────────────────────
# # BƯỚC 4: Physical Groups — curves (BC)
# # ─────────────────────────────────────────────
# print("\n=== BƯỚC 4: Physical Groups (curves BC) ===")

# def curves_at_y(y_target, tol=TOL):
#     found = []
#     for (dim, tag) in gmsh.model.getEntities(1):
#         xmn,ymn,zmn,xmx,ymx,zmx = gmsh.model.getBoundingBox(1, tag)
#         if abs(ymn - y_target) < tol and abs(ymx - y_target) < tol:
#             found.append(tag)
#     return found

# root_curves = curves_at_y(0)
# tip_curves  = curves_at_y(3000)

# if root_curves:
#     gmsh.model.addPhysicalGroup(1, root_curves, tag=TAG_ROOT, name="root")
#     print(f"  ✓  root (y=0)    tag={TAG_ROOT} → {len(root_curves)} curves")
# else:
#     print(f"  ⚠  root → không tìm thấy")

# if tip_curves:
#     gmsh.model.addPhysicalGroup(1, tip_curves, tag=TAG_TIP, name="tip")
#     print(f"  ✓  tip  (y=3000) tag={TAG_TIP} → {len(tip_curves)} curves")
# else:
#     print(f"  ⚠  tip  → không tìm thấy")

# # ─────────────────────────────────────────────
# # BƯỚC 5: Mesh + Export
# # ─────────────────────────────────────────────
# print("\n=== BƯỚC 5: Mesh ===")

# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE / 4)
# gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
# gmsh.option.setNumber("Mesh.MinimumCirclePoints", 20)
# gmsh.option.setNumber("Mesh.Algorithm", 6)

# gmsh.model.mesh.generate(2)

# nodes = len(gmsh.model.mesh.getNodes()[0])
# print(f"  ✓  {nodes} nodes")

# gmsh.write(OUTPUT_MSH)
# print(f"  ✓  Saved: {OUTPUT_MSH}")

# # ─────────────────────────────────────────────
# # TỔNG KẾT
# # ─────────────────────────────────────────────
# print("\n=== TỔNG KẾT ===")
# print(f"  Shared edges mới : {delta}")
# print(f"  Nodes            : {nodes}")
# print(f"  Output           : {OUTPUT_MSH}")
# print("""
#   Dùng trong FEniCSx:

#     from dolfinx.io import gmshio
#     domain, cell_tags, facet_tags = gmshio.read_from_msh(
#         "wing_connected.msh", MPI.COMM_WORLD, gdim=3
#     )

#     # Assign vật liệu theo tag
#     V0      = fem.functionspace(domain, ("DG", 0))
#     t_func  = fem.Function(V0)
#     E_func  = fem.Function(V0)

#     mat = {
#     #   tag  t(m)    E(Pa)
#         1: (2e-3,  70e9),   # skin
#         2: (3e-3,  70e9),   # rib0
#         3: (3e-3,  70e9),   # rib750
#         4: (3e-3,  70e9),   # rib1500
#         5: (3e-3,  70e9),   # rib2250
#         6: (3e-3,  70e9),   # rib3000
#         7: (4e-3,  70e9),   # mainspar
#         8: (3e-3,  70e9),   # tailspar
#     }
#     for tag, (t_val, E_val) in mat.items():
#         cells = cell_tags.find(tag)
#         t_func.x.array[cells] = t_val
#         E_func.x.array[cells] = E_val

#     # BC ngàm tại root
#     root_dofs = fem.locate_dofs_topological(
#         V, facet_tags.dim, facet_tags.find(11)
#     )
# """)

# gmsh.finalize()
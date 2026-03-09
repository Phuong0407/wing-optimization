import gmsh
import sys
from collections import defaultdict

FILES = [
    ("upper",    "skin.stp"),
    ("rib0",     "rib0.stp"),
    ("rib750",   "rib750.stp"),
    ("rib1500",  "rib1500.stp"),
    ("rib2250",  "rib2250.stp"),
    ("rib3000",  "rib3000.stp"),
    ("mainspar", "spar_main.stp"),
    ("tailspar", "spar_tail.stp"),
]

SURF_TAGS = {
    "upper"   : 14,
    "lower"   : 15,
    "rib0"    : 38,
    "rib750"  : 39,
    "rib1500" : 40,
    "rib2250" : 41,
    "rib3000" : 42,
    "mainspar": 43,
    "tailspar": 44,
}

RIB_CURVE_TAGS = {
    "rib0"   : 45,
    "rib750" : 46,
    "rib1500": 47,
    "rib2250": 48,
    "rib3000": 49,
}

SPAR_CURVE_TAGS = {
    "mainspar": 50,
    "tailspar": 51,
}

TAG_ROOT   = 52
MESH_SIZE  = 20.0
MESH_ALGO  = 6
OUTPUT_MSH = "./InputData/wing_new.msh"
OCC_TOL    = 1e-2
BC_TOL     = 1.0


print("\n" + "="*60)
print("STEP 1 — Import .stp files")
print("="*60)

gmsh.initialize()
gmsh.model.add("wing")

gmsh.option.setNumber("Geometry.Tolerance",        OCC_TOL)
gmsh.option.setNumber("Geometry.ToleranceBoolean", OCC_TOL)

tag_map = {}

for name, filepath in FILES:
    tags_before = {t for (_, t) in gmsh.model.getEntities(2)}
    try:
        gmsh.model.occ.importShapes(filepath)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"  FAIL  {filepath} -> {e}")
        gmsh.finalize()
        sys.exit(1)
    tags_after = {t for (_, t) in gmsh.model.getEntities(2)}
    new_tags   = tags_after - tags_before
    tag_map[name] = new_tags
    print(f"  ok  {name:10s}  ({filepath})  ->  geo tags {sorted(new_tags)}")

tag_map["lower"] = set()


print("\n" + "="*60)
print("STEP 2 — Fragment")
print("="*60)

all_surfs_before = gmsh.model.getEntities(dim=2)
n_edges_before   = len(gmsh.model.getEntities(dim=1))

result, mapping = gmsh.model.occ.fragment(all_surfs_before, [])
gmsh.model.occ.synchronize()

n_surfs_after = len(gmsh.model.getEntities(dim=2))
n_edges_after = len(gmsh.model.getEntities(dim=1))

print(f"  Surfaces : {len(all_surfs_before)} -> {n_surfs_after}")
print(f"  Edges    : {n_edges_before} -> {n_edges_after}"
      f"  (delta = {n_edges_after - n_edges_before:+d})")

old_to_new = {}
for i, (dim, old_tag) in enumerate(all_surfs_before):
    old_to_new[old_tag] = [t for (d, t) in mapping[i] if d == 2]

for name in tag_map:
    updated = set()
    for old_tag in tag_map[name]:
        updated.update(old_to_new.get(old_tag, [old_tag]))
    tag_map[name] = updated

print()
for name, tags in tag_map.items():
    print(f"  {name:10s} -> {sorted(tags)}")

missing = [n for n, tags in tag_map.items() if not tags and n != "lower"]
if missing:
    print(f"\n  ERROR: empty tag_map after fragment for: {missing}")
    gmsh.finalize()
    sys.exit(1)


print("\n" + "="*60)
print("STEP 3 — Remove stale physical groups")
print("="*60)

before_surf  = gmsh.model.getPhysicalGroups(2)
before_curve = gmsh.model.getPhysicalGroups(1)

gmsh.model.removePhysicalGroups([])

after_surf  = gmsh.model.getPhysicalGroups(2)
after_curve = gmsh.model.getPhysicalGroups(1)

print(f"  Surface groups : {len(before_surf)} -> {len(after_surf)}")
print(f"  Curve  groups  : {len(before_curve)} -> {len(after_curve)}")
print(f"  ok  All stale physical groups removed")


print("\n" + "="*60)
print("STEP 4 — Connectivity check")
print("="*60)

edge_to_surfaces = defaultdict(list)
for (dim, edge_tag) in gmsh.model.getEntities(1):
    upward, _ = gmsh.model.getAdjacencies(dim, edge_tag)
    for surf_tag in upward:
        edge_to_surfaces[edge_tag].append(surf_tag)

shared_edges = [(e, s) for e, s in edge_to_surfaces.items() if len(s) >= 2]
print(f"  Shared edges (>= 2 surfaces) : {len(shared_edges)}")

if not shared_edges:
    print("  ERROR: no shared edges — fragment did not connect components.")
    print("  Increase OCC_TOL or check that .stp boundaries coincide.")
    gmsh.finalize()
    sys.exit(1)

surface_graph = defaultdict(set)
for edge, surfs in shared_edges:
    for i in range(len(surfs)):
        for j in range(i + 1, len(surfs)):
            surface_graph[surfs[i]].add(surfs[j])
            surface_graph[surfs[j]].add(surfs[i])

all_surf_tags = [t for (_, t) in gmsh.model.getEntities(2)]
visited, n_components = set(), 0
for s in all_surf_tags:
    if s not in visited:
        n_components += 1
        stack = [s]
        while stack:
            cur = stack.pop()
            if cur not in visited:
                visited.add(cur)
                stack.extend(surface_graph[cur] - visited)

print(f"  Connected components         : {n_components}")
if n_components != 1:
    print("  ERROR: geometry is not one connected shell.")
    gmsh.finalize()
    sys.exit(1)
print("  ok  ONE connected shell")


print("\n" + "="*60)
print("STEP 5 — Split upper / lower skin")
print("="*60)

upper_tags = set()
lower_tags = set()

for t in tag_map["upper"]:
    xmn, ymn, zmn, xmx, ymx, zmx = gmsh.model.getBoundingBox(2, t)
    center = [(xmn + xmx) / 2, (ymn + ymx) / 2, (zmn + zmx) / 2]
    _, uv = gmsh.model.getClosestPoint(2, t, center)
    normal = gmsh.model.getNormal(t, uv)
    if normal[2] >= 0:
        upper_tags.add(t)
    else:
        lower_tags.add(t)

tag_map["upper"] = upper_tags
tag_map["lower"] = lower_tags

print(f"  upper : {len(upper_tags)} surface(s)  {sorted(upper_tags)}")
print(f"  lower : {len(lower_tags)} surface(s)  {sorted(lower_tags)}")

if not upper_tags or not lower_tags:
    print("  WARN: could not separate upper/lower skin.")
    print("  Check that skin spans both positive and negative z.")


print("\n" + "="*60)
print("STEP 6 — Physical Groups (surfaces)")
print("="*60)

for name, phys_tag in SURF_TAGS.items():
    geo_tags = sorted(tag_map.get(name, []))
    if geo_tags:
        gmsh.model.addPhysicalGroup(2, geo_tags, tag=phys_tag, name=name)
        print(f"  ok  {name:10s}  phys={phys_tag}  ->  {len(geo_tags)} surface(s)  {geo_tags}")
    else:
        print(f"  WARN  {name:10s}  phys={phys_tag}  ->  no surfaces found")


print("\n" + "="*60)
print("STEP 7 — Physical Groups (curves)")
print("="*60)


def all_curves_at_y(y_target, tol=BC_TOL):
    found = []
    for (dim, tag) in gmsh.model.getEntities(1):
        _, ymn, _, _, ymx, _ = gmsh.model.getBoundingBox(1, tag)
        if abs(ymn - y_target) < tol and abs(ymx - y_target) < tol:
            found.append(tag)
    return found


def curves_of_component_at_y(component_names, y_target, tol=BC_TOL):
    comp_surfs = set()
    for name in component_names:
        comp_surfs.update(tag_map.get(name, set()))

    found = []
    for (dim, tag) in gmsh.model.getEntities(1):
        _, ymn, _, _, ymx, _ = gmsh.model.getBoundingBox(1, tag)
        if abs(ymn - y_target) < tol and abs(ymx - y_target) < tol:
            upward, _ = gmsh.model.getAdjacencies(dim, tag)
            if set(upward) & comp_surfs:
                found.append(tag)
    return found


root_curves = all_curves_at_y(0.0)

print(f"  root curves found : {len(root_curves)}")

if root_curves:
    gmsh.model.addPhysicalGroup(1, root_curves, tag=TAG_ROOT, name="root")
    print(f"  ok  root       phys={TAG_ROOT}  ->  {len(root_curves)} curves")
else:
    print(f"  WARN  root not found at y=0  (BC_TOL={BC_TOL} mm)")

RIB_Y = {
    "rib0"   : (0.0,    45),
    "rib750" : (750.0,  46),
    "rib1500": (1500.0, 47),
    "rib2250": (2250.0, 48),
    "rib3000": (3000.0, 49),
}
for rib_name, (y_pos, curve_tag) in RIB_Y.items():
    rib_curves = curves_of_component_at_y(
        [rib_name, "upper", "lower", "mainspar", "tailspar"], y_pos
    )
    if rib_curves:
        gmsh.model.addPhysicalGroup(1, rib_curves, tag=curve_tag,
                                    name=f"{rib_name}c")
        print(f"  ok  {rib_name:10s}c  phys={curve_tag}  "
              f"->  {len(rib_curves)} curves")
    else:
        print(f"  WARN  {rib_name}c not found at y={y_pos}")

for spar_name, curve_tag in SPAR_CURVE_TAGS.items():
    spar_surfs = tag_map.get(spar_name, set())
    spar_curves = []
    for (dim, tag) in gmsh.model.getEntities(1):
        upward, _ = gmsh.model.getAdjacencies(dim, tag)
        if set(upward) & spar_surfs:
            spar_curves.append(tag)
    if spar_curves:
        gmsh.model.addPhysicalGroup(1, spar_curves, tag=curve_tag,
                                    name=f"{spar_name}c")
        print(f"  ok  {spar_name:10s}c  phys={curve_tag}  "
              f"->  {len(spar_curves)} curves")
    else:
        print(f"  WARN  {spar_name}c not found")


print("\n" + "="*60)
print("STEP 8 — Physical group summary")
print("="*60)

all_curve_groups = gmsh.model.getPhysicalGroups(1)
all_surf_groups  = gmsh.model.getPhysicalGroups(2)
print(f"  Surface groups : {[t for (_, t) in all_surf_groups]}")
print(f"  Curve  groups  : {[t for (_, t) in all_curve_groups]}")

if TAG_ROOT not in [t for (_, t) in all_curve_groups]:
    print(f"\n  ERROR: TAG_ROOT={TAG_ROOT} was not registered — root BC will be empty!")
    gmsh.finalize()
    sys.exit(1)
print(f"  ok  TAG_ROOT={TAG_ROOT} present")


print("\n" + "="*60)
print("STEP 9 — Mesh")
print("="*60)

gmsh.option.setNumber("Mesh.CharacteristicLengthMax",           MESH_SIZE)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin",           MESH_SIZE / 4)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
gmsh.option.setNumber("Mesh.MinimumCirclePoints",               20)
gmsh.option.setNumber("Mesh.Algorithm",                         MESH_ALGO)

gmsh.model.mesh.generate(2)

n_nodes = len(gmsh.model.mesh.getNodes()[0])
print(f"  Nodes : {n_nodes}")

gmsh.write(OUTPUT_MSH)
print(f"  Saved -> {OUTPUT_MSH}")


print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Output       : {OUTPUT_MSH}")
print(f"  Nodes        : {n_nodes}")
print(f"  Shared edges : {len(shared_edges)}")
print(f"  Components   : {n_components}")
print("""
  Physical group tags for wing.py:

    Surfaces (cell_tags):
      14 -> upper skin    39 -> rib750     43 -> mainspar
      15 -> lower skin    40 -> rib1500    44 -> tailspar
      38 -> rib0          41 -> rib2250
                          42 -> rib3000

    Curves (facet_tags):
      45 -> ribrootc    46 -> rib750c    47 -> rib1500c
      48 -> rib2250c    49 -> ribtailc
      50 -> mainsparc   51 -> tailsparc
      52 -> root BC (clamped)

  Next steps:
      python wing.py   ->  FEM solver
""")

gmsh.finalize()
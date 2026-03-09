import numpy as np

Y_RIBS      = [0, 750, 1500, 2250, 3000]
SPAR_RATIOS = {"main": 0.25, "tail": 0.75}
WINGFILE    = "wingsolid.stp"

def get_chord_bounds_at_y(y_rib):
    """Lấy xmin, xmax của airfoil tại Y station."""
    gmsh.initialize()
    gmsh.model.add("probe")
    gmsh.model.occ.importShapes(WINGFILE)
    gmsh.model.occ.synchronize()
    wing = gmsh.model.getEntities(dim=3)

    SIZE = 10000.0
    pts  = [
        gmsh.model.occ.addPoint(-SIZE,  y_rib, -SIZE),
        gmsh.model.occ.addPoint( SIZE,  y_rib, -SIZE),
        gmsh.model.occ.addPoint( SIZE,  y_rib,  SIZE),
        gmsh.model.occ.addPoint(-SIZE,  y_rib,  SIZE),
    ]
    loop  = gmsh.model.occ.addCurveLoop([
        gmsh.model.occ.addLine(pts[i], pts[(i+1)%4]) for i in range(4)
    ])
    plane = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    result, _ = gmsh.model.occ.intersect(
        [(2, plane)], wing,
        removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()

    xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(2, result[0][1])
    gmsh.finalize()
    return xmin, xmax

# --- Bước 1: Thu thập chord bounds tại mỗi Y ---
print("Sampling chord bounds...")
chord_data = {}  # y → (xmin, xmax)
for y in Y_RIBS:
    xmin, xmax = get_chord_bounds_at_y(y)
    chord_data[y] = (xmin, xmax)
    print(f"  Y={y:4d}:  xmin={xmin:.1f}  xmax={xmax:.1f}  chord={xmax-xmin:.1f}")

# --- Bước 2: Tạo spar cho mỗi ratio ---
for spar_name, ratio in SPAR_RATIOS.items():
    print(f"\nCreating {spar_name} spar (x={ratio})...")

    gmsh.initialize()
    gmsh.model.add(f"spar_{spar_name}")
    gmsh.model.occ.importShapes(WINGFILE)
    gmsh.model.occ.synchronize()
    wing = gmsh.model.getEntities(dim=3)

    # Tính x_spar tại mỗi Y station
    spar_points = []
    for y in Y_RIBS:
        xmin, xmax = chord_data[y]
        x_spar = xmin + ratio * (xmax - xmin)
        spar_points.append((x_spar, y))
        print(f"  Y={y}: x_spar = {x_spar:.1f}")

    # --- Tạo cutting surface: ruled quad qua tất cả Y stations ---
    SIZE_Z = 10000.0
    wire_pts_top = []
    wire_pts_bot = []

    for (x_s, y_s) in spar_points:
        wire_pts_top.append(gmsh.model.occ.addPoint(x_s, y_s,  SIZE_Z))
        wire_pts_bot.append(gmsh.model.occ.addPoint(x_s, y_s, -SIZE_Z))

    # Đường trên và dưới (spline theo sweep của cánh)
    line_top = gmsh.model.occ.addSpline(wire_pts_top)
    line_bot = gmsh.model.occ.addSpline(wire_pts_bot)

    # Hai cạnh đầu và cuối (thẳng đứng)
    line_start = gmsh.model.occ.addLine(wire_pts_bot[0],  wire_pts_top[0])
    line_end   = gmsh.model.occ.addLine(wire_pts_top[-1], wire_pts_bot[-1])

    loop    = gmsh.model.occ.addCurveLoop([line_bot, line_end, -line_top, -line_start])
    # BSpline surface thay vì plane (handle sweep)
    surface = gmsh.model.occ.addSurfaceFilling(loop)
    gmsh.model.occ.synchronize()

    # Intersect với wing solid
    result, _ = gmsh.model.occ.intersect(
        [(2, surface)], wing,
        removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()

    if not result:
        print(f"  WARNING: No intersection for {spar_name} spar!")
        gmsh.finalize()
        continue

    # Mesh và export
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 20)
    gmsh.model.mesh.generate(2)

    gmsh.write(f"spar_{spar_name}.msh")
    gmsh.write(f"spar_{spar_name}.stp")
    print(f"  Saved: spar_{spar_name}.msh / .stp")
    gmsh.finalize()

print("\nDone.")

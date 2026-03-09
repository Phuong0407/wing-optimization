import gmsh
import sys
import numpy as np
from scipy.interpolate import interp1d





RIBFILE     = "rib0.stp"
OUTFILE_MSH = "rib_frame.msh"
OUTFILE_STP = "rib_frame.stp"

OFFSETX         = 40.0
OFFSETZ         = 20.0
STRUT_W         = 20.0
FILLET_R        = 8.0
MESH_SIZE       = 10.0

# Vi tri cac strut doc (% chord tu LE)
STRUT_POSITIONS = [0.30, 0.45, 0.60, 0.75]

# ============================================================
# HELPER: rectangle voi 4 goc bo trong XZ plane
# ============================================================
def make_rounded_rect(x0, x1, z0, z1, y, r):
    if (x1-x0) < 2*r+1 or (z1-z0) < 2*r+1:
        return None
    p = [
        gmsh.model.occ.addPoint(x0+r, y, z0  ),
        gmsh.model.occ.addPoint(x1-r, y, z0  ),
        gmsh.model.occ.addPoint(x1,   y, z0+r),
        gmsh.model.occ.addPoint(x1,   y, z1-r),
        gmsh.model.occ.addPoint(x1-r, y, z1  ),
        gmsh.model.occ.addPoint(x0+r, y, z1  ),
        gmsh.model.occ.addPoint(x0,   y, z1-r),
        gmsh.model.occ.addPoint(x0,   y, z0+r),
    ]
    c = [
        gmsh.model.occ.addPoint(x0+r, y, z0+r),
        gmsh.model.occ.addPoint(x1-r, y, z0+r),
        gmsh.model.occ.addPoint(x1-r, y, z1-r),
        gmsh.model.occ.addPoint(x0+r, y, z1-r),
    ]
    l = [
        gmsh.model.occ.addLine(p[0], p[1]),
        gmsh.model.occ.addLine(p[2], p[3]),
        gmsh.model.occ.addLine(p[4], p[5]),
        gmsh.model.occ.addLine(p[6], p[7]),
    ]
    a = [
        gmsh.model.occ.addCircleArc(p[1], c[1], p[2]),
        gmsh.model.occ.addCircleArc(p[3], c[2], p[4]),
        gmsh.model.occ.addCircleArc(p[5], c[3], p[6]),
        gmsh.model.occ.addCircleArc(p[7], c[0], p[0]),
    ]
    loop = gmsh.model.occ.addCurveLoop([
        l[0], a[0], l[1], a[1], l[2], a[2], l[3], a[3]
    ])
    return gmsh.model.occ.addPlaneSurface([loop])

# ============================================================
# STEP 1 — Import rib
# ============================================================
gmsh.initialize()
gmsh.model.add("rae2822_frame")

gmsh.model.occ.importShapes(RIBFILE)
gmsh.model.occ.synchronize()

surfs = gmsh.model.getEntities(2)
if not surfs:
    print("No surface found.")
    gmsh.finalize()
    sys.exit(1)

rib = surfs[0]
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(rib[0], rib[1])
chord  = xmax - xmin
height = zmax - zmin
y_rib  = ymin

print(f"chord={chord:.1f}, height={height:.1f}")

# ============================================================
# STEP 2 — Lay airfoil profile
# ============================================================
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
gmsh.model.mesh.generate(1)

bounds  = gmsh.model.getBoundary([(2, rib[1])], oriented=False)
all_pts = []
for b in bounds:
    _, coords, _ = gmsh.model.mesh.getNodes(1, abs(b[1]))
    all_pts.append(coords.reshape(-1, 3))

all_pts = np.vstack(all_pts)
xs = all_pts[:, 0]
zs = all_pts[:, 2]

x_sample = np.linspace(xmin+5, xmax-5, 500)
upper_z, lower_z = [], []
tol = chord * 0.02

for x in x_sample:
    mask = np.abs(xs - x) < tol
    if mask.sum() < 2:
        upper_z.append(np.nan)
        lower_z.append(np.nan)
    else:
        z_local = zs[mask]
        upper_z.append(z_local.max())
        lower_z.append(z_local.min())

upper_z = np.array(upper_z)
lower_z = np.array(lower_z)
valid   = ~np.isnan(upper_z) & ~np.isnan(lower_z)

f_upper = interp1d(x_sample[valid], upper_z[valid],
                   bounds_error=False, fill_value="extrapolate")
f_lower = interp1d(x_sample[valid], lower_z[valid],
                   bounds_error=False, fill_value="extrapolate")

gmsh.model.mesh.clear()

# ============================================================
# STEP 3 — Tao holes truc tiep voi rounded corners
#
# Cac boundaries:
# x_left  = xmin + OFFSETX           (vien LE)
# x_right = xmax - OFFSETX           (vien TE)
# struts tai STRUT_POSITIONS
#
# Holes nam GIUA cac struts:
# hole i = [strut_i + STRUT_W/2, strut_{i+1} - STRUT_W/2]
# Hole dau tien = [xmin+OFFSETX, strut_0 - STRUT_W/2]
# Hole cuoi     = [strut_last + STRUT_W/2, xmax-OFFSETX]
# ============================================================
x_struts = [xmin + p * chord for p in STRUT_POSITIONS]

# Tat ca cac boundary x (LE, struts, TE)
x_bounds = (
    [xmin + OFFSETX]
    + x_struts
    + [xmax - OFFSETX]
)

def make_adaptive_hole(x0, x1, y, f_upper, f_lower, offsetz, r, n=20):
    """Hole voi top/bot di theo airfoil profile + fillet 4 goc"""
    
    # Sample theo x
    x_inner = np.linspace(x0+r, x1-r, n)

    # === 4 goc fillet (arc tron) ===
    n_arc = 6
    pts = []

    # Bottom-left arc
    z_bl = float(f_lower(x0)) + offsetz
    cx, cz = x0+r, z_bl+r
    for t in np.linspace(np.pi, 3*np.pi/2, n_arc):
        pts.append(gmsh.model.occ.addPoint(
            cx + r*np.cos(t), y, cz + r*np.sin(t)))

    # Bottom edge theo lower surface
    for x in x_inner:
        pts.append(gmsh.model.occ.addPoint(
            x, y, float(f_lower(x)) + offsetz))

    # Bottom-right arc
    z_br = float(f_lower(x1)) + offsetz
    cx, cz = x1-r, z_br+r
    for t in np.linspace(3*np.pi/2, 2*np.pi, n_arc):
        pts.append(gmsh.model.occ.addPoint(
            cx + r*np.cos(t), y, cz + r*np.sin(t)))

    # Right vertical
    z_bot_r = float(f_lower(x1)) + offsetz + r
    z_top_r = float(f_upper(x1)) - offsetz - r
    if z_top_r > z_bot_r:
        for z in np.linspace(z_bot_r, z_top_r, 4):
            pts.append(gmsh.model.occ.addPoint(x1, y, z))

    # Top-right arc
    z_tr = float(f_upper(x1)) - offsetz
    cx, cz = x1-r, z_tr-r
    for t in np.linspace(0, np.pi/2, n_arc):
        pts.append(gmsh.model.occ.addPoint(
            cx + r*np.cos(t), y, cz + r*np.sin(t)))

    # Top edge theo upper surface (phai → trai)
    for x in reversed(x_inner):
        pts.append(gmsh.model.occ.addPoint(
            x, y, float(f_upper(x)) - offsetz))

    # Top-left arc
    z_tl = float(f_upper(x0)) - offsetz
    cx, cz = x0+r, z_tl-r
    for t in np.linspace(np.pi/2, np.pi, n_arc):
        pts.append(gmsh.model.occ.addPoint(
            cx + r*np.cos(t), y, cz + r*np.sin(t)))

    # Left vertical
    z_top_l = float(f_upper(x0)) - offsetz - r
    z_bot_l = float(f_lower(x0)) + offsetz + r
    if z_top_l > z_bot_l:
        for z in np.linspace(z_top_l, z_bot_l, 4):
            pts.append(gmsh.model.occ.addPoint(x0, y, z))

    if len(pts) < 4:
        return None

    spline = gmsh.model.occ.addBSpline(pts + [pts[0]])
    loop   = gmsh.model.occ.addCurveLoop([spline])
    return gmsh.model.occ.addPlaneSurface([loop])

holes = []
for i in range(len(x_bounds) - 1):
    if i == 0:
        # Khoang LE → strut dau
        x0 = x_bounds[0]
        x1 = x_bounds[1] - STRUT_W/2
    elif i == len(x_bounds) - 2:
        # Khoang strut cuoi → TE
        x0 = x_bounds[-2] + STRUT_W/2
        x1 = x_bounds[-1]
    else:
        # Khoang giua 2 struts
        x0 = x_bounds[i]   + STRUT_W/2
        x1 = x_bounds[i+1] - STRUT_W/2

    x_mid = (x0 + x1) / 2
    z_top = float(f_upper(x_mid)) - OFFSETZ
    z_bot = float(f_lower(x_mid)) + OFFSETZ

    if z_top - z_bot < 2*FILLET_R + 5:
        print(f"  Skipping hole {i}: too thin")
        continue
    if x1 - x0 < 2*FILLET_R + 5:
        print(f"  Skipping hole {i}: too narrow")
        continue

    print(f"  Hole {i}: x=[{x0:.1f},{x1:.1f}], z=[{z_bot:.1f},{z_top:.1f}]")

    # surf = make_rounded_rect(x0, x1, z_bot, z_top, y_rib, FILLET_R)
    # Thay make_rounded_rect bang:
    surf = make_adaptive_hole(x0, x1, y_rib, f_upper, f_lower, OFFSETZ, FILLET_R)
    if surf is not None:
        holes.append((2, surf))

gmsh.model.occ.synchronize()
print(f"Total holes: {len(holes)}")

# ============================================================
# STEP 4 — Cut: rib - holes
# ============================================================
if holes:
    result, _ = gmsh.model.occ.cut(
        [rib],
        holes,
        removeObject=True,
        removeTool=True
    )
    gmsh.model.occ.synchronize()
    print(f"Cut result: {result}")

# ============================================================
# STEP 5 — Mesh + export
# ============================================================
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
gmsh.option.setNumber("Mesh.MinimumCirclePoints", 20)
gmsh.model.mesh.generate(2)

gmsh.write(OUTFILE_MSH)
gmsh.write(OUTFILE_STP)
print(f"Saved: {OUTFILE_MSH}")
print(f"Saved: {OUTFILE_STP}")

gmsh.finalize()
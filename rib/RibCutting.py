import gmsh
import sys
import numpy as np
from scipy.interpolate import interp1d


def make_adaptive_hole(x0, x1, y, f_upper, f_lower, offsetz, r, n=20):
    x_inner = np.linspace(x0+r, x1-r, n)

    n_arc = 6
    pts = []

    z_bl = float(f_lower(x0)) + offsetz
    cx, cz = x0+r, z_bl+r
    for t in np.linspace(np.pi, 3*np.pi/2, n_arc):
        pts.append(gmsh.model.occ.addPoint(
            cx + r*np.cos(t), y, cz + r*np.sin(t)))

    for x in x_inner:
        pts.append(gmsh.model.occ.addPoint(
            x, y, float(f_lower(x)) + offsetz))

    z_br = float(f_lower(x1)) + offsetz
    cx, cz = x1-r, z_br+r
    for t in np.linspace(3*np.pi/2, 2*np.pi, n_arc):
        pts.append(gmsh.model.occ.addPoint(
            cx + r*np.cos(t), y, cz + r*np.sin(t)))

    z_bot_r = float(f_lower(x1)) + offsetz + r
    z_top_r = float(f_upper(x1)) - offsetz - r
    if z_top_r > z_bot_r:
        for z in np.linspace(z_bot_r, z_top_r, 4):
            pts.append(gmsh.model.occ.addPoint(x1, y, z))

    z_tr = float(f_upper(x1)) - offsetz
    cx, cz = x1-r, z_tr-r
    for t in np.linspace(0, np.pi/2, n_arc):
        pts.append(gmsh.model.occ.addPoint(
            cx + r*np.cos(t), y, cz + r*np.sin(t)))

    for x in reversed(x_inner):
        pts.append(gmsh.model.occ.addPoint(
            x, y, float(f_upper(x)) - offsetz))

    z_tl = float(f_upper(x0)) - offsetz
    cx, cz = x0+r, z_tl-r
    for t in np.linspace(np.pi/2, np.pi, n_arc):
        pts.append(gmsh.model.occ.addPoint(
            cx + r*np.cos(t), y, cz + r*np.sin(t)))

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


OFFSETX         = 15.0
OFFSETZ         = 13.5
STRUT_W         = 15.0
FILLET_R        = 2.0
MESH_SIZE       = 2.5
STRUT_POSITIONS = [0.25, 0.50, 0.75]


RIBIND      = 2250
RIBFILE     = f"rib{RIBIND}.stp"
OUTFILE_MSH = f"rib{RIBIND}_frame.msh"
OUTFILE_STP = f"rib{RIBIND}_frame.stp"

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
print(chord)
print(height)


gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
gmsh.model.mesh.generate(1)

bounds  = gmsh.model.getBoundary([(2, rib[1])], oriented=False)
all_pts = []
for b in bounds:
    _, coords, _ = gmsh.model.mesh.getNodes(1, abs(b[1]))
    all_pts.append(coords.reshape(-1, 3))

all_pts = np.vstack(all_pts)
xs = all_pts[:, 0]
zs = all_pts[:, 2]

x_sample = np.linspace(xmin+5, xmax-5, 1000)
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

f_upper = interp1d(x_sample[valid], upper_z[valid], bounds_error=False, fill_value="extrapolate")
f_lower = interp1d(x_sample[valid], lower_z[valid], bounds_error=False, fill_value="extrapolate")

gmsh.model.mesh.clear()




x_struts  = [xmin + p * chord for p in STRUT_POSITIONS]
x_bounds  = [xmin + OFFSETX] + x_struts + [xmax - OFFSETX]

holes = []
for i in range(len(x_bounds) - 1):
    if i == 0:
        x0 = x_bounds[0]
        x1 = x_bounds[1] - STRUT_W/2
    elif i == len(x_bounds) - 2:
        x0 = x_bounds[-2] + STRUT_W/2
        x1 = x_bounds[-1]
    else:
        x0 = x_bounds[i]   + STRUT_W/2
        x1 = x_bounds[i+1] - STRUT_W/2

    x_mid = (x0 + x1) / 2
    z_top = float(f_upper(x_mid)) - OFFSETZ
    z_bot = float(f_lower(x_mid)) + OFFSETZ

    if z_top - z_bot < 2*FILLET_R + 5:
        print(f"  Skipping hole {i}: too thin ({z_top-z_bot:.1f} mm)")
        continue
    if x1 - x0 < 2*FILLET_R + 5:
        print(f"  Skipping hole {i}: too narrow ({x1-x0:.1f} mm)")
        continue

    print(f"  Hole {i}: x=[{x0:.1f},{x1:.1f}]")

    surf = make_adaptive_hole(x0, x1, y_rib, f_upper, f_lower, OFFSETZ, FILLET_R)
    if surf is not None:
        holes.append((2, surf))

gmsh.model.occ.synchronize()
print(f"Total holes: {len(holes)}")


if holes:
    result, _ = gmsh.model.occ.cut(
        [rib],
        holes,
        removeObject=True,
        removeTool=True
    )
    gmsh.model.occ.synchronize()
    print(f"Cut result: {result}")

gmsh.model.occ.synchronize()

all_curves = gmsh.model.getEntities(1)
fillet_curves = []
for c in all_curves:
    xmn, ymn, zmn, xmx, ymx, zmx = gmsh.model.getBoundingBox(1, c[1])
    length = np.sqrt((xmx-xmn)**2 + (zmx-zmn)**2)
    if length < 3 * FILLET_R:
        fillet_curves.append(c[1])

print(f"Fillet curves: {len(fillet_curves)}")

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", fillet_curves)
gmsh.model.mesh.field.setNumber(1, "Sampling", 20)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField",   1)
gmsh.model.mesh.field.setNumber(2, "SizeMin",   FILLET_R / 2)
gmsh.model.mesh.field.setNumber(2, "SizeMax",   MESH_SIZE)
gmsh.model.mesh.field.setNumber(2, "DistMin",   FILLET_R)
gmsh.model.mesh.field.setNumber(2, "DistMax",   3 * FILLET_R)

gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints",         0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature",      0)

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
import gmsh

gmsh.initialize()
gmsh.model.add("rib")

# Import wing solid
gmsh.model.occ.importShapes("wingsolid.stp")
gmsh.model.occ.synchronize()

# Tạo plane cắt tại vị trí rib
y_rib = 500.0  # đổi theo bounding box của bạn
size  = 2000.0

p1 = gmsh.model.occ.addPoint(-size, y_rib, -size)
p2 = gmsh.model.occ.addPoint( size, y_rib, -size)
p3 = gmsh.model.occ.addPoint( size, y_rib,  size)
p4 = gmsh.model.occ.addPoint(-size, y_rib,  size)
l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p1)
loop  = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
plane = gmsh.model.occ.addPlaneSurface([loop])

gmsh.model.occ.synchronize()

# Cắt lấy intersection
wing_vol = gmsh.model.getEntities(dim=3)
result, _ = gmsh.model.occ.intersect(
    [(2, plane)],
    wing_vol,
    removeObject=True,
    removeTool=True
)

gmsh.model.occ.synchronize()

# Lấy boundary curves của rib surface
rib_surface = result[0][1]
gmsh.model.occ.synchronize()

# Export ra .geo
gmsh.write("rib_contour.geo_unrolled")
print("Saved: rib_contour.geo_unrolled")

# Hoặc export sang .brep để dùng lại
gmsh.write("rib_contour.stp")
print("Saved: rib_contour.stp")

gmsh.finalize()




import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add("rib")

gmsh.model.occ.importShapes("rib.stp")
gmsh.model.occ.synchronize()

surfs = gmsh.model.getEntities(dim=2)
rib_tag = surfs[0][1]

xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, rib_tag)
chord  = xmax - xmin  # 915.6mm
y_rib  = ymin

# === Các đường cắt theo chord (normalized 0→1, scale theo chord) ===
x_cuts_norm = [0.20, 0.30, 0.42, 0.55, 0.68, 0.80, 0.90]
x_cuts = [xmin + v * chord for v in x_cuts_norm]

# Offset inward: top và bottom boundary của rib
# Dựa theo bounding box: top ≈ zmax*0.7, bot ≈ zmin*0.7
offset = 12.0  # mm, độ dày viền rib
z_top  = zmax - offset
z_bot  = zmin + offset
r      = 8.0   # fillet radius mm

holes = []
for i in range(len(x_cuts) - 1):
    x0 = x_cuts[i]   + offset  # left
    x1 = x_cuts[i+1] - offset  # right
    z0 = z_bot                  # bottom
    z1 = z_top                  # top

    # Tạo hình chữ nhật với fillet góc bằng cách dùng
    # các đường thẳng + arc ở 4 góc
    # Corner centers
    c1 = gmsh.model.occ.addPoint(x0+r, y_rib, z0+r)  # bottom-left
    c2 = gmsh.model.occ.addPoint(x1-r, y_rib, z0+r)  # bottom-right
    c3 = gmsh.model.occ.addPoint(x1-r, y_rib, z1-r)  # top-right
    c4 = gmsh.model.occ.addPoint(x0+r, y_rib, z1-r)  # top-left

    # 4 điểm trên các cạnh thẳng
    p1  = gmsh.model.occ.addPoint(x0+r, y_rib, z0)
    p2  = gmsh.model.occ.addPoint(x1-r, y_rib, z0)
    p3  = gmsh.model.occ.addPoint(x1,   y_rib, z0+r)
    p4  = gmsh.model.occ.addPoint(x1,   y_rib, z1-r)
    p5  = gmsh.model.occ.addPoint(x1-r, y_rib, z1)
    p6  = gmsh.model.occ.addPoint(x0+r, y_rib, z1)
    p7  = gmsh.model.occ.addPoint(x0,   y_rib, z1-r)
    p8  = gmsh.model.occ.addPoint(x0,   y_rib, z0+r)

    # 4 cạnh thẳng
    l1 = gmsh.model.occ.addLine(p1, p2)  # bottom
    l2 = gmsh.model.occ.addLine(p3, p4)  # right
    l3 = gmsh.model.occ.addLine(p5, p6)  # top
    l4 = gmsh.model.occ.addLine(p7, p8)  # left

    # 4 arc fillet tại 4 góc
    a1 = gmsh.model.occ.addCircleArc(p2, c2, p3)  # bottom-right
    a2 = gmsh.model.occ.addCircleArc(p4, c3, p5)  # top-right
    a3 = gmsh.model.occ.addCircleArc(p6, c4, p7)  # top-left
    a4 = gmsh.model.occ.addCircleArc(p8, c1, p1)  # bottom-left

    loop = gmsh.model.occ.addCurveLoop([l1, a1, l2, a2, l3, a3, l4, a4])
    hole_surf = gmsh.model.occ.addPlaneSurface([loop])
    holes.append((2, hole_surf))

gmsh.model.occ.synchronize()

# Boolean cut: rib - tất cả holes
result, _ = gmsh.model.occ.cut(
    [(2, rib_tag)],
    holes,
    removeObject=True,
    removeTool=True
)

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10.0)
gmsh.model.mesh.generate(2)
gmsh.write("rib_optimized.msh")
gmsh.write("rib_optimized.stp")
print("Done!")
gmsh.finalize()
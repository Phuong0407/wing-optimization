import meshio
import numpy as np
import matplotlib.pyplot as plt

# Đọc kết quả final
mesh = meshio.read("topopt/rib_0_rho_final.xdmf")

# Density field (DG0 = giá trị trên mỗi cell)
# Lưu ý: DG0 data có thể nằm trong cell_data
try:
    density = mesh.point_data["density"]
    points  = mesh.points
except:
    density = mesh.cell_data["density"][0]
    # Tính centroid của mỗi cell
    cells   = mesh.cells_dict["triangle"]  # hoặc "tetra"
    points  = mesh.points[cells].mean(axis=1)

print(f"Density range: {density.min():.3f} → {density.max():.3f}")
print(f"Points shape:  {points.shape}")


# Threshold: solid nếu density > 0.5
solid_mask = density > 0.5
solid_pts  = points[solid_mask]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colormap density
sc = axes[0].scatter(points[:, 0]*1e3, points[:, 2]*1e3,
                     c=density, cmap='RdBu_r', s=2)
plt.colorbar(sc, ax=axes[0], label='Density')
axes[0].set_title('Density field (topo opt result)')
axes[0].set_xlabel('x [mm]')
axes[0].set_ylabel('z [mm]')
axes[0].set_aspect('equal')

# Solid vs void
axes[1].scatter(solid_pts[:, 0]*1e3, solid_pts[:, 2]*1e3,
                c='steelblue', s=2, label='Solid')
axes[1].scatter(points[~solid_mask, 0]*1e3,
                points[~solid_mask, 2]*1e3,
                c='lightgray', s=1, alpha=0.3, label='Void')
axes[1].set_title(f'Solid structure (ρ > 0.5)\nVol fraction = {solid_mask.mean():.2f}')
axes[1].set_xlabel('x [mm]')
axes[1].set_ylabel('z [mm]')
axes[1].set_aspect('equal')
plt.legend()
plt.tight_layout()
plt.savefig("rib_topology_result.png", dpi=200)
plt.show()


from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
import shapely

# Lấy boundary của vùng solid (2D: dùng x-z plane)
solid_2d = solid_pts[:, [0, 2]]  # chỉ lấy x và z

# Tạo polygon từ điểm solid dùng alpha shape / buffer
from shapely.geometry import MultiPoint
mp      = MultiPoint(solid_2d)
outline = mp.buffer(0.003)  # buffer 3mm để tạo solid region

# Export sang DXF để import vào CATIA
import ezdxf
doc = ezdxf.new('R2010')
msp = doc.modelspace()

# Lấy exterior boundary
if hasattr(outline, 'geoms'):
    polygons = list(outline.geoms)
else:
    polygons = [outline]

for poly in polygons:
    coords = list(poly.exterior.coords)
    # Vẽ polyline trong DXF
    msp.add_lwpolyline(
        [(x*1e3, z*1e3) for x, z in coords],  # convert to mm
        close=True
    )

doc.saveas("rib_outline.dxf")
print("Saved: rib_outline.dxf → Import vào CATIA Sketch")

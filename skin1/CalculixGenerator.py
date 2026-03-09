"""
CalculixGenerator.py
====================
Generate a Calculix .inp file from MappedTraction.xdmf

Supports S3 (triangle) and S6 (triangle6)
Units: SI (m, Pa, N) throughout

Correct keyword order (Calculix is strict):
    *NODE
    *ELEMENT
    *NSET
    *MATERIAL        <- BEFORE *STEP
    *SHELL SECTION   <- BEFORE *STEP
    *STEP
      *STATIC
      *BOUNDARY
      *CLOAD
      *NODE FILE
      *EL FILE
      *NODE PRINT
    *END STEP
"""

import meshio
import numpy as np
from pathlib import Path
import vtk
from vtk.util.numpy_support import numpy_to_vtk

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRACTION_FILE = "2ndMappedTraction.xdmf"
OUTPUT_FILE   = "wing_calculix.inp"

E     = 210e9   # Pa (SI)
NU    = 0.3
THICK = 1e-3    # m (1mm)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Reading MappedTraction.xdmf ...")
mesh     = meshio.read(TRACTION_FILE)
points   = mesh.points                  # (N, 3) in meters
traction = mesh.point_data["traction"]  # (N, 3) in Pa

if "triangle6" in mesh.cells_dict:
    triangles = mesh.cells_dict["triangle6"]  # (M, 6)
    elem_type = "S6"
else:
    triangles = mesh.cells_dict["triangle"]   # (M, 3)
    elem_type = "S3"

print(f"  Points    : {points.shape}")
print(f"  Elements  : {triangles.shape}  ->  {elem_type}")
print(f"  Traction  : {traction.min():.2f} -> {traction.max():.2f} Pa")

# ─────────────────────────────────────────────
# FIX ORIENTATION VIA VTK
# VTK ne comprend pas triangle6, donc on utilise
# les 3 corner nodes pour detecter les flips,
# puis on applique le flip sur le triangle6 complet
# ─────────────────────────────────────────────
print("\nFixing element orientation via VTK ...")

corner_triangles = triangles[:, :3].copy()

# Build VTK polydata avec corner nodes seulement
poly = vtk.vtkPolyData()
vtk_pts = vtk.vtkPoints()
vtk_pts.SetData(numpy_to_vtk(points.astype(np.float64)))
poly.SetPoints(vtk_pts)

cells_vtk = vtk.vtkCellArray()
for conn in corner_triangles:
    tri = vtk.vtkTriangle()
    tri.GetPointIds().SetId(0, int(conn[0]))
    tri.GetPointIds().SetId(1, int(conn[1]))
    tri.GetPointIds().SetId(2, int(conn[2]))
    cells_vtk.InsertNextCell(tri)
poly.SetPolys(cells_vtk)

# Auto fix orientation
nf = vtk.vtkPolyDataNormals()
nf.SetInputData(poly)
nf.AutoOrientNormalsOn()
nf.ConsistencyOn()
nf.SplittingOff()
nf.Update()
poly_fixed = nf.GetOutput()

# Recuperer les corner nodes fixes
id_list = vtk.vtkIdList()
poly_fixed.GetPolys().InitTraversal()
fixed_corners = []
while poly_fixed.GetPolys().GetNextCell(id_list):
    fixed_corners.append([id_list.GetId(i) for i in range(3)])
fixed_corners = np.array(fixed_corners)

# Detecter les elements flippes
flipped = (
    (fixed_corners[:, 0] != corner_triangles[:, 0]) |
    (fixed_corners[:, 1] != corner_triangles[:, 1]) |
    (fixed_corners[:, 2] != corner_triangles[:, 2])
)
print(f"  Elements flipped: {flipped.sum()} / {len(triangles)}")

# Appliquer le flip sur triangle complet
fixed_triangles = triangles.copy()
if elem_type == "S6":
    # S6 node order: [n0, n1, n2, n01, n12, n20]
    # Flip swap n1<->n2 et n01<->n20 (midside nodes suivent)
    fixed_triangles[flipped] = triangles[flipped][:, [0, 2, 1, 4, 3, 5]]
else:
    fixed_triangles[flipped] = triangles[flipped][:, [0, 2, 1]]

triangles = fixed_triangles

# ─────────────────────────────────────────────
# ROOT NODES (clamped BC) — y = ymin
# ─────────────────────────────────────────────
ymin       = points[:, 1].min()
tol        = 1e-3 * (points[:, 1].max() - ymin)
root_nodes = np.where(np.abs(points[:, 1] - ymin) < tol)[0]
print(f"  Root nodes: {len(root_nodes)}  (ymin = {ymin:.6f} m)")
root_nodes_ccx = root_nodes + 1

# ─────────────────────────────────────────────
# NODAL FORCES = traction * lumped area
# Lumping scheme:
#   S3: area/3 pour chaque node
#   S6: corner = area/12, midside = area*3/12
#       (exact integration for T6)
# ─────────────────────────────────────────────
print("\nComputing nodal forces ...")
nodal_area = np.zeros(len(points))

for conn in triangles:
    p0, p1, p2 = points[conn[0]], points[conn[1]], points[conn[2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))

    if elem_type == "S6":
        for j in range(3):      # corner nodes
            nodal_area[conn[j]] += area / 12.0
        for j in range(3, 6):   # midside nodes
            nodal_area[conn[j]] += area * 3.0 / 12.0
    else:
        for n in conn:
            nodal_area[n] += area / 3.0

nodal_force = traction * nodal_area[:, np.newaxis]  # (N, 3) in N

print(f"  Total force: "
      f"Fx={nodal_force[:,0].sum():.4e}  "
      f"Fy={nodal_force[:,1].sum():.4e}  "
      f"Fz={nodal_force[:,2].sum():.4e} N")

# ─────────────────────────────────────────────
# WRITE .INP
# ─────────────────────────────────────────────
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
print(f"\nWriting {OUTPUT_FILE} ...")

with open(OUTPUT_FILE, "w") as f:

    f.write("** ============================================================\n")
    f.write("** Calculix input - Wing shell under aerodynamic traction\n")
    f.write("** Source: MappedTraction.xdmf\n")
    f.write(f"** Element type : {elem_type}\n")
    f.write(f"** Thickness    : {THICK*1e3:.1f} mm\n")
    f.write("** Units        : SI (m, Pa, N)\n")
    f.write("** ============================================================\n\n")

    # 1. NODES
    f.write("*NODE, NSET=ALL_NODES\n")
    for i, (x, y, z) in enumerate(points):
        f.write(f"{i+1}, {x:.10e}, {y:.10e}, {z:.10e}\n")
    f.write("\n")

    # 2. ELEMENTS
    f.write(f"*ELEMENT, TYPE={elem_type}, ELSET=WING\n")
    for i, conn in enumerate(triangles):
        node_ids = ", ".join(str(int(n) + 1) for n in conn)
        f.write(f"{i+1}, {node_ids}\n")
    f.write("\n")

    # 3. NSET ROOT
    f.write("*NSET, NSET=ROOT\n")
    for chunk_start in range(0, len(root_nodes_ccx), 16):
        chunk = root_nodes_ccx[chunk_start:chunk_start + 16]
        f.write(", ".join(str(n) for n in chunk) + "\n")
    f.write("\n")

    # 4. MATERIAL
    f.write("*MATERIAL, NAME=STEEL\n")
    f.write("*ELASTIC\n")
    f.write(f"{E:.6e}, {NU}\n\n")

    # 5. SHELL SECTION
    f.write("*SHELL SECTION, ELSET=WING, MATERIAL=STEEL\n")
    f.write(f"{THICK:.6e}\n\n")

    # 6. STEP
    f.write("*STEP\n")
    f.write("*STATIC\n\n")

    # 7. BOUNDARY CONDITIONS
    f.write("** Clamped BC at wing root\n")
    f.write("*BOUNDARY\n")
    f.write("ROOT, 1, 6, 0.0\n\n")

    # 8. CLOAD
    f.write("** Aerodynamic nodal forces (traction * lumped area) in N\n")
    f.write("*CLOAD\n")
    for i in range(len(points)):
        fx, fy, fz = nodal_force[i]
        if abs(fx) > 1e-20:
            f.write(f"{i+1}, 1, {fx:.10e}\n")
        if abs(fy) > 1e-20:
            f.write(f"{i+1}, 2, {fy:.10e}\n")
        if abs(fz) > 1e-20:
            f.write(f"{i+1}, 3, {fz:.10e}\n")
    f.write("\n")

    # 9. OUTPUT
    f.write("*NODE FILE\n")
    f.write("U, RF\n\n")

    f.write("*EL FILE\n")
    f.write("S, E\n\n")

    f.write("*NODE PRINT, NSET=ALL_NODES\n")
    f.write("U\n\n")

    f.write("*END STEP\n")

print(f"Done! -> {OUTPUT_FILE}")
print(f"\nTo run    : ccx wing_calculix")
print(f"To convert: ccx2paraview wing_calculix.frd wing_calculix")
print(f"To view   : paraview wing_calculix.pvd")

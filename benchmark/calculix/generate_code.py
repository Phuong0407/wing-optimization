import meshio
import numpy as np

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pathlib import Path


TRACTION_FILE = "MappedTraction.xdmf"
OUTPUT_FILE   = "WingCalculix.inp"

YOUNG = 210E9
NU    = 0.3
THICK = 10E-3



print("Reading MappedTraction.xdmf ...")
MESH     = meshio.read(TRACTION_FILE)
POINT    = MESH.points
TRACTION = MESH.point_data["traction"]

if "triangle6" in MESH.cells_dict:
  TRIANGLE = MESH.cells_dict["triangle6"]
  ELEMTYPE = "S6"
else:
  TRIANGLE = MESH.cells_dict["triangle"]
  ELEMTYPE = "S3"

triangles_before = TRIANGLE.copy()

poly = vtk.vtkPolyData()

VTKPOINT = vtk.vtkPoints()
VTKPOINT.SetData(numpy_to_vtk(POINT))
poly.SetPoints(VTKPOINT)

cells = vtk.vtkCellArray()
for conn in TRIANGLE:
  tri = vtk.vtkTriangle()
  tri.GetPointIds().SetId(0, conn[0])
  tri.GetPointIds().SetId(1, conn[1])
  tri.GetPointIds().SetId(2, conn[2])
  cells.InsertNextCell(tri)
poly.SetPolys(cells)

NORMAL_FILTER = vtk.vtkPolyDataNormals()
NORMAL_FILTER.SetInputData(poly)
NORMAL_FILTER.AutoOrientNormalsOn()
NORMAL_FILTER.ConsistencyOn()
NORMAL_FILTER.SplittingOff()
NORMAL_FILTER.Update()
poly = NORMAL_FILTER.GetOutput()

cells = poly.GetPolys()
cells.InitTraversal()
idList = vtk.vtkIdList()
fixed_triangles = []
while cells.GetNextCell(idList):
  fixed_triangles.append([idList.GetId(i) for i in range(3)])
triangles = np.array(fixed_triangles)

print(f"  Triangles before: {triangles_before.shape}")
print(f"  Triangles after : {triangles.shape}")
diff = (triangles_before != triangles).any(axis=1).sum()
print(f"  Elements flipped: {diff}")

ymin            = POINT[:, 1].min()
tol             = 1e-3 * (POINT[:, 1].max() - ymin)
ROOT_NODES      = np.where(np.abs(POINT[:, 1] - ymin) < tol)[0]
ROOT_NODES_CCX  = ROOT_NODES + 1



NODAL_AREA = np.zeros(len(POINT))
for conn in triangles:
  P0, P1, P2 = POINT[conn[0]], POINT[conn[1]], POINT[conn[2]]
  AREA  = 0.5 * np.linalg.norm(np.cross(P1 - P0, P2 - P0))
  SHARED_AREA = AREA / len(conn)
  for n in conn:
    NODAL_AREA[n] += SHARED_AREA
NODAL_FORCE = TRACTION * NODAL_AREA[:, np.newaxis]



Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
print(f"\nWriting {OUTPUT_FILE} ...")

with open(OUTPUT_FILE, "w") as OUTFILE:
  OUTFILE.write("*NODE, NSET=ALL_NODES\n")
  for i, (x, y, z) in enumerate(POINT):
    OUTFILE.write(f"{i+1}, {x:.6e}, {y:.6e}, {z:.6e}\n")
  OUTFILE.write("\n")

  OUTFILE.write(f"*ELEMENT, TYPE={ELEMTYPE}, ELSET=WING\n")
  for i, conn in enumerate(triangles):
    node_ids = ", ".join(str(n + 1) for n in conn)
    OUTFILE.write(f"{i+1}, {node_ids}\n")
  OUTFILE.write("\n")

  OUTFILE.write("*NSET, NSET=ROOT\n")
  for chunk_start in range(0, len(ROOT_NODES_CCX), 16):
    chunk = ROOT_NODES_CCX[chunk_start:chunk_start + 16]
    OUTFILE.write(", ".join(str(n) for n in chunk) + "\n")
  OUTFILE.write("\n")

  OUTFILE.write("*MATERIAL, NAME=STEEL\n")
  OUTFILE.write("*ELASTIC\n")
  OUTFILE.write(f"{YOUNG}, {NU}\n\n")

  OUTFILE.write("*SHELL SECTION, ELSET=WING, MATERIAL=STEEL\n")
  OUTFILE.write(f"{THICK}\n\n")

  OUTFILE.write("*STEP\n")
  OUTFILE.write("*STATIC\n\n")

  OUTFILE.write("*BOUNDARY\n")
  OUTFILE.write("ROOT, 1, 6, 0.0\n\n")

  OUTFILE.write("** LUMPED AERODYNAMIC NODAL FORCES\n")
  OUTFILE.write("*CLOAD\n")
  for i in range(len(POINT)):
    FX, FY, FZ = NODAL_FORCE[i]
    if abs(FX) > 1e-12:
      OUTFILE.write(f"{i+1}, 1, {FX:.6e}\n")
    if abs(FY) > 1e-12:
      OUTFILE.write(f"{i+1}, 2, {FY:.6e}\n")
    if abs(FZ) > 1e-12:
      OUTFILE.write(f"{i+1}, 3, {FZ:.6e}\n")
  OUTFILE.write("\n")

  OUTFILE.write("*NODE FILE\n")
  OUTFILE.write("U, RF\n\n")

  OUTFILE.write("*EL FILE\n")
  OUTFILE.write("S, E\n\n")

  OUTFILE.write("*NODE PRINT, NSET=ALL_NODES\n")
  OUTFILE.write("U\n\n")

  OUTFILE.write("*END STEP\n")
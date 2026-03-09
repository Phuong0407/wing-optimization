import vtk
import meshio
import numpy as np
import triangle as tr
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path

WING_STL_FILE = "wing.stl"
SCALE         = 1E-3
OUTPUT_DIR    = Path(".")
RIB_FRACTIONS = [0.01, 0.33, 0.50, 0.67, 0.99]

MAX_AREA = 0.000005

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load and scale STL
reader = vtk.vtkSTLReader()
reader.SetFileName(WING_STL_FILE)
reader.Update()

transform = vtk.vtkTransform()
transform.Scale(SCALE, SCALE, SCALE)
tf = vtk.vtkTransformPolyDataFilter()
tf.SetTransform(transform)
tf.SetInputConnection(reader.GetOutputPort())
tf.Update()
poly = tf.GetOutput()

pts_all = vtk_to_numpy(poly.GetPoints().GetData())
y_min   = pts_all[:, 1].min()
y_max   = pts_all[:, 1].max()
span    = y_max - y_min

for i, frac in enumerate(RIB_FRACTIONS):
    y_rib = y_min + frac * span

    # Cut at Y = y_rib
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, y_rib, 0.0)
    plane.SetNormal(0.0, 1.0, 0.0)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(poly)
    cutter.Update()

    cut_out = cutter.GetOutput()
    n       = cut_out.GetNumberOfPoints()
    raw     = vtk_to_numpy(cut_out.GetPoints().GetData())

    # Walk edge graph to get ordered closed boundary
    adj     = {k: [] for k in range(n)}
    id_list = vtk.vtkIdList()
    for c in range(cut_out.GetNumberOfCells()):
        cut_out.GetCellPoints(c, id_list)
        if id_list.GetNumberOfIds() == 2:
            a, b = id_list.GetId(0), id_list.GetId(1)
            adj[a].append(b)
            adj[b].append(a)

    visited    = np.zeros(n, dtype=bool)
    loop       = [0]
    visited[0] = True
    current    = 0
    while True:
        moved = False
        for nb in adj[current]:
            if not visited[nb]:
                loop.append(nb)
                visited[nb] = True
                current     = nb
                moved       = True
                break
        if not moved:
            break

    # Ordered boundary in XZ plane
    ordered    = raw[loop]
    vertices   = ordered[:, [0, 2]]   # (N, 2) in XZ

    # Boundary segments: 0→1, 1→2, ..., N-1→0
    N        = len(vertices)
    segments = np.array([[j, (j+1) % N] for j in range(N)])

    # Triangle: 'p' = PSLG, 'q' = quality (min 20° angle), 'a' = max area
    mesh_input = {"vertices": vertices, "segments": segments}
    # mesh_out   = tr.triangulate(mesh_input, f"pqa{MAX_AREA}")
    mesh_out = tr.triangulate(mesh_input, f"pqa{MAX_AREA}q30")

    pts_2d = mesh_out["vertices"]               # (M, 2)
    tris   = mesh_out["triangles"].astype(int)  # (T, 3)

    # Restore 3D
    pts_3d       = np.zeros((len(pts_2d), 3))
    pts_3d[:, 0] = pts_2d[:, 0]   # X chord
    pts_3d[:, 1] = y_rib           # Y span
    pts_3d[:, 2] = pts_2d[:, 1]   # Z thickness

    mesh = meshio.Mesh(
        points=pts_3d,
        cells=[("triangle", tris)],
    )
    meshio.write(str(OUTPUT_DIR / f"rib_{i}.xdmf"), mesh)
    meshio.write(str(OUTPUT_DIR / f"rib_{i}.vtk"),  mesh)
    print(f"rib_{i}  y={y_rib:.3f}m  pts={len(pts_3d)}  tris={len(tris)}")
    
ordered    = raw[loop]
x_chord    = ordered[:, 0]
z_thick    = ordered[:, 2]
chord      = x_chord.max() - x_chord.min()
thickness  = z_thick.max() - z_thick.min()
area_approx = chord * thickness * 0.12   # ~12% t/c cho NACA thông thường

print(chord)



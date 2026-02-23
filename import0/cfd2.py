import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

# ── Cell 1: Đọc CFD VTP ──
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("wing.vtp")
reader.Update()
triangulate = vtk.vtkTriangleFilter()
triangulate.SetInputData(reader.GetOutput())
triangulate.Update()
polydata = triangulate.GetOutput()

cfd_pts = vtk_to_numpy(polydata.GetPoints().GetData())
p_cfd = vtk_to_numpy(polydata.GetPointData().GetArray("p")).reshape(-1)
print("CFD points:", cfd_pts.shape)
print("Pressure range:", p_cfd.min(), p_cfd.max())

# ── Cell 2: Đọc structural mesh từ meshio (giữ winding order) ──
import meshio
msh = meshio.read("wingTRI.msh")
pts = msh.points * 1e-3
tri_all = msh.cells_dict["triangle"]
tags = msh.cell_data_dict["gmsh:physical"]["triangle"]

# Fix normal orientation dùng physical tags
fixed_tri = []
for idx, (i, j, k) in enumerate(tri_all):
    x0, x1, x2 = pts[i], pts[j], pts[k]
    nA = np.cross(x1 - x0, x2 - x0)
    if tags[idx] == 14:      # upper: normal +z
        if nA[2] < 0:
            fixed_tri.append([k, j, i])
        else:
            fixed_tri.append([i, j, k])
    else:                     # lower tag 15: normal -z
        if nA[2] > 0:
            fixed_tri.append([k, j, i])
        else:
            fixed_tri.append([i, j, k])
fixed_tri = np.array(fixed_tri)

# ── Cell 3: Interpolate pressure lên structural mesh ──
from scipy.spatial import cKDTree
tree = cKDTree(cfd_pts)
dist, nn_idx = tree.query(pts, k=4)
weights = 1.0 / (dist + 1e-12)
weights /= weights.sum(axis=1, keepdims=True)
p_struct = (weights * p_cfd[nn_idx]).sum(axis=1)
print("Interpolated pressure range:", p_struct.min(), p_struct.max())

# ── Cell 4: Tính lực từ fixed_tri (meshio, winding order preserved) ──
F = np.zeros(3)
A_total = 0.0
for i, j, k in fixed_tri:
    x0, x1, x2 = pts[i], pts[j], pts[k]
    nA = np.cross(x1 - x0, x2 - x0) * 0.5
    dS = np.linalg.norm(nA)
    if dS == 0:
        continue
    ptri = (p_struct[i] + p_struct[j] + p_struct[k]) / 3.0
    F += -ptri * nA
    A_total += dS

print("Surface area:", A_total)
print("Force from structural mesh [N]:", F)
print("Force flipped [N]:", -F)
print("Force from OpenFOAM        [N]: [ 892, -4449, 17799]")

# ── Cell 5: Export pressure lên FEniCS mesh để dùng làm BC ──
import fenics as fe
wing_mesh = fe.Mesh()
with fe.XDMFFile("wingTRI_fixed.xdmf") as f:
    f.read(wing_mesh)

V = fe.FunctionSpace(wing_mesh, "CG", 1)
p_func = fe.Function(V)
# Map p_struct (indexed by meshio vertex) sang FEniCS dof ordering
p_func.vector()[:] = p_struct[fe.dof_to_vertex_map(V)]

with fe.XDMFFile("pressure_on_structure.xdmf") as f:
    f.write(p_func)
print("Saved pressure_on_structure.xdmf")

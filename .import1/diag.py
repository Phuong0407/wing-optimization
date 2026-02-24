# VERIFICATION WHETHER TIPCAP CONTRIBUTES TO ERROR IN WINGSPAN FORCE (FY)

import numpy as np
import pyvista as pv

# ── Load OpenFOAM VTP ────────────────────────────────────────────
mesh_vtp = pv.read("wing.vtp")
pts_of   = np.array(mesh_vtp.points)
p_of     = np.array(mesh_vtp.point_data["p"])
wss_of   = np.array(mesh_vtp.point_data["wallShearStress"])

# Tính normals - dùng vtk trực tiếp như trước (đã proven working)
import vtk
from vtk.util.numpy_support import vtk_to_numpy

data_reader = vtk.vtkXMLPolyDataReader()
data_reader.SetFileName("wing.vtp")
data_reader.Update()
poly = data_reader.GetOutput()

triangulate = vtk.vtkTriangleFilter()
triangulate.SetInputData(poly)
triangulate.Update()
poly = triangulate.GetOutput()

normal_filter = vtk.vtkPolyDataNormals()
normal_filter.SetInputData(poly)
normal_filter.ComputePointNormalsOn()
normal_filter.ComputeCellNormalsOff()
normal_filter.AutoOrientNormalsOn()
normal_filter.ConsistencyOn()
normal_filter.SplittingOff()
normal_filter.Update()
poly = normal_filter.GetOutput()

pts_of = vtk_to_numpy(poly.GetPoints().GetData())
p_of   = vtk_to_numpy(poly.GetPointData().GetArray("p"))
wss_of = vtk_to_numpy(poly.GetPointData().GetArray("wallShearStress"))
n_of   = vtk_to_numpy(poly.GetPointData().GetArray("Normals"))

# Extract triangles
cells = poly.GetPolys()
cells.InitTraversal()
idList = vtk.vtkIdList()
tris = []
while cells.GetNextCell(idList):
    tris.append([idList.GetId(i) for i in range(3)])
tris = np.array(tris)

print(f"Points    : {pts_of.shape}")
print(f"Triangles : {tris.shape}")
print(f"y range   : {pts_of[:,1].min():.4f} → {pts_of[:,1].max():.4f}")

# ── Traction ─────────────────────────────────────────────────────
p_inf   = 54019.55
p_gauge = p_of - p_inf
wss_t   = -wss_of - np.sum(-wss_of*n_of, axis=1, keepdims=True)*n_of
traction = -p_gauge[:,None]*n_of + wss_t

# ── Classify triangles: tip cap vs wing skin ──────────────────────
y_max = pts_of[:,1].max()
tri_cy = (pts_of[tris[:,0],1] + pts_of[tris[:,1],1] + pts_of[tris[:,2],1]) / 3.0

tip_tri_mask  = tri_cy > (y_max - 0.01)
skin_tri_mask = ~tip_tri_mask

print(f"Tip cap triangles : {tip_tri_mask.sum()}")
print(f"Wing skin triangles: {skin_tri_mask.sum()}")

# ── Integrate ────────────────────────────────────────────────────
def integrate_force(tris_sel):
    v0 = pts_of[tris_sel[:,0]]
    v1 = pts_of[tris_sel[:,1]]
    v2 = pts_of[tris_sel[:,2]]
    areas    = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
    t_center = (traction[tris_sel[:,0]] +
                traction[tris_sel[:,1]] +
                traction[tris_sel[:,2]]) / 3.0
    return np.sum(t_center * areas[:,None], axis=0)

F_skin  = integrate_force(tris[skin_tri_mask])
F_tip   = integrate_force(tris[tip_tri_mask])
F_total = F_skin + F_tip
F_of    = np.array([1133.08, -4433.40, 17824.76])

print(f"\n=== Force Decomposition ===")
print(f"{'':12} {'Fx':>10} {'Fy':>10} {'Fz':>10}")
print(f"{'-'*44}")
print(f"{'Wing skin':12} {F_skin[0]:10.2f} {F_skin[1]:10.2f} {F_skin[2]:10.2f}")
print(f"{'Tip cap':12} {F_tip[0]:10.2f}  {F_tip[1]:10.2f} {F_tip[2]:10.2f}")
print(f"{'Total':12} {F_total[0]:10.2f} {F_total[1]:10.2f} {F_total[2]:10.2f}")
print(f"{'OpenFOAM':12} {F_of[0]:10.2f} {F_of[1]:10.2f} {F_of[2]:10.2f}")

print(f"\n=== Error after tip cap ===")
for i, l in enumerate(["Fx","Fy","Fz"]):
    e = abs(F_total[i]-F_of[i])/(abs(F_of[i])+1e-10)*100
    print(f"{l}: {e:.1f}%")


import numpy as np

# Dùng lại data từ trên
F_of = np.array([1133.08, -4433.40, 17824.76])

# Case 1: p_gauge (current approach)
trac_gauge = -p_gauge[:,None]*n_of + wss_t

# Case 2: absolute pressure pRef=0 (như OpenFOAM)
trac_abs   = -p_of[:,None]*n_of + wss_t

def integrate_all(traction):
    v0=pts_of[tris[:,0]]; v1=pts_of[tris[:,1]]; v2=pts_of[tris[:,2]]
    areas = 0.5*np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
    tc = (traction[tris[:,0]]+traction[tris[:,1]]+traction[tris[:,2]])/3.0
    return np.sum(tc*areas[:,None], axis=0)

F_gauge = integrate_all(trac_gauge)
F_abs   = integrate_all(trac_abs)

# Integral of normals
v0=pts_of[tris[:,0]]; v1=pts_of[tris[:,1]]; v2=pts_of[tris[:,2]]
areas = 0.5*np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
nc = (n_of[tris[:,0]]+n_of[tris[:,1]]+n_of[tris[:,2]])/3.0
int_n = np.sum(nc*areas[:,None], axis=0)

print(f"Integral of normals: {int_n}")
print(f"p_inf * int_n      : {p_inf * int_n}")
print(f"\nF_gauge = {F_gauge}")
print(f"F_abs   = {F_abs}")
print(f"F_of    = {F_of}")
print(f"\nF_abs - F_gauge = {F_abs - F_gauge}")
print(f"p_inf * int_n   = {p_inf * int_n}  ← phải match line trên")
print(f"\nConclusion:")
print(f"F_abs_y  = {F_abs[1]:.2f} N  vs  OF_y = {F_of[1]:.2f} N")
print(f"Remaining Fy error = {abs(F_abs[1]-F_of[1]):.2f} N")

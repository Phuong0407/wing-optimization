import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

def skew(v):
    """Skew-symmetric matrix so that skew(v) @ f == v x f"""
    vx, vy, vz = v
    return np.array([[0, -vz,  vy],
                     [vz,  0, -vx],
                     [-vy, vx,  0]], dtype=float)

def cfd_force_moment_from_vtp(vtp_path: str, p_name="p", r0=np.zeros(3)):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(reader.GetOutput())
    tri.Update()
    poly = tri.GetOutput()

    pts = vtk_to_numpy(poly.GetPoints().GetData())
    p = vtk_to_numpy(poly.GetPointData().GetArray(p_name)).reshape(-1)

    polys = poly.GetPolys()
    polys.InitTraversal()
    idList = vtk.vtkIdList()

    F = np.zeros(3)
    M = np.zeros(3)
    A = 0.0

    while polys.GetNextCell(idList):
        ids = [idList.GetId(i) for i in range(3)]
        x0, x1, x2 = pts[ids]
        nA = 0.5 * np.cross(x1 - x0, x2 - x0)      # area vector
        ptri = float(p[ids].mean())                # point-pressure averaged
        ftri = -ptri * nA
        rc = (x0 + x1 + x2) / 3.0
        F += ftri
        M += np.cross(rc - r0, ftri)
        A += np.linalg.norm(nA)

    return F, M, A

import numpy as np

def distribute_forces_constrained(pts: np.ndarray,
                                  F_target: np.ndarray,
                                  M_target: np.ndarray,
                                  r0: np.ndarray,
                                  weights: np.ndarray | None = None):
    """
    pts: (N,3) node coordinates (meters)
    F_target: (3,) desired total force
    M_target: (3,) desired total moment about r0
    r0: (3,) reference point for moment
    weights: (N,) positive weights; larger => penalize force at node more
             If None -> uniform weights.
    Returns f_nodes: (N,3)
    """
    N = pts.shape[0]
    F_target = np.asarray(F_target, float).reshape(3)
    M_target = np.asarray(M_target, float).reshape(3)
    r0 = np.asarray(r0, float).reshape(3)

    if weights is None:
        weights = np.ones(N, dtype=float)
    else:
        weights = np.asarray(weights, float).reshape(N)
        if np.any(weights <= 0):
            raise ValueError("weights must be > 0")

    # Build A (6 x 3N)
    A = np.zeros((6, 3*N), dtype=float)

    # Force constraints: sum f_i = F
    for i in range(N):
        A[0:3, 3*i:3*i+3] = np.eye(3)

    # Moment constraints: sum (r_i-r0) x f_i = M
    for i in range(N):
        ri = pts[i] - r0
        A[3:6, 3*i:3*i+3] = skew(ri)

    b = np.hstack([F_target, M_target])  # (6,)

    # Quadratic term: minimize f^T W f, W is diagonal block with weights
    # KKT:
    # [ 2W   A^T ] [f]   [0]
    # [ A     0  ] [λ] = [b]
    # where W is (3N x 3N) diagonal with weights repeated 3 times.
    Wdiag = np.repeat(weights, 3)            # (3N,)
    K11 = np.diag(2.0 * Wdiag)               # (3N,3N)
    K12 = A.T                                # (3N,6)
    K21 = A                                  # (6,3N)
    K22 = np.zeros((6, 6), dtype=float)      # (6,6)

    K = np.block([[K11, K12],
                  [K21, K22]])
    rhs = np.hstack([np.zeros(3*N), b])

    sol = np.linalg.solve(K, rhs)
    f = sol[:3*N].reshape(N, 3)

    # sanity checks
    F_check = f.sum(axis=0)
    M_check = np.sum(np.cross(pts - r0, f), axis=0)

    return f, F_check, M_check

import meshio
import numpy as np

# --- read structural nodes (meters) ---
msh = meshio.read("wingTRI2.msh")
pts_s = msh.points * 1e-3   # meters

# --- compute CFD resultant ---
r0 = np.array([0.0, 0.0, 0.0])  # chọn mốc moment (đổi tùy bạn: root, SOB...)
F_cfd, M_cfd, A_cfd = cfd_force_moment_from_vtp("wing.vtp", p_name="p", r0=r0)

print("CFD area:", A_cfd)
print("F_cfd:", F_cfd)
print("M_cfd:", M_cfd)

# --- distribute to structural nodes, force+moment conservative ---
# weights: để lực không dồn vào 1 chỗ. Có thể lấy theo "lumped area" nếu bạn có; tạm thời uniform.
f_nodes, F_chk, M_chk = distribute_forces_constrained(
    pts=pts_s,
    F_target=F_cfd,
    M_target=M_cfd,
    r0=r0,
    weights=None
)

print("Check sum force:", F_chk, " error:", F_chk - F_cfd)
print("Check sum moment:", M_chk, " error:", M_chk - M_cfd)

# f_nodes là nodal load (N) bạn đưa vào FEM
np.save("nodal_forces.npy", f_nodes)
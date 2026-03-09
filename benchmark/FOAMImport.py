import os
import vtk
import meshio
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
from dolfinx import fem
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

VERBOSE = False



def import_foam_traction(FOAMFILE, XDMFFILE=None, VERBOSE=False):
  FOAMFILE = "wing.vtp"
  FOAMREADER = vtk.vtkXMLPolyDataReader()
  FOAMREADER.SetFileName(FOAMFILE)
  FOAMREADER.Update()
  poly = FOAMREADER.GetOutput()

  TRIANGULATION = vtk.vtkTriangleFilter()
  TRIANGULATION.SetInputData(poly)
  TRIANGULATION.Update()
  poly = TRIANGULATION.GetOutput()

  NORMAL_FILTER = vtk.vtkPolyDataNormals()
  NORMAL_FILTER.SetInputData(poly)
  NORMAL_FILTER.ComputePointNormalsOn()
  NORMAL_FILTER.ComputeCellNormalsOff()
  NORMAL_FILTER.AutoOrientNormalsOn()
  NORMAL_FILTER.AutoOrientNormalsOff()
  NORMAL_FILTER.ConsistencyOn()
  NORMAL_FILTER.SplittingOff()
  NORMAL_FILTER.Update()
  POLY = NORMAL_FILTER.GetOutput()

  POINTS  = vtk_to_numpy(POLY.GetPoints().GetData())
  P       = vtk_to_numpy(POLY.GetPointData().GetArray("p"))
  WSS     = vtk_to_numpy(POLY.GetPointData().GetArray("wallShearStress"))
  NORMALS = vtk_to_numpy(POLY.GetPointData().GetArray("Normals"))
  NORMALS  = -NORMALS

  if VERBOSE:
    print(f"Points  : {POINTS.shape}")
    print(f"p       : {P.shape}")
    print(f"wss     : {WSS.shape}")
    print(f"normals : {NORMALS.shape}")

  TRACTION     = -P[:,np.newaxis] * NORMALS + WSS
  TRACTIONMAG  = np.linalg.norm(TRACTION, axis=1)

  if VERBOSE:
    print(f"p range            : {P.min():.2f} -> {P.max():.2f} Pa")
    print(f"traction magnitude : {TRACTIONMAG.min():.2f} -> {TRACTIONMAG.max():.2f} Pa")

  CELLS = POLY.GetPolys()
  CELLS.InitTraversal()
  IDLIST = vtk.vtkIdList()

  TRIANGLES = []
  while CELLS.GetNextCell(IDLIST):
    TRIANGLES.append([IDLIST.GetId(i) for i in range(3)])
  TRIANGLES = np.array(TRIANGLES)

  print("Points:", POINTS.shape)
  print("Triangles:", TRIANGLES.shape)

  mesh = meshio.Mesh(
    points=POINTS,
    cells=[("triangle", TRIANGLES)],
    point_data={
      "p"       : P,
      "normals" : NORMALS,
      "wss"     : WSS,
      "traction": TRACTION,
    }
  )

  meshio.write(XDMFFILE, mesh)
  print(f"Export the wing-patch traction from FOAM to the file {XDMFFILE}")



def is_xdmffile_valie(path):
  if not os.path.exists(path):
    return False
  try:
    mesh = meshio.read(path)
  except Exception:
    return False
  if "triangle" not in [c.type for c in mesh.cells]:
    return False
  if "traction" not in mesh.point_data:
    return False
  traction = mesh.point_data["traction"]
  if traction.ndim != 2 or traction.shape[1] != 3:
    return False
  return True

def load_traction_from_xdmf(XDMFFILE, DOMAIN, GDIM,):
  FOAM = meshio.read(XDMFFILE)
  FOAMPoint = FOAM.points
  FOAMTraction = FOAM.point_data["traction"]
  
  VTTraction = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))
  FTraction  = fem.Function(VTTraction, name="traction")
  
  DOFCOORDS = VTTraction.tabulate_dof_coordinates()
  TREE = cKDTree(FOAMPoint)
  _, ID = TREE.query(DOFCOORDS, k=1)
  FTraction.x.array[:] = FOAMTraction[ID].flatten()
  FTraction.x.scatter_forward()
  return FTraction

def map_traction(foamfile, meshfile, mappedfile):
  FOAMMESH     = meshio.read(foamfile)
  FOAMPOINT    = FOAMMESH.points
  FOAMTRI      = FOAMMESH.cells_dict["triangle"]
  FOAMTRACTION = FOAMMESH.point_data["traction"]

  FEMMESH      = meshio.read(meshfile)
  FEMPOINT     = FEMMESH.points * 1E-3
  FEMTRI       = FEMMESH.cells_dict["triangle"]
  
  TREE = cKDTree(FOAMPOINT)
  DISTS, IDS = TREE.query(FEMPOINT, k=10)
  WEIGHTS = 1.0 / (DISTS + 1e-12)
  WEIGHTS /= WEIGHTS.sum(axis=1, keepdims=True)
  MAPPEDTRACTION = np.einsum('ni,nij->nj', WEIGHTS, FOAMTRACTION[IDS])

  FEM_FORCE  = np.zeros(3)
  FOAM_FORCE = np.zeros(3)
  for TRI in FEMTRI:
    P0, P1, P2 = FEMPOINT[TRI]
    AREA = 0.5 * np.linalg.norm(np.cross(P1-P0, P2-P0))
    FEM_FORCE += MAPPEDTRACTION[TRI].mean(axis=0) * AREA
  for TRI in FOAMTRI:
    P0, P1, P2 = FOAMPOINT[TRI]
    AREA = 0.5 * np.linalg.norm(np.cross(P1-P0, P2-P0))
    FOAM_FORCE += FOAMTRACTION[TRI].mean(axis=0) * AREA

  print("FOAM force [N]:", FOAM_FORCE)
  print("FEM  force [N]:", FEM_FORCE)
  print("Error:", np.linalg.norm(FEM_FORCE-FOAM_FORCE)/np.linalg.norm(FOAM_FORCE))

  # Thử k=1 (nearest neighbor thuần) để xem baseline
  DISTS, IDS = TREE.query(FEMPOINT, k=1)
  MappedTraction_k1 = FOAMTRACTION[IDS]

  # Thử k=10
  DISTS, IDS = TREE.query(FEMPOINT, k=10)
  WEIGHTS = 1.0 / (DISTS + 1e-12)
  WEIGHTS /= WEIGHTS.sum(axis=1, keepdims=True)
  MappedTraction_k10 = np.einsum('ni,nij->nj', WEIGHTS, FOAMTRACTION[IDS])

  # So sánh X force
  for k_name, MT in [("k=1", MappedTraction_k1), ("k=10", MappedTraction_k10)]:
      F = np.zeros(3)
      for TRI in FEMTRI:
          P0, P1, P2 = FEMPOINT[TRI]
          AREA = 0.5 * np.linalg.norm(np.cross(P1-P0, P2-P0))
          F += MT[TRI].mean(axis=0) * AREA
      print(f"{k_name} FEM force [N]:", F)


  MappedMesh = meshio.Mesh(
    points=FEMPOINT,
    cells=[("triangle",FEMMESH.cells_dict["triangle"])],
    point_data={"traction": MAPPEDTRACTION}
  )
  meshio.write(mappedfile, MappedMesh)
  
def map_traction_conservative(foamfile, meshfile, outfile):
  FOAMMesh     = meshio.read(foamfile)
  FOAMPts      = FOAMMesh.points
  FOAMTri      = FOAMMesh.cells_dict["triangle"]
  FOAMTraction = FOAMMesh.point_data["traction"]

  FEMMesh = meshio.read(meshfile)
  FEMPts  = FEMMesh.points * 1e-3
  FEMTri  = FEMMesh.cells_dict["triangle"]

  # Tính centroid và traction tại centroid của mỗi FOAM triangle
  FOAM_CENTROIDS = FOAMPts[FOAMTri].mean(axis=1)          # (N_foam, 3)
  FOAM_TRI_TRACT = FOAMTraction[FOAMTri].mean(axis=1)     # (N_foam, 3)
  FOAM_TRI_AREA  = 0.5 * np.linalg.norm(
      np.cross(FOAMPts[FOAMTri[:,1]] - FOAMPts[FOAMTri[:,0]],
                FOAMPts[FOAMTri[:,2]] - FOAMPts[FOAMTri[:,0]]), axis=1
  )                                                        # (N_foam,)

  # Tính centroid của mỗi FEM triangle
  FEM_CENTROIDS = FEMPts[FEMTri].mean(axis=1)             # (N_fem, 3)
  FEM_TRI_AREA  = 0.5 * np.linalg.norm(
      np.cross(FEMPts[FEMTri[:,1]] - FEMPts[FEMTri[:,0]],
                FEMPts[FEMTri[:,2]] - FEMPts[FEMTri[:,0]]), axis=1
  )                                                        # (N_fem,)

  # Với mỗi FOAM triangle, tìm FEM triangle gần nhất
  # rồi accumulate lực (conservative mapping)
  TREE = cKDTree(FEM_CENTROIDS)
  _, FEM_TRI_ID = TREE.query(FOAM_CENTROIDS)

  # Accumulate force từ FOAM → FEM triangles
  FEM_TRI_FORCE = np.zeros((len(FEMTri), 3))
  for i, fid in enumerate(FEM_TRI_ID):
      FEM_TRI_FORCE[fid] += FOAM_TRI_TRACT[i] * FOAM_TRI_AREA[i]

  # Convert force → traction trên FEM triangle
  FEM_TRI_TRACTION = FEM_TRI_FORCE / (FEM_TRI_AREA[:,np.newaxis] + 1e-12)

  # Verify
  FOAM_FORCE = np.sum(FOAM_TRI_TRACT * FOAM_TRI_AREA[:,np.newaxis], axis=0)
  FEM_FORCE  = np.sum(FEM_TRI_FORCE, axis=0)
  print("FOAM force [N]:", FOAM_FORCE)
  print("FEM  force [N]:", FEM_FORCE)
  print("Error:", np.linalg.norm(FEM_FORCE-FOAM_FORCE)/np.linalg.norm(FOAM_FORCE))

  # Lump sang nodal force
  NUM_POINT   = len(FEMPts)
  NODAL_FORCE = np.zeros((NUM_POINT, 3))
  for i, TRI in enumerate(FEMTri):
      for j in TRI:
          NODAL_FORCE[j] += FEM_TRI_FORCE[i] / 3.0

  # Convert nodal force → nodal traction để lưu vào xdmf
  # (dùng tributary area)
  TRIBUTARY_AREA = np.zeros(NUM_POINT)
  for i, TRI in enumerate(FEMTri):
      for j in TRI:
          TRIBUTARY_AREA[j] += FEM_TRI_AREA[i] / 3.0

  NODAL_TRACTION = NODAL_FORCE / (TRIBUTARY_AREA[:,np.newaxis] + 1e-12)

  MappedMesh = meshio.Mesh(
      points=FEMPts,
      cells=[("triangle", FEMTri)],
      point_data={"traction": NODAL_TRACTION}
  )
  meshio.write(outfile, MappedMesh)
  print("Saved:", outfile)
  return NODAL_FORCE
  
def import_constant_pressure_traction(P_INTERNAL, MESHFILE, XDMFFILE, VERBOSE=False):
  mesh = meshio.read(MESHFILE)
  POINTS    = mesh.points
  TRIANGLES = mesh.cells_dict["triangle"]

  if VERBOSE:
      print(f"Points    : {POINTS.shape}")
      print(f"Triangles : {TRIANGLES.shape}")

  # ── Normale sortante = direction radiale (x, y, 0) normalisée ────────────
  XY      = POINTS[:, :2]                        # (N, 2)
  R       = np.linalg.norm(XY, axis=1, keepdims=True)
  R       = np.where(R < 1e-15, 1.0, R)          # évite division par zéro
  NORMALS = np.hstack([XY / R, np.zeros((len(POINTS), 1))])  # (N, 3)

  # ── Traction = p * n  (pression interne → force vers l'extérieur) ────────
  TRACTION    = P_INTERNAL * NORMALS             # (N, 3)
  TRACTIONMAG = np.linalg.norm(TRACTION, axis=1)

  if VERBOSE:
      print(f"p constant         : {P_INTERNAL} Pa")
      print(f"traction magnitude : {TRACTIONMAG.min():.4f} -> {TRACTIONMAG.max():.4f} Pa")

  # ── Écriture xdmf ────────────────────────────────────────────────────────
  out_mesh = meshio.Mesh(
      points=POINTS,
      cells=[("triangle", TRIANGLES)],
      point_data={
          "normals" : NORMALS,
          "traction": TRACTION,
      }
  )
  meshio.write(XDMFFILE, out_mesh)

  if VERBOSE:
      print(f"Traction écrite dans : {XDMFFILE}")

  return XDMFFILE
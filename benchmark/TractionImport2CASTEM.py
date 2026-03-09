import meshio
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from FOAMImport import map_traction, map_traction_conservative

MESHFILE = "wing.msh"
FOAMFILE = "FOAMData.xdmf"
MAPFILE  = "MappedTraction.xdmf"
# map_traction(FOAMFILE, MESHFILE, MAPFILE)
map_traction_conservative(FOAMFILE, MESHFILE, MAPFILE)

MAPPEDMESH = meshio.read(MAPFILE)
MAPPEDPOINT = MAPPEDMESH.points
MAPPEDTRIANGLE = MAPPEDMESH.cells_dict["triangle"]
MAPPEDTRACTION = MAPPEDMESH.point_data["traction"]
 
NUM_POINT = len(MAPPEDPOINT)
NODAL_FORCE = np.zeros((NUM_POINT, 3))

print("start lump force from distributed traction")

for TRI in MAPPEDTRIANGLE:
  P0, P1, P2 = MAPPEDPOINT[TRI]
  AREA = 0.5 * np.linalg.norm(np.cross(P1 - P0, P2 - P0))
  AVG_TRACTION = MAPPEDTRACTION[TRI].mean(axis=0)
  TRIANGLE_FORCE = AREA * AVG_TRACTION
  for i in TRI:
    NODAL_FORCE[i] += TRIANGLE_FORCE / 3.0

print("start lump force from distributed traction")

FEM_FORCE = np.sum(NODAL_FORCE, axis=0)

from scipy.spatial import cKDTree
CASTEMMesh  = meshio.read("wing.med", file_format="med")
CASTEMPoint = CASTEMMesh.points


TREE = cKDTree(MAPPEDPOINT)
DIST, IDX = TREE.query(CASTEMPoint)

print("Max distance FEM→CASTEM [m]:", DIST.max())
print("Mean distance FEM→CASTEM [m]:", DIST.mean())

CASTEMForce = NODAL_FORCE[IDX]

print("Force range [N]:")
print("Fx:", CASTEMForce[:,0].min(), "->", CASTEMForce[:,0].max())
print("Fy:", CASTEMForce[:,1].min(), "->", CASTEMForce[:,1].max())
print("Fz:", CASTEMForce[:,2].min(), "->", CASTEMForce[:,2].max())
print("Total force [N]:", CASTEMForce.sum(axis=0))
print("Reference  [N]: [184, -4294, 17846]")

data = np.column_stack([CASTEMPoint, CASTEMForce])
np.savetxt(
    "CASTEMForce.csv",
    data,
    delimiter=";",
    header="x;y;z;fx;fy;fz",
    comments=""
)
print("Saved: CASTEMForce.csv")
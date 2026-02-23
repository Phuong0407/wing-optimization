import meshio
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

FOAMDATAFILE = "ExtractedFOAMData.xdmf"
FOAMMesh     = meshio.read(FOAMDATAFILE)
FOAMPts      = FOAMMesh.points
FOAMTraction = FOAMMesh.point_data["traction"]

FEMMESHFILE  = "FEMMesh.xdmf"
FEMMesh      = meshio.read(FEMMESHFILE)
FEMPts       = FEMMesh.points

interp_x = NearestNDInterpolator(FOAMPts, FOAMTraction[:, 0])
interp_y = NearestNDInterpolator(FOAMPts, FOAMTraction[:, 1])
interp_z = NearestNDInterpolator(FOAMPts, FOAMTraction[:, 2])

MappedTraction = np.column_stack([
  interp_x(FEMPts),
  interp_y(FEMPts),
  interp_z(FEMPts)
])

MappedMesh = meshio.Mesh(
  points=FEMPts,
  cells=[("triangle",FEMMesh.cells_dict["triangle"])],
  point_data={"traction": MappedTraction}
)
meshio.write("MappedTraction.xdmf", MappedMesh)
print("Saved: MappedTraction.xdmf")

import meshio
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RBFInterpolator

FOAMDATAFILE = "FOAMData.xdmf"
FOAMMESH     = meshio.read(FOAMDATAFILE)
FOAMPOINT    = FOAMMESH.points
FOAMTRACTION = FOAMMESH.point_data["traction"]

FEMMESHFILE  = "2ndwing.xdmf"
FEMMESH      = meshio.read(FEMMESHFILE)
FEMPOINT       = FEMMESH.points

# interp_x = NearestNDInterpolator(FOAMPOINT, FOAMTraction[:, 0])
# interp_y = NearestNDInterpolator(FOAMPOINT, FOAMTraction[:, 1])
# interp_z = NearestNDInterpolator(FOAMPOINT, FOAMTraction[:, 2])
# MappedTraction = np.column_stack([interp_x(FEMPOINT), interp_y(FEMPOINT), interp_z(FEMPOINT)])

interp = RBFInterpolator(FOAMPOINT,FOAMTRACTION,kernel='thin_plate_spline',neighbors=20)
MappedTraction = interp(FEMPOINT)

MappedMesh = meshio.Mesh(
    points=FEMPOINT,
    cells=[("triangle6", FEMMESH.cells_dict["triangle6"])],
    point_data={"traction": MappedTraction}
)
meshio.write("2ndMappedTraction.xdmf", MappedMesh)

print(f"Points   : {FEMPOINT.shape}")
print(f"Elements : {FEMMESH.cells_dict['triangle6'].shape}")
print("Saved: 2ndMappedTraction.xdmf")
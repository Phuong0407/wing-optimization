import meshio
from scipy.spatial import cKDTree
import numpy as np

foam = meshio.read("../data/FOAM/ExtractedFOAMData.xdmf")
foam_pts   = foam.points
foam_tract = foam.point_data["traction"]

print(f"FOAM pts range   : {foam_pts.min(axis=0)} -> {foam_pts.max(axis=0)}")
print(f"Traction range   : {foam_tract.min():.4e} -> {foam_tract.max():.4e} Pa")
print(f"Traction mean mag: {np.linalg.norm(foam_tract, axis=1).mean():.4e} Pa")

# Sau khi map vào dolfinx
print(f"f_traction array range: {f_traction.x.array.min():.4e} -> {f_traction.x.array.max():.4e}")

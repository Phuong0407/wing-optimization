import meshio
import numpy as np

foam = meshio.read("../../data/FOAM/ExtractedFOAMData.xdmf")
fem  = meshio.read("../../data/FORCE/MappedTraction.xdmf")

# Tổng lực từ FOAM (trên mesh FOAM gốc)
foam_pts = foam.points
foam_tri = foam.cells_dict["triangle"]
foam_T   = foam.point_data["traction"]

area_foam = np.zeros(len(foam_pts))
for t in foam_tri:
    v0,v1,v2 = foam_pts[t[0]], foam_pts[t[1]], foam_pts[t[2]]
    a = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0))
    area_foam[t] += a / 3.

F_foam = (foam_T * area_foam[:,np.newaxis]).sum(axis=0)
print(f"Luc tren mesh FOAM goc : Fx={F_foam[0]:.3e}, Fy={F_foam[1]:.3e}, Fz={F_foam[2]:.3e}")

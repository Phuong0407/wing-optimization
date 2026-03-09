import meshio
import numpy as np

foam = meshio.read("../FOAMData.xdmf")
mapped = meshio.read("MappedTraction.xdmf")

t1 = foam.point_data["traction"]
t2 = mapped.point_data["traction"]

print(f"FOAMData    : {t1.min():.4f} -> {t1.max():.4f} Pa, mean={np.linalg.norm(t1,axis=1).mean():.4f}")
print(f"MappedTract : {t2.min():.4f} -> {t2.max():.4f} Pa, mean={np.linalg.norm(t2,axis=1).mean():.4f}")

# convert_to_med.py
import meshio

m = meshio.read("wing.msh")

# Scale mm -> m (giống WingFEMMeshExporter.py của bạn)
m.points = m.points * 1e-3

meshio.write("../data/CAD/wing.med", m)
print("Exported wing.med")

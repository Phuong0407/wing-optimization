import meshio
import numpy as np

INDICES = [0, 750, 1500, 2250, 3000]

for indices in INDICES:
    MESHFILE = f"rib{indices}.msh"
    MESH  = meshio.read(MESHFILE)
    POINT = MESH.points * 1e-3

    mesh_out = meshio.Mesh(
    points=POINT,
    cells=[("triangle", MESH.cells_dict["triangle"])]
    )

    print("==================== EXPORT MESH TO STL ====================")
    bbox = np.ptp(POINT, axis=0)
    print("bbox =", bbox)
    print("number of points =", len(POINT))
    meshio.write(f"wrib{indices}.stl", mesh_out, file_format="stl")
    print("write mesh file to stl file: DONE")
    print("==================== EXPORT MESH TO STL ====================")
    
MESHFILE = "skin.msh"
MESH  = meshio.read(MESHFILE)
POINT = MESH.points * 1e-3

mesh_out = meshio.Mesh(
points=POINT,
cells=[("triangle", MESH.cells_dict["triangle"])]
)

print("==================== EXPORT MESH TO STL ====================")
bbox = np.ptp(POINT, axis=0)
print("bbox =", bbox)
print("number of points =", len(POINT))
meshio.write("skin.stl", mesh_out, file_format="stl")
print("write mesh file to stl file: DONE")
print("==================== EXPORT MESH TO STL ====================")

MESHFILE = "spar_main.msh"
MESH  = meshio.read(MESHFILE)
POINT = MESH.points * 1e-3

mesh_out = meshio.Mesh(
points=POINT,
cells=[("triangle", MESH.cells_dict["triangle"])]
)

print("==================== EXPORT MESH TO STL ====================")
bbox = np.ptp(POINT, axis=0)
print("bbox =", bbox)
print("number of points =", len(POINT))
meshio.write("spar_main.stl", mesh_out, file_format="stl")
print("write mesh file to stl file: DONE")
print("==================== EXPORT MESH TO STL ====================")

MESHFILE = "spar_tail.msh"
MESH  = meshio.read(MESHFILE)
POINT = MESH.points * 1e-3

mesh_out = meshio.Mesh(
points=POINT,
cells=[("triangle", MESH.cells_dict["triangle"])]
)

print("==================== EXPORT MESH TO STL ====================")
bbox = np.ptp(POINT, axis=0)
print("bbox =", bbox)
print("number of points =", len(POINT))
meshio.write("spar_tail.stl", mesh_out, file_format="stl")
print("write mesh file to stl file: DONE")
print("==================== EXPORT MESH TO STL ====================")
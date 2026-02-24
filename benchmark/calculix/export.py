import meshio
import numpy as np
import os



ROOT_TAG          = 11        # gmsh physical tag for the clamped root
MATERIAL_NAME     = "STEEL"
THICKNESS         = 6e-3
E_MODULUS         = 210e9
POISSON_RATIO     = 0.3



FEM_MESH_FILE     = "../../data/CAD/FEMMesh.xdmf"
TRACTION_FILE     = "../../data/FORCE/MappedTraction.xdmf"
OUTPUT_DIR        = "."

print("Reading FEM mesh ...")
FEM_MESH  = meshio.read(FEM_MESH_FILE)
MeshPoint = FEM_MESH.points
Triangle  = FEM_MESH.cells_dict["triangle"]
print("Finish reading FEM mesh ...")



print("Reading mapped traction ...")
TRACTION_MESH = meshio.read(TRACTION_FILE)
Traction      = TRACTION_MESH.point_data["traction"]
assert Traction.shape[0] == MeshPoint.shape[0], \
    "Traction and mesh node counts do not match! Re-run TractionMapping.py"
print("Finish reading mapped traction ...")


# Identify root nodes (physical tag = ROOT_TAG)
phys          = FEM_MESH.cell_data_dict["gmsh:physical"]["triangle"]
ROOT_ELEMS    = np.where(phys == ROOT_TAG)[0]
ROOT_NODE_IDS = np.unique(Triangle[ROOT_ELEMS].flatten())   # zero-indexed
print(f"  Root nodes (tag {ROOT_TAG}): {len(ROOT_NODE_IDS)}")

# Compute nodal areas, this is for nodal force
def compute_nodal_areas(pts, tri):
    """Distribute element area equally to the 3 corner nodes."""
    nodal_area = np.zeros(len(pts))
    for t in tri:
        v0, v1, v2 = MeshPoint[t[0]], pts[t[1]], pts[t[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        nodal_area[t] += area / 3.0
    return nodal_area


print("Computing nodal areas ...")
nodal_area  = compute_nodal_areas(MeshPoint, Triangle)
nodal_force = Traction * nodal_area[:, np.newaxis]
total_force = nodal_force.sum(axis=0)
print(f"  Total applied force: Fx={total_force[0]:.3e} N, "
      f"Fy={total_force[1]:.3e} N, Fz={total_force[2]:.3e} N")

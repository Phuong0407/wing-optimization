import numpy as np
import meshio
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# Load OpenFOAM traction data
msh_FOAM      = meshio.read("wing_surface.xdmf")
FOAM_pts      = msh_FOAM.points                      # (N_FOAM, 3)
FOAM_traction = msh_FOAM.point_data["traction"]      # (N_FOAM, 3)

# Load GMSH structural mesh
msh_struct = meshio.read("wing_skin_fenics.xdmf")
struct_pts  = msh_struct.points                   # (N_struct, 3)

# Interpolate traction từ FOAM mesh → structural mesh
# Dùng nearest neighbor (robust hơn cho surface 3D)
interp_x = NearestNDInterpolator(FOAM_pts, FOAM_traction[:, 0])
interp_y = NearestNDInterpolator(FOAM_pts, FOAM_traction[:, 1])
interp_z = NearestNDInterpolator(FOAM_pts, FOAM_traction[:, 2])

traction_mapped = np.column_stack([
    interp_x(struct_pts),
    interp_y(struct_pts),
    interp_z(struct_pts)
])

print(f"Traction mapped: {traction_mapped.shape}")
print(f"Traction mag max: {np.linalg.norm(traction_mapped, axis=1).max():.2f} Pa")

# Save mesh + traction cho FEniCS
mesh_final = meshio.Mesh(
    points=struct_pts,
    cells=[("triangle", msh_struct.cells_dict["triangle"])],
    point_data={"traction": traction_mapped}
)
meshio.write("wing_final.xdmf", mesh_final)
print("Saved: wing_final.xdmf")

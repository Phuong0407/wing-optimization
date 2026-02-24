from dolfinx import io
from mpi4py import MPI
import numpy as np
import meshio

# Read mesh with dolfinx
with io.XDMFFile(MPI.COMM_WORLD, "../data/CAD/FEMMesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

fenics_points = mesh.geometry.x

# Read mesh with meshio
m = meshio.read("../data/CAD/FEMMesh.xdmf")

# Compare
same = np.allclose(m.points, fenics_points)

print("Same ordering:", same)
print("Max difference:", np.max(np.abs(m.points - fenics_points)))


print("meshio  points range:", m.points.min(axis=0), m.points.max(axis=0))
print("dolfinx points range:", fenics_points.min(axis=0), fenics_points.max(axis=0))
print("meshio  n_points:", m.points.shape)
print("dolfinx n_points:", fenics_points.shape)

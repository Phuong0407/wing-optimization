from mpi4py import MPI
from dolfinx.io import gmsh

def load_mesh(filename, gdim=3):
  MESH = gmsh.read_from_msh(
    filename,
    comm=MPI.COMM_WORLD,
    gdim=gdim
  )
  DOMAIN = MESH.mesh
  DOMAIN.geometry.x[:] *= 1E-3
  return DOMAIN, MESH.cell_tags, MESH.facet_tags
from dolfinx import fem
from dolfinx.io import gmsh
from mpi4py import MPI
import ufl
import basix

from helper import local_frame

class WingModel:
    def __init__(self, mesh_file, comm, meshunit = "mm"):
        self.comm       = comm
        MESHREADER      = gmsh.read_from_msh(mesh_file, self.comm, rank=0, gdim=3)
        self.mesh       = MESHREADER.mesh
        self.cell_tags  = MESHREADER.cell_tags
        self.facet_tags = MESHREADER.facet_tags
        self.gdim       = self.mesh.geometry.dim
        self.tdim       = self.mesh.topology.dim
        self.fdim       = self.tdim - 1
        if meshunit == "mm":
            self.mesh.geometry.x[:] *= 1E-3

        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)

    def function_space(self):
        Ue = basix.ufl.element("P",  self.mesh.basix_cell(), 2, shape=(self.gdim,))
        Re = basix.ufl.element("CR", self.mesh.basix_cell(), 1, shape=(self.gdim,))
        self.V = fem.functionspace(self.mesh, basix.ufl.mixed_element([Ue, Re]))

        self.v = fem.Function(self.V)
        self.u, self.theta = ufl.split(self.v)

        self.v_ = ufl.TestFunction(self.V)
        self.u_, self.theta_ = ufl.split(self.v_)

        self.dv = ufl.TrialFunction(self.V)
        
    def local_frame(self):
        self.e1, self.e2, self.e3 = local_frame(self.mesh)
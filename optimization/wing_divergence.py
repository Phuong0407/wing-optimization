from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np

class WingDivergenceSolver:

    def __init__(self, model):

        self.m = model
        self.comm = MPI.COMM_WORLD

    def solve(self, design):

        self.m.build_materials(design)

        K = self.m.assemble_stiffness()
        Ka = self.m.assemble_aero_stiffness()

        eps = SLEPc.EPS().create(self.comm)

        eps.setOperators(K, Ka)

        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

        eps.setDimensions(5)

        eps.solve()

        lam = eps.getEigenvalue(0)

        q_div = lam.real

        rho = 1.225

        V_div = np.sqrt(2*q_div/rho)

        if self.comm.rank == 0:
            print("Divergence dynamic pressure =", q_div)
            print("Divergence speed =", V_div, "m/s")

        return V_div
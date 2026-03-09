from mpi4py import MPI
from slepc4py import SLEPc
import numpy as np


class WingModalSolver:

    def __init__(self, model, n_modes=1):
        self.m = model
        self.comm = MPI.COMM_WORLD
        self.n_modes = n_modes

    def solve(self, design):

        self.m.build_materials(design)

        K = self.m.assemble_stiffness()
        M = self.m.assemble_mass()

        if self.comm.rank == 0:
            print("K nnz =", K.getInfo()["nz_used"])
            print("M nnz =", M.getInfo()["nz_used"])

        eps = SLEPc.EPS().create(self.comm)

        eps.setOperators(K, M)

        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        eps.setDimensions(self.n_modes)

        # ===== QUAN TRỌNG =====
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(0.0)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)

        eps.setFromOptions()

        eps.solve()

        nconv = eps.getConverged()

        freqs = []

        if self.comm.rank == 0:
            print("\n===== Modal frequencies =====")

        for i in range(min(nconv, self.n_modes)):

            lam = eps.getEigenvalue(i)

            omega = np.sqrt(lam.real)
            f = omega / (2*np.pi)

            freqs.append(f)

            if self.comm.rank == 0:
                print(f"Mode {i+1:2d}: {f:8.3f} Hz")

        return freqs
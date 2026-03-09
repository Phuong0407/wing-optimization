import numpy as np
from mpi4py import MPI

from wing_static import WingStaticSolver


class WingOptimizer:

    def __init__(self, model, debug=True):

        self.model = model
        self.solver = WingStaticSolver(model)
        self.comm = MPI.COMM_WORLD

        self.debug = debug
        
        # constraint limits
        self.u_limit = 0.15     # meters
        self.penalty = 1e6
        self.eval_count = 0
        self.eval_limit = 20 if self.debug else 10000


    def evaluate(self, x):
        self.eval_count += 1
        if self.eval_count > self.eval_limit:
            raise RuntimeError("Debug evaluation limit reached")

        # x now contains t_skin, t_spar, and then individual t_ribs
        t_skin = x[0]
        t_spar = x[1]
        t_ribs = x[2:] # This will be a slice for all rib thicknesses

        design = dict(
            t_skin=t_skin,
            t_spar=t_spar,
            t_ribs=t_ribs, # Pass as a list/tuple
            skin_layup=[0,45,-45,90,90,-45,45,0],
            spar_layup=[0,45,-45,90,90,-45,45,0],
            rib_layup =[0,45,-45,90,90,-45,45,0],
        )

        res = self.solver.solve(design, export=False)

        mass = res.mass
        u = res.u_max

        FI = max(res.FI.values())

        cost = mass

        if u > self.u_limit:
            cost += self.penalty*(u-self.u_limit)**2

        if FI > 1:
            cost += self.penalty*(FI-1)**2

        if self.comm.rank == 0:
            rib_str = ", ".join([f"{tr:.3e}" for tr in t_ribs])
            print(
                f"x=(t_skin={t_skin:.3e}, t_spar={t_spar:.3e}, t_ribs=[{rib_str}])  mass={mass:.3f}  u={u:.4f}  FI={FI:.3f}  cost={cost:.3f}"
            )

        return cost


    def optimize(self):

        # initial guess
        num_ribs = len(self.model.TAG_RIBS)
        initial_t_rib = 0.75e-3 # Default initial thickness for each rib
        x = np.array([0.75e-3, 0.75e-3] + [initial_t_rib] * num_ribs)

        step = 0.1e-3 if self.debug else 0.25e-3

        best_cost = self.evaluate(x)

        max_iter = 3 if self.debug else 20

        for _ in range(max_iter):

            improved = False

            for i in range(len(x)): # Iterate over all design variables

                for s in [-step, step]:

                    x_trial = x.copy()
                    x_trial[i] += s

                    if x_trial[i] <= 0:
                        continue

                    cost = self.evaluate(x_trial)

                    if cost < best_cost:
                        x = x_trial
                        best_cost = cost
                        improved = True

            if not improved:
                step *= 0.5

            if step < 0.05e-3:
                break

        return x, best_cost
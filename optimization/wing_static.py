# wing_static.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import ufl
from dolfinx.fem.petsc import NonlinearProblem


@dataclass
class StaticResult:
    solution: Any          # fem.Function on V (mixed)
    mass: float
    u_max: float
    FI: Dict[str, float]


class WingStaticSolver:
    """
    Static solver wrapper: uses an existing WingComputationModel instance,
    but keeps "analysis logic" outside the model.

    Expected interface from model:
      - build_materials(design)
      - build_internal_form() -> (a_int, mass_form_scalar)
      - build_external_load() -> L_ext
      - bc_extras dict possibly containing spring_form/moment_form
      - model.v (unknown function), model.dv (trial), model.V, model.mesh...
      - BCS list
      - petsc_options dict
      - postprocess(export=True) -> dict with keys mass, u_max, FI
    """

    def __init__(self, model):
        self.m = model

    def _build_residual_and_tangent(self):
        a_int = self.m.build_internal_form()
        L_ext = self.m.build_external_load()
        a_int, L_ext = self.m.apply_bc_contributions(a_int, L_ext)
        # BC “extra” contributions (spring/moment) are currently added in solve() :contentReference[oaicite:2]{index=2}
        if isinstance(self.m.bc_extras, dict):
            a_int += self.m.bc_extras.get("spring_form", 0)
            L_ext += self.m.bc_extras.get("moment_form", 0)

        residual = a_int - L_ext
        tangent = ufl.derivative(residual, self.m.model.v, self.m.model.dv)
        return residual, tangent

    def solve(self, design: Dict[str, Any], export: bool = True) -> StaticResult:
        # 1) materials
        self.m.build_materials(design)

        # 2) forms
        residual, tangent = self._build_residual_and_tangent()

        # 3) nonlinear solve (Newton)
        problem = NonlinearProblem(
            residual,
            self.m.model.v,
            bcs=self.m.BCS,
            J=tangent,
            petsc_options_prefix="wing",
            petsc_options=self.m.petsc_options,
        )
        problem.solve()  # uses PETSc SNES options already in your model :contentReference[oaicite:3]{index=3}

        # 4) postprocess (kept as-is)
        out = self.m.postprocess(export=export)

        return StaticResult(
            solution=self.m.model.v,
            mass=float(out["mass"]),
            u_max=float(out["u_max"]),
            FI=dict(out["FI"]),
        )
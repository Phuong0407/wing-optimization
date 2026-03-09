import ufl
from dolfinx import fem
from ShellMaterial import ShellMaterial

class IsotropicShell(ShellMaterial):
  def __init__(self, domain, thickness, E, nu):
    self.h  = fem.Constant(domain, thickness)
    self.E  = fem.Constant(domain, E)
    self.nu = fem.Constant(domain, nu)

    self.mu = self.E / (2 * (1 + self.nu))
    lmbda   = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    self.lmbda_ps = 2 * lmbda * self.mu / (lmbda + 2 * self.mu)

  def _plane_stress(self, e):
    tdim = e.ufl_shape[0]
    return (
      self.lmbda_ps * ufl.tr(e) * ufl.Identity(tdim) + 2 * self.mu * e
    )

  def membrane_stress(self, eps):
    return self.h * self._plane_stress(eps)

  def bending_stress(self, kappa):
    return self.h**3 / 12 * self._plane_stress(kappa)

  def shear_stress(self, gamma):
      return self.mu * self.h * gamma
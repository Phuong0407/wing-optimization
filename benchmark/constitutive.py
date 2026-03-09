import ufl
from dolfinx import fem

def ShellLinearElastic(shell, thickness, Young, Poisson):
  thickness = fem.Constant(shell, thickness)
  E = fem.Constant(shell, Young)
  nu = fem.Constant(shell, Poisson)
  lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
  mu = E / 2 / (1 + nu)
  lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)
  return E, nu, lmbda, mu, lmbda_ps

def plane_stress_elasticity(e, mu, lmbda_ps, tdim):
  return lmbda_ps * ufl.tr(e) * ufl.Identity(tdim) + 2 * mu * e

def ShellStress(eps, kappa, gamma, thickness, mu):
  N = thickness * plane_stress_elasticity(eps)
  M = thickness**3 / 12 * plane_stress_elasticity(kappa)
  Q = mu * thickness * gamma
"""
dst_only.py
═══════════════════════════════════════════════════════════════════════
Minimal MITC3 / DST shell element for Reissner–Mindlin shells
Lightweight version for large wing models.

• P1 displacement
• P1 rotation
• 1-point shear integration (DST trick)
• Linear solver (no SNES)
• No extra strategies

This is the fastest stable shell formulation for triangular meshes.
═══════════════════════════════════════════════════════════════════════
"""

import ufl
import basix
from types import SimpleNamespace
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem


# ════════════════════════════════════════════════════════════════════════
# LOCAL FRAME (symbolic only — no DG interpolation)
# ════════════════════════════════════════════════════════════════════════

def normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))

def local_frame(domain):
    J  = ufl.Jacobian(domain)
    t1 = ufl.as_vector([J[0, 0], J[1, 0], J[2, 0]])
    t2 = ufl.as_vector([J[0, 1], J[1, 1], J[2, 1]])
    e3 = normalize(ufl.cross(t1, t2))

    ey = ufl.as_vector([0, 1, 0])
    ez = ufl.as_vector([0, 0, 1])
    e1_trial = ufl.cross(ey, e3)
    e1 = ufl.conditional(
        ufl.lt(ufl.sqrt(ufl.dot(e1_trial, e1_trial)), 0.5),
        ez,
        normalize(e1_trial),
    )
    e2 = normalize(ufl.cross(e3, e1))
    return e1, e2, e3


# ════════════════════════════════════════════════════════════════════════
# FUNCTION SPACE — P1/P1
# ════════════════════════════════════════════════════════════════════════

def build_space(domain):
    cell = domain.basix_cell()
    gdim = domain.geometry.dim

    Ue = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
    Te = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))

    V  = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Te]))
    v  = fem.Function(V)
    dv = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)

    ndof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"[DST] DOFs: {ndof}")
    return V, v, dv, v_


# ════════════════════════════════════════════════════════════════════════
# KINEMATICS
# ════════════════════════════════════════════════════════════════════════

def hstack(vecs):
    return ufl.as_matrix([[vi[i] for i in range(len(vi))] for vi in vecs]).T

def shell_strains(u, theta, e1, e2, e3):
    P = hstack([e1, e2])

    def tgrad(w):
        return ufl.dot(ufl.grad(w), P)

    t_gu = ufl.dot(P.T, tgrad(u))
    eps  = ufl.sym(t_gu)

    beta  = ufl.cross(e3, theta)
    kappa = ufl.sym(ufl.dot(P.T, tgrad(beta)))

    gamma = tgrad(ufl.dot(u, e3)) - ufl.dot(P.T, beta)

    return eps, kappa, gamma


# ════════════════════════════════════════════════════════════════════════
# MATERIAL — isotropic only (CLT can be re-added later)
# ════════════════════════════════════════════════════════════════════════

def isotropic_material(domain, E, nu, thickness):
    h  = fem.Constant(domain, float(thickness))
    E  = fem.Constant(domain, float(E))
    nu = fem.Constant(domain, float(nu))

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu    = E / (2 * (1 + nu))
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)

    return SimpleNamespace(
        h=h, E=E, nu=nu,
        lmbda_ps=lmbda_ps, mu=mu
    )


def plane_stress(mat, e):
    I = ufl.Identity(2)
    return mat.lmbda_ps * ufl.tr(e) * I + 2 * mat.mu * e


# ════════════════════════════════════════════════════════════════════════
# WEAK FORM — MITC3 / DST
# ════════════════════════════════════════════════════════════════════════

def build_forms(domain, v, v_, mat, f_ext=None):

    u, theta = ufl.split(v)
    u_, theta_ = ufl.split(v_)

    e1, e2, e3 = local_frame(domain)

    eps,  kappa,  gamma  = shell_strains(u,  theta,  e1, e2, e3)
    eps_, kappa_, gamma_ = shell_strains(u_, theta_, e1, e2, e3)

    N = mat.h * plane_stress(mat, eps)
    M = mat.h**3 / 12.0 * plane_stress(mat, kappa)
    Q = mat.mu * mat.h * gamma

    # Full integration for membrane/bending
    dx_mb = ufl.Measure("dx", domain=domain,
                        metadata={"quadrature_degree": 2})

    # DST trick: 1-point integration for shear
    dx_s  = ufl.Measure("dx", domain=domain,
                        metadata={"quadrature_degree": 1})

    a_mb = (ufl.inner(N, eps_) +
            ufl.inner(M, kappa_)) * dx_mb

    a_s  = ufl.inner(Q, gamma_) * dx_s

    a = a_mb + a_s

    if f_ext is not None:
        L = ufl.dot(f_ext, u_) * dx_mb
    else:
        L = 0

    return a, L


# ════════════════════════════════════════════════════════════════════════
# SOLVER
# ════════════════════════════════════════════════════════════════════════

def solve(domain, V, a, L, bcs):

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    v = problem.solve()
    return v
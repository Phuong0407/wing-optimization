"""
dst_fenics.py
═════════════════════════════════════════════════════════════════════════════
Discrete Shear Triangle (DST) for Reissner-Mindlin shells in FEniCSx
─────────────────────────────────────────────────────────────────────────────

Three formulations are implemented, from simplest to most accurate:

  Strategy A — DKT  (Discrete Kirchhoff Triangle)
      Thin-plate limit.  Shear γ=0 strongly (penalty).
      No locking because shear energy is absent.
      Suitable only for t/L < 1/50.

  Strategy B — MITC3  (equivalent to DST via Selective Reduced Integration)
      P1 displacement + P1 rotation.
      Shear term integrated with 1-point (centroid) rule → locks DOF space.
      Mathematically equivalent to the original Batoz–Lardeur DST.
      RECOMMENDED for thin-to-moderately-thick shells.

  Strategy C — CR (Crouzeix-Raviart rotations)  ← already in wing.py
      P2 displacement + CR rotation.
      CR DOFs live at edge midpoints, exactly as in DST shear constraint.
      Most accurate; used in the existing project code.

─────────────────────────────────────────────────────────────────────────────
Theoretical Background
─────────────────────────────────────────────────────────────────────────────

Reissner-Mindlin weak form (3D shell):

  Find (u, θ) ∈ V such that ∀ (u*, θ*):

  ∫ [ N : ε(u*)  +  M : κ(θ*)  +  Q · γ* ] dΩ  =  ∫ f · u* dΩ

where
  ε  = sym( Pᵀ ∇u P )          membrane strain   (2×2)
  κ  = sym( Pᵀ ∇β P )          bending curvature  (2×2),  β = e3 × θ
  γ  = Pᵀ ∇(u·e3) − Pᵀ β      transverse shear   (2,)
  P  = [e1 | e2]                tangent projection (3×2)

Shear locking arises when P1/P1 elements are used because:
  dim(ker γ_h) → 0  as  t → 0  (the element cannot reproduce Kirchhoff modes)

DST fix: replace γ by its projection onto an edge-constant space
  γ_DST = Π_h γ,  with  Π_h : L²(T) → P0(edges)

This projection is implicitly achieved by 1-point (centroid) quadrature
on the shear term (Strategy B), or explicitly by using CR shape functions
whose DOFs are on edge midpoints (Strategy C).

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────

  # Run built-in patch tests and convergence study:
  python dst_fenics.py

  # Import into wing.py / skin.py and swap element strategy:
  from dst_fenics import build_shell_space, shell_strains_dst, DSTStrategy
  V, v, dv, v_ = build_shell_space(domain, strategy=DSTStrategy.MITC3)

─────────────────────────────────────────────────────────────────────────────
References
─────────────────────────────────────────────────────────────────────────────
  [1] Batoz & Lardeur (1989) IJNME 28, 533-553  — original DST paper
  [2] Bathe & Dvorkin (1985) IJNME 21, 367-383  — MITC family
  [3] Chapelle & Bathe (2003) "The FE Analysis of Shells" — theory
  [4] Bleyer (2018) dolfinx-shells demos — CR-based 3D shells
═════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import enum
from types import SimpleNamespace

import numpy as np
import ufl
import basix
from dolfinx import fem, mesh, io
from mpi4py import MPI


# ══════════════════════════════════════════════════════════════════════════════
# 1.  STRATEGY ENUM
# ══════════════════════════════════════════════════════════════════════════════

class DSTStrategy(enum.Enum):
    DKT   = "dkt"    # Discrete Kirchhoff Triangle (thin only)
    MITC3 = "mitc3"  # Selective reduced integration  ← DST equivalent
    CR    = "cr"     # Crouzeix-Raviart rotations     ← wing.py default


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOCAL FRAME (identical to wing.py)
# ══════════════════════════════════════════════════════════════════════════════

def normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))


def _local_frame_ufl(domain):
    """UFL symbolic local frame on a curved shell."""
    t  = ufl.Jacobian(domain)
    t1 = ufl.as_vector([t[0, 0], t[1, 0], t[2, 0]])
    t2 = ufl.as_vector([t[0, 1], t[1, 1], t[2, 1]])
    e3 = normalize(ufl.cross(t1, t2))
    ey = ufl.as_vector([0, 1, 0])
    ez = ufl.as_vector([0, 0, 1])
    e1_trial = ufl.cross(ey, e3)
    norm_e1  = ufl.sqrt(ufl.dot(e1_trial, e1_trial))
    e1 = ufl.conditional(ufl.lt(norm_e1, 0.5), ez, normalize(e1_trial))
    e2 = normalize(ufl.cross(e3, e1))
    return e1, e2, e3


def local_frame(domain, gdim=3):
    FRAME = _local_frame_ufl(domain)
    VT    = fem.functionspace(domain, ("DG", 0, (gdim,)))
    V0, _ = VT.sub(0).collapse()
    BASIS = [fem.Function(VT, name=f"e{i+1}") for i in range(gdim)]
    for i in range(gdim):
        expr = fem.Expression(FRAME[i], V0.element.interpolation_points)
        BASIS[i].interpolate(expr)
    return BASIS[0], BASIS[1], BASIS[2]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FUNCTION SPACE FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_shell_space(domain, strategy: DSTStrategy = DSTStrategy.MITC3):
    """
    Build the mixed function space (displacement, rotation) for the chosen
    DST strategy on a 3-D shell mesh.

    Parameters
    ----------
    domain   : dolfinx Mesh (2-D manifold embedded in 3-D)
    strategy : DSTStrategy enum

    Returns
    -------
    V   : mixed FunctionSpace
    v   : Function (unknown)
    dv  : TrialFunction
    v_  : TestFunction
    """
    cell  = domain.basix_cell()
    gdim  = domain.geometry.dim

    if strategy == DSTStrategy.DKT:
        # ── Discrete Kirchhoff Triangle ──────────────────────────────────────
        # P2 displacement + P1 rotation (same polynomial order as MITC3 in
        # bending — shear locking cured by penalty γ→0 approach)
        # Element: (P2)^3 × (P1)^3
        Ue = basix.ufl.element("Lagrange", cell, 2, shape=(gdim,))
        Te = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
        print("[DST] Strategy: DKT  — P2 disp / P1 rot + large-penalty shear")

    elif strategy == DSTStrategy.MITC3:
        # ── MITC3 / DST ───────────────────────────────────────────────────────
        # P1 displacement + P1 rotation
        # The DST shear projection is achieved by 1-point quadrature
        # on the shear energy term (see shell_strains_dst / measures below).
        # Element: (P1)^3 × (P1)^3
        Ue = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
        Te = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
        print("[DST] Strategy: MITC3 — P1 disp / P1 rot + reduced shear integration")

    elif strategy == DSTStrategy.CR:
        # ── Crouzeix-Raviart ──────────────────────────────────────────────────
        # P2 displacement + CR1 rotation
        # CR DOFs lie at edge midpoints → naturally encodes the DST discrete
        # shear constraint without any projection step.
        # This is the formulation already used in wing.py / skin.py.
        # Element: (P2)^3 × (CR1)^3
        Ue = basix.ufl.element("Lagrange",        cell, 2, shape=(gdim,))
        Te = basix.ufl.element("Crouzeix-Raviart", cell, 1, shape=(gdim,))
        print("[DST] Strategy: CR   — P2 disp / CR1 rot (wing.py default)")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    V   = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Te]))
    v   = fem.Function(V)
    dv  = ufl.TrialFunction(V)
    v_  = ufl.TestFunction(V)

    n_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"[DST] DOFs (global): {n_dofs}")
    return V, v, dv, v_


# ══════════════════════════════════════════════════════════════════════════════
# 4.  KINEMATICS (shared across strategies)
# ══════════════════════════════════════════════════════════════════════════════

def hstack(vecs):
    return ufl.as_matrix([[vi[i] for i in range(len(vi))] for vi in vecs]).T

def tangent_projection(e1, e2):
    return hstack([e1, e2])

def tangential_gradient(w, P):
    return ufl.dot(ufl.grad(w), P)

def membrane_strain(u, P):
    t_gu = ufl.dot(P.T, tangential_gradient(u, P))
    return ufl.sym(t_gu), t_gu

def bending_strain(theta, e3, P):
    beta = ufl.cross(e3, theta)
    return ufl.sym(ufl.dot(P.T, tangential_gradient(beta, P)))

def shear_strain(u, theta, e3, P):
    beta = ufl.cross(e3, theta)
    return tangential_gradient(ufl.dot(u, e3), P) - ufl.dot(P.T, beta)

def compute_drilling_strain(t_gu, theta, e3):
    return (t_gu[0, 1] - t_gu[1, 0]) / 2.0 + ufl.dot(theta, e3)


def shell_strains_dst(u, theta, e1, e2, e3):
    """
    Return all shell strains.  The returned `gamma` uses the full (un-projected)
    expression; for MITC3 the projection is achieved at the integration level
    by passing `dx_shear` (1-point quadrature) to the weak form builder.
    For DKT, a penalty parameter replaces Q entirely.
    For CR, no modification needed — the CR space is already locking-free.
    """
    P               = tangent_projection(e1, e2)
    eps, t_gu       = membrane_strain(u, P)
    kappa           = bending_strain(theta, e3, P)
    gamma           = shear_strain(u, theta, e3, P)
    drilling        = compute_drilling_strain(t_gu, theta, e3)
    return eps, kappa, gamma, drilling


# ══════════════════════════════════════════════════════════════════════════════
# 5.  INTEGRATION MEASURES
#     The DST / MITC3 trick lives entirely here.
# ══════════════════════════════════════════════════════════════════════════════

def dst_measures(domain, strategy: DSTStrategy, cell_tags=None, subdomain_tag=None):
    """
    Return (dx_membrane_bending, dx_shear) integration measures.

    For MITC3 / DST:
        dx_membrane_bending — full quadrature (degree 4 by default)
        dx_shear            — 1-point centroid quadrature  ← THE DST TRICK

    For DKT and CR:
        Both measures use full quadrature (shear is either penalised or
        already locking-free via the CR space).

    The 1-point centroid rule for triangles integrates constant functions
    exactly (P0).  Because the shear strain field in P1/P1 is linear and
    the test function is also linear, their product is quadratic — but we
    intentionally under-integrate it at P0 level.  This under-integration
    is mathematically equivalent to replacing γ by its cell-average
    (a P0 projection), which is exactly what the DST discrete shear
    constraint achieves at edge midpoints.

    Parameters
    ----------
    domain        : dolfinx Mesh
    strategy      : DSTStrategy
    cell_tags     : optional MeshTags for subdomain integrals (multi-material)
    subdomain_tag : int tag to use with cell_tags

    Returns
    -------
    dx_mb   : Measure for membrane + bending + drilling
    dx_s    : Measure for transverse shear
    """
    kw_full  = {"quadrature_degree": 4}
    # 1-point rule for triangles: centroid (1/3, 1/3), weight = 1
    kw_shear = {"quadrature_degree": 1}  # degree-1 = 1 point for simplices

    if cell_tags is not None and subdomain_tag is not None:
        dx_mb = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags,
                            subdomain_id=subdomain_tag,
                            metadata=kw_full)
        if strategy == DSTStrategy.MITC3:
            dx_s = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags,
                               subdomain_id=subdomain_tag,
                               metadata=kw_shear)
        else:
            dx_s = dx_mb
    else:
        dx_mb = ufl.Measure("dx", domain=domain, metadata=kw_full)
        if strategy == DSTStrategy.MITC3:
            dx_s = ufl.Measure("dx", domain=domain, metadata=kw_shear)
        else:
            dx_s = dx_mb

    return dx_mb, dx_s


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MATERIAL MODELS (isotropic + CLT, same as wing.py)
# ══════════════════════════════════════════════════════════════════════════════

def isotropic_material(thickness, young, poisson, domain):
    h        = fem.Constant(domain, float(thickness))
    E        = fem.Constant(domain, float(young))
    nu       = fem.Constant(domain, float(poisson))
    lmbda    = E * nu / (1 + nu) / (1 - 2 * nu)
    mu       = E / 2 / (1 + nu)
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)
    return SimpleNamespace(h=h, E=E, nu=nu, lmbda=lmbda, mu=mu,
                           lmbda_ps=lmbda_ps, kind="isotropic")


def _plane_stress_iso(mat, e):
    tdim = e.ufl_shape[0]
    return mat.lmbda_ps * ufl.tr(e) * ufl.Identity(tdim) + 2 * mat.mu * e


def to_voigt(e):
    return ufl.as_vector([e[0, 0], e[1, 1], 2.0 * e[0, 1]])


def from_voigt(v):
    return ufl.as_tensor([[v[0], v[2]], [v[2], v[1]]])


def stress_resultants(mat, eps, kappa, gamma):
    if mat.kind == "isotropic":
        N = mat.h * _plane_stress_iso(mat, eps)
        M = mat.h**3 / 12.0 * _plane_stress_iso(mat, kappa)
        Q = mat.mu * mat.h * gamma
    elif mat.kind == "clt":
        eps_v   = to_voigt(eps)
        kappa_v = to_voigt(kappa)
        N_v = ufl.dot(mat.A_ufl, eps_v) + ufl.dot(mat.B_ufl, kappa_v)
        M_v = ufl.dot(mat.B_ufl, eps_v) + ufl.dot(mat.D_ufl, kappa_v)
        N   = from_voigt(N_v)
        M   = from_voigt(M_v)
        Q   = ufl.dot(mat.As_ufl, gamma)
    else:
        raise ValueError(f"Unknown material: {mat.kind}")
    return N, M, Q


def drilling_terms(mat, domain, drilling_strain):
    h_mesh    = ufl.CellDiameter(domain)
    G_eff     = mat.mu if mat.kind == "isotropic" \
                else fem.Constant(domain, mat.G_eff)
    h         = mat.h if mat.kind == "isotropic" \
                else fem.Constant(domain, mat.H)
    stiffness = G_eff * h**3 / h_mesh**2
    return stiffness, stiffness * drilling_strain


# ══════════════════════════════════════════════════════════════════════════════
# 7.  WEAK FORM BUILDER
#     Single function that builds the bilinear/linear forms for any strategy.
# ══════════════════════════════════════════════════════════════════════════════

def build_weak_form(domain, v, v_, mat, e1, e2, e3,
                    strategy: DSTStrategy,
                    f_ext=None,
                    cell_tags=None, subdomain_tag=None,
                    dkt_penalty_factor=1e3):
    """
    Build the residual UFL form for the chosen DST strategy.

    Parameters
    ----------
    domain             : dolfinx Mesh
    v                  : Function (solution)
    v_                 : TestFunction
    mat                : material namespace (isotropic_material or clt_material)
    e1, e2, e3         : local frame Functions
    strategy           : DSTStrategy
    f_ext              : external traction Function (optional)
    cell_tags          : MeshTags for subdomain integration (optional)
    subdomain_tag      : int tag for cell_tags (optional)
    dkt_penalty_factor : penalty multiplier for DKT shear suppression

    Returns
    -------
    residual  : UFL Form
    """
    u,  theta  = ufl.split(v)
    u_, theta_ = ufl.split(v_)

    # eps,  kappa,  gamma,  drilling  = shell_strains_dst(u,  theta,  e1, e2, e3)
    # eps_, kappa_, gamma_, drilling_ = (
    #     ufl.derivative(eps,      v, v_),
    #     ufl.derivative(kappa,    v, v_),
    #     ufl.derivative(gamma,    v, v_),
    #     ufl.replace(compute_drilling_strain(
    #         ufl.dot(tangent_projection(e1, e2).T,
    #                 tangential_gradient(u, tangent_projection(e1, e2))),
    #         theta, e3
    #     ), {v: v_}),
    # )
    u, theta = ufl.split(v)
    u_, theta_ = ufl.split(v_)

    eps,  kappa,  gamma,  drilling  = shell_strains_dst(u,  theta,  e1, e2, e3)
    eps_, kappa_, gamma_, drilling_ = shell_strains_dst(u_, theta_, e1, e2, e3)
    
    # Recompute test-side drilling cleanly
    P_  = tangent_projection(e1, e2)
    t_gu_ = ufl.dot(P_.T, tangential_gradient(u_, P_))
    drilling_ = compute_drilling_strain(t_gu_, theta_, e3)

    N, M, Q        = stress_resultants(mat, eps, kappa, gamma)
    drill_k, drill_s = drilling_terms(mat, domain, drilling)

    dx_mb, dx_s = dst_measures(domain, strategy, cell_tags, subdomain_tag)

    # ── Internal virtual work ──────────────────────────────────────────────
    if strategy == DSTStrategy.DKT:
        # Shear is penalised to zero: Q_pen = penalty * h_mesh^(-2) * γ
        # This enforces the Kirchhoff constraint γ → 0 weakly.
        h_mesh  = ufl.CellDiameter(domain)
        h_val   = mat.h if mat.kind == "isotropic" \
                  else fem.Constant(domain, mat.H)
        G_val   = mat.mu if mat.kind == "isotropic" \
                  else fem.Constant(domain, mat.G_eff)
        pen     = fem.Constant(domain,
                               float(dkt_penalty_factor)) * G_val * h_val / h_mesh**2
        Q_pen   = pen * gamma       # penalised shear resultant
        gamma_q = gamma_            # test shear (no projection needed for penalty)
        a_int = (
              ufl.inner(N, eps_)
            + ufl.inner(M, kappa_)
            + ufl.inner(Q_pen, gamma_q)
            + drill_s * drilling_
        ) * dx_mb

    elif strategy == DSTStrategy.MITC3:
        # ── THE DST TRICK ─────────────────────────────────────────────────
        # Membrane + bending with full quadrature.
        # Shear with 1-point (centroid) quadrature → P0 projection of γ.
        # dx_s already carries the 1-point metadata.
        a_mb = (
              ufl.inner(N, eps_)
            + ufl.inner(M, kappa_)
            + drill_s * drilling_
        ) * dx_mb
        a_s  = ufl.inner(Q, gamma_) * dx_s
        a_int = a_mb + a_s

    elif strategy == DSTStrategy.CR:
        # CR elements are already locking-free → full quadrature everywhere.
        a_int = (
              ufl.inner(N, eps_)
            + ufl.inner(M, kappa_)
            + ufl.inner(Q, gamma_)
            + drill_s * drilling_
        ) * dx_mb

    # ── External virtual work ──────────────────────────────────────────────
    if f_ext is not None:
        L_ext = ufl.dot(f_ext, u_) * dx_mb
    else:
        L_ext = fem.Constant(domain, 0.0) * u_[0] * dx_mb

    return a_int - L_ext


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SOLVER WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def solve_shell(residual, v, bcs, tangent=None):
    """
    Solve the shell problem with a direct MUMPS/LU solver.

    Returns the converged Function v (displacement + rotation).
    """
    # from dolfinx.fem.petsc import NonlinearProblem
    # if tangent is None:
    #     dv = ufl.TrialFunction(v.function_space)
    #     tangent = ufl.derivative(residual, v, dv)

    # problem = NonlinearProblem(
    #     F=residual, u=v, bcs=bcs, J=tangent,
    #     petsc_options={
    #         "ksp_type"                  : "preonly",
    #         "pc_type"                   : "lu",
    #         "pc_factor_mat_solver_type" : "mumps",
    #         "snes_type"                 : "newtonls",
    #         "snes_rtol"                 : 1e-10,
    #         "snes_atol"                 : 1e-10,
    #         "snes_max_it"               : 20,
    #     },
    # )
    
    from dolfinx.fem.petsc import LinearProblem

    problem = LinearProblem(a, L, bcs=bcs,
                        petsc_options={
                            "ksp_type": "preonly",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                        })
    v = problem.solve()

    # problem.solve()
    reason = problem.solver.getConvergedReason()
    n_it   = problem.solver.getIterationNumber()
    assert reason > 0, f"[SNES] Did not converge (reason {reason})"
    print(f"[SNES] Converged in {n_it} iteration(s),  reason = {reason}")
    problem.solver.destroy()
    return v


# ══════════════════════════════════════════════════════════════════════════════
# 9.  VALIDATION — PATCH TEST + CONVERGENCE ON A SQUARE PLATE
#
#     Classical benchmark: simply-supported square plate under uniform pressure.
#     Analytical (Navier) solution for thin plate:
#         w_max = 0.00406 q L^4 / (D)     (D = bending stiffness)
#
#     We compare all three strategies against the reference.
# ══════════════════════════════════════════════════════════════════════════════

def _make_square_plate(nx, gdim=3):
    """
    Unit square plate mesh, embedded in 3-D (z=0 plane).
    """
    msh2d = mesh.create_unit_square(
        MPI.COMM_WORLD, nx, nx,
        cell_type=mesh.CellType.triangle,
        ghost_mode=mesh.GhostMode.shared_facet,
    )
    # Lift to 3-D by appending z=0 column
    coords2d = msh2d.geometry.x
    coords3d = np.hstack([coords2d[:, :2],
                          np.zeros((coords2d.shape[0], 1))])
    msh3d = mesh.create_mesh(
        MPI.COMM_WORLD,
        msh2d.topology.connectivity(2, 0).array.reshape(-1, 3),
        coords3d,
        ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1,
                                   shape=(3,)))
    )
    return msh3d


def _simply_supported_bcs(V, domain):
    """
    Clamp the full boundary (conservative for patch test).
    Returns list of DirichletBC.
    """
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    Vu, _  = V.sub(0).collapse()
    Vt, _  = V.sub(1).collapse()
    dofs_u = fem.locate_dofs_topological((V.sub(0), Vu), fdim, boundary_facets)
    dofs_t = fem.locate_dofs_topological((V.sub(1), Vt), fdim, boundary_facets)
    uD     = fem.Function(Vu);  uD.x.array[:] = 0.0
    tD     = fem.Function(Vt);  tD.x.array[:] = 0.0
    return [
        fem.dirichletbc(uD, dofs_u, V.sub(0)),
        fem.dirichletbc(tD, dofs_t, V.sub(1)),
    ]


def convergence_study(strategies=None, n_list=None, t=0.01):
    """
    Run a convergence study for all strategies on a simply-supported square plate.

    Plate: L=1 m, uniform pressure q=1 Pa, t=0.01 m (thin plate).
    Analytical max deflection (thin plate theory):
        w_ref = 0.00406 * q * L^4 / D

    Parameters
    ----------
    strategies : list of DSTStrategy  (default: all three)
    n_list     : list of mesh sizes   (default: [4, 8, 16, 32])
    t          : plate thickness [m]

    Returns
    -------
    results : dict  strategy -> list of (ndof, w_max, error_%)
    """
    if strategies is None:
        strategies = list(DSTStrategy)
    if n_list is None:
        n_list = [4, 8, 16, 32]

    # Material
    E, nu = 1e6, 0.3
    D     = E * t**3 / (12 * (1 - nu**2))
    q     = 1.0
    w_ref = 0.00406 * q * 1.0**4 / D    # analytical
    print(f"\n{'═'*65}")
    print(f"  CONVERGENCE STUDY — simply-supported square plate")
    print(f"  t={t:.3f} m   E={E:.0f} Pa   nu={nu}   D={D:.4e} N·m")
    print(f"  Analytical w_max = {w_ref:.6e} m")
    print(f"{'═'*65}")

    results = {s: [] for s in strategies}

    for strategy in strategies:
        print(f"\n  ── Strategy: {strategy.value.upper()} ──────────────────────")
        for nx in n_list:
            domain = _make_square_plate(nx)
            mat    = isotropic_material(t, E, nu, domain)

            V, v, dv, v_ = build_shell_space(domain, strategy)
            e1, e2, e3   = local_frame(domain, gdim=3)
            bcs          = _simply_supported_bcs(V, domain)

            # Uniform pressure in -z direction
            VF   = fem.functionspace(domain, ("Lagrange", 1, (3,)))
            fext = fem.Function(VF)
            fext.interpolate(lambda x: np.vstack([
                np.zeros_like(x[0]),
                np.zeros_like(x[0]),
                -q * np.ones_like(x[0]),
            ]))

            residual = build_weak_form(
                domain, v, v_, mat, e1, e2, e3,
                strategy=strategy, f_ext=fext,
            )
            tangent  = ufl.derivative(residual, v, dv)
            v        = solve_shell(residual, v, bcs, tangent)

            # Extract max z-displacement
            disp = v.sub(0).collapse()
            arr  = disp.x.array.reshape(-1, 3)
            w_max = float(MPI.COMM_WORLD.allreduce(
                np.abs(arr[:, 2]).max(), op=MPI.MAX
            ))
            ndof  = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
            err   = abs(w_max - w_ref) / w_ref * 100
            results[strategy].append((ndof, w_max, err))
            print(f"    nx={nx:3d}  DOFs={ndof:8d}  "
                  f"w_max={w_max:.4e}  err={err:6.2f}%  "
                  f"(ref={w_ref:.4e})")

    # Summary table
    print(f"\n{'═'*65}")
    print("  SUMMARY TABLE — error (%) vs mesh refinement")
    print(f"  {'Strategy':<10}", end="")
    for nx in n_list:
        print(f"  nx={nx:2d}  ", end="")
    print()
    for strategy, data in results.items():
        print(f"  {strategy.value.upper():<10}", end="")
        for _, _, err in data:
            print(f"  {err:6.2f}%  ", end="")
        print()
    print(f"{'═'*65}\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 10.  HOW TO PLUG INTO wing.py / skin.py
# ══════════════════════════════════════════════════════════════════════════════

INTEGRATION_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║          HOW TO USE DST STRATEGIES IN wing.py / skin.py                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STEP 1 — Import this module at the top of wing.py                          ║
║  ─────────────────────────────────────────────────                          ║
║  from dst_fenics import (                                                   ║
║      DSTStrategy, build_shell_space,                                        ║
║      shell_strains_dst, build_weak_form, dst_measures,                      ║
║      solve_shell,                                                           ║
║  )                                                                          ║
║                                                                              ║
║  STEP 2 — Choose a strategy (one line change)                               ║
║  ────────────────────────────────────────────                               ║
║  STRATEGY = DSTStrategy.MITC3  # or .DKT or .CR                            ║
║                                                                              ║
║  STEP 3 — Build the function space                                          ║
║  ─────────────────────────────────                                          ║
║  V, v, dv, v_ = build_shell_space(DOMAIN, strategy=STRATEGY)               ║
║  u, theta      = ufl.split(v)                                               ║
║  u_, theta_    = ufl.split(v_)                                              ║
║                                                                              ║
║  STEP 4 — Build the weak form (replaces the manual a_int + L_ext block)    ║
║  ──────────────────────────────────────────────────────────────────────     ║
║  residual = build_weak_form(                                                ║
║      DOMAIN, v, v_, MAT_SKIN, E1, E2, E3,                                  ║
║      strategy  = STRATEGY,                                                  ║
║      f_ext     = FTraction,                                                 ║
║      cell_tags = CELL_TAGS,                                                 ║
║      subdomain_tag = TAG_SKIN,                                              ║
║  )                                                                          ║
║                                                                              ║
║  STEP 5 — Solve (unchanged)                                                 ║
║  ─────────────────────────                                                  ║
║  tangent = ufl.derivative(residual, v, dv)                                  ║
║  v       = solve_shell(residual, v, BCS, tangent)                           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ELEMENT COMPARISON                                                          ║
║  ─────────────────                                                           ║
║  Strategy │ Disp  │ Rot │ DOFs/cell │ Anti-locking mechanism                ║
║  DKT      │  P2   │  P1 │    27     │ Penalty enforces γ → 0               ║
║  MITC3    │  P1   │  P1 │    18     │ 1-pt shear quadrature (= DST proj.)  ║
║  CR       │  P2   │  CR │    33     │ CR edge-DOFs = DST edge constraints   ║
║                                                                              ║
║  RECOMMENDATION:                                                             ║
║  • MITC3 for coarse meshes, fastest convergence per DOF                     ║
║  • CR    for accuracy (higher-order, already used in wing.py)               ║
║  • DKT   only for thickness t/L < 1/100                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN — run validation if executed directly
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(INTEGRATION_GUIDE)

    # Run convergence study for MITC3 and CR
    # (DKT excluded because it diverges for t/L = 0.01 without careful tuning)
    convergence_study(
        strategies=[DSTStrategy.MITC3, DSTStrategy.CR],
        n_list=[4, 8, 16],
        t=0.01,    # thin plate  → λ = t/L = 0.01
    )
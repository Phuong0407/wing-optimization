import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl

from scipy.optimize import minimize

# -----------------------------
# USER: define your component layups (angles fixed)
# -----------------------------
SKIN_ANGLES = [0, 45, -45, 90, 90, -45, 45, 0]     # example symmetric
MS_ANGLES   = [0, 45, -45, 0, 0, -45, 45, 0]       # main spar example
TS_ANGLES   = [0, 0, 0, 0]                         # tail spar example
RIB_ANGLES  = [90, 90, 90, 90]                     # rib example

# -----------------------------
# USER: density (for mass objective) + component areas
# If you already have per-cell-tags integration, replace these with dx(tag) integrals.
# -----------------------------
RHO = 1600.0  # kg/m^3 example CFRP

# Example: if you have cell tags per component, define them here
# (replace with your real ids)
TAG_SKIN = 1
TAG_MS   = 2
TAG_TS   = 3
TAG_RIB  = 4

# If you have cell_tags + dx = ufl.Measure("dx", domain=DOMAIN, subdomain_data=CELL_TAGS)
# then mass can be computed with dx(TAG_*)

# -----------------------------
# USER: constraints
# -----------------------------
U_LIM = 0.12      # m, example max deflection limit
BUCK_MIN = 1.0    # min buckling factor
FI_MAX  = 1.0     # Tsai-Wu <= 1

# -----------------------------
# Helper: build material object for each component (scalar thickness per ply in component)
# -----------------------------
def build_materials_from_x(x):
    """
    x = [t_skin, t_ms, t_ts, t_rib]  (meters)
    """
    t_skin, t_ms, t_ts, t_rib = map(float, x)

    MAT_SKIN = clt_material(
        layup_angles=SKIN_ANGLES,
        t_ply=t_skin,                 # scalar => all plies in skin same thickness
        E1=_E1, E2=_E2, G12=_G12, nu12=_nu12,
        G13=_G12, G23=_G12*0.5, kappa_s=5/6
    )

    MAT_MS = clt_material(
        layup_angles=MS_ANGLES,
        t_ply=t_ms,
        E1=_E1, E2=_E2, G12=_G12, nu12=_nu12,
        G13=_G12, G23=_G12*0.5, kappa_s=5/6
    )

    MAT_TS = clt_material(
        layup_angles=TS_ANGLES,
        t_ply=t_ts,
        E1=_E1, E2=_E2, G12=_G12, nu12=_nu12,
        G13=_G12, G23=_G12*0.5, kappa_s=5/6
    )

    MAT_RIB = clt_material(
        layup_angles=RIB_ANGLES,
        t_ply=t_rib,
        E1=_E1, E2=_E2, G12=_G12, nu12=_nu12,
        G13=_G12, G23=_G12*0.5, kappa_s=5/6
    )

    return MAT_SKIN, MAT_MS, MAT_TS, MAT_RIB


# -----------------------------
# Helper: assemble residual/tangent using *component-wise* internal energy
# You MUST have separate eps,kappa,gamma definitions; and stress_resultants accepts MAT and strains.
# -----------------------------
def rebuild_problem_for_x(x):
    global residual, tangent, a_int, problem

    MAT_SKIN, MAT_MS, MAT_TS, MAT_RIB = build_materials_from_x(x)

    # Stress resultants per component (use SAME strain measures but integrate on component subdomains)
    N_sk, M_sk, Q_sk = stress_resultants(MAT_SKIN, eps, kappa, gamma)
    N_ms, M_ms, Q_ms = stress_resultants(MAT_MS, eps, kappa, gamma)
    N_ts, M_ts, Q_ts = stress_resultants(MAT_TS, eps, kappa, gamma)
    N_rb, M_rb, Q_rb = stress_resultants(MAT_RIB, eps, kappa, gamma)

    # Drilling (if you use it)
    _, drill_sk = drilling_terms(MAT_SKIN, DOMAIN, drilling_strain)
    _, drill_ms = drilling_terms(MAT_MS,   DOMAIN, drilling_strain)
    _, drill_ts = drilling_terms(MAT_TS,   DOMAIN, drilling_strain)
    _, drill_rb = drilling_terms(MAT_RIB,  DOMAIN, drilling_strain)

    # Internal energy on subdomains (dx(TAG_*))
    a_int = (
        (ufl.inner(N_sk, eps_) + ufl.inner(M_sk, kappa_) + ufl.inner(Q_sk, gamma_) + drill_sk * drilling_strain_) * dx(TAG_SKIN)
        + (ufl.inner(N_ms, eps_) + ufl.inner(M_ms, kappa_) + ufl.inner(Q_ms, gamma_) + drill_ms * drilling_strain_) * dx(TAG_MS)
        + (ufl.inner(N_ts, eps_) + ufl.inner(M_ts, kappa_) + ufl.inner(Q_ts, gamma_) + drill_ts * drilling_strain_) * dx(TAG_TS)
        + (ufl.inner(N_rb, eps_) + ufl.inner(M_rb, kappa_) + ufl.inner(Q_rb, gamma_) + drill_rb * drilling_strain_) * dx(TAG_RIB)
    )

    residual = a_int - L_ext
    tangent  = ufl.derivative(residual, v, dv)

    # Rebuild nonlinear problem (same solver options as you used)
    problem = NonlinearProblem(
        F=residual,
        u=v,
        bcs=BCS,
        J=tangent,
        petsc_options_prefix="opt",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_type": "newtonls",
        },
    )


# -----------------------------
# Metrics you need (you must implement / plug your existing ones)
# -----------------------------
def compute_mass_from_x(x):
    """Mass objective with simple shell mass rho * total_thickness * area, integrated per component."""
    t_skin, t_ms, t_ts, t_rib = map(float, x)

    # For shell-like model: total laminate thickness H = nplies * t_ply (since scalar per ply in that component)
    H_skin = len(SKIN_ANGLES) * t_skin
    H_ms   = len(MS_ANGLES)   * t_ms
    H_ts   = len(TS_ANGLES)   * t_ts
    H_rib  = len(RIB_ANGLES)  * t_rib

    one = fem.Constant(DOMAIN, 1.0)
    A_skin = fem.assemble_scalar(fem.form(one * dx(TAG_SKIN)))
    A_ms   = fem.assemble_scalar(fem.form(one * dx(TAG_MS)))
    A_ts   = fem.assemble_scalar(fem.form(one * dx(TAG_TS)))
    A_rib  = fem.assemble_scalar(fem.form(one * dx(TAG_RIB)))

    mass = RHO * (H_skin*A_skin + H_ms*A_ms + H_ts*A_ts + H_rib*A_rib)
    return mass


def solve_static_for_x(x):
    """Solve nonlinear static problem for given thickness x."""
    rebuild_problem_for_x(x)

    # good practice: reset initial guess (or warm start from previous v)
    # If you want speed, REMOVE the reset and let Newton warm-start.
    v.x.array[:] = 0.0
    v.x.scatter_forward()

    problem.solve()


def compute_umax():
    """Return max displacement magnitude."""
    u_h, _ = ufl.split(v)
    # If you already have post-processing code, plug it here.
    # Quick-and-dirty: assemble max over dofs of u component function space
    Vu, _ = V.sub(0).collapse()
    u_fun = fem.Function(Vu)
    u_fun.x.array[:] = v.x.array[V.sub(0).dofmap.list.flatten()]  # may need proper extraction in your setup

    # safer: compute magnitude at dofs
    arr = u_fun.x.array
    # For vector u, you'd need per-component extraction; replace with your known correct method.
    return float(np.max(np.abs(arr)))


def compute_tsai_wu_max():
    """Plug your existing Tsai-Wu evaluation here and return max FI."""
    # You already computed Tsai-Wu in your post earlier; reuse that code.
    # Return a float.
    FImax = composite_failure_tsai_wu_max(...)  # <-- replace
    return float(FImax)


def compute_buckling_factor_min():
    """Use your existing buckling routine; return smallest positive lambda."""
    lam = compute_buckling_factor_from_current_state(...)  # <-- replace with your buckling function
    return float(lam)


# -----------------------------
# Objective + constraints for COBYLA
# COBYLA expects constraints g(x) >= 0
# -----------------------------
def objective(x):
    # You can also add small regularization if you see thickness going to bounds.
    return compute_mass_from_x(x)


def con_FI(x):
    # FI_MAX - FI(x) >= 0
    solve_static_for_x(x)
    fi = compute_tsai_wu_max()
    return FI_MAX - fi


def con_buckling(x):
    # buck(x) - BUCK_MIN >= 0
    solve_static_for_x(x)
    lam = compute_buckling_factor_min()
    return lam - BUCK_MIN


def con_deflection(x):
    # U_LIM - umax(x) >= 0
    solve_static_for_x(x)
    u = compute_umax()
    return U_LIM - u


# -----------------------------
# Bounds with COBYLA: implement as constraints
# -----------------------------
def bound_constraints(x, tmin, tmax):
    cons = []
    for i in range(len(x)):
        cons.append(lambda z, i=i: z[i] - tmin[i])   # z[i] >= tmin
        cons.append(lambda z, i=i: tmax[i] - z[i])   # z[i] <= tmax
    return cons


# -----------------------------
# Run optimization
# -----------------------------
# initial guess (meters)
x0 = np.array([0.25e-3, 0.25e-3, 0.25e-3, 0.25e-3], dtype=float)

# bounds (meters) - tune these
tmin = np.array([0.10e-3, 0.10e-3, 0.10e-3, 0.10e-3])
tmax = np.array([1.50e-3, 2.00e-3, 2.00e-3, 2.00e-3])

constraints = [
    {"type": "ineq", "fun": con_FI},
    {"type": "ineq", "fun": con_buckling},
    {"type": "ineq", "fun": con_deflection},
]

for bc in bound_constraints(x0, tmin, tmax):
    constraints.append({"type": "ineq", "fun": bc})

res = minimize(
    objective,
    x0,
    method="COBYLA",
    constraints=constraints,
    options={"maxiter": 50, "rhobeg": 0.2e-3, "disp": True},
)

if MPI.COMM_WORLD.rank == 0:
    print("Optimization success:", res.success)
    print("Message:", res.message)
    print("x* [t_skin, t_ms, t_ts, t_rib] =", res.x)
    print("mass* =", res.fun)
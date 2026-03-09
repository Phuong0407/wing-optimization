"""
buckling_analysis.py
====================
Linear buckling analysis for the composite wingbox shell.
Standalone — mirrors opti8var.py setup exactly.

Eigenvalue problem:
    K_e · φ = λ · K_g · φ

    K_e  : elastic stiffness  (tangent of nonlinear residual at u₀)
    K_g  : geometric stiffness assembled from pre-buckling N₀
    λ    : buckling load factor  (λ > 1 → safe under design load)

Key implementation choices (validated on skin-only case):
    1. K_e from ufl.derivative(residual)  — consistent with opti8var solver
    2. K_g membrane-only:  ∫ N₀ : (∇ₛw_trial ⊗ ∇ₛw_test) dΩ
    3. BC handling: K_e with bcs=BCS (diag=1), K_g zeroed manually (diag=0)
    4. GHIEP problem type  — handles indefinite K_g (mixed tension/compression)
    5. setTarget(2.0)      — shift past spurious negative cluster near 0

CLI:
    python buckling_analysis.py                 # full wingbox
    python buckling_analysis.py --skin-only      # monocoque comparison
    python buckling_analysis.py --n-modes 10
    mpirun -n 4 python buckling_analysis.py

Dependencies: dolfinx, ufl, basix, petsc4py, slepc4py, mpi4py, meshio, scipy
"""

import argparse
import csv
from functools import reduce
from pathlib import Path
from types import SimpleNamespace

import basix
import meshio
import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.io import gmsh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from mpi4py import MPI
from scipy.spatial import cKDTree

try:
    from petsc4py import PETSc
    from slepc4py import SLEPc
except ImportError:
    raise ImportError("slepc4py / petsc4py required: conda install -c conda-forge slepc4py")

# ─────────────────────────────────────────────────────────────
# MPI / verbosity
# ─────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank

VERBOSE = True
def vprint(*a, **kw):
    if VERBOSE and rank == 0:
        print(*a, **kw)

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skin-only", action="store_true")
parser.add_argument("--n-modes",   type=int,   default=10)
parser.add_argument("--target",    type=float, default=2.0,
                    help="SLEPc shift target (default 2.0, increase if modes missing)")
args = parser.parse_args()

SKIN_ONLY = args.skin_only
N_MODES   = args.n_modes
TARGET    = args.target

vprint("=" * 62)
vprint("  LINEAR BUCKLING ANALYSIS — COMPOSITE WINGBOX")
vprint(f"  Config  : {'SKIN-ONLY' if SKIN_ONLY else 'FULL WINGBOX'}")
vprint(f"  Modes   : {N_MODES}   Target shift : {TARGET}")
vprint("=" * 62)

# ─────────────────────────────────────────────────────────────
# PATHS  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
MESHUNIT   = "mm"
INPUT_DIR  = Path("InputData")
OUTPUT_DIR = Path("Result")
MESHFILE   = INPUT_DIR / "wingbox.msh"
MAPFILE    = INPUT_DIR / "MappedTraction.xdmf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PHYSICAL TAGS  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
TAG_UPPER      = 14;  TAG_LOWER    = 15
TAG_RIB0       = 38;  TAG_RIB750   = 39
TAG_RIB1500    = 40;  TAG_RIB2250  = 41;  TAG_RIB3000 = 42
TAG_MAINSPAR   = 43;  TAG_TAILSPAR = 44
TAG_RIB0_CURVE = 45
TAG_SKIN       = [TAG_UPPER, TAG_LOWER]

# ─────────────────────────────────────────────────────────────
# MESH
# ─────────────────────────────────────────────────────────────
MESH_IO    = gmsh.read_from_msh(MESHFILE, comm=MPI.COMM_WORLD, gdim=3)
DOMAIN     = MESH_IO.mesh
CELL_TAGS  = MESH_IO.cell_tags
FACET_TAGS = MESH_IO.facet_tags
TDIM       = DOMAIN.topology.dim
FDIM       = TDIM - 1
GDIM       = DOMAIN.geometry.dim

if MESHUNIT == "mm":
    DOMAIN.geometry.x[:] *= 1e-3

vprint(f"\n[MESH] Cells={DOMAIN.topology.index_map(TDIM).size_global}"
       f"  Vertices={DOMAIN.topology.index_map(0).size_global}")

# ─────────────────────────────────────────────────────────────
# LOCAL FRAME  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
def normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))

def local_frame_ufl(domain):
    t  = ufl.Jacobian(domain)
    t1 = ufl.as_vector([t[0,0], t[1,0], t[2,0]])
    t2 = ufl.as_vector([t[0,1], t[1,1], t[2,1]])
    e3 = normalize(ufl.cross(t1, t2))
    ey = ufl.as_vector([0, 1, 0])
    ez = ufl.as_vector([0, 0, 1])
    e1_trial = ufl.cross(ey, e3)
    norm_e1  = ufl.sqrt(ufl.dot(e1_trial, e1_trial))
    e1 = ufl.conditional(ufl.lt(norm_e1, 0.5), ez, normalize(e1_trial))
    e2 = normalize(ufl.cross(e3, e1))
    return e1, e2, e3

def local_frame(domain):
    FRAME = local_frame_ufl(domain)
    VT    = fem.functionspace(domain, ("DG", 0, (3,)))
    V0, _ = VT.sub(0).collapse()
    BASIS = [fem.Function(VT, name=f"e{i+1}") for i in range(3)]
    for i in range(3):
        BASIS[i].interpolate(fem.Expression(FRAME[i], V0.element.interpolation_points))
    return BASIS[0], BASIS[1], BASIS[2]

E1, E2, E3 = local_frame(DOMAIN)

# ─────────────────────────────────────────────────────────────
# SHELL KINEMATICS  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
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
    return (t_gu[0,1] - t_gu[1,0]) / 2 + ufl.dot(theta, e3)

def shell_strains(u, theta, e1, e2, e3):
    P               = tangent_projection(e1, e2)
    eps, t_gu       = membrane_strain(u, P)
    kappa           = bending_strain(theta, e3, P)
    gamma           = shear_strain(u, theta, e3, P)
    drilling_strain = compute_drilling_strain(t_gu, theta, e3)
    return eps, kappa, gamma, drilling_strain

# ─────────────────────────────────────────────────────────────
# MATERIAL — VOIGT + CLT  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
def to_voigt(e):
    return ufl.as_vector([e[0,0], e[1,1], 2.0*e[0,1]])

def from_voigt(v):
    return ufl.as_tensor([[v[0], v[2]], [v[2], v[1]]])

def _Q_ply(E1, E2, G12, nu12):
    nu21 = nu12 * E2 / E1
    d    = 1 - nu12 * nu21
    return np.array([[E1/d, nu12*E2/d, 0],
                     [nu12*E2/d, E2/d, 0],
                     [0, 0, G12]])

def _Qbar_ply(Q, angle):
    a    = np.radians(angle)
    s, c = np.sin(a), np.cos(a)
    T = np.array([[ c**2,  s**2,  2*c*s],
                  [ s**2,  c**2, -2*c*s],
                  [-c*s,   c*s,  c**2-s**2]])
    R    = np.diag([1.0, 1.0, 2.0])
    Rinv = np.diag([1.0, 1.0, 0.5])
    return np.linalg.inv(T) @ Q @ R @ T @ Rinv

def compute_ABD(layup, t_ply, E1, E2, G12, nu12):
    Q = _Q_ply(E1, E2, G12, nu12)
    H = t_ply * len(layup)
    z = -H / 2.0
    A = np.zeros((3,3)); B = np.zeros((3,3)); D = np.zeros((3,3))
    for angle in layup:
        Qb     = _Qbar_ply(Q, angle)
        z0, z1 = z, z + t_ply
        A += Qb * (z1 - z0)
        B += Qb * (z1**2 - z0**2) / 2.0
        D += Qb * (z1**3 - z0**3) / 3.0
        z  = z1
    return A, B, D, H

def clt_composite(layup, t_ply, E1, E2, G12, nu12,
                  G13=None, G23=None, kappa_s=5/6,
                  label="CLT", verbose=False, domain=None):
    if G13 is None: G13 = G12
    if G23 is None: G23 = G12 * 0.5
    A_np, B_np, D_np, H = compute_ABD(layup, t_ply, E1, E2, G12, nu12)
    As_np = kappa_s * H * np.array([[G13, 0.0], [0.0, G23]])
    G_eff = float(A_np[2,2]) / H
    if verbose:
        vprint(f"[{label}] H={H*1e3:.2f} mm  A11={A_np[0,0]/1e6:.1f} MPa·m"
               f"  D11={D_np[0,0]:.3f} N·m²")
    return SimpleNamespace(
        kind="clt", H=fem.Constant(domain, H), t_ply=t_ply,
        G_eff=fem.Constant(domain, G_eff),
        A_ufl=fem.Constant(domain, A_np),
        B_ufl=fem.Constant(domain, B_np),
        D_ufl=fem.Constant(domain, D_np),
        As_ufl=fem.Constant(domain, As_np),
        _layup_angles=layup,
        _E1=E1, _E2=E2, _G12=G12, _nu12=nu12,
        _G13=G13, _G23=G23, _kappa_s=kappa_s,
    )

def stress_resultants(mat, eps, kappa, gamma):
    eps_v   = to_voigt(eps)
    kappa_v = to_voigt(kappa)
    N = from_voigt(ufl.dot(mat.A_ufl, eps_v)   + ufl.dot(mat.B_ufl, kappa_v))
    M = from_voigt(ufl.dot(mat.B_ufl, eps_v)   + ufl.dot(mat.D_ufl, kappa_v))
    Q = ufl.dot(mat.As_ufl, gamma)
    return N, M, Q

def drilling_terms(mat, domain, drilling_strain):
    h_mesh = ufl.CellDiameter(domain)
    stiffness = mat.G_eff * mat.H**3 / h_mesh**2
    return stiffness, stiffness * drilling_strain

# ─────────────────────────────────────────────────────────────
# MATERIAL CONSTANTS  (AS4/3501-6)
# ─────────────────────────────────────────────────────────────
CM_YNG1 = 181.0e9;  CM_YNG2 = 10.30e9
CM_G12  = 7.170e9;  CM_NU12 = 0.28

# ─────────────────────────────────────────────────────────────
# LAYUPS  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
SKIN_LAYUP = [0,  0,  45, -45, -45,  45,  0, 0]
SPAR_LAYUP = [0, 45, -45,   0,   0, -45, 45, 0]
RIB_LAYUP  = [0, 45, -45,  90,  90, -45, 45, 0]

# ─────────────────────────────────────────────────────────────
# OPTIMAL THICKNESSES from opti8var result [m]
# ─────────────────────────────────────────────────────────────
T_OPT = {
    "skin"  : 7.300e-3,  "rib0"   : 2.212e-3,
    "rib750": 1.839e-3,  "rib1500": 1.674e-3,
    "rib2250":4.491e-3,  "rib3000": 9.696e-3,
    "mspar" : 3.495e-3,  "tspar"  : 2.785e-3,
}

def _mat(layup, t_total, label):
    return clt_composite(layup, t_total/len(layup),
        CM_YNG1, CM_YNG2, CM_G12, CM_NU12,
        G13=CM_G12, G23=CM_G12*0.5, kappa_s=5/6,
        label=label, verbose=VERBOSE, domain=DOMAIN)

vprint("\n[MAT] Laminates …")
MAT_SKIN      = _mat(SKIN_LAYUP, T_OPT["skin"],    "SKIN")
MAT_SPAR_MAIN = _mat(SPAR_LAYUP, T_OPT["mspar"],   "MAINSPAR")
MAT_SPAR_TAIL = _mat(SPAR_LAYUP, T_OPT["tspar"],   "TAILSPAR")
MAT_RIB0      = _mat(RIB_LAYUP,  T_OPT["rib0"],    "RIB0")
MAT_RIB750    = _mat(RIB_LAYUP,  T_OPT["rib750"],  "RIB750")
MAT_RIB1500   = _mat(RIB_LAYUP,  T_OPT["rib1500"],"RIB1500")
MAT_RIB2250   = _mat(RIB_LAYUP,  T_OPT["rib2250"],"RIB2250")
MAT_RIB3000   = _mat(RIB_LAYUP,  T_OPT["rib3000"],"RIB3000")

MATS_FULL = {
    TAG_UPPER:MAT_SKIN,    TAG_LOWER:MAT_SKIN,
    TAG_RIB0:MAT_RIB0,     TAG_RIB750:MAT_RIB750,
    TAG_RIB1500:MAT_RIB1500, TAG_RIB2250:MAT_RIB2250,
    TAG_RIB3000:MAT_RIB3000,
    TAG_MAINSPAR:MAT_SPAR_MAIN, TAG_TAILSPAR:MAT_SPAR_TAIL,
}
MATS_SKIN = {TAG_UPPER:MAT_SKIN, TAG_LOWER:MAT_SKIN}
MATS = MATS_SKIN if SKIN_ONLY else MATS_FULL

# ─────────────────────────────────────────────────────────────
# FUNCTION SPACE  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
Ue = basix.ufl.element("P",  DOMAIN.basix_cell(), 2, shape=(GDIM,))
Te = basix.ufl.element("CR", DOMAIN.basix_cell(), 1, shape=(GDIM,))
V  = fem.functionspace(DOMAIN, basix.ufl.mixed_element([Ue, Te]))
v  = fem.Function(V)          # will hold pre-buckling solution
dv = ufl.TrialFunction(V)
v_ = ufl.TestFunction(V)
u,     theta     = ufl.split(v)
u_,    theta_    = ufl.split(v_)

dx = ufl.Measure("dx", domain=DOMAIN, subdomain_data=CELL_TAGS)

# ─────────────────────────────────────────────────────────────
# BOUNDARY CONDITIONS — clamp root  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
ROOT = FACET_TAGS.find(TAG_RIB0_CURVE)
Vu, _ = V.sub(0).collapse()
dofs_u = fem.locate_dofs_topological((V.sub(0), Vu), FDIM, ROOT)
bc_u   = fem.dirichletbc(fem.Function(Vu), dofs_u, V.sub(0))
Vt, _ = V.sub(1).collapse()
dofs_t = fem.locate_dofs_topological((V.sub(1), Vt), FDIM, ROOT)
bc_t   = fem.dirichletbc(fem.Function(Vt), dofs_t, V.sub(1))
BCS    = [bc_u, bc_t]

# ─────────────────────────────────────────────────────────────
# TRACTION LOAD  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
def load_traction_xdmf(xdmffile, domain):
    if rank == 0:
        data  = meshio.read(str(xdmffile))
        pts   = data.points
        tract = data.point_data["traction"]
    else:
        pts = tract = None
    pts   = comm.bcast(pts,   root=0)
    tract = comm.bcast(tract, root=0)
    VT     = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    f      = fem.Function(VT, name="traction")
    coords = VT.tabulate_dof_coordinates()
    _, idx = cKDTree(pts).query(coords, k=1)
    f.x.array[:] = tract[idx].reshape(-1)
    f.x.scatter_forward()
    return f

FTraction = load_traction_xdmf(MAPFILE, DOMAIN)

# ─────────────────────────────────────────────────────────────
# NONLINEAR RESIDUAL — same as opti8var.py
# Build kinematics ONCE (outside tag loop — avoids JIT slowdown)
# ─────────────────────────────────────────────────────────────
eps, kappa, gamma, drilling_strain = shell_strains(u, theta, E1, E2, E3)
eps_             = ufl.derivative(eps,           v, v_)
kappa_           = ufl.derivative(kappa,         v, v_)
gamma_           = ufl.derivative(gamma,         v, v_)
drilling_strain_ = ufl.replace(drilling_strain, {v: v_})

form_pieces = []
for tag, mat in MATS.items():
    N_t, M_t, Q_t = stress_resultants(mat, eps, kappa, gamma)
    _, drill_t     = drilling_terms(mat, DOMAIN, drilling_strain)
    form_pieces.append((
        ufl.inner(N_t, eps_)
        + ufl.inner(M_t, kappa_)
        + ufl.inner(Q_t, gamma_)
        + drill_t * drilling_strain_
    ) * dx(tag))

L_ext    = sum(ufl.dot(FTraction, u_) * dx(tag) for tag in TAG_SKIN)
residual = reduce(lambda a, b: a+b, form_pieces) - L_ext
tangent  = ufl.derivative(residual, v, dv)

# ─────────────────────────────────────────────────────────────
# STEP 1  PRE-BUCKLING STATIC SOLVE
# Reuse the same NonlinearProblem setup as opti8var.py
# ─────────────────────────────────────────────────────────────
vprint("\n[STEP 1] Pre-buckling static solve …")

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

problem = NonlinearProblem(
    F=residual, u=v, bcs=BCS, J=tangent,
    petsc_options_prefix="buck_",
    petsc_options={
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_type": "newtonls",
        "snes_rtol": 1e-8, "snes_atol": 1e-8,
        "snes_max_it": 25,
        "mat_mumps_icntl_14": 80,
        "mat_mumps_icntl_23": 4000,
    }
)

JIT_OPTS = {"cache_dir": ".jit_cache"}
vprint("  [JIT] Compiling forms … (cached after first run)")
problem.solve()
v.x.scatter_forward()

# Verify tip deflection
Vu_c = V.sub(0).collapse()[0]
u_sol = v.sub(0).collapse()
coords_u = Vu_c.tabulate_dof_coordinates()
u_arr    = u_sol.x.array.reshape(-1, 3)
tip_mask = np.abs(coords_u[:, 1] - 3.0) < 0.05
uz_tip   = comm.allreduce(
    float(np.abs(u_arr[tip_mask, 2]).max()) if tip_mask.any() else 0.0,
    op=MPI.MAX)
vprint(f"  uz_tip = {uz_tip*1e3:.2f} mm  (expect ~150 mm for full wingbox)")

# ─────────────────────────────────────────────────────────────
# STEP 2  ELASTIC STIFFNESS  K_e
# Use tangent of converged nonlinear residual — consistent with
# the static solve and with opti8var.py material definitions.
# ─────────────────────────────────────────────────────────────
vprint("\n[STEP 2] Assembling K_e (tangent stiffness) …")

K_form = fem.form(tangent, jit_options=JIT_OPTS)
K_e    = assemble_matrix(K_form, bcs=BCS)   # BCs: diag=1 → invertible
K_e.assemble()
vprint("  K_e assembled ✓")

# ─────────────────────────────────────────────────────────────
# STEP 3  GEOMETRIC STIFFNESS  K_g
#
# Membrane-only formulation (validated):
#   a_G(w, v) = ∫ N₀ : (∇ₛ w_n ⊗ ∇ₛ v_n) dΩ
#
# where:
#   N₀    = pre-buckling membrane resultant (2×2, from converged u₀)
#   w_n   = w · e₃   (out-of-plane trial displacement)
#   ∇ₛ f  = tangential_gradient(f, P)  (surface gradient)
#
# BC handling (critical):
#   K_g assembled WITHOUT BCs first, then rows/cols of constrained
#   DOFs are explicitly zeroed with diagonal = 0.0
#   (diagonal=1 would shift eigenvalues; diagonal=0 removes them)
# ─────────────────────────────────────────────────────────────
vprint("\n[STEP 3] Assembling K_g (geometric stiffness) …")

u_h, theta_h = ufl.split(v)           # pre-buckling solution
P = tangent_projection(E1, E2)

# Pre-buckling stress resultants per subdomain
w_trial = ufl.TrialFunction(V)
v_test  = ufl.TestFunction(V)
u_w, _  = ufl.split(w_trial)
u_v, _  = ufl.split(v_test)

w_n    = ufl.dot(u_w, E3)             # out-of-plane trial
v_n    = ufl.dot(u_v, E3)             # out-of-plane test
grad_w = tangential_gradient(w_n, P)  # ∇ₛ w_n  (2-vector)
grad_v = tangential_gradient(v_n, P)  # ∇ₛ v_n

kg_pieces = []
for tag, mat in MATS.items():
    # N₀ from pre-buckling membrane + bending strains
    eps0, _  = membrane_strain(u_h, P)
    kappa0   = bending_strain(theta_h, E3, P)
    N0 = from_voigt(
        ufl.dot(mat.A_ufl, to_voigt(eps0))
        + ufl.dot(mat.B_ufl, to_voigt(kappa0))
    )
    # Membrane geometric stiffness: N₀ : (∇ₛw ⊗ ∇ₛv)
    kg_pieces.append(ufl.inner(ufl.dot(N0, grad_w), grad_v) * dx(tag))

KG_form = fem.form(reduce(lambda a, b: a+b, kg_pieces), jit_options=JIT_OPTS)
vprint("  [JIT] Geometric form compiled ✓")

# Assemble K_g WITHOUT BC enforcement
K_g = assemble_matrix(KG_form)
K_g.assemble()

# Manually zero constrained DOF rows+cols with diag=0
# (diag=0 → those DOFs play no role in eigenproblem, unlike diag=1)
bc_dofs = np.unique(np.concatenate([bc.dof_indices()[0] for bc in BCS]))
K_g.zeroRowsColumns(bc_dofs, 0.0)

vprint("  K_g assembled + BCs zeroed ✓")

# ─────────────────────────────────────────────────────────────
# STEP 4  GENERALIZED EIGENVALUE PROBLEM
#         K_e · φ = λ · K_g · φ
#
# Problem type: GHIEP
#   Generalized Hermitian Indefinite Eigenvalue Problem.
#   Correct choice when K_g is indefinite (mixed tension/compression).
#   Unlike GHEP, GHIEP does not require B (= K_g) to be positive definite.
#
# Shift target:
#   Spurious near-zero / negative eigenvalues exist due to:
#     (a) tension zones where N₀ > 0 → K_g negative semi-definite locally
#     (b) rigid-body modes not fully suppressed by K_g zeros
#   Solution: setTarget(2.0) shifts past this cluster.
#   Increase --target if physical modes are missing.
# ─────────────────────────────────────────────────────────────
vprint(f"\n[STEP 4] SLEPc GHIEP — target={TARGET}, nev={N_MODES} …")

eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setOperators(K_e, K_g)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHIEP)

ST = eigensolver.getST()
ST.setType(SLEPc.ST.Type.SINVERT)

ksp_st = ST.getKSP()
ksp_st.setType("preonly")
pc_st = ksp_st.getPC()
pc_st.setType("lu")
pc_st.setFactorSolverType("mumps")

eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
eigensolver.setTarget(TARGET)
eigensolver.setDimensions(nev=N_MODES, ncv=max(6*N_MODES, 60))
eigensolver.setTolerances(tol=1e-6, max_it=1000)
eigensolver.setFromOptions()

eigensolver.solve()
nconv = eigensolver.getConverged()
vprint(f"  Converged eigenpairs : {nconv}")

# ─────────────────────────────────────────────────────────────
# STEP 5  EXTRACT + REPORT RESULTS
# ─────────────────────────────────────────────────────────────
vprint("\n" + "=" * 62)
vprint(f"  BUCKLING RESULTS — {'SKIN-ONLY' if SKIN_ONLY else 'FULL WINGBOX'}")
vprint("=" * 62)

vr = K_e.createVecRight()
vi = K_e.createVecRight()

all_pairs = []
for i in range(nconv):
    lam = eigensolver.getEigenpair(i, vr, vi)
    phi = fem.Function(V, name=f"mode_{i+1}")
    phi.x.petsc_vec.setArray(vr.getArray())
    phi.x.scatter_forward()
    all_pairs.append((lam.real, lam.imag, phi))

# Keep positive eigenvalues (physical buckling), sort ascending
physical = sorted([(r, im, phi) for r, im, phi in all_pairs if r > 0],
                  key=lambda x: x[0])

vprint(f"  {'Mode':>5}   {'λ_cr':>10}   {'Im(λ)':>10}   Status")
vprint(f"  {'-'*5}   {'-'*10}   {'-'*10}   {'-'*20}")

lambdas     = []
mode_shapes = []
for i, (lam_r, lam_i, phi) in enumerate(physical[:N_MODES]):
    lambdas.append(lam_r)
    mode_shapes.append(phi)
    imag_ok = "✓" if abs(lam_i) < 1e-3 * abs(lam_r) else "⚠ large imag"
    status  = ("<<< CRITICAL >>>" if i == 0
               else "⚠ below load" if lam_r < 1.0
               else "low margin"   if lam_r < 1.5
               else "OK")
    vprint(f"  {i+1:>5}   {lam_r:>10.4f}   {lam_i:>+10.2e}   {status}  {imag_ok}")

if lambdas:
    lam_cr = lambdas[0]
    vprint(f"\n  Critical buckling factor  λ_cr = {lam_cr:.4f}")
    vprint(f"  → Buckles at {lam_cr:.2f}× the applied aerodynamic load")
    if   lam_cr < 1.0: vprint("  !! UNSAFE — buckles before design load !!")
    elif lam_cr < 1.5: vprint("  ⚠  Low safety margin")
    else:              vprint(f"  ✓  SF_buckling = {lam_cr:.2f}")
else:
    vprint(f"\n  ⚠  No positive eigenvalue found near target={TARGET}")
    neg = sorted([r for r, _, _ in all_pairs], reverse=True)
    if neg:
        vprint(f"  Most positive found: {neg[0]:.4f}")
        vprint(f"  → Try: python buckling_analysis.py --target {neg[0]*1.5:.1f}")

# ─────────────────────────────────────────────────────────────
# STEP 6  EXPORT XDMF (Paraview)
# Mode shapes are P2 → interpolate to P1 for XDMF compatibility
# ─────────────────────────────────────────────────────────────
label     = "skin_only" if SKIN_ONLY else "wingbox"
xdmf_path = OUTPUT_DIR / f"buckling_modes_{label}.xdmf"
V_out     = fem.functionspace(DOMAIN, ("Lagrange", 1, (GDIM,)))

with io.XDMFFile(comm, xdmf_path, "w") as xf:
    xf.write_mesh(DOMAIN)
    for i, phi in enumerate(mode_shapes):
        phi_u  = phi.sub(0).collapse()
        phi_p1 = fem.Function(V_out, name=f"mode_{i+1}")
        phi_p1.interpolate(phi_u)
        xf.write_function(phi_p1, float(i + 1))

vprint(f"\n  Mode shapes → {xdmf_path}")

# ─────────────────────────────────────────────────────────────
# STEP 7  ANALYTICAL PANEL BUCKLING CHECK
# ─────────────────────────────────────────────────────────────
vprint("\n" + "=" * 62)
vprint("  ANALYTICAL PANEL BUCKLING  σ/σ_cr  per component")
vprint("=" * 62)

RIB_SPACING = 0.75;  SPAR_HEIGHT = 0.40;  RIB_PANEL = 0.40

def panel_buckling_index(eps_cells, mat, panel_width, buckling_k=4.0):
    t        = float(mat.H.value)
    E_eff    = float(mat.D_ufl.value[0,0]) * 12 / t**3
    sigma_cr = buckling_k * np.pi**2 * E_eff / (12*(1-0.3**2)) * (t/panel_width)**2
    eps_xx   = eps_cells[:, 0]
    comp     = eps_xx[eps_xx < 0]
    if len(comp) == 0: return 0.0
    return float(mat.A_ufl.value[0,0]) * np.abs(comp).max() / t / sigma_cr

DG0    = fem.functionspace(DOMAIN, ("DG", 0, (3,)))
eps_fn = fem.Function(DG0)
kap_fn = fem.Function(DG0)
eps_fn.interpolate(fem.Expression(to_voigt(eps), DG0.element.interpolation_points))
kap_fn.interpolate(fem.Expression(to_voigt(bending_strain(theta, E3, tangent_projection(E1, E2))),
                                   DG0.element.interpolation_points))
eps0_all  = eps_fn.x.array.reshape(-1, 3)
kappa_all = kap_fn.x.array.reshape(-1, 3)

TAG_INFO = {
    TAG_UPPER   :("Upper skin",  MAT_SKIN,      RIB_SPACING, 4.00),
    TAG_LOWER   :("Lower skin",  MAT_SKIN,      RIB_SPACING, 4.00),
    TAG_MAINSPAR:("Main spar",   MAT_SPAR_MAIN, SPAR_HEIGHT,  5.35),
    TAG_TAILSPAR:("Tail spar",   MAT_SPAR_TAIL, SPAR_HEIGHT,  5.35),
    TAG_RIB750  :("Rib  750mm",  MAT_RIB750,    RIB_PANEL,   6.97),
    TAG_RIB1500 :("Rib 1500mm",  MAT_RIB1500,   RIB_PANEL,   6.97),
    TAG_RIB2250 :("Rib 2250mm",  MAT_RIB2250,   RIB_PANEL,   6.97),
    TAG_RIB3000 :("Rib 3000mm",  MAT_RIB3000,   RIB_PANEL,   6.97),
}

vprint(f"  {'Component':>12}  {'b [m]':>7}  {'BI=σ/σ_cr':>10}  Status")
vprint(f"  {'-'*12}  {'-'*7}  {'-'*10}  {'-'*12}")

panel_results = {}
for tag, (name, mat, b, k) in TAG_INFO.items():
    if SKIN_ONLY and tag not in TAG_SKIN: continue
    cells = CELL_TAGS.find(tag)
    if len(cells) == 0: continue
    bi   = panel_buckling_index(eps0_all[cells], mat, b, k)
    bi_g = comm.allreduce(bi, op=MPI.MAX)
    status = "!! BUCKLED !!" if bi_g > 1.0 else "marginal" if bi_g > 0.7 else "OK"
    vprint(f"  {name:>12}  {b:>7.3f}  {bi_g:>10.4f}  {status}")
    panel_results[name] = bi_g

# ─────────────────────────────────────────────────────────────
# STEP 8  CSV SUMMARY
# ─────────────────────────────────────────────────────────────
if rank == 0:
    csv_path = OUTPUT_DIR / f"buckling_summary_{label}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "name", "value", "unit", "config"])
        for i, lam in enumerate(lambdas):
            w.writerow(["eigenmode", f"mode_{i+1}", round(lam,6), "lambda_cr", label])
        for name, bi in panel_results.items():
            w.writerow(["panel_BI", name, round(bi,6), "sigma/sigma_cr", label])
    vprint(f"\n  Summary CSV → {csv_path}")

vprint("\n[DONE]\n")
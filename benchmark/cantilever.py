"""
Cantilever Plate Benchmark
==========================
Identifies the factor-of-2 bug by comparing:
  - Approach 1: Residual form with ufl.derivative (current wing.py style)
  - Approach 2: Direct bilinear form (correct approach)

Analytical solution for clamped plate under uniform pressure:
    w_tip = q * L^4 / (8 * D)
    D = E * t^3 / (12 * (1 - nu^2))

Expected output:
    Approach 1 ratio ~2.0  → bug confirmed in residual form
    Approach 2 ratio ~1.0  → direct bilinear form is correct

Run with:
    python3 cantilever_benchmark.py
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import dolfinx.fem.petsc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
import ufl
import basix

# ── Parameters ────────────────────────────────────────────────────────────────
L    = 1.0      # plate length [m]
W    = 0.1      # plate width  [m]
T    = 1e-3     # thickness    [m]
E_v  = 210e9    # Young modulus [Pa]
NU_v = 0.3      # Poisson ratio [-]
Q    = 1000.0   # uniform pressure [Pa]
NL   = 20       # elements along length
NW   = 4        # elements along width

# ── Analytical solution ───────────────────────────────────────────────────────
D     = E_v * T**3 / (12.0 * (1.0 - NU_v**2))
w_ana = Q * L**4 / (8.0 * D)
print("=" * 60)
print("CANTILEVER PLATE BENCHMARK")
print("=" * 60)
print(f"Plate: L={L} m, W={W} m, t={T} m")
print(f"Material: E={E_v:.2e} Pa, nu={NU_v}")
print(f"Load: q={Q} Pa")
print(f"Flexural rigidity D = {D:.4e} N.m")
print(f"Analytical tip deflection = {w_ana:.6e} m")
print()

# ── 3D plate mesh in XY plane ─────────────────────────────────────────────────
def create_plate_mesh_3d(L, W, NL, NW):
    """Create a flat triangular mesh in the XY plane (z=0)."""
    nx, ny = NL + 1, NW + 1
    x = np.linspace(0, L, nx)
    y = np.linspace(0, W, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    cells = []
    def nid(i, j): return j * nx + i
    for j in range(NW):
        for i in range(NL):
            cells.append([nid(i,j),   nid(i+1,j),   nid(i+1,j+1)])
            cells.append([nid(i,j),   nid(i+1,j+1), nid(i,j+1)  ])
    return points, np.array(cells, dtype=np.int64)

pts, cells = create_plate_mesh_3d(L, W, NL, NW)
ufl_mesh   = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(3,)))
domain     = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, pts, ufl_mesh)

gdim = 3
tdim = domain.topology.dim   # = 2 (surface)
fdim = tdim - 1               # = 1 (edges)

n_nodes = domain.topology.index_map(0).size_global
n_cells = domain.topology.index_map(tdim).size_global
print(f"Mesh: {n_nodes} nodes, {n_cells} triangles")

# ── Material constants ────────────────────────────────────────────────────────
thick    = fem.Constant(domain, T)
E        = fem.Constant(domain, E_v)
nu       = fem.Constant(domain, NU_v)
mu_c     = E / (2.0 * (1.0 + nu))
lmbda    = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
lmbda_ps = 2.0 * lmbda * mu_c / (lmbda + 2.0 * mu_c)

# ── Local frame (e1,e2 in-plane, e3 normal) ───────────────────────────────────
def normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))

def local_frame(mesh):
    t  = ufl.Jacobian(mesh)
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

e1, e2, e3 = local_frame(domain)
P_plane = ufl.as_matrix([[e1[i], e2[i]] for i in range(gdim)])  # 3×2

# ── Shell kinematics ──────────────────────────────────────────────────────────
def t_grad(u):
    """Tangential gradient: returns 2×2 matrix."""
    return ufl.dot(ufl.grad(u), P_plane)

def membrane_strain(u):
    return ufl.sym(ufl.dot(P_plane.T, t_grad(u)))

def bending_strain(theta):
    beta = ufl.cross(e3, theta)
    return ufl.sym(ufl.dot(P_plane.T, t_grad(beta)))

def shear_strain(u, theta):
    beta = ufl.cross(e3, theta)
    return t_grad(ufl.dot(u, e3)) - ufl.dot(P_plane.T, beta)

def drilling_strain(u, theta):
    tgu = ufl.dot(P_plane.T, t_grad(u))
    return (tgu[0,1] - tgu[1,0]) / 2.0 + ufl.dot(theta, e3)

def plane_stress(eps):
    return lmbda_ps * ufl.tr(eps) * ufl.Identity(tdim) + 2.0 * mu_c * eps

h = ufl.CellDiameter(domain)
dx = ufl.dx

# ── Function space (P2 displacement + CR rotation) ───────────────────────────
Ue = basix.ufl.element("P",  domain.basix_cell(), 2, shape=(gdim,))
Te = basix.ufl.element("CR", domain.basix_cell(), 1, shape=(gdim,))
V  = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Te]))
print(f"DOFs: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
print()

# ── Boundary conditions: clamp at x=0 ────────────────────────────────────────
domain.topology.create_connectivity(fdim, tdim)

def at_x0(x): return np.isclose(x[0], 0.0)
clamp_facets = mesh.locate_entities_boundary(domain, fdim, at_x0)

Vu, _ = V.sub(0).collapse()
Vt, _ = V.sub(1).collapse()
clamp_dofs_u = fem.locate_dofs_topological((V.sub(0), Vu), fdim, clamp_facets)
clamp_dofs_t = fem.locate_dofs_topological((V.sub(1), Vt), fdim, clamp_facets)

zero_u = fem.Function(Vu); zero_u.x.array[:] = 0.0
zero_t = fem.Function(Vt); zero_t.x.array[:] = 0.0
bcs = [fem.dirichletbc(zero_u, clamp_dofs_u, V.sub(0)),
       fem.dirichletbc(zero_t, clamp_dofs_t, V.sub(1))]

# ── Solver helper ─────────────────────────────────────────────────────────────
def solve(a_form, L_form, bcs, label):
    A = assemble_matrix(fem.form(a_form), bcs=bcs)
    A.assemble()
    b = assemble_vector(fem.form(L_form))
    apply_lifting(b, [fem.form(a_form)], [bcs])
    b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    sol = A.createVecRight()
    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.solve(b, sol)

    v_fn = fem.Function(V)
    v_fn.x.array[:] = sol.getArray()

    # Max tip deflection (x ≈ L)
    coords = Vu.tabulate_dof_coordinates()
    u_arr  = v_fn.sub(0).collapse().x.array.reshape(-1, gdim)
    tip    = coords[:, 0] > 0.99 * L
    uz_tip = np.abs(u_arr[tip, 2]).max() if tip.any() else np.abs(u_arr[:, 2]).max()

    # Energy diagnostics
    Ku = A.createVecRight()
    A.mult(sol, Ku)
    uKu = sol.dot(Ku)          # u^T K u
    Fu  = b.dot(sol)           # f^T u  (after set_bc, reaction DOFs zeroed)

    K_norm = A.norm()

    print(f"── {label} ──")
    print(f"  Max tip UZ          = {uz_tip:.6e} m")
    print(f"  Analytical          = {w_ana:.6e} m")
    print(f"  Ratio tip/analytical= {uz_tip/w_ana:.4f}  (should be ~1.0)")
    print(f"  u^T K u             = {uKu:.4e} J")
    print(f"  f^T u               = {Fu:.4e} J")
    print(f"  f^T u / (u^T K u)   = {Fu/uKu:.4f}  (should be 1.0)")
    print(f"  K Frobenius norm    = {K_norm:.4e}")
    print()
    return uz_tip, K_norm

# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — Residual form + ufl.derivative  (current wing.py style)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("APPROACH 1: Residual form  (ufl.derivative of Wint)")
print("=" * 60)

v1      = fem.Function(V)
u1, th1 = ufl.split(v1)
v1_     = ufl.TestFunction(V)
u1_, th1_ = ufl.split(v1_)
dv1     = ufl.TrialFunction(V)

eps1   = membrane_strain(u1)
kap1   = bending_strain(th1)
gam1   = shear_strain(u1, th1)
drl1   = drilling_strain(u1, th1)

N1 = thick * plane_stress(eps1)
M1 = thick**3 / 12.0 * plane_stress(kap1)
Q1 = mu_c * thick * gam1
Sd1 = (E * thick**3 / h**2) * drl1

# Variation of strains via derivative
dpe1 = ufl.derivative(eps1, v1, v1_)
dpk1 = ufl.derivative(kap1, v1, v1_)
dpg1 = ufl.derivative(gam1, v1, v1_)
dpd1 = ufl.replace(drl1, {v1: v1_})

# Internal virtual work residual
Wint1 = (ufl.inner(N1, dpe1) + ufl.inner(M1, dpk1) +
         ufl.inner(Q1, dpg1) + Sd1 * dpd1) * dx

# Tangent stiffness via derivative of residual
a1 = ufl.derivative(Wint1, v1, dv1)

# External load: uniform pressure -Z
L1 = ufl.inner(ufl.as_vector([0.0, 0.0, -Q]), u1_) * dx

uz1, Kn1 = solve(a1, L1, bcs, "Approach 1 (residual + derivative)")

# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — Direct bilinear form  (trial/test functions explicitly)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("APPROACH 2: Direct bilinear form (trial × test)")
print("=" * 60)

du2, dth2 = ufl.TrialFunctions(V)
u2_, th2_ = ufl.TestFunctions(V)

# Strains of trial functions
eps2  = membrane_strain(du2)
kap2  = bending_strain(dth2)
gam2  = shear_strain(du2, dth2)
drl2  = drilling_strain(du2, dth2)

# Strains of test functions
eps2_ = membrane_strain(u2_)
kap2_ = bending_strain(th2_)
gam2_ = shear_strain(u2_, th2_)
drl2_ = drilling_strain(u2_, th2_)

# Stresses from trial strains
N2  = thick * plane_stress(eps2)
M2  = thick**3 / 12.0 * plane_stress(kap2)
Q2  = mu_c * thick * gam2
Sd2 = (E * thick**3 / h**2) * drl2

a2 = (ufl.inner(N2, eps2_) + ufl.inner(M2, kap2_) +
      ufl.inner(Q2, gam2_) + Sd2 * drl2_) * dx
L2 = ufl.inner(ufl.as_vector([0.0, 0.0, -Q]), u2_) * dx

uz2, Kn2 = solve(a2, L2, bcs, "Approach 2 (direct bilinear)")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Analytical tip deflection : {w_ana:.6e} m")
print(f"Approach 1 tip deflection : {uz1:.6e} m   ratio={uz1/w_ana:.4f}")
print(f"Approach 2 tip deflection : {uz2:.6e} m   ratio={uz2/w_ana:.4f}")
print(f"K1 / K2 norm ratio        : {Kn1/Kn2:.4f}  (1.0=same, 2.0=A1 twice stiffer)")
print()
if abs(uz1/w_ana - 2.0) < 0.1:
    print("DIAGNOSIS: Approach 1 gives 2× too large displacement")
    print("  → ufl.derivative(Residual, v, dv) produces K/2, not K")
    print("  → FIX: Replace with Approach 2 (direct bilinear form)")
elif abs(uz1/w_ana - 0.5) < 0.1:
    print("DIAGNOSIS: Approach 1 gives 2× too small displacement")
    print("  → Stiffness is doubled somewhere")
elif abs(uz1/w_ana - 1.0) < 0.1:
    print("DIAGNOSIS: Both approaches agree with analytical")
    print("  → Wing factor-of-2 is not in the weak form — check traction mapping")
else:
    print(f"DIAGNOSIS: Unexpected ratio {uz1/w_ana:.4f} — check formulation")

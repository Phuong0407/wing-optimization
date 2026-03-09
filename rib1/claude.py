"""
RibTopOpt.py
=============
Topology optimization of a rib (plane stress 2D).
Traction: spatially varying, interpolated from FOAM wing skin
          onto every FEniCS node of the rib mesh.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from scipy.interpolate import NearestNDInterpolator
import ufl
import meshio
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

RIB_XDMF  = "../data/CAD/ribs/rib_2.xdmf"
FOAM_XDMF = "../data/FOAM/ExtractedFOAMData.xdmf"
RIB_INDEX = 2

VOLFRAC   = 0.4
PENAL     = 3.0
E0        = 210e9
Emin      = E0 * 1e-9
NU        = 0.3
MOVE      = 0.2
MAX_ITER  = 200
TOL       = 1e-3


# ─────────────────────────────────────────────────────────────────────
# LOAD MESH INTO FENICS FIRST (need node coords for interpolation)
# ─────────────────────────────────────────────────────────────────────

with XDMFFile(MPI.COMM_WORLD, RIB_XDMF, "r") as f:
    domain = f.read_mesh(name="Grid")

gdim = domain.geometry.dim   # 3
tdim = domain.topology.dim   # 2
print(f"Mesh: {domain.topology.index_map(tdim).size_local} cells, gdim={gdim}")

# FEniCS node coordinates (CG1 nodes = mesh geometry nodes)
V   = fem.functionspace(domain, ("CG", 1, (gdim,)))
DG0 = fem.functionspace(domain, ("DG", 0))

# Scalar CG1 for interpolation
V1  = fem.functionspace(domain, ("CG", 1))

# Node coordinates of the rib (ordered by FEniCS dof numbering)
node_coords = V1.tabulate_dof_coordinates()   # (N_nodes, 3)
y_rib       = float(np.mean(node_coords[:, 1]))


print(f"Rib y = {y_rib:.4f} m,  nodes = {len(node_coords)}")


# ─────────────────────────────────────────────────────────────────────
# INTERPOLATE FOAM TRACTION ONTO EVERY RIB NODE
# ─────────────────────────────────────────────────────────────────────

foam      = meshio.read(FOAM_XDMF)
foam_pts  = foam.points               # (N, 3) metres
foam_trac = foam.point_data["traction"]  # (N, 3) Pa

# Only use FOAM points near y_rib (±15% span) for speed + accuracy
y_span = foam_pts[:, 1].max() - foam_pts[:, 1].min()
band   = 0.15 * y_span
mask   = np.abs(foam_pts[:, 1] - y_rib) < band
print(f"FOAM points in band: {mask.sum()} / {len(foam_pts)}")

interp_x = NearestNDInterpolator(foam_pts[mask], foam_trac[mask, 0])
interp_y = NearestNDInterpolator(foam_pts[mask], foam_trac[mask, 1])
interp_z = NearestNDInterpolator(foam_pts[mask], foam_trac[mask, 2])

# Evaluate at every FEniCS CG1 node
trac_x = interp_x(node_coords)   # (N_nodes,)
trac_y = interp_y(node_coords)
trac_z = interp_z(node_coords)

print(f"Traction X: {trac_x.min():.2f} -> {trac_x.max():.2f} Pa")
print(f"Traction Z: {trac_z.min():.2f} -> {trac_z.max():.2f} Pa")

# Store as 3 scalar FEniCS Functions, then combine into vector Function
f_func  = fem.Function(V, name="traction")

# Fill vector function: dofs are interleaved [x0,y0,z0, x1,y1,z1, ...]
f_func.x.array[0::gdim] = trac_x   # X component at each node
f_func.x.array[1::gdim] = trac_y   # Y component (spanwise, ~0 for in-plane)
f_func.x.array[2::gdim] = trac_z   # Z component at each node

f_func.x.scatter_forward()


# ─────────────────────────────────────────────────────────────────────
# SIMP + PLANE STRESS
# ─────────────────────────────────────────────────────────────────────

rho = fem.Function(DG0, name="density")
rho.x.array[:] = VOLFRAC

def simp(rho):
    return Emin + rho**PENAL * (E0 - Emin)

def simp_deriv(rho):
    return PENAL * rho**(PENAL - 1) * (E0 - Emin)

def strain(u):
    return ufl.sym(ufl.grad(u))

def stress(u, rho_f):
    E        = simp(rho_f)
    lmbda    = E * NU / ((1 + NU) * (1 - 2*NU))
    mu       = E / (2 * (1 + NU))
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)
    return lmbda_ps * ufl.tr(strain(u)) * ufl.Identity(gdim) + 2*mu*strain(u)


# ─────────────────────────────────────────────────────────────────────
# BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────────────

pts_np = domain.geometry.x
chord  = pts_np[:, 0].max() - pts_np[:, 0].min()
x_min  = pts_np[:, 0].min()

def spar_boundary(x):
    front = np.abs(x[0] - (x_min + 0.25 * chord)) < 0.02 * chord
    rear  = np.abs(x[0] - (x_min + 0.75 * chord)) < 0.02 * chord
    return front | rear

spar_dofs = fem.locate_dofs_geometrical(V, spar_boundary)
u_zero    = fem.Function(V)
bcs       = [fem.dirichletbc(u_zero, spar_dofs)]


# ─────────────────────────────────────────────────────────────────────
# VARIATIONAL FORMS
# Spatially varying traction: f_func varies per node on ds
# ─────────────────────────────────────────────────────────────────────

u      = ufl.TrialFunction(V)
v_test = ufl.TestFunction(V)

a_form = ufl.inner(stress(u, rho), strain(v_test)) * ufl.dx
L_form = ufl.dot(f_func, v_test) * ufl.ds   # f_func varies in space


# ─────────────────────────────────────────────────────────────────────
# OC UPDATE
# ─────────────────────────────────────────────────────────────────────

def OC_update(rho_old, dc, volfrac, move=MOVE):
    l1, l2 = 1e-9, 1e9
    while (l2 - l1) / (l1 + l2) > 1e-6:
        lmid    = 0.5 * (l1 + l2)
        rho_new = np.clip(
            rho_old * np.sqrt(np.maximum(-dc / lmid, 0)),
            np.maximum(rho_old - move, 1e-3),
            np.minimum(rho_old + move, 1.0),
        )
        if rho_new.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


# ─────────────────────────────────────────────────────────────────────
# OPTIMIZATION LOOP
# ─────────────────────────────────────────────────────────────────────

results_dir = Path("../data/CAD/ribs/topopt")
results_dir.mkdir(parents=True, exist_ok=True)

with io.VTKFile(MPI.COMM_WORLD,
                str(results_dir / f"rib_{RIB_INDEX}_topopt.pvd"), "w") as vtk:
    vtk.write_mesh(domain)

    for it in range(MAX_ITER):

        prob = LinearProblem(a_form, L_form, bcs=bcs,
                             petsc_options={"ksp_type": "cg",
                                            "pc_type":  "gamg"})
        uh = prob.solve()

        # Compliance = ∫ f·u ds  (work done by traction)
        compliance = fem.assemble_scalar(
            fem.form(ufl.dot(f_func, uh) * ufl.ds)
        )

        # Sensitivity
        dE      = simp_deriv(rho)
        dc_form = fem.form(-dE * ufl.inner(strain(uh), strain(uh)) * ufl.dx)
        dc_vec  = fem.assemble_vector(dc_form)

        rho_old        = rho.x.array.copy()
        rho.x.array[:] = OC_update(rho_old, dc_vec.array, VOLFRAC)

        change = np.max(np.abs(rho.x.array - rho_old))
        print(f"Iter {it:3d}  C={compliance:.4e}  "
              f"vol={rho.x.array.mean():.3f}  change={change:.5f}")

        if it % 10 == 0:
            vtk.write_function(rho, float(it))

        if change < TOL:
            print(f"Converged at iteration {it}")
            break

    vtk.write_function(rho, float(it))

with XDMFFile(MPI.COMM_WORLD,
              str(results_dir / f"rib_{RIB_INDEX}_rho_final.xdmf"), "w") as f:
    f.write_mesh(domain)
    f.write_function(rho)

print(f"Done -> {results_dir}")
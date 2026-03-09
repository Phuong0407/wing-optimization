import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import NearestNDInterpolator
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem import Expression, Function, functionspace, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.io import XDMFFile
from dolfinx.mesh import compute_midpoints
from petsc4py import PETSc
import ufl
from ufl import sym, grad, tr, Identity, inner, dot, dx
import meshio
from pathlib import Path

RIB_INDEX    = 0
RIB_XDMF     = f"rib_{RIB_INDEX}.xdmf"
FOAM_XDMF    = "FOAMData.xdmf"
FREE_VOLFRAC = 0.6
P_START      = 1.0
P_MAX        = 5.0
P_RAMP       = 80
LAME_NU      = 0.3
E0           = 210e9
FILTER_R     = 0.020
MAX_ITER     = 500
TOL          = 1e-3

with XDMFFile(MPI.COMM_WORLD, RIB_XDMF, "r") as f:
    domain = f.read_mesh(name="Grid")

gdim      = domain.geometry.dim
tdim      = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
midpoints = compute_midpoints(domain, tdim, np.arange(num_cells))
pts_np    = domain.geometry.x

CG1    = functionspace(domain, ("CG", 1, (gdim,)))
DG0    = functionspace(domain, ("DG", 0))
rho    = Function(DG0, name="density")
energy = Function(DG0, name="strain_energy")

# ── LOAD: DP từ FOAM ──────────────────────────────────────────────────
V1          = functionspace(domain, ("CG", 1))
node_coords = V1.tabulate_dof_coordinates()
y_rib       = float(np.mean(node_coords[:, 1]))

foam      = meshio.read(FOAM_XDMF)
foam_pts  = foam.points
foam_p    = foam.point_data["p"]
foam_n    = foam.point_data["normals"]
y_span    = foam_pts[:,1].max() - foam_pts[:,1].min()
band      = np.abs(foam_pts[:,1] - y_rib) < 0.15 * y_span
nz        = foam_n[:, 2]

foam_upper = band & (nz >  0.1)
foam_lower = band & (nz < -0.1)

p_upper = NearestNDInterpolator(
    foam_pts[foam_upper], foam_p[foam_upper])(node_coords)
p_lower = NearestNDInterpolator(
    foam_pts[foam_lower], foam_p[foam_lower])(node_coords)
dp = p_lower - p_upper

f_func = Function(CG1, name="load")
f_func.x.array[0::gdim] = 0.0
f_func.x.array[1::gdim] = 0.0
f_func.x.array[2::gdim] = dp
f_func.x.scatter_forward()
print(f"DP mean={dp.mean():.1f} Pa | norm={np.linalg.norm(f_func.x.array):.3e}")

# ── CONSTITUTIVE ──────────────────────────────────────────────────────
lame_mu  = E0 / (2*(1+LAME_NU))
lame_lam = E0*LAME_NU / ((1+LAME_NU)*(1-2*LAME_NU))
lame_lps = 2*lame_lam*lame_mu / (lame_lam+2*lame_mu)

def epsilon(u): return sym(grad(u))
def sigma(u):   return rho*(lame_lps*tr(epsilon(u))*Identity(gdim)
                             + 2*lame_mu*epsilon(u))
def psi(u, v):  return inner(sigma(u), epsilon(v))

# ── BC ────────────────────────────────────────────────────────────────
chord = pts_np[:,0].max() - pts_np[:,0].min()
x_le  = pts_np[:,0].min()
front_dofs = locate_dofs_geometrical(
    CG1, lambda x: np.abs(x[0]-(x_le+0.25*chord)) < 0.05*chord)
rear_dofs  = locate_dofs_geometrical(
    CG1, lambda x: np.abs(x[0]-(x_le+0.75*chord)) < 0.05*chord)
bcs = [fem.dirichletbc(Function(CG1), np.union1d(front_dofs, rear_dofs))]

# ── PASSIVE MASK ──────────────────────────────────────────────────────
z_min    = pts_np[:,2].min()
z_max    = pts_np[:,2].max()
FLANGE_T = 0.05 * (z_max - z_min)
passive_mask = (
    (midpoints[:,2] < z_min + FLANGE_T) |
    (midpoints[:,2] > z_max - FLANGE_T)
)
free_mask = ~passive_mask
print(f"Passive={passive_mask.sum()}/{num_cells}={passive_mask.mean()*100:.1f}%")

x_var               = np.full(num_cells, FREE_VOLFRAC)
x_var[passive_mask] = 1.0

# ── FORMS & SOLVER ────────────────────────────────────────────────────
u      = ufl.TrialFunction(CG1)
v_test = ufl.TestFunction(CG1)
a_form = form(psi(u, v_test) * dx)
L_form = form(dot(f_func, v_test) * dx)

A = assemble_matrix(a_form, bcs=bcs); A.assemble()
b = assemble_vector(L_form)
apply_lifting(b, [a_form], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.GAMG)
solver.setTolerances(rtol=1e-8)
solver.setFromOptions()
uh = Function(CG1, name="displacement")

# ── FILTER ────────────────────────────────────────────────────────────
tree     = KDTree(midpoints)
dist_mat = tree.sparse_distance_matrix(tree, FILTER_R).tocsr()
dist_mat.data = np.maximum(FILTER_R - dist_mat.data, 0)
omega     = dist_mat
omega_sum = np.array(omega.sum(1)).flatten()

# ── VOLUMES ───────────────────────────────────────────────────────────
vol_vec      = fem.assemble_vector(form(ufl.TestFunction(DG0) * dx))
cell_volumes = vol_vec.array.copy()
free_volume  = cell_volumes[free_mask].sum()

# ── LOOP ──────────────────────────────────────────────────────────────
results_dir = Path("topopt"); results_dir.mkdir(exist_ok=True)
best_compliance = np.inf
best_x          = x_var.copy()
change_hist     = []

with io.VTKFile(MPI.COMM_WORLD,
                str(results_dir/f"rib_{RIB_INDEX}_topopt.pvd"), "w") as vtk:
    vtk.write_mesh(domain)
    for it in range(MAX_ITER):
        P_SIMP = min(P_MAX, P_START + it*(P_MAX-P_START)/P_RAMP)

        # ── Adaptive MOVE ──────────────────────────────────────────────
        if it < P_RAMP:
            MOVE = 0.2                                    # explore
        else:
            MOVE = max(0.2 * (0.97**(it-P_RAMP)), 0.01) # refine

        rho.x.array[:] = x_var ** P_SIMP
        A.zeroEntries()
        assemble_matrix(A, a_form, bcs=bcs); A.assemble()

        with b.localForm() as bl: bl.set(0)
        assemble_vector(b, L_form)
        apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        energy.interpolate(
            Expression(psi(uh,uh), DG0.element.interpolation_points))
        compliance = float(energy.x.array @ cell_volumes)

        raw_sens    = P_SIMP*(x_var**(P_SIMP-1))*np.maximum(energy.x.array,0)
        sensitivity = (omega @ raw_sens) / (omega_sum + 1e-30)

        # OC bisection
        l1, l2 = 1e-9, 1e9
        x_new  = x_var.copy()
        while (l2-l1)/(l1+l2) > 1e-5:
            lmid  = 0.5*(l1+l2)
            Be    = sensitivity / (lmid*cell_volumes + 1e-30)
            x_new = np.clip(np.clip(x_var*np.sqrt(Be),
                                    x_var-MOVE, x_var+MOVE), 0.001, 1.0)
            x_new[passive_mask] = 1.0
            vol = (x_new[free_mask]*cell_volumes[free_mask]).sum()/free_volume
            if vol > FREE_VOLFRAC: l1 = lmid
            else:                  l2 = lmid

        change = float(np.max(np.abs(x_new[free_mask]-x_var[free_mask])))
        x_var  = x_new.copy()
        grey   = float(np.logical_and(0.15<x_var[free_mask],
                        x_var[free_mask]<0.85).sum()) / free_mask.sum()

        if compliance < best_compliance:
            best_compliance = compliance
            best_x          = x_var.copy()
            marker = " ✓"
        else:
            marker = ""

        change_hist.append(change)
        print(f"Iter {it:3d} | p={P_SIMP:.1f} | C={compliance:.4e}{marker} | "
              f"vol={vol:.3f} | Δx={change:.4f} | grey={grey:.3f} | mv={MOVE:.3f}")

        if it % 20 == 0:
            rho.x.array[:] = x_var
            vtk.write_function(rho, float(it))

        if it > P_RAMP and change < TOL:
            print(f"✓ Converged at iter {it}"); break

    x_var = best_x.copy()
    rho.x.array[:] = x_var
    vtk.write_function(rho, float(it))

with XDMFFile(MPI.COMM_WORLD,
              str(results_dir/f"rib_{RIB_INDEX}_rho_final.xdmf"), "w") as f:
    f.write_mesh(domain)
    f.write_function(rho)

print(f"✓ Done | best_C={best_compliance:.4e}")
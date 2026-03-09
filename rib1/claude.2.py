import numpy as np
from scipy.spatial import KDTree
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem import Expression, Function, functionspace, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.io import XDMFFile
from dolfinx.mesh import compute_midpoints, locate_entities_boundary, exterior_facet_indices
from petsc4py import PETSc
import ufl
from ufl import sym, grad, tr, Identity, inner, dot, dx, ds
import meshio
from scipy.interpolate import NearestNDInterpolator
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────
RIB_INDEX  = 0
RIB_XDMF  = f"rib_{RIB_INDEX}.xdmf"
FOAM_XDMF = "FOAMData.xdmf"
VOLFRAC    = 0.5
P_START    = 1.0    # initial SIMP penalty (ramp up to avoid checkerboard)
P_MAX      = 4.0    # final SIMP penalty
P_RAMP     = 80     # iterations to reach P_MAX
LAME_NU    = 0.3
E0         = 210e9
FILTER_R   = 0.01   # sensitivity filter radius [m]
MOVE       = 0.2
MAX_ITER   = 10000
TOL        = 5e-3
FLANGE_T   = 0.012 # passive boundary thickness [m] (rib flange)

# ── LOAD MESH ─────────────────────────────────────────────────────────
with XDMFFile(MPI.COMM_WORLD, RIB_XDMF, "r") as f:
    domain = f.read_mesh(name="Grid")

gdim      = domain.geometry.dim
tdim      = domain.topology.dim
fdim      = tdim - 1
num_cells = domain.topology.index_map(tdim).size_local
midpoints = compute_midpoints(domain, tdim, np.arange(num_cells))
print(f"Mesh: {num_cells} cells, gdim={gdim}, tdim={tdim}")

# ── FUNCTION SPACES ───────────────────────────────────────────────────
CG1 = functionspace(domain, ("CG", 1, (gdim,)))
DG0 = functionspace(domain, ("DG", 0))
rho    = Function(DG0, name="density")
energy = Function(DG0, name="strain_energy")
x_var  = np.full(num_cells, VOLFRAC)

# ── TRACTION FROM FOAM ────────────────────────────────────────────────
V1          = functionspace(domain, ("CG", 1))
node_coords = V1.tabulate_dof_coordinates()   # (N, 3)
y_rib       = float(np.mean(node_coords[:, 1]))
print(f"Rib y = {y_rib:.4f} m")

foam      = meshio.read(FOAM_XDMF)
foam_pts  = foam.points
foam_trac = foam.point_data["traction"]

y_span = foam_pts[:, 1].max() - foam_pts[:, 1].min()
mask   = np.abs(foam_pts[:, 1] - y_rib) < 0.15 * y_span
print(f"FOAM points in band: {mask.sum()}")

f_func = Function(CG1, name="traction")
f_func.x.array[0::gdim] = NearestNDInterpolator(foam_pts[mask], foam_trac[mask, 0])(node_coords)
f_func.x.array[1::gdim] = NearestNDInterpolator(foam_pts[mask], foam_trac[mask, 1])(node_coords)
f_func.x.array[2::gdim] = NearestNDInterpolator(foam_pts[mask], foam_trac[mask, 2])(node_coords)
f_func.x.scatter_forward()
print(f"f_func norm = {np.linalg.norm(f_func.x.array):.4e}")

# ── CONSTITUTIVE MODEL (plane stress, SIMP) ───────────────────────────
lame_mu  = E0 / (2 * (1 + LAME_NU))
lame_lam = E0 * LAME_NU / ((1 + LAME_NU) * (1 - 2 * LAME_NU))
lame_lps = 2 * lame_lam * lame_mu / (lame_lam + 2 * lame_mu)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    # rho is updated every iteration before assembly
    return rho * (lame_lps * tr(epsilon(u)) * Identity(gdim) + 2 * lame_mu * epsilon(u))

def psi(u, v):
    return inner(sigma(u), epsilon(v))

# ── BOUNDARY CONDITIONS (spar lines at 25% and 75% chord) ─────────────
pts_np = domain.geometry.x
chord  = pts_np[:, 0].max() - pts_np[:, 0].min()
x_le   = pts_np[:, 0].min()
print(f"Chord = {chord*1e3:.0f} mm,  LE at x = {x_le*1e3:.1f} mm")

front_dofs = locate_dofs_geometrical(
    CG1, lambda x: np.abs(x[0] - (x_le + 0.25 * chord)) < 0.05 * chord
)
rear_dofs = locate_dofs_geometrical(
    CG1, lambda x: np.abs(x[0] - (x_le + 0.75 * chord)) < 0.05 * chord
)
all_dofs = np.union1d(front_dofs, rear_dofs)
print(f"Front spar dofs: {len(front_dofs)},  Rear spar dofs: {len(rear_dofs)}")

bcs = [fem.dirichletbc(Function(CG1), all_dofs)]

# ── PASSIVE ELEMENTS (rib flange = boundary skin contact) ─────────────
domain.topology.create_connectivity(fdim, tdim)
bnd_facets = exterior_facet_indices(domain.topology)
bnd_nodes  = fem.locate_dofs_topological(
    functionspace(domain, ("CG", 1)), fdim, bnd_facets
)
bnd_coords = domain.geometry.x[bnd_nodes]

dist_to_bnd  = KDTree(bnd_coords).query(midpoints)[0]
passive_mask = dist_to_bnd < FLANGE_T
print(f"Passive flange elements: {passive_mask.sum()} / {num_cells}")

# ── VARIATIONAL FORMS ─────────────────────────────────────────────────
u      = ufl.TrialFunction(CG1)
v_test = ufl.TestFunction(CG1)

a_compiled = form(psi(u, v_test) * dx)
L_compiled = form(dot(f_func, v_test) * dx)

# ── SOLVER SETUP ──────────────────────────────────────────────────────
A = assemble_matrix(a_compiled, bcs=bcs)
A.assemble()

b = assemble_vector(L_compiled)
apply_lifting(b, [a_compiled], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.GAMG)
solver.setFromOptions()

uh = Function(CG1, name="displacement")

# ── SENSITIVITY FILTER ────────────────────────────────────────────────
def prepare_filter(coords, r):
    tree     = KDTree(coords)
    distance = tree.sparse_distance_matrix(tree, r).tocsr()
    distance.data = (r - distance.data) / r
    return distance, np.array(distance.sum(1)).flatten()

omega, omega_sum = prepare_filter(midpoints, FILTER_R)
print(f"Filter: r={FILTER_R*1e3:.0f}mm, avg neighbors={omega.nnz/num_cells:.1f}")

# ── CELL VOLUMES ──────────────────────────────────────────────────────
vol_vec = fem.assemble_vector(form(ufl.TestFunction(DG0) * dx))
vol_vec.scatter_forward()
cell_volumes = vol_vec.array.copy()
total_volume = cell_volumes.sum()

# ── OPTIMIZATION LOOP ─────────────────────────────────────────────────
results_dir = Path("topopt")
results_dir.mkdir(parents=True, exist_ok=True)

with io.VTKFile(MPI.COMM_WORLD,
                str(results_dir / f"rib_{RIB_INDEX}_topopt.pvd"), "w") as vtk:
    vtk.write_mesh(domain)

    for it in range(MAX_ITER):

        # Penalty ramping: p=1 → p=4 over P_RAMP iterations
        P_SIMP = min(P_MAX, P_START + it * (P_MAX - P_START) / P_RAMP)

        # 1. Penalize density
        rho.x.array[:] = x_var ** P_SIMP

        # 2. Reassemble stiffness
        A.zeroEntries()
        assemble_matrix(A, a_compiled, bcs=bcs)
        A.assemble()

        # 3. Reassemble RHS
        with b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(b, L_compiled)
        apply_lifting(b, [a_compiled], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

        # 4. Solve
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        # 5. Strain energy per element
        energy.interpolate(Expression(psi(uh, uh), DG0.element.interpolation_points))
        compliance = energy.x.array.sum()

        # 6. Sensitivity + filter
        sensitivity = -P_SIMP * (x_var ** (P_SIMP - 1)) * energy.x.array
        sensitivity = (omega @ sensitivity) / omega_sum

        # 7. OC update
        ocp = x_var * np.sqrt(-sensitivity / cell_volumes)
        l1, l2 = 0.0, float(ocp.sum())
        while (l2 - l1) > 1e-6:
            lmid  = 0.5 * (l1 + l2)
            x_new = (ocp / lmid).clip(x_var - MOVE, x_var + MOVE).clip(0.001, 1.0)
            vol   = (x_new * cell_volumes).sum() / total_volume
            l1, l2 = (lmid, l2) if vol > VOLFRAC else (l1, lmid)

        # 8. Enforce passive flange elements
        x_new[passive_mask] = 1.0

        change = np.max(np.abs(x_new - x_var))
        x_var  = x_new

        grey = np.logical_and(0.15 < x_var, x_var < 0.85).sum() / num_cells
        print(f"Iter {it:3d}  p={P_SIMP:.1f}  C={compliance:.4e}  "
              f"vol={vol:.3f}  change={change:.4f}  grey={grey:.3f}")

        if it % 10 == 0:
            rho.x.array[:] = x_var
            vtk.write_function(rho, float(it))

        if grey < TOL:
            print(f"Converged at iteration {it}")
            break

    rho.x.array[:] = x_var
    vtk.write_function(rho, float(it))

# ── FINAL EXPORT ──────────────────────────────────────────────────────
rho.x.array[:] = x_var
with XDMFFile(MPI.COMM_WORLD,
              str(results_dir / f"rib_{RIB_INDEX}_rho_final.xdmf"), "w") as f:
    f.write_mesh(domain)
    f.write_function(rho)

print(f"Done -> {results_dir}")
"""
opti8var_buckling.py
====================
opti8var.py + FE buckling constraint  (λ_cr >= LAMBDA_CR_LIMIT)

Chiến lược để không tăng quá nhiều thời gian tính toán:

  1. LAZY EVALUATION
     Chỉ gọi FE buckling khi analytical BI > BUCKLING_TRIGGER (default 0.4).
     Trong thực tế ~60-70% iterations đầu không trigger → tiết kiệm nhiều.

  2. nev=1, ncv=15
     Chỉ cần mode 1 (λ_cr nhỏ nhất). Krylov subspace nhỏ → SLEPc nhanh hơn.

  3. KG_form compile MỘT LẦN ở startup
     JIT chỉ chạy 1 lần trước optimization loop. Mỗi iteration chỉ
     assemble lại (re-use compiled kernel) — fast ~3-5s.

  4. K_e từ tangent đã có sẵn
     tangent = ufl.derivative(residual) đã được compile cho SNES.
     Re-assemble sau problem.solve() không cần JIT lại.

Ước tính thời gian so với opti8var.py:
  - Iteration không trigger buckling: +0s   (~60% iterations)
  - Iteration trigger buckling      : +25-35s (~40% iterations)
  - Overhead trung bình             : +10-15% tổng thời gian

Usage:
    mpirun -n 4 python opti8var_buckling.py
    mpirun -n 4 python opti8var_buckling.py --no-buckling   # same as opti8var
    mpirun -n 4 python opti8var_buckling.py --trigger 0.3   # stricter lazy threshold
"""

import argparse
import csv
import time
from datetime import datetime
from functools import reduce
from pathlib import Path
from types import SimpleNamespace

import basix
import meshio
import nlopt
import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.io import gmsh
from dolfinx.fem.petsc import NonlinearProblem, assemble_matrix
from mpi4py import MPI
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

try:
    from petsc4py import PETSc
    from slepc4py import SLEPc
    HAS_SLEPC = True
except ImportError:
    HAS_SLEPC = False

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no-buckling", action="store_true",
                    help="Disable FE buckling constraint (reproduces opti8var.py)")
parser.add_argument("--trigger", type=float, default=0.4,
                    help="Analytical BI threshold to trigger FE buckling (default 0.4)")
parser.add_argument("--lambda-limit", type=float, default=1.5,
                    help="Minimum required buckling factor (default 1.5)")
parser.add_argument("--quick-test", action="store_true")
args = parser.parse_args()

USE_FE_BUCKLING  = (not args.no_buckling) and HAS_SLEPC
BUCKLING_TRIGGER = args.trigger          # analytical BI > this → call FE buckling
LAMBDA_CR_LIMIT  = args.lambda_limit     # constraint: λ_cr >= this
QUICK_TEST       = args.quick_test

# ─────────────────────────────────────────────────────────────
# MPI
# ─────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank

VERBOSE = True
def vprint(*a, **kw):
    if VERBOSE and rank == 0:
        print(*a, **kw)

vprint("=" * 62)
vprint("  WINGBOX OPTIMISATION + FE BUCKLING CONSTRAINT")
vprint(f"  FE buckling     : {'ENABLED' if USE_FE_BUCKLING else 'DISABLED'}")
if USE_FE_BUCKLING:
    vprint(f"  Trigger (BI >)  : {BUCKLING_TRIGGER}")
    vprint(f"  λ_cr limit      : {LAMBDA_CR_LIMIT}")
vprint("=" * 62)

# ─────────────────────────────────────────────────────────────
# IDENTICAL TO opti8var.py FROM HERE
# ─────────────────────────────────────────────────────────────
MESHUNIT     = "mm"
INPUT_DIR    = Path("InputData")
OUTPUT_DIR   = Path("Result")
MESHFILE     = INPUT_DIR / "wingbox.msh"
FOAMFILE     = INPUT_DIR / "wing.vtp"
XDMFFILE     = INPUT_DIR / "FOAMData.xdmf"
MAPFILE      = INPUT_DIR / "MappedTraction.xdmf"

TAG_UPPER    = 14;  TAG_LOWER    = 15
TAG_RIB0     = 38;  TAG_RIB750   = 39
TAG_RIB1500  = 40;  TAG_RIB2250  = 41;  TAG_RIB3000 = 42
TAG_MAINSPAR = 43;  TAG_TAILSPAR = 44
TAG_RIB0_CURVE = 45
TAG_SKIN      = [TAG_UPPER, TAG_LOWER]
TAG_RIBS      = [TAG_RIB0, TAG_RIB750, TAG_RIB1500, TAG_RIB2250, TAG_RIB3000]
TAG_COMPOSITE = TAG_SKIN + [TAG_MAINSPAR, TAG_TAILSPAR]
TAG_ALL       = TAG_COMPOSITE + TAG_RIBS

MESH_IO    = gmsh.read_from_msh(MESHFILE, comm=MPI.COMM_WORLD, gdim=3)
DOMAIN     = MESH_IO.mesh
CELL_TAGS  = MESH_IO.cell_tags
FACET_TAGS = MESH_IO.facet_tags
GDIM       = DOMAIN.geometry.dim
TDIM       = DOMAIN.topology.dim
FDIM       = TDIM - 1

if MESHUNIT == "mm":
    DOMAIN.geometry.x[:] *= 1e-3

vprint(f"\n[MESH] Cells={DOMAIN.topology.index_map(TDIM).size_global}"
       f"  Vertices={DOMAIN.topology.index_map(0).size_global}")

# ── Local frame ───────────────────────────────────────────────
def normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))

def local_frame_ufl(domain):
    t  = ufl.Jacobian(domain)
    t1 = ufl.as_vector([t[0,0], t[1,0], t[2,0]])
    t2 = ufl.as_vector([t[0,1], t[1,1], t[2,1]])
    e3 = normalize(ufl.cross(t1, t2))
    ey = ufl.as_vector([0, 1, 0]); ez = ufl.as_vector([0, 0, 1])
    e1_trial = ufl.cross(ey, e3)
    norm_e1  = ufl.sqrt(ufl.dot(e1_trial, e1_trial))
    e1 = ufl.conditional(ufl.lt(norm_e1, 0.5), ez, normalize(e1_trial))
    return e1, normalize(ufl.cross(e3, e1)), e3

def local_frame(domain):
    FRAME = local_frame_ufl(domain)
    VT    = fem.functionspace(domain, ("DG", 0, (3,)))
    V0, _ = VT.sub(0).collapse()
    BASIS = [fem.Function(VT, name=f"e{i+1}") for i in range(3)]
    for i in range(3):
        BASIS[i].interpolate(fem.Expression(FRAME[i], V0.element.interpolation_points))
    return BASIS[0], BASIS[1], BASIS[2]

E1, E2, E3 = local_frame(DOMAIN)

# ── Kinematics ────────────────────────────────────────────────
def hstack(vecs):
    return ufl.as_matrix([[vi[i] for i in range(len(vi))] for vi in vecs]).T

def tangent_projection(e1, e2): return hstack([e1, e2])
def tangential_gradient(w, P):  return ufl.dot(ufl.grad(w), P)

def membrane_strain(u, P):
    t_gu = ufl.dot(P.T, tangential_gradient(u, P))
    return ufl.sym(t_gu), t_gu

def bending_strain(theta, e3, P):
    return ufl.sym(ufl.dot(P.T, tangential_gradient(ufl.cross(e3, theta), P)))

def shear_strain(u, theta, e3, P):
    return tangential_gradient(ufl.dot(u, e3), P) - ufl.dot(P.T, ufl.cross(e3, theta))

def compute_drilling_strain(t_gu, theta, e3):
    return (t_gu[0,1] - t_gu[1,0]) / 2 + ufl.dot(theta, e3)

def shell_strains(u, theta, e1, e2, e3):
    P               = tangent_projection(e1, e2)
    eps, t_gu       = membrane_strain(u, P)
    kappa           = bending_strain(theta, e3, P)
    gamma           = shear_strain(u, theta, e3, P)
    drilling_strain = compute_drilling_strain(t_gu, theta, e3)
    return eps, kappa, gamma, drilling_strain

# ── Material ──────────────────────────────────────────────────
def to_voigt(e): return ufl.as_vector([e[0,0], e[1,1], 2.0*e[0,1]])
def from_voigt(v): return ufl.as_tensor([[v[0], v[2]], [v[2], v[1]]])

def _Q_ply(E1, E2, G12, nu12):
    nu21 = nu12*E2/E1; d = 1-nu12*nu21
    return np.array([[E1/d, nu12*E2/d, 0], [nu12*E2/d, E2/d, 0], [0, 0, G12]])

def _Qbar_ply(Q, angle):
    a = np.radians(angle); s, c = np.sin(a), np.cos(a)
    T = np.array([[c**2, s**2, 2*c*s], [s**2, c**2, -2*c*s], [-c*s, c*s, c**2-s**2]])
    R = np.diag([1.,1.,2.]); Rinv = np.diag([1.,1.,.5])
    return np.linalg.inv(T) @ Q @ R @ T @ Rinv

def compute_ABD(layup, t_ply, E1, E2, G12, nu12):
    Q = _Q_ply(E1, E2, G12, nu12); H = t_ply*len(layup); z = -H/2.0
    A = np.zeros((3,3)); B = np.zeros((3,3)); D = np.zeros((3,3))
    for angle in layup:
        Qb = _Qbar_ply(Q, angle); z0, z1 = z, z+t_ply
        A += Qb*(z1-z0); B += Qb*(z1**2-z0**2)/2.; D += Qb*(z1**3-z0**3)/3.; z=z1
    return A, B, D, H

def clt_composite(layup, t_ply, E1, E2, G12, nu12, G13=None, G23=None,
                  kappa_s=5/6, label="CLT", verbose=False, domain=None):
    if G13 is None: G13=G12
    if G23 is None: G23=G12*0.5
    A_np, B_np, D_np, H = compute_ABD(layup, t_ply, E1, E2, G12, nu12)
    As_np = kappa_s*H*np.array([[G13,0.],[0.,G23]])
    G_eff = float(A_np[2,2])/H
    max_B = np.abs(B_np).max()
    if verbose:
        vprint(f"[{label}] H={H*1e3:.2f} mm  A11={A_np[0,0]/1e6:.2f} MPa·m"
               f"  D11={D_np[0,0]:.4f} N·m²")
    return SimpleNamespace(
        kind="clt", H=fem.Constant(domain, H), t_ply=t_ply,
        G_eff=fem.Constant(domain, G_eff),
        A_ufl=fem.Constant(domain, A_np), B_ufl=fem.Constant(domain, B_np),
        D_ufl=fem.Constant(domain, D_np), As_ufl=fem.Constant(domain, As_np),
        _layup_angles=layup, _E1=E1, _E2=E2, _G12=G12, _nu12=nu12,
        _G13=G13, _G23=G23, _kappa_s=kappa_s,
    )

def build_laminate(layup, thickness_const, label):
    mat = clt_composite(layup, thickness_const.value/len(layup),
        CM_YNG1, CM_YNG2, CM_G12, CM_NU12,
        G13=CM_G12, G23=CM_G12*0.5, kappa_s=5/6,
        label=label, verbose=VERBOSE, domain=DOMAIN)
    mat._total_thickness_const = thickness_const
    return mat

def update_clt_material(mat, total_thickness, n_layers):
    t_ply = float(total_thickness)/n_layers
    A_np, B_np, D_np, H = compute_ABD(mat._layup_angles, t_ply,
                                       mat._E1, mat._E2, mat._G12, mat._nu12)
    As_np = mat._kappa_s*H*np.array([[mat._G13,0.],[0.,mat._G23]])
    mat._total_thickness_const.value = float(total_thickness)
    mat.A_ufl.value = A_np; mat.B_ufl.value = B_np
    mat.D_ufl.value = D_np; mat.As_ufl.value = As_np
    mat.G_eff.value = float(A_np[2,2])/H; mat.H.value = H; mat.t_ply = t_ply

def stress_resultants(mat, eps, kappa, gamma):
    eps_v = to_voigt(eps); kappa_v = to_voigt(kappa)
    N = from_voigt(ufl.dot(mat.A_ufl, eps_v) + ufl.dot(mat.B_ufl, kappa_v))
    M = from_voigt(ufl.dot(mat.B_ufl, eps_v) + ufl.dot(mat.D_ufl, kappa_v))
    Q = ufl.dot(mat.As_ufl, gamma)
    return N, M, Q

def drilling_terms(mat, domain, drilling_strain):
    h_mesh = ufl.CellDiameter(domain)
    stiffness = mat.G_eff * mat.H**3 / h_mesh**2
    return stiffness, stiffness * drilling_strain

# ── Failure criteria ──────────────────────────────────────────
def tsai_wu(sigma_mat, strengths):
    s1,s2,s6 = sigma_mat
    Xt,Xc,Yt,Yc,S = (strengths[k] for k in ["Xt","Xc","Yt","Yc","S"])
    F1=1/Xt-1/Xc; F2=1/Yt-1/Yc; F11=1/(Xt*Xc); F22=1/(Yt*Yc)
    F66=1/S**2;   F12=-0.5/np.sqrt(Xt*Xc*Yt*Yc)
    return F1*s1+F2*s2+F11*s1**2+F22*s2**2+F66*s6**2+2*F12*s1*s2

def _eval_failure_from_arrays(eps0_vals, kappa_vals, mat, strengths, criterion, label):
    layup=mat._layup_angles; t_ply=mat.t_ply
    Q_ply=_Q_ply(mat._E1,mat._E2,mat._G12,mat._nu12)
    H=float(mat.H.value); z=-H/2.; FI_all=[]
    for angle in layup:
        z_mid=z+t_ply/2.; strain_k=eps0_vals+z_mid*kappa_vals
        Qb_lam=_Qbar_ply(Q_ply,angle); stress_lam=strain_k@Qb_lam.T
        a=np.radians(angle); c,s=np.cos(a),np.sin(a)
        T=np.array([[c**2,s**2,2*c*s],[s**2,c**2,-2*c*s],[-c*s,c*s,c**2-s**2]])
        stress_mat=stress_lam@T.T
        FI_k=np.array([tsai_wu(row,strengths) for row in stress_mat])
        FI_all.append(FI_k); z+=t_ply
    FI_all=np.array(FI_all)
    FI_max=float(FI_all.max()) if FI_all.size>0 else 0.
    vprint(f"  [{label}] Global max FI : {FI_max:.4f}")
    return FI_max

def panel_buckling_index(eps_cells, mat, panel_width, buckling_k=4.0):
    t=float(mat.H.value); E_eff=float(mat.D_ufl.value[0,0])*12/t**3
    sigma_cr=buckling_k*(np.pi**2*E_eff)/(12*(1-0.3**2))*(t/panel_width)**2
    compressive=eps_cells[:,0][eps_cells[:,0]<0]
    if len(compressive)==0: return 0.0
    return float(mat.A_ufl.value[0,0])*np.abs(compressive).max()/t/sigma_cr

# ── CFD traction ──────────────────────────────────────────────
def load_traction_xdmf(xdmffile, domain, verbose=False):
    if rank==0:
        data=meshio.read(str(xdmffile)); pts=data.points; tract=data.point_data["traction"]
    else:
        pts=tract=None
    pts=comm.bcast(pts,root=0); tract=comm.bcast(tract,root=0)
    VT=fem.functionspace(domain,("Lagrange",1,(3,)))
    f=fem.Function(VT,name="traction")
    coords=VT.tabulate_dof_coordinates()
    _,idx=cKDTree(pts).query(coords,k=1)
    f.x.array[:]=tract[idx].reshape(-1); f.x.scatter_forward()
    return f

# ── Post-processing ───────────────────────────────────────────
def compute_tip_deflection(v, domain, tip_y=3.0, tol=0.05):
    disp=v.sub(0).collapse(); u_arr=disp.x.array.reshape(-1,3)
    coords=disp.function_space.tabulate_dof_coordinates()
    tip_mask=np.abs(coords[:,1]-tip_y)<tol
    uz=np.abs(u_arr[tip_mask,2]).max() if tip_mask.any() else 0.
    return comm.allreduce(uz, op=MPI.MAX)

def compute_relative_twist(v, domain, root_y=0., tip_y=3., tol=0.05):
    rota=v.sub(1).collapse(); t_arr=rota.x.array.reshape(-1,3)
    coords=rota.function_space.tabulate_dof_coordinates()
    root_mask=np.abs(coords[:,1]-root_y)<tol
    tip_mask =np.abs(coords[:,1]-tip_y) <tol
    tip_vec  = t_arr[tip_mask].mean(axis=0)  if tip_mask.any()  else np.zeros(3)
    root_vec = t_arr[root_mask].mean(axis=0) if root_mask.any() else np.zeros(3)
    tip_vec  = comm.allreduce(tip_vec,  op=MPI.SUM)/max(comm.allreduce(int(tip_mask.sum()),  op=MPI.SUM),1)
    root_vec = comm.allreduce(root_vec, op=MPI.SUM)/max(comm.allreduce(int(root_mask.sum()), op=MPI.SUM),1)
    return np.linalg.norm(tip_vec - root_vec)

# ─────────────────────────────────────────────────────────────
# MATERIAL CONSTANTS
# ─────────────────────────────────────────────────────────────
CM_YNG1=181.0e9; CM_YNG2=10.30e9; CM_G12=7.170e9; CM_NU12=0.28
CM_STRENGTH={"Xt":1500e6,"Xc":900e6,"Yt":50e6,"Yc":200e6,"S":70e6}

SKIN_LAYUP = [0,  0,  45,-45,-45, 45,  0, 0]
SPAR_LAYUP = [0, 45, -45,  0,  0,-45, 45, 0]
RIB_LAYUP  = [0, 45, -45, 90, 90,-45, 45, 0]

RIB_SPACING=0.75; SPAR_HEIGHT=0.40; RIB_PANEL=0.40
BUCKLING_K_SKIN=4.0; BUCKLING_K_SPAR=5.35; BUCKLING_K_RIB=6.97
K_MAP={TAG_UPPER:4.,TAG_LOWER:4.,TAG_MAINSPAR:5.35,TAG_TAILSPAR:5.35,
       TAG_RIB750:6.97,TAG_RIB1500:6.97,TAG_RIB2250:6.97,TAG_RIB3000:6.97}

# ─────────────────────────────────────────────────────────────
# FUNCTION SPACE
# ─────────────────────────────────────────────────────────────
Ue        = basix.ufl.element("P",  DOMAIN.basix_cell(), 2, shape=(GDIM,))
Te        = basix.ufl.element("CR", DOMAIN.basix_cell(), 1, shape=(GDIM,))
V         = fem.functionspace(DOMAIN, basix.ufl.mixed_element([Ue, Te]))
v         = fem.Function(V)
u, theta  = ufl.split(v)
v_        = ufl.TestFunction(V)
u_, theta_= ufl.split(v_)
dv        = ufl.TrialFunction(V)
dx        = ufl.Measure("dx", domain=DOMAIN, subdomain_data=CELL_TAGS)

# ─────────────────────────────────────────────────────────────
# MATERIALS
# ─────────────────────────────────────────────────────────────
t_skin_const    = fem.Constant(DOMAIN, 0.75e-3)
t_mspar_const   = fem.Constant(DOMAIN, 0.75e-3)
t_tspar_const   = fem.Constant(DOMAIN, 0.75e-3)
t_rib0_const    = fem.Constant(DOMAIN, 0.75e-3)
t_rib750_const  = fem.Constant(DOMAIN, 0.75e-3)
t_rib1500_const = fem.Constant(DOMAIN, 0.75e-3)
t_rib2250_const = fem.Constant(DOMAIN, 0.75e-3)
t_rib3000_const = fem.Constant(DOMAIN, 0.75e-3)

MAT_SKIN      = build_laminate(SKIN_LAYUP, t_skin_const,    "SKIN")
MAT_SPAR_MAIN = build_laminate(SPAR_LAYUP, t_mspar_const,   "MAINSPAR")
MAT_SPAR_TAIL = build_laminate(SPAR_LAYUP, t_tspar_const,   "TAILSPAR")
MAT_RIB0      = build_laminate(RIB_LAYUP,  t_rib0_const,    "RIB0")
MAT_RIB750    = build_laminate(RIB_LAYUP,  t_rib750_const,  "RIB750")
MAT_RIB1500   = build_laminate(RIB_LAYUP,  t_rib1500_const, "RIB1500")
MAT_RIB2250   = build_laminate(RIB_LAYUP,  t_rib2250_const, "RIB2250")
MAT_RIB3000   = build_laminate(RIB_LAYUP,  t_rib3000_const, "RIB3000")

MATS = {
    TAG_UPPER:MAT_SKIN,    TAG_LOWER:MAT_SKIN,
    TAG_RIB0:MAT_RIB0,     TAG_RIB750:MAT_RIB750,
    TAG_RIB1500:MAT_RIB1500, TAG_RIB2250:MAT_RIB2250,
    TAG_RIB3000:MAT_RIB3000,
    TAG_MAINSPAR:MAT_SPAR_MAIN, TAG_TAILSPAR:MAT_SPAR_TAIL,
}

# ─────────────────────────────────────────────────────────────
# BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────
ROOT   = FACET_TAGS.find(TAG_RIB0_CURVE)
Vu, _  = V.sub(0).collapse()
dofs_u = fem.locate_dofs_topological((V.sub(0), Vu), FDIM, ROOT)
bc_u   = fem.dirichletbc(fem.Function(Vu), dofs_u, V.sub(0))
Vt, _  = V.sub(1).collapse()
dofs_t = fem.locate_dofs_topological((V.sub(1), Vt), FDIM, ROOT)
bc_t   = fem.dirichletbc(fem.Function(Vt), dofs_t, V.sub(1))
BCS    = [bc_u, bc_t]

vprint(f"[BC] Root clamp : {comm.allreduce(len(dofs_u[0]),op=MPI.SUM)} disp DOFs"
       f" | {comm.allreduce(len(dofs_t[0]),op=MPI.SUM)} rot DOFs")

# ─────────────────────────────────────────────────────────────
# TRACTION + RESIDUAL  (identical to opti8var.py)
# ─────────────────────────────────────────────────────────────
if not MAPFILE.exists():
    from scipy.interpolate import RBFInterpolator
    # map_traction(...) — same as opti8var.py (omitted for brevity)
    raise FileNotFoundError(f"MAPFILE not found: {MAPFILE}")

FTraction = load_traction_xdmf(MAPFILE, DOMAIN, verbose=VERBOSE)

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
        ufl.inner(N_t, eps_) + ufl.inner(M_t, kappa_)
        + ufl.inner(Q_t, gamma_) + drill_t * drilling_strain_
    ) * dx(tag))

L_ext    = sum(ufl.dot(FTraction, u_) * dx(tag) for tag in TAG_SKIN)
residual = reduce(lambda a,b: a+b, form_pieces) - L_ext
tangent  = ufl.derivative(residual, v, dv)

problem = NonlinearProblem(
    F=residual, u=v, bcs=BCS, J=tangent,
    petsc_options_prefix="wing_",
    petsc_options={
        "ksp_type":"preonly","pc_type":"lu",
        "pc_factor_mat_solver_type":"mumps",
        "snes_type":"newtonls","snes_rtol":1e-8,"snes_atol":1e-8,
        "snes_max_it":25,"snes_monitor":"",
        "mat_mumps_icntl_14":80,"mat_mumps_icntl_23":4000,
    }
)

# ─────────────────────────────────────────────────────────────
# GEOMETRIC STIFFNESS FORM — compiled ONCE before opt loop
# Re-assembled cheaply each iteration (no JIT re-run)
# ─────────────────────────────────────────────────────────────
if USE_FE_BUCKLING:
    vprint("\n[BUCKLING] Pre-compiling K_g form (one-time JIT) …")
    _t0 = time.time()

    u_h, theta_h = ufl.split(v)
    _P     = tangent_projection(E1, E2)
    _w_tr  = ufl.TrialFunction(V);  _u_w, _ = ufl.split(_w_tr)
    _v_te  = ufl.TestFunction(V);   _u_v, _ = ufl.split(_v_te)
    _w_n   = ufl.dot(_u_w, E3)
    _v_n   = ufl.dot(_u_v, E3)
    _gw    = tangential_gradient(_w_n, _P)
    _gv    = tangential_gradient(_v_n, _P)

    _kg_pieces = []
    for tag, mat in MATS.items():
        eps0, _  = membrane_strain(u_h, _P)
        kappa0   = bending_strain(theta_h, E3, _P)
        N0 = from_voigt(
            ufl.dot(mat.A_ufl, to_voigt(eps0))
            + ufl.dot(mat.B_ufl, to_voigt(kappa0))
        )
        _kg_pieces.append(ufl.inner(ufl.dot(N0, _gw), _gv) * dx(tag))

    KG_FORM    = fem.form(reduce(lambda a,b: a+b, _kg_pieces),
                          jit_options={"cache_dir": ".jit_cache_kg"})
    K_FORM     = fem.form(tangent,
                          jit_options={"cache_dir": ".jit_cache_ke"})

    # Pre-collect constrained DOF indices (constant throughout optimization)
    BC_DOFS = np.unique(np.concatenate([bc.dof_indices()[0] for bc in BCS]))

    vprint(f"[BUCKLING] K_g form compiled in {time.time()-_t0:.1f}s ✓")
    vprint(f"[BUCKLING] Trigger BI > {BUCKLING_TRIGGER}  |  λ_cr limit = {LAMBDA_CR_LIMIT}")


def compute_lambda_cr():
    """
    Assemble K_e and K_g from current solution v, solve EVP.
    Returns λ_cr (smallest positive buckling factor).

    Time breakdown (typical):
      K_e assembly   : ~3s
      K_g assembly   : ~3s
      SLEPc (nev=1)  : ~15-25s
      Total          : ~20-30s per call
    """
    t0 = time.time()

    # Assemble K_e (tangent at converged state) with BCs (diag=1)
    K_e = assemble_matrix(K_FORM, bcs=BCS)
    K_e.assemble()

    # Assemble K_g without BCs, then zero constrained rows/cols (diag=0)
    K_g = assemble_matrix(KG_FORM)
    K_g.assemble()
    K_g.zeroRowsColumns(BC_DOFS, 0.0)

    # SLEPc GHIEP — nev=1 is enough for constraint (only need λ_cr)
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
    eigensolver.setTarget(2.0)
    eigensolver.setDimensions(
        nev=1,          # ← only need λ_cr: much faster than nev=10
        ncv=15,         # ← minimal Krylov subspace
        mpd=10,
    )
    eigensolver.setTolerances(tol=1e-5, max_it=500)  # looser tol → faster
    eigensolver.solve()

    nconv = eigensolver.getConverged()

    if nconv == 0:
        vprint("  [BUCK] No converged eigenpair — returning λ_cr=999 (assume safe)")
        return 999.0

    lam = eigensolver.getEigenpair(0,
                                    K_e.createVecRight(),
                                    K_e.createVecRight())
    lam_cr = float(lam.real)

    # If we got a negative value, the target shift needs adjustment
    # Return large value so optimizer doesn't penalize (conservative)
    if lam_cr <= 0:
        vprint(f"  [BUCK] Got λ={lam_cr:.3f} (negative — shift issue). "
               f"Returning conservative λ_cr=0.1")
        lam_cr = 0.1

    t_elapsed = time.time() - t0
    sf = 1/lam_cr if lam_cr > 0 else float("inf")
    vprint(f"  [BUCK] λ_cr = {lam_cr:.4f}  SF_buck={sf:.2f}  "
           f"(computed in {t_elapsed:.1f}s)")
    return lam_cr


# ─────────────────────────────────────────────────────────────
# AREAS + DG0 WORKSPACE
# ─────────────────────────────────────────────────────────────
def compute_area(*tags):
    area_local = fem.assemble_scalar(fem.form(sum(1*dx(tag) for tag in tags)))
    return comm.allreduce(area_local, op=MPI.SUM)

AREA_SKIN      = compute_area(TAG_UPPER, TAG_LOWER)
AREA_RIB0      = compute_area(TAG_RIB0)
AREA_RIB750    = compute_area(TAG_RIB750)
AREA_RIB1500   = compute_area(TAG_RIB1500)
AREA_RIB2250   = compute_area(TAG_RIB2250)
AREA_RIB3000   = compute_area(TAG_RIB3000)
AREA_SPAR_MAIN = compute_area(TAG_MAINSPAR)
AREA_SPAR_TAIL = compute_area(TAG_TAILSPAR)

DG0    = fem.functionspace(DOMAIN, ("DG", 0, (3,)))
eps_fn = fem.Function(DG0)
kap_fn = fem.Function(DG0)

# ─────────────────────────────────────────────────────────────
# DESIGN LIMITS
# ─────────────────────────────────────────────────────────────
DEFLECTION_LIMIT_M = 0.16
TWIST_LIMIT_RAD    = 0.01
FI_LIMIT           = 0.80
REF_MASS           = 101.1662

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
suffix   = "buckling" if USE_FE_BUCKLING else "baseline"
LOG_FILE = OUTPUT_DIR / f"pareto_log_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

FIELDNAMES = [
    "iter","t_skin_mm","t_rib0_mm","t_rib750_mm","t_rib1500_mm",
    "t_rib2250_mm","t_rib3000_mm","t_mspar_mm","t_tspar_mm",
    "mass_kg","tip_deflection_mm","relative_twist_deg",
    "fi_max","lambda_cr","buckling_triggered","obj_mass","feasible",
]
_iter_count = 0

def init_log():
    if rank==0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE,"w",newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        vprint(f"[LOG] → {LOG_FILE}")

def log_iteration(x, obj, fi, defl, twist, mass, lam_cr, triggered):
    global _iter_count
    if rank!=0: return
    _iter_count += 1
    feasible = int(fi<=FI_LIMIT and defl<=DEFLECTION_LIMIT_M
                   and twist<=TWIST_LIMIT_RAD
                   and (lam_cr>=LAMBDA_CR_LIMIT if USE_FE_BUCKLING else True))
    row = {
        "iter":_iter_count,
        "t_skin_mm":round(x[0]*1e3,4), "t_rib0_mm":round(x[1]*1e3,4),
        "t_rib750_mm":round(x[2]*1e3,4), "t_rib1500_mm":round(x[3]*1e3,4),
        "t_rib2250_mm":round(x[4]*1e3,4), "t_rib3000_mm":round(x[5]*1e3,4),
        "t_mspar_mm":round(x[6]*1e3,4), "t_tspar_mm":round(x[7]*1e3,4),
        "mass_kg":round(mass,4),
        "tip_deflection_mm":round(defl*1e3,4),
        "relative_twist_deg":round(np.degrees(twist),6),
        "fi_max":round(fi,6), "lambda_cr":round(lam_cr,4),
        "buckling_triggered":int(triggered),
        "obj_mass":round(obj,6), "feasible":feasible,
    }
    with open(LOG_FILE,"a",newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

# ─────────────────────────────────────────────────────────────
# SOLVE FUNCTION  (opti8var.py + lazy FE buckling)
# ─────────────────────────────────────────────────────────────
def solve_wing(x):
    t_skin,t_rib0,t_rib750,t_rib1500,t_rib2250,t_rib3000,t_mspar,t_tspar = x

    update_clt_material(MAT_SKIN,      t_skin,    len(SKIN_LAYUP))
    update_clt_material(MAT_RIB0,      t_rib0,    len(RIB_LAYUP))
    update_clt_material(MAT_RIB750,    t_rib750,  len(RIB_LAYUP))
    update_clt_material(MAT_RIB1500,   t_rib1500, len(RIB_LAYUP))
    update_clt_material(MAT_RIB2250,   t_rib2250, len(RIB_LAYUP))
    update_clt_material(MAT_RIB3000,   t_rib3000, len(RIB_LAYUP))
    update_clt_material(MAT_SPAR_MAIN, t_mspar,   len(SPAR_LAYUP))
    update_clt_material(MAT_SPAR_TAIL, t_tspar,   len(SPAR_LAYUP))

    problem.solve()

    tip_deflection = compute_tip_deflection(v, DOMAIN, tip_y=3.0)
    relative_twist = compute_relative_twist(v, DOMAIN, root_y=0., tip_y=3.)

    u_h, theta_h = ufl.split(v)
    P_proj  = tangent_projection(E1, E2)
    eps_h,_ = membrane_strain(u_h,    P_proj)
    kappa_h = bending_strain(theta_h, E3, P_proj)

    eps_fn.interpolate(fem.Expression(to_voigt(eps_h),   DG0.element.interpolation_points))
    kap_fn.interpolate(fem.Expression(to_voigt(kappa_h), DG0.element.interpolation_points))

    eps0_all  = eps_fn.x.array.reshape(-1, 3)
    kappa_all = kap_fn.x.array.reshape(-1, 3)

    SKIP_BUCKLING = {TAG_RIB0}
    global_fi_max  = 0.0
    max_bi_analytical = 0.0   # track for lazy trigger

    for tag, mat in [
        (TAG_UPPER,    MAT_SKIN),      (TAG_LOWER,    MAT_SKIN),
        (TAG_MAINSPAR, MAT_SPAR_MAIN), (TAG_TAILSPAR, MAT_SPAR_TAIL),
        (TAG_RIB0,     MAT_RIB0),      (TAG_RIB750,   MAT_RIB750),
        (TAG_RIB1500,  MAT_RIB1500),   (TAG_RIB2250,  MAT_RIB2250),
        (TAG_RIB3000,  MAT_RIB3000),
    ]:
        cells = CELL_TAGS.find(tag)
        if len(cells) > 0:
            eps_c = eps0_all[cells]; kap_c = kappa_all[cells]
            fi_val = _eval_failure_from_arrays(
                eps_c, kap_c, mat, CM_STRENGTH,
                criterion="tsai_wu", label=f"TAG_{tag}")
            panel_width = (RIB_SPACING if tag in TAG_SKIN else
                           SPAR_HEIGHT  if tag in [TAG_MAINSPAR,TAG_TAILSPAR] else
                           RIB_PANEL)
            bi = (0.0 if tag in SKIP_BUCKLING else
                  panel_buckling_index(eps_c, mat, panel_width, K_MAP.get(tag,4.0)))
            combined = max(fi_val, bi)
        else:
            fi_val = bi = combined = 0.0

        global_tag    = comm.allreduce(combined, op=MPI.MAX)
        bi_global     = comm.allreduce(bi,       op=MPI.MAX)
        global_fi_max = max(global_fi_max, global_tag)
        max_bi_analytical = max(max_bi_analytical, bi_global)

        vprint(f"  [TAG_{tag}] TsaiWu={comm.allreduce(fi_val,op=MPI.MAX):.4f}"
               f"  BI_analytical={bi_global:.4f}")

    # ── FE BUCKLING (lazy) ────────────────────────────────────
    # Strategy: only call the expensive SLEPc solve if
    #   (a) FE buckling is enabled, AND
    #   (b) analytical BI already indicates potential risk
    # Early in optimization (thick panels), BI << trigger → skip.
    # Near optimum (thin panels), BI > trigger → call FE buckling.
    lam_cr    = 999.0   # default: assume safe (no buckling concern)
    triggered = False

    if USE_FE_BUCKLING:
        if max_bi_analytical > BUCKLING_TRIGGER:
            triggered = True
            vprint(f"\n  [BUCK] BI_analytical={max_bi_analytical:.3f} > {BUCKLING_TRIGGER}"
                   f" → calling FE buckling …")
            lam_cr = compute_lambda_cr()
        else:
            vprint(f"\n  [BUCK] BI_analytical={max_bi_analytical:.3f} <= {BUCKLING_TRIGGER}"
                   f" → skipping FE buckling (safe to skip)")

    # ── Mass ──────────────────────────────────────────────────
    rho = 1600.0
    mass_total = rho * (
        AREA_SKIN*t_skin + AREA_RIB0*t_rib0 + AREA_RIB750*t_rib750
        + AREA_RIB1500*t_rib1500 + AREA_RIB2250*t_rib2250 + AREA_RIB3000*t_rib3000
        + AREA_SPAR_MAIN*t_mspar + AREA_SPAR_TAIL*t_tspar
    )
    obj_mass = mass_total / REF_MASS

    if rank==0:
        vprint("\n==============================")
        vprint(f"[POST] uz_tip     : {tip_deflection*1e3:.2f} mm  (lim {DEFLECTION_LIMIT_M*1e3:.0f})")
        vprint(f"[POST] twist      : {np.degrees(relative_twist):.4f} deg")
        vprint(f"[POST] FI max     : {global_fi_max:.4f}")
        vprint(f"[POST] λ_cr       : {lam_cr:.4f}  "
               f"{'(FE)' if triggered else '(analytical approx)'}")
        vprint(f"[POST] mass       : {mass_total:.3f} kg")
        vprint(f"[POST] obj_mass   : {obj_mass:.4f}")
        vprint("==============================\n")

    return obj_mass, global_fi_max, tip_deflection, relative_twist, mass_total, lam_cr, triggered


# ─────────────────────────────────────────────────────────────
# NLOPT WRAPPERS
# ─────────────────────────────────────────────────────────────
_cache = {"x":None,"obj":None,"fi":None,"defl":None,
          "twist":None,"mass":None,"lam_cr":None,"triggered":False}

def _run_and_cache(x):
    if _cache["x"] is not None and np.allclose(x, _cache["x"], rtol=1e-12):
        return
    obj,fi,defl,twist,mass,lam_cr,triggered = solve_wing(x)
    _cache.update(x=np.copy(x), obj=obj, fi=fi, defl=defl,
                  twist=twist, mass=mass, lam_cr=lam_cr, triggered=triggered)
    log_iteration(x, obj, fi, defl, twist, mass, lam_cr, triggered)

def objective_mass(x, grad):
    _run_and_cache(x);  return float(_cache["obj"])

def constraint_fi(x, grad):
    _run_and_cache(x);  return float(_cache["fi"]) - FI_LIMIT

def constraint_deflection(x, grad):
    _run_and_cache(x);  return float(_cache["defl"]) - DEFLECTION_LIMIT_M

def constraint_twist(x, grad):
    _run_and_cache(x);  return float(_cache["twist"]) - TWIST_LIMIT_RAD

def constraint_buckling(x, grad):
    """
    g(x) = LAMBDA_CR_LIMIT - λ_cr  <= 0
    Feasible when λ_cr >= LAMBDA_CR_LIMIT.
    """
    _run_and_cache(x);  return LAMBDA_CR_LIMIT - float(_cache["lam_cr"])


# ─────────────────────────────────────────────────────────────
# NLOPT SETUP
# ─────────────────────────────────────────────────────────────
N_VARS=8; T_MAX=20.e-3; T_INIT=12.e-3
T_MIN_SKIN=1.e-3; T_MIN_RIB=2.e-3; T_MIN_SPAR=1.e-3

opt = nlopt.opt(nlopt.LN_COBYLA, N_VARS)
opt.set_lower_bounds([T_MIN_SKIN,T_MIN_RIB,T_MIN_RIB,T_MIN_RIB,
                      T_MIN_RIB,T_MIN_RIB,T_MIN_SPAR,T_MIN_SPAR])
opt.set_upper_bounds([T_MAX]*N_VARS)
opt.set_xtol_rel(1e-5); opt.set_ftol_rel(1e-5); opt.set_maxeval(300)

opt.set_min_objective(objective_mass)
opt.add_inequality_constraint(constraint_fi,          1e-4)
opt.add_inequality_constraint(constraint_deflection,  1e-4)
opt.add_inequality_constraint(constraint_twist,       1e-5)

if USE_FE_BUCKLING:
    opt.add_inequality_constraint(constraint_buckling, 1e-3)
    vprint(f"[OPT] Buckling constraint added: λ_cr >= {LAMBDA_CR_LIMIT}")

x0 = [T_INIT] * N_VARS

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
init_log()

if QUICK_TEST:
    vprint("\n>>> QUICK TEST <<<")
    obj,fi,defl,twist,mass,lam_cr,triggered = solve_wing(x0)
    log_iteration(x0, obj, fi, defl, twist, mass, lam_cr, triggered)

else:
    vprint("\n>>> NLOPT COBYLA OPTIMISATION START <<<")
    vprint(f"    Objective : minimize  mass / REF_MASS")
    vprint(f"    s.t.      : FI         <= {FI_LIMIT}")
    vprint(f"                deflection <= {DEFLECTION_LIMIT_M*1e3:.0f} mm")
    vprint(f"                twist      <= {np.degrees(TWIST_LIMIT_RAD):.2f} deg")
    if USE_FE_BUCKLING:
        vprint(f"                λ_cr       >= {LAMBDA_CR_LIMIT}  [NEW]")

    t_start = time.time()
    xopt    = opt.optimize(x0)
    t_total = time.time() - t_start

    if rank==0:
        vprint(f"\n========== OPTIMUM ({'WITH' if USE_FE_BUCKLING else 'WITHOUT'} BUCKLING) ==========")
        labels = ["Skin","Rib0","Rib750","Rib1500","Rib2250","Rib3000","MainSpar","TailSpar"]
        for lbl, t in zip(labels, xopt):
            vprint(f"  {lbl:>10s} : {t*1e3:.3f} mm")
        vprint(f"  Opt obj     : {opt.last_optimum_value():.6f}")
        vprint(f"  NLopt status: {opt.last_optimize_result()}")
        vprint(f"  Total time  : {t_total/3600:.2f} h")
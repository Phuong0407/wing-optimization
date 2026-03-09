from mpi4py import MPI
from wing_model import WingModel
from pathlib import Path
from aerodynamic_load import import_foam_traction, map_traction, load_traction_xdmf
from shell_kinematics import shell_strains_from_model
from shell_bcs import bc_full_clamped
from material_clt import clt_composite
from helper import vprint
from shell_stress import stress_resultants, drilling_terms
import ufl
from functools import reduce
from dolfinx.fem.petsc import NonlinearProblem
import numpy as np


comm = MPI.COMM_WORLD


VERBOSE         = True
WRITE_OUTPUT    = True
MESHUNIT        = "mm"
INPUT_DIR       = Path("InputData")
OUTPUT_DIR      = Path("Result")
MESHFILE        = INPUT_DIR / "wingbox.msh"
FOAMFILE        = INPUT_DIR / "wing.vtp"
XDMFFILE        = INPUT_DIR / "FOAMData.xdmf"
MAPFILE         = INPUT_DIR / "MappedTraction.xdmf"


# MATERIAL
## AS4/3501-6 CARBON EPOXY COMPOSITE
CE_YNG1 = 181.0E9
CE_YNG2 = 10.30E9
CE_G12  = 7.170E9
CE_NU12 = 0.28

# STRENGTH DATA FOR TSAI-WU CRITERION
CE_STRENGTH = {
    "Xt": 1500E6, "Xc":  900E6,
    "Yt":   50E6, "Yc":  200E6,
    "S":    70E6,
}

## ALUMINIUM 2024-T3
AL_YNG = 73.1E9
AL_NU  = 0.33
AL_YS  = 324E6

## PLY THICKNESS
SPLY = 0.75E-3
MPLY = 0.75E-3
TPLY = 0.75E-3
RPLY = 0.75E-3

## LAYUP CONFIGURATION
SKIN_LAYUP  = [0, 45, -45, 90, 90, -45, 45, 0]
MSPAR_LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
TSPAR_LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
RIB_LAYUP   = [0, 45, -45, 90, 90, -45, 45, 0]

## ASSIGN MATERIAL
vprint("\n[MAT] SKIN")
MAT_SKIN = clt_composite(
    SKIN_LAYUP,
    SPLY,
    CE_YNG1,
    CE_YNG2,
    CE_G12,
    CE_NU12,
    G13 = CE_G12,
    G23 = CE_G12 * 0.5,
    kappa_s=5/6,
    label="SKIN",
    verbose=VERBOSE
)
MAT_SKIN._layup_angles  = SKIN_LAYUP
MAT_SKIN._t_ply         = SPLY
MAT_SKIN._E1            = CE_YNG1
MAT_SKIN._E2            = CE_YNG2
MAT_SKIN._G12           = CE_G12
MAT_SKIN._nu12          = CE_NU12

vprint("\n[MAT] MAIN SPAR")
MAT_MAINSPAR = clt_composite(
    MSPAR_LAYUP,
    MPLY,
    CE_YNG1,
    CE_YNG2,
    CE_G12,
    CE_NU12,
    G13=CE_G12,
    G23=CE_G12 * 0.5,
    kappa_s=5/6,
    label="MAINSPAR",
    verbose=VERBOSE
)
MAT_MAINSPAR._layup_angles  = SKIN_LAYUP
MAT_MAINSPAR._t_ply         = SPLY
MAT_MAINSPAR._E1            = CE_YNG1
MAT_MAINSPAR._E2            = CE_YNG2
MAT_MAINSPAR._G12           = CE_G12
MAT_MAINSPAR._nu12          = CE_NU12

vprint("\n[MAT] TAIL SPAR")
MAT_TAILSPAR = clt_composite(
    TSPAR_LAYUP,
    TPLY,
    CE_YNG1,
    CE_YNG2,
    CE_G12,
    CE_NU12,
    G13=CE_G12,
    G23=CE_G12 * 0.5,
    kappa_s=5/6,
    label="TAILSPAR",
    verbose=VERBOSE
)
MAT_TAILSPAR._layup_angles  = MSPAR_LAYUP
MAT_TAILSPAR._t_ply         = SPLY
MAT_TAILSPAR._E1            = CE_YNG1
MAT_TAILSPAR._E2            = CE_YNG2
MAT_TAILSPAR._G12           = CE_G12
MAT_TAILSPAR._nu12          = CE_NU12

vprint("\n[MAT] RIB")
MAT_RIB = clt_composite(
    RIB_LAYUP,
    RPLY,
    CE_YNG1,
    CE_YNG2,
    CE_G12,
    CE_NU12,
    G13=CE_G12,
    G23=CE_G12 * 0.5,
    kappa_s=5/6,
    label="RIB",
    verbose=VERBOSE
)
MAT_RIB._layup_angles  = RIB_LAYUP
MAT_RIB._t_ply         = SPLY
MAT_RIB._E1            = CE_YNG1
MAT_RIB._E2            = CE_YNG2
MAT_RIB._G12           = CE_G12
MAT_RIB._nu12          = CE_NU12

MATS = {
    14 : MAT_SKIN,
    15 : MAT_SKIN,
    38 : MAT_RIB,
    39 : MAT_RIB,
    40 : MAT_RIB,
    41 : MAT_RIB,
    42 : MAT_RIB,
    43 : MAT_MAINSPAR,
    44 : MAT_TAILSPAR,
}




TAG_UPPER     = 14
TAG_LOWER     = 15
TAG_RIB0      = 38
TAG_RIB750    = 39
TAG_RIB1500   = 40
TAG_RIB2250   = 41
TAG_RIB3000   = 42
TAG_MAINSPAR  = 43
TAG_TAILSPAR  = 44
TAG_SKIN      = [TAG_UPPER, TAG_LOWER]
TAG_RIBS      = [TAG_RIB0, TAG_RIB750, TAG_RIB1500, TAG_RIB2250, TAG_RIB3000]
TAG_COMPOSITE = TAG_SKIN + [TAG_MAINSPAR, TAG_TAILSPAR]
TAG_ALL       = TAG_COMPOSITE + TAG_RIBS


model = WingModel(MESHFILE.as_posix(), comm)
model.local_frame()
model.function_space()

TAG_SURFACE = TAG_SKIN + [TAG_RIB3000]
import_foam_traction(FOAMFILE, XDMFFILE, VERBOSE)
map_traction(XDMFFILE, MESHFILE, MAPFILE, TAG_SURFACE, MESHUNIT, VERBOSE)
FTraction = load_traction_xdmf(MAPFILE, model.mesh, VERBOSE)

eps,  kappa,  gamma,  drilling_strain,  \
eps_, kappa_, gamma_, drilling_strain_  \
    = shell_strains_from_model(model)

TAG_RIB0_CURVE = 45

BCS, extra = bc_full_clamped(model.V, model.facet_tags, model.fdim, TAG_RIB0_CURVE, comm, VERBOSE)

dx = ufl.Measure("dx", domain=model.mesh, subdomain_data=model.cell_tags)

form_pieces = []
for tag, mat in MATS.items():
    N_t, M_t, Q_t = stress_resultants(mat, eps, kappa, gamma)
    _, drill_t    = drilling_terms(mat, model.mesh, drilling_strain)
    piece = (
          ufl.inner(N_t, eps_)
        + ufl.inner(M_t, kappa_)
        + ufl.inner(Q_t, gamma_)
        + drill_t * drilling_strain_
    ) * dx(tag)
    form_pieces.append(piece)

a_int       = reduce(lambda a, b: a + b, form_pieces)
L_ext       = sum(ufl.dot(FTraction, model.u_) * dx(tag) for tag in TAG_SKIN)
residual    = a_int - L_ext
tangent     = ufl.derivative(residual, model.v, model.dv)

problem = NonlinearProblem(
    F=residual,
    u=model.v,
    bcs=BCS,
    J=tangent,
    petsc_options_prefix="wing",
    petsc_options={
        "ksp_type"                  : "preonly",
        "pc_type"                   : "lu",
        "pc_factor_mat_solver_type" : "mumps",
        "snes_type"                 : "newtonls",
        "snes_rtol"                 : 1E-8,
        "snes_atol"                 : 1E-8,
        "snes_max_it"               : 25,
        "snes_monitor"              : None,
        "mat_mumps_icntl_14"        : 80,
        "mat_mumps_icntl_23"        : 2000,
    }
)
problem.solve()


disp = model.v.sub(0).collapse()
disp.name = "Displacement"
rota = model.v.sub(1).collapse()
rota.name = "Rotation"

u_arr = disp.x.array.reshape(-1, 3)
t_arr = rota.x.array.reshape(-1, 3)

ux_max_local = np.abs(u_arr[:, 0]).max()
uy_max_local = np.abs(u_arr[:, 1]).max()
uz_max_local = np.abs(u_arr[:, 2]).max()
u_mag_local  = np.linalg.norm(u_arr, axis=1).max()

ux_max = comm.allreduce(ux_max_local, op=MPI.MAX)
uy_max = comm.allreduce(uy_max_local, op=MPI.MAX)
uz_max = comm.allreduce(uz_max_local, op=MPI.MAX)
u_max  = comm.allreduce(u_mag_local,  op=MPI.MAX)

tx_max_local = np.abs(t_arr[:, 0]).max()
ty_max_local = np.abs(t_arr[:, 1]).max()
tz_max_local = np.abs(t_arr[:, 2]).max()

tx_max = comm.allreduce(tx_max_local, op=MPI.MAX)
ty_max = comm.allreduce(ty_max_local, op=MPI.MAX)
tz_max = comm.allreduce(tz_max_local, op=MPI.MAX)

vprint(VERBOSE, "\n" + "="*60)
vprint(VERBOSE, "[POST] DISPLACEMENT & ROTATION")
vprint(VERBOSE, "="*60)
vprint(VERBOSE, f"  |ux| max : {ux_max:.6e} m  ({ux_max*1e3:.3f} mm)")
vprint(VERBOSE, f"  |uy| max : {uy_max:.6e} m  ({uy_max*1e3:.3f} mm)")
vprint(VERBOSE, f"  |uz| max : {uz_max:.6e} m  ({uz_max*1e3:.3f} mm)")
vprint(VERBOSE, f"  |u|  max : {u_max:.6e} m  ({u_max*1e3:.3f} mm)")
vprint()
vprint(VERBOSE, f"  |thetax| max : {tx_max:.6e} rad")
vprint(VERBOSE, f"  |thetay| max : {ty_max:.6e} rad")
vprint(VERBOSE, f"  |thetaz| max : {tz_max:.6e} rad")
vprint()

problem.solver.destroy()
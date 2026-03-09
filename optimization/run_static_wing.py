# run_static_wing.py
from mpi4py import MPI

from wing_computation_model import WingComputationModel, TagsConfig, ClampedBC
from wing_static import WingStaticSolver

comm = MPI.COMM_WORLD

MESH_FILE  = "InputData/wingbox.msh"
FOAM_FILE  = "InputData/wing.vtp"
OUTPUT_DIR = "Result"

tags = TagsConfig(
    skin=(14, 15),
    ribs=(38, 39, 40, 41, 42),
    spars=(43, 44),
)

bc = ClampedBC(root_tag=45)

model = WingComputationModel(
    mesh_file=MESH_FILE,
    foam_file=FOAM_FILE,
    output_dir=OUTPUT_DIR,
    tags=tags,
    traction_map_tags=[14, 15, 42],
    bc=bc,
    verbose=True,
)

design = dict(
    t_skin=0.75e-3,
    t_spar=0.75e-3,
    t_rib=0.75e-3,
    skin_layup=[0, 45, -45, 90, 90, -45, 45, 0],
    spar_layup=[0, 45, -45, 90, 90, -45, 45, 0],
    rib_layup =[0, 45, -45, 90, 90, -45, 45, 0],
)

solver = WingStaticSolver(model)
res = solver.solve(design, export=True)

if comm.rank == 0:
    print(f"[STATIC] mass = {res.mass:.6f} kg")
    print(f"[STATIC] u_max = {res.u_max*1e3:.3f} mm")
    for k, v in res.FI.items():
        print(f"[STATIC] {k}: FI_max = {v:.4f}")
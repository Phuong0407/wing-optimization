import numpy as np
from mpi4py import MPI

from wing_computation_model import (
    WingComputationModel,
    TagsConfig,
    ClampedBC,
)

# ---- postprocess funcs ----
from postprocess import (
    displacement_summary,
    composite_failure_summary,
    print_summary,
    export_results,
)



comm = MPI.COMM_WORLD


# =========================================================
# INPUT FILES
# =========================================================
MESH_FILE  = "InputData/wingbox.msh"
FOAM_FILE  = "InputData/wing.vtp"
OUTPUT_DIR = "Result"


# =========================================================
# WING TAG CONFIGURATION
# =========================================================
tags = TagsConfig(
    skin=(14, 15),
    ribs=(38, 39, 40, 41, 42),
    spars=(43, 44),
)


# =========================================================
# BOUNDARY CONDITION
# =========================================================
bc = ClampedBC(root_tag=45)


# =========================================================
# CREATE MODEL
# =========================================================
model = WingComputationModel(
    mesh_file=MESH_FILE,
    foam_file=FOAM_FILE,
    output_dir=OUTPUT_DIR,
    tags=tags,
    traction_map_tags=[14, 15, 42],
    bc=bc,
    verbose=True,
)


# =========================================================
# DESIGN VARIABLES
# =========================================================
design = dict(
    t_skin=0.75e-3,
    t_spar=0.75e-3,
    t_rib=0.75e-3,

    skin_layup=[0, 45, -45, 90, 90, -45, 45, 0],
    spar_layup=[0, 45, -45, 90, 90, -45, 45, 0],
    rib_layup =[0, 45, -45, 90, 90, -45, 45, 0],
)


# =========================================================
# SOLVE STATIC WING
# =========================================================
result = model.solve(design)

# IMPORTANT:
# model.solve() phải set model.MATS dạng dict với keys "SKIN","SPAR","RIB"
# Nếu class bạn chưa set như vậy, bạn phải sửa trong class (xem note cuối).


# =========================================================
# POSTPROCESS (direct calls, outside)
# =========================================================

model.postprocess(export=True)
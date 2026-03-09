from mpi4py import MPI

from wing_computation_model import (
    WingComputationModel,
    TagsConfig,
    ClampedBC
)

from wing_optimization import WingOptimizer


comm = MPI.COMM_WORLD


tags = TagsConfig(
    skin=(14,15),
    ribs=(38,39,40,41,42),
    spars=(43,44),
)

bc = ClampedBC(root_tag=45)


model = WingComputationModel(
    mesh_file="InputData/wingbox.msh",
    foam_file="InputData/wing.vtp",
    output_dir="Result",
    tags=tags,
    traction_map_tags=[14,15,42],
    bc=bc,
    verbose=True,
)


optimizer = WingOptimizer(model)

x_opt, cost = optimizer.optimize()


if comm.rank == 0:

    print("\n===================================")
    print("Optimization result")
    print("===================================")

    print("t_skin =", x_opt[0])
    print("t_spar =", x_opt[1])
    for i, t_rib_val in enumerate(x_opt[2:]):
        print(f"t_rib_{i} =", t_rib_val)
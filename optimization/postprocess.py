import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from pathlib import Path
from helper import vprint



def displacement_summary(v, comm=MPI.COMM_WORLD):

    disp = v.sub(0).collapse()
    rota = v.sub(1).collapse()

    gdim = disp.function_space.element.value_shape[0]

    u_arr = disp.x.array.reshape(-1, gdim)
    t_arr = rota.x.array.reshape(-1, gdim)

    ux_local = np.abs(u_arr[:, 0]).max()
    uy_local = np.abs(u_arr[:, 1]).max()
    uz_local = np.abs(u_arr[:, 2]).max()
    u_local  = np.linalg.norm(u_arr, axis=1).max()

    tx_local = np.abs(t_arr[:, 0]).max()
    ty_local = np.abs(t_arr[:, 1]).max()
    tz_local = np.abs(t_arr[:, 2]).max()

    ux_max = comm.allreduce(ux_local, op=MPI.MAX)
    uy_max = comm.allreduce(uy_local, op=MPI.MAX)
    uz_max = comm.allreduce(uz_local, op=MPI.MAX)
    u_max  = comm.allreduce(u_local,  op=MPI.MAX)

    tx_max = comm.allreduce(tx_local, op=MPI.MAX)
    ty_max = comm.allreduce(ty_local, op=MPI.MAX)
    tz_max = comm.allreduce(tz_local, op=MPI.MAX)

    vprint("\n" + "="*60)
    vprint("[POST] DISPLACEMENT & ROTATION")
    vprint("="*60)

    vprint(f"  |ux| max : {ux_max:.6e} m  ({ux_max*1e3:.3f} mm)")
    vprint(f"  |uy| max : {uy_max:.6e} m  ({uy_max*1e3:.3f} mm)")
    vprint(f"  |uz| max : {uz_max:.6e} m  ({uz_max*1e3:.3f} mm)")
    vprint(f"  |u|  max : {u_max:.6e} m  ({u_max*1e3:.3f} mm)")
    vprint()

    vprint(f"  |thetax| max : {tx_max:.6e} rad")
    vprint(f"  |thetay| max : {ty_max:.6e} rad")
    vprint(f"  |thetaz| max : {tz_max:.6e} rad")
    vprint()

    return u_max



def composite_failure_summary(domain, v, cell_tags, regions,
                              strengths, recover_failure_fn):
    vprint("="*60)
    vprint("[POST] COMPOSITE FAILURE (TSAI-WU)")
    vprint("="*60)

    fi_results = {}

    for tag, material, label in regions:
        if isinstance(tag, list):
            cell_list = []
            for t in tag:
                cells = cell_tags.find(t)
                if len(cells) > 0:
                    cell_list.append(cells)
            cells = np.concatenate(cell_list) if cell_list else np.array([], dtype=np.int32)
        else:
            cells = cell_tags.find(tag)
        vprint(f"\n  {label}  ({len(cells)} cells)")

        if len(cells) == 0:
            fi_results[label] = 0.0
            continue

        fi_max, _ = recover_failure_fn(
            domain,
            v,
            material,
            cells,
            strengths,
            criterion="tsai_wu",
            label=label,
        )
        fi_results[label] = fi_max

    return fi_results


def print_summary(u_max, fi_results):
    vprint("\n" + "="*60)
    vprint("[POST] SUMMARY")
    vprint("="*60)
    vprint(f"  Max displacement magnitude : {u_max*1e3:.3f} mm\n")
    vprint("  Composite regions — Tsai-Wu:")
    for label, fi in fi_results.items():
        sf = 1/fi if fi > 0 else float("inf")
        flag = "  *** FAILURE ***" if fi >= 1.0 else ""
        vprint(f"    {label:<12}  FI = {fi:.4f}   SF = {sf:.2f}{flag}")
    vprint()



def export_results(domain, v, cell_tags, tag_all,
                   gdim, write_output, comm=MPI.COMM_WORLD):
    if not write_output:
        return
    result_folder = Path("Result")
    result_folder.mkdir(exist_ok=True, parents=True)

    disp = v.sub(0).collapse()
    rota = v.sub(1).collapse()

    Vout = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

    disp_out = fem.Function(Vout)
    disp_out.interpolate(disp)
    disp_out.name = "Displacement"

    rota_out = fem.Function(Vout)
    rota_out.interpolate(rota)
    rota_out.name = "Rotation"

    with io.XDMFFile(comm, result_folder / "results.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(disp_out)
        xdmf.write_function(rota_out)

    vprint(f"[EXPORT] Results -> {result_folder/'results.xdmf'}")

    DG0 = fem.functionspace(domain, ("DG", 0))
    mat_field = fem.Function(DG0, name="MaterialTag")
    for tag in tag_all:
        cells = cell_tags.find(tag)
        if len(cells) > 0:
            mat_field.x.array[cells] = float(tag)
    mat_field.x.scatter_forward()
    with io.XDMFFile(comm, result_folder / "material_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(mat_field)
    vprint(f"[EXPORT] Material tags -> {result_folder/'material_tags.xdmf'}")
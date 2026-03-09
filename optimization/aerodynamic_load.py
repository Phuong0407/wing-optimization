import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import meshio
from helper import vprint
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from dolfinx import fem


def import_foam_traction(foamfile, xdmffile):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        vprint()
        vprint(f"[FOAM LOAD IMPORT] Reading OpenFOAM {foamfile}.")
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(foamfile))
        reader.Update()
        vprint(f"[FOAM LOAD IMPORT] Done Reading OpenFOAM {foamfile}.")
        poly = reader.GetOutput()

        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly)
        tri.Update()
        poly = tri.GetOutput()

        points = vtk_to_numpy(poly.GetPoints().GetData())
        p      = vtk_to_numpy(poly.GetPointData().GetArray("p"))
        wss    = vtk_to_numpy(poly.GetPointData().GetArray("wallShearStress"))

        cells = poly.GetPolys()
        cells.InitTraversal()
        idList    = vtk.vtkIdList()
        triangles = []
        while cells.GetNextCell(idList):
            triangles.append([idList.GetId(i) for i in range(3)])
        triangles = np.array(triangles, dtype=np.int64)

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOff()
        normals.AutoOrientNormalsOn()
        normals.Update()
        poly_n = normals.GetOutput()

        n_unit = vtk_to_numpy(poly_n.GetCellData().GetArray("Normals"))

        P0 = points[triangles[:,0]]
        P1 = points[triangles[:,1]]
        P2 = points[triangles[:,2]]

        area = 0.5 * np.linalg.norm(np.cross(P1-P0, P2-P0), axis=1)

        p_tri   = p[triangles].mean(axis=1)
        wss_tri = wss[triangles].mean(axis=1)

        traction_cell  = -p_tri[:, None] * n_unit + wss_tri

        nodal_traction = np.zeros_like(points)
        nodal_area     = np.zeros(len(points))

        for i, tri_nodes in enumerate(triangles):
            a_third = area[i] / 3.0
            t_i = traction_cell[i]
            for j in tri_nodes:
                nodal_traction[j] += t_i * a_third
                nodal_area[j]     += a_third
        nodal_traction /= nodal_area[:, None]

        # if verbose:
        #     total_force = (traction_cell * area[:, None]).sum(axis=0)    
        #     vprint("[FOAM LOAD IMPORT] Total OpenFOAM force:", total_force)

        meshio.write(
            str(xdmffile),
            meshio.Mesh(
                points=points,
                cells=[("triangle", triangles)],
                point_data={"traction": nodal_traction},
            ),
        )
        vprint(f"[FOAM LOAD IMPORT] Exported traction -> {xdmffile}")
        vprint()
    comm.barrier()


def map_traction(foamfile, femfile, outfile, skin_phys_tags, unit="mm"):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        vprint()
        vprint(f"[TRACTION MAP] Reading {foamfile}.")
        fm = meshio.read(str(foamfile))
        fp = fm.points
        ft = fm.point_data["traction"]
        foam_tri = fm.cells_dict.get("triangle", None)
        vprint(f"[TRACTION MAP] Done Reading {foamfile}.")
    else:
        fp = None
        ft = None
        foam_tri = None

    fp = comm.bcast(fp, root=0)
    ft = comm.bcast(ft, root=0)
    foam_tri = comm.bcast(foam_tri, root=0)

    vprint(f"[TRACTION MAP] Reading gmsh File: {femfile}.")
    sm = meshio.read(str(femfile))
    all_triangles = sm.cells_dict.get("triangle", np.zeros((0, 3), dtype=np.int64))
    vprint(f"[TRACTION MAP] Done Reading gmsh File: {femfile}.")

    skin_triangles = []
    phys_data = None
    for key in ["gmsh:physical", "cell_tags"]:
        if key in sm.cell_data:
            phys_data = sm.cell_data[key]
            break

    if phys_data is not None:
        tri_blocks = [c.data for c in sm.cells if c.type == "triangle"]
        tag_blocks = [d for d, c in zip(phys_data, sm.cells) if c.type == "triangle"]
        for block_cells, block_tags in zip(tri_blocks, tag_blocks):
            mask = np.isin(block_tags, list(skin_phys_tags))
            skin_triangles.append(block_cells[mask])

        skin_triangles = np.concatenate(skin_triangles) if skin_triangles else all_triangles
    else:
        if rank == 0:
            vprint("[TRACTION MAP] Warning: no physical tag data — using all triangles")
        skin_triangles = all_triangles

    skin_node_idx = np.unique(skin_triangles)
    sp_all = sm.points.astype(np.float64)
    if unit == "mm":
        sp_all *= 1e-3
    sp = sp_all[skin_node_idx]

    # Setting degree = -1 removes all singular matrix
    interp = RBFInterpolator(fp, ft, kernel="thin_plate_spline", neighbors=10, degree=-1)
    skin_traction = interp(sp)

    nodal_traction = np.zeros((len(sp_all), 3), dtype=np.float64)
    nodal_traction[skin_node_idx] = skin_traction

    trib_area = np.zeros(len(sp_all))
    for tri_nodes in skin_triangles:
        P0, P1, P2 = sp_all[tri_nodes[0]], sp_all[tri_nodes[1]], sp_all[tri_nodes[2]]
        area = 0.5 * np.linalg.norm(np.cross(P1 - P0, P2 - P0))
        trib_area[tri_nodes] += area / 3.0

    nodal_force = nodal_traction * trib_area[:, None]

    if rank == 0 and foam_tri is not None:
        foam_area = 0.5 * np.linalg.norm(
            np.cross(fp[foam_tri[:, 1]] - fp[foam_tri[:, 0]],
                     fp[foam_tri[:, 2]] - fp[foam_tri[:, 0]]),
            axis=1
        )
        foam_tract = ft[foam_tri].mean(axis=1)
        foam_force = (foam_tract * foam_area[:, None]).sum(axis=0)

        fem_force = nodal_force.sum(axis=0)
        err = np.linalg.norm(fem_force - foam_force) / np.linalg.norm(foam_force)

        vprint(f"[TRACTION MAP] FOAM force [N] : {foam_force}")
        vprint(f"[TRACTION MAP] FEM  force [N] : {fem_force}")
        vprint(f"[TRACTION MAP] Force error    : {err*100:.2f}%")
        vprint(f"[TRACTION MAP] Skin nodes     : {len(skin_node_idx)} / {len(sp_all)} total")

    if rank == 0:
        meshio.write(
            str(outfile),
            meshio.Mesh(
                points=sp_all,
                cells=[("triangle", all_triangles)],
                point_data={"traction": nodal_traction},
            ),
        )
        vprint(f"[TRACTION MAP] Saved =====> {outfile}")
        vprint()

    comm.barrier()
    return nodal_force


def load_traction_xdmf(xdmffile, domain):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        vprint(f"[TRACTION LOAD] Reading OpenFOAM {xdmffile}.")
        data  = meshio.read(str(xdmffile))
        pts   = data.points
        tract = data.point_data["traction"]
        vprint(f"[TRACTION LOAD] Done Reading OpenFOAM {xdmffile}.")
    else:
        pts = None
        tract = None

    pts   = comm.bcast(pts, root=0)
    tract = comm.bcast(tract, root=0)

    VT = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    f  = fem.Function(VT, name="traction")
    coords = VT.tabulate_dof_coordinates()

    if rank == 0:
        vprint()
        vprint(f"[TRACTION LOAD] FEM  pts range : {domain.geometry.x.min(axis=0)} => {domain.geometry.x.max(axis=0)}")
        vprint(f"[TRACTION LOAD] XDMF pts range : {pts.min(axis=0)} => {pts.max(axis=0)}")

    tree = cKDTree(pts)
    dist, idx = tree.query(coords, k=1)

    # if verbose:
    dmax = comm.allreduce(dist.max(), op=MPI.MAX)
    if rank == 0:
        vprint(f"[TRACTION LOAD] Max KDTree dist : {dmax:.4e} m (should be < 1E-3 m)")
        vprint()

    f.x.array[:] = tract[idx].reshape(-1)
    f.x.scatter_forward()

    if comm.rank == 0:
        max_traction_magnitude = np.linalg.norm(tract, axis=1).max()
        vprint(f"[TRACTION LOAD] Max traction magnitude: {max_traction_magnitude:.3e} N")
    return f
from dolfinx import fem
from mpi4py import MPI
import ufl


def root_facets(facet_tags, tag_rib_curve):
    return facet_tags.find(tag_rib_curve)


def clamp_displacement(V, fdim, root_facets_idx):
    Vu, _   = V.sub(0).collapse()
    dofs_u  = fem.locate_dofs_topological((V.sub(0), Vu), fdim, root_facets_idx)
    bc_u    = fem.dirichletbc(fem.Function(Vu), dofs_u, V.sub(0))
    return bc_u, dofs_u

def clamp_rotation(V, fdim, root_facets_idx):
    Vt, _   = V.sub(1).collapse()
    dofs_t  = fem.locate_dofs_topological((V.sub(1), Vt), fdim, root_facets_idx)
    bc_t    = fem.dirichletbc(fem.Function(Vt), dofs_t, V.sub(1))
    return bc_t, dofs_t



def bc_full_clamped(V, facet_tags, fdim, tag_root, comm=None, verbose=True):
    comm = MPI.COMM_WORLD if comm is None else comm

    root = root_facets(facet_tags, tag_root)
    bc_u, dofs_u = clamp_displacement(V, fdim, root)
    bc_t, dofs_t = clamp_rotation(V, fdim, root)
    BCS = [bc_u, bc_t]

    if verbose:
        global_u = comm.allreduce(len(dofs_u[0]))
        global_t = comm.allreduce(len(dofs_t[0]))

        if comm.rank == 0:
            print(f"[BC] Full clamp (tag={tag_root})")
            print(f"      displacement DOFs = {global_u}")
            print(f"      rotation DOFs     = {global_t}")

    extras = {"ROOT": root}
    return BCS, extras



def bc_torsional_spring(mesh, V, facet_tags, fdim, tag_root,
                        k_theta_value,rotational_spring_components, ds,
                        comm=None, verbose=True):

    comm = MPI.COMM_WORLD if comm is None else comm

    root = root_facets(facet_tags, tag_root)
    bc_u, dofs_u = clamp_displacement(V, fdim, root)
    BCS = [bc_u]

    root_ds     = ds(tag_root)
    k_theta     = fem.Constant(mesh, float(k_theta_value))
    v           = ufl.TrialFunction(V)
    v_          = ufl.TestFunction(V)
    _, theta    = ufl.split(v)
    _, theta_   = ufl.split(v_)
    comps       = list(rotational_spring_components)
    spring_form = 0 * root_ds
    for c in comps:
        spring_form += k_theta * theta[c] * theta_[c] * root_ds

    if verbose:
        global_u = comm.allreduce(len(dofs_u[0]))
        if comm.rank == 0:
            print(f"[BC] Root torsional spring (tag={tag_root})")
            print(f"      displacement DOFs = {global_u}")
            print(f"      k_theta = {float(k_theta_value):.3e}")
            print(f"      components = {comps}")

    extras = dict(ROOT=root, spring_form=spring_form)
    return BCS, extras



def bc_prescribed_moment(mesh, V, facet_tags, fdim, tag_root, M_applied_val, ds,
                         comm=None, verbose=True):

    comm = MPI.COMM_WORLD if comm is None else comm

    root = root_facets(facet_tags, tag_root)
    bc_u, dofs_u = clamp_displacement(V, fdim, root)
    BCS = [bc_u]

    M_applied   = fem.Constant(mesh, tuple(float(x) for x in M_applied_val))
    root_ds     = ds(tag_root)
    v_          = ufl.TestFunction(V)
    _, theta_   = ufl.split(v_)
    moment_form = ufl.dot(M_applied, theta_) * root_ds

    if verbose:
        global_u = comm.allreduce(len(dofs_u[0]))
        if comm.rank == 0:
            print(f"[BC] Root prescribed moment (tag={tag_root})")
            print(f"      displacement DOFs = {global_u}")
            print(f"      M_applied = {tuple(M_applied_val)}")

    extras = dict(ROOT=root, moment_form=moment_form)
    return BCS, extras
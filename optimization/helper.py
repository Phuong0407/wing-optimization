import ufl
from dolfinx import fem
from mpi4py import MPI
import numpy as np

def vprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

def vprint_global(verbose, label, values, op=MPI.SUM):
    comm = MPI.COMM_WORLD

    vals = np.atleast_1d(values).astype(np.float64)
    global_vals = np.empty_like(vals)

    comm.Allreduce(vals, global_vals, op=op)

    if verbose and comm.rank == 0:
        if len(global_vals) == 1:
            print(f"{label} {global_vals[0]}")
        else:
            joined = " | ".join(f"{v:g}" for v in global_vals)
            print(f"{label} {joined}")


def normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))

def local_frame_ufl(domain):
    t  = ufl.Jacobian(domain)
    t1 = ufl.as_vector([t[0, 0], t[1, 0], t[2, 0]])
    t2 = ufl.as_vector([t[0, 1], t[1, 1], t[2, 1]])
    e3 = normalize(ufl.cross(t1, t2))
    ey = ufl.as_vector([0, 1, 0])
    ez = ufl.as_vector([0, 0, 1])
    e1_trial = ufl.cross(ey, e3)
    norm_e1  = ufl.sqrt(ufl.dot(e1_trial, e1_trial))
    e1 = ufl.conditional(ufl.lt(norm_e1, 0.5), ez, normalize(e1_trial))
    e2 = normalize(ufl.cross(e3, e1))
    return e1, e2, e3

def local_frame(domain):
    frame = local_frame_ufl(domain)
    VT    = fem.functionspace(domain, ("DG", 0, (3,)))
    V0, _ = VT.sub(0).collapse()
    basis = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(3)]
    for i in range(3):
        e_exp = fem.Expression(frame[i], V0.element.interpolation_points)
        basis[i].interpolate(e_exp)
    return basis[0], basis[1], basis[2]


def hstack(vecs):
    return ufl.as_matrix([[vi[i] for i in range(len(vi))] for vi in vecs]).T

def tangent_projection(e1, e2):
    return hstack([e1, e2])


def to_voigt(e):
    return ufl.as_vector([e[0, 0], e[1, 1], 2.0 * e[0, 1]])

def from_voigt(v):
    return ufl.as_tensor([[v[0], v[2]], [v[2], v[1]]])
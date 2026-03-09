import ufl
import dolfinx
from dolfinx import fem
from petsc4py import PETSc

def normalize(v):
  return v / ufl.sqrt(ufl.dot(v, v))

def _local_frame(domain):
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

def local_frame(domain,gdim):
  FRAME = _local_frame(domain)
  VT = fem.functionspace(domain, ("DG", 0, (gdim,)))
  V0, _ = VT.sub(0).collapse()
  BASIS_VECTORS = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(gdim)]
  for i in range(gdim):
    e_exp = fem.Expression(FRAME[i], V0.element.interpolation_points)
    BASIS_VECTORS[i].interpolate(e_exp)
  E1, E2, E3 = BASIS_VECTORS
  return E1, E2, E3

from petsc4py import PETSc

def project_scalar(expr, space, domain):
    w  = ufl.TrialFunction(space)
    q  = ufl.TestFunction(space)

    a = fem.form(ufl.inner(w, q) * ufl.dx)
    L = fem.form(ufl.inner(expr, q) * ufl.dx)

    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L)

    fn = fem.Function(space)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.solve(b, fn.x.petsc_vec)
    fn.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)
    return fn
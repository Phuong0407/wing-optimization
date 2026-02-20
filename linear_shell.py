from dolfin import *
from ufl import Jacobian, replace
import numpy as np

# material parameters
thick = Constant(0.01)
E = Constant(210e9)
rho = Constant(7850.0)
nu = Constant(0.3)
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)
g = Constant((0.0, 0.0, -9.81)) 
f = rho * thick * g

filename = "wingTRI.xdmf"

mesh = Mesh()
mvc  = MeshValueCollection("size_t", mesh, 1)

with XDMFFile("wingTRI.xdmf") as infile:
  print(type(mesh))
  infile.read(mesh)
  infile.read(mvc, "gmsh:physical")

import numpy as np
bbox = np.ptp(mesh.coordinates(), axis=0)
print("bbox (Δx, Δy, Δz) =", bbox)

facets = cpp.mesh.MeshFunctionSizet(mesh, mvc)

ys = mesh.coordinates()[:, 1]
ymin = ys.min()
print("ymin =", ymin)

class Root(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[1], ymin, 1e-6)

facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
Root().mark(facets, 11)
root_mesh = SubMesh(mesh, facets, 11)

def local_frame(mesh):
  t = Jacobian(mesh)
  if mesh.geometric_dimension() == 2:
    t1 = as_vector([t[0, 0], t[1, 0], 0])
    t2 = as_vector([t[0, 1], t[1, 1], 0])
  else:
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]])
    t2 = as_vector([t[0, 1], t[1, 1], t[2, 1]])
  e3 = cross(t1, t2)
  e3 /= sqrt(dot(e3, e3))
  ey = as_vector([0, 1, 0])
  ez = as_vector([0, 0, 1])
  e1 = cross(ey, e3)
  norm_e1 = sqrt(dot(e1, e1))
  e1 = conditional(lt(norm_e1, 0.5), ez, e1 / norm_e1)

  e2 = cross(e3, e1)
  e2 /= sqrt(dot(e2, e2))
  return e1, e2, e3

frame = local_frame(mesh)
e1, e2, e3 = frame
VT = VectorFunctionSpace(mesh, "DG", 0, dim=3)
# V = FunctionSpace(mesh, MixedElement([Ue, Te]))

ffile = XDMFFile("outputw.xdmf")
ffile.parameters["functions_share_mesh"] = True
for (i, ei) in enumerate(frame):
    ei = Function(VT, name="e{}".format(i + 1))
    ei.assign(project(frame[i], VT))
    ffile.write(ei, 0)


Ue = VectorElement("Lagrange", mesh.ufl_cell(), 2, dim=3)
Te = VectorElement("CR", mesh.ufl_cell(), 1, dim=3)
V = FunctionSpace(mesh, MixedElement([Ue, Te]))
v = Function(V)
u, theta = split(v)

v_ = TestFunction(V)
u_, theta_ = split(v_)
dv = TrialFunction(V)

def vstack(vectors):
    """Stack a list of vectors vertically."""
    return as_matrix([[v[i] for i in range(len(v))] for v in vectors])

def hstack(vectors):
    """Stack a list of vectors horizontally."""
    return vstack(vectors).T

P_plane = hstack([e1, e2])

def t_grad(u):
    """Tangential gradient operator"""
    g = grad(u)
    return dot(g, P_plane)

t_gu = dot(P_plane.T, t_grad(u))
eps = sym(t_gu)
kappa = sym(dot(P_plane.T, t_grad(cross(e3, theta))))
gamma = t_grad(dot(u, e3)) - dot(P_plane.T, cross(e3, theta))

eps_ = replace(eps, {v: v_})
kappa_ = replace(kappa, {v: v_})
gamma_ = replace(gamma, {v: v_})

def plane_stress_elasticity(e):
    return lmbda_ps * tr(e) * Identity(2) + 2 * mu * e

N = thick * plane_stress_elasticity(eps)
M = thick ** 3 / 12 * plane_stress_elasticity(kappa)
Q = mu * thick * gamma

drilling_strain = (t_gu[0, 1] - t_gu[1, 0]) / 2 - dot(theta, e3)
drilling_strain_ = replace(drilling_strain, {v: v_})
drilling_stress = E * thick ** 3 * drilling_strain

zero = Constant((0.0, 0.0, 0.0))
bc_u_root = DirichletBC(V.sub(0), zero, facets, 11)
bc_t_root = DirichletBC(V.sub(1), zero, facets, 11)
bcs = [bc_u_root, bc_t_root]

Wdef = (
    inner(N, eps_)
    + inner(M, kappa_)
    + dot(Q, gamma_)
    + drilling_stress * drilling_strain_
) * dx
a = derivative(Wdef, v, dv)
Wext = dot(f, u_) * dx

solve(a == Wext, v, bcs)

u = v.sub(0, True)
u.rename("Displacement", "u")
theta = v.sub(1, True)
theta.rename("Rotation", "theta")
ffile.write(u, 0)
ffile.write(theta, 0)
ffile.close()
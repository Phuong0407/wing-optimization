from dolfin import *

mesh = Mesh()
with XDMFFile("wingTRI.xdmf") as infile:
  infile.read(mesh)

element = MixedElement([VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        FiniteElement("N1curl", triangle, 1)])

Q = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)

Q_F = Q.full_space

q_ = Function(Q_F)
theta_, w_, R_gamma_, p_ = split(q_)
q = TrialFunction(Q_F)
q_t = TestFunction(Q_F)

E = Constant(1E11)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)
t = Constant(0.001)

k = sym(grad(theta_))
D = (E*t**3)/(12.0*(1.0 - nu**2))
psi_b = 0.5*D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)

psi_s = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)

f = Constant(1.0)
W_ext = inner(f*t**3, w_)*dx

dSp = Measure('dS', metadata={'quadrature_degree': 1})
dsp = Measure('ds', metadata={'quadrature_degree': 1})

n = FacetNormal(mesh)
t = as_vector((-n[1], n[0]))

inner_e = lambda x, y: (inner(x, t)*inner(y, t))('+')*dSp + \
                       (inner(x, t)*inner(y, t))('-')*dSp + \
                       (inner(x, t)*inner(y, t))*dsp

Pi_R = inner_e(gamma - R_gamma_, p_)

Pi = psi_b*dx + psi_s*dx + Pi_R - W_ext

dPi = derivative(Pi, q_, q_t)
J = derivative(dPi, q_, q)

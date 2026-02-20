import numpy as np
from dolfin import *
from fenics_shells import *

mesh = Mesh()
with XDMFFile("cylinder.xdmf") as f:
    f.read(mesh)

assert mesh.topology().dim() == 2
assert mesh.geometry().dim() == 3

R = 1.0
L = 5.0
t = Constant(0.02)
E = Constant(70e9)
nu = Constant(0.33)
mu = E/(2*(1+nu))
lmbda = 2*mu*nu/(1-2*nu)
G = mu

element = MixedElement([
    VectorElement("Lagrange", mesh.ufl_cell(), 1, dim=2),
    VectorElement("Lagrange", mesh.ufl_cell(), 2, dim=2),
    FiniteElement("Lagrange", mesh.ufl_cell(), 1),
    FiniteElement("N1curl", mesh.ufl_cell(), 1),
    FiniteElement("N1curl", mesh.ufl_cell(), 1)
])

U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
U_F = U.full_space
U_P = U.projected_space

u  = TrialFunction(U_F)
u_t= TestFunction(U_F)
u_ = Function(U_F)

v_, beta_, w_, Rgamma_, p_ = split(u_)
z_ = as_vector([v_[0], v_[1], w_])

# # ---- Reference normal from mesh (curved surface) ----
# # Tangential gradient via projection onto surface
# n = FacetNormal(mesh)
# # Surface gradient: grad_T u = (I - n⊗n) grad u
# I3 = Identity(3)
# P = I3 - outer(n, n)

# # ---- Linearized strain measures (small strain on curved surface) ----
# # Deformation gradient on surface:
# F = P*grad(z_) + P  # linearized around identity on tangent plane

# # Membrane strain (linearized Green)
# e = sym(P*grad(z_))  # tangential symmetric gradient

# # Curvature change (linearized)
# # director ~ rotation beta (small): d ≈ n + small terms
# # linearized bending strain via grad(beta) on surface:
# k = sym(P*grad(as_vector([beta_[0], beta_[1], 0.0])))

# # Shear (linearized)
# gamma = P*as_vector([beta_[0], beta_[1], 0.0]) + P*grad(w_)

# # ---- Isotropic constitutive (plane stress-like on surface) ----
# S = lambda X: 2.0*mu*X + (lmbda*tr(X))*Identity(3)

# psi_N = 0.5*t*inner(S(e), e)
# psi_K = 0.5*(t**3/12.0)*inner(S(k), k)
# psi_T = 0.5*t*mu*inner(Rgamma_, Rgamma_)

# # Durán–Liberman reduction
# inner_e = lambda x, y: (inner(x, y))*dx  # đơn giản cho test linear
# L_R = inner_e(gamma - Rgamma_, p_)

# # ---- Boundary conditions ----
# # Clamp at z=0
# def left(x, on_boundary):
#     return on_boundary and near(x[2], 0.0, 1e-6)

# bc_v = DirichletBC(U.sub(0), Constant((0.0, 0.0)), left)
# bc_b = DirichletBC(U.sub(1), Constant((0.0, 0.0)), left)
# bc_w = DirichletBC(U.sub(2), Constant(0.0), left)
# bcs = [bc_v, bc_b, bc_w]

# # ---- Torsional traction at z=L ----
# # tangential direction e_theta = (-y/R, x/R, 0)
# x = SpatialCoordinate(mesh)
# e_theta = as_vector([-x[1]/R, x[0]/R, 0.0])

# Tval = 1e5  # applied torque
# tau = Constant(Tval/(2*np.pi*R**2*t))  # Bredt shear flow / t

# # Mark right boundary
# class Right(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and near(x[2], L, 1e-6)

# facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
# Right().mark(facets, 1)
# ds = Measure("ds", subdomain_data=facets)

# W_ext = inner(tau*e_theta, z_)*ds(1)

# # ---- Variational problem ----
# Lagr = (psi_N + psi_K + psi_T)*dx + L_R - W_ext
# F_form = derivative(Lagr, u_, u_t)
# J_form = derivative(F_form, u_, u)

# u_p_ = Function(U_P)
# problem = ProjectedNonlinearProblem(U_P, F_form, u_, u_p_, bcs=bcs, J=J_form)

# solver = NewtonSolver()
# solver.parameters["linear_solver"] = "mumps"
# solver.parameters["relative_tolerance"] = 1e-8
# solver.parameters["absolute_tolerance"] = 1e-12
# solver.solve(problem, u_p_.vector())

# # ---- Postprocess: twist angle at z=L ----
# v_h, beta_h, w_h, Rg_h, p_h = u_.split(deepcopy=True)

# # lấy một điểm trên mép phải, ví dụ (R,0,L)
# pt = np.array([R, 0.0, L])
# ux = v_h(pt)[0]
# uy = v_h(pt)[1]

# theta_num = np.arctan2(uy, R+ux)/L

# theta_ana = Tval*L/(4*np.pi**2*R**3*float(t)*float(G))

# print("Twist per unit length (num) =", theta_num)
# print("Twist per unit length (ana) =", theta_ana)
# print("Relative error =", abs(theta_num-theta_ana)/theta_ana)

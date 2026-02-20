from dolfin import *
from ufl import Jacobian, replace
import numpy as np
import meshio

# material parameters
thick = Constant(1e-3)
E = Constant(210e9)
nu = Constant(0.3)
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)

# loading (self-weight)
f = Constant((0, 0.2, -1))


# ### Loading the mesh and computing a local tangent frame
# 
# We now load the mesh as well as the facet MeshFunction from the XDMF files generated from a Gmsh script using Gmsh's Python API (see [this notebook](I_beam_gmsh.ipynb)).
# 
# We then define the `local_frame` function which returns the UFL representation of the local frame  $(\be_1,\be_2,\be_3)$. We first use UFL's `Jacobian` function to retrieve the Jacobian of the mapping from reference cells to spatial coordinates. Contrary to the `fenics-shells` approach which uses a global analytical mapping between the reference domain and the shell surface, we use here the reference element mapping of the finite-element method. For a shell, the Jacobian is of size $3\times 2$ with both columns $\bt_1,\bt_2$ being a vector of the local tangent plane. We therefore compute $\be_3$, the normal to the mid-surface from $\bt_1\times \bt_2$. We then have to choose a convention to compute $\be_1,\be_2$, knowing $\be_3$. Our convention is that $\be_1$ lies in the plane orthogonal to $\be_Y$ and $\be_3$. If $\be_3$ happens to be colinear to $\be_Y$, we set $\be_1=\be_Z$. $\be_2$ then follows from $\be_2=\be_3\times\be_1$.

# In[11]:


filename = "I_beam.xdmf"

mesh = Mesh()
with XDMFFile(filename) as mesh_file:
    mesh_file.read(mesh)


mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile(filename.replace(".", "_facet_region.")) as infile:
    infile.read(mvc, "name_to_read")
facets = MeshFunction("size_t", mesh, mvc)

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

ffile = XDMFFile("output.xdmf")
ffile.parameters["functions_share_mesh"] = True
for (i, ei) in enumerate(frame):
    ei = Function(VT, name="e{}".format(i + 1))
    ei.assign(project(frame[i], VT))
    ffile.write(ei, 0)


# The local frame is then stored in an output XDMFFile. Note that when it is represented via `DG0` Functions but Paraview does not seem to handle very well such fields on a surface shell. The fields seem to be displayed as `CG1` Functions, inducing erratic orientations near the boundary. Overall, the orientation is however consistent with what has been defined ($\be_1$ in blue, $\be_2$ in green, $\be_3$ in red).
# 
# <center><img src="local_frame.png" width="800"></center>
# 
# 
# ### FunctionSpace choice and strain measures
# 
# We now use the afore-mentioned `P2/CR`interpolation for the displacement $\bu$ and the rotation $\btheta$ variables using a `MixedElement`.

# In[12]:


Ue = VectorElement("Lagrange", mesh.ufl_cell(), 2, dim=3)
Te = VectorElement("CR", mesh.ufl_cell(), 1, dim=3)
V = FunctionSpace(mesh, MixedElement([Ue, Te]))

v = Function(V)
u, theta = split(v)

v_ = TestFunction(V)
u_, theta_ = split(v_)
dv = TrialFunction(V)


# We then define the in-plane projector and the in-plane (tangential) gradient operator ($\tilde\nabla$ above) as follows:

# In[13]:


def vstack(vectors):
    """Stack a list of vectors vertically."""
    return as_matrix([[v[i] for i in range(len(v))] for v in vectors])


def hstack(vectors):
    """Stack a list of vectors horizontally."""
    return vstack(vectors).T


# In-plane projection
P_plane = hstack([e1, e2])

def t_grad(u):
    """Tangential gradient operator"""
    g = grad(u)
    return dot(g, P_plane)


# We then extract gradient of the in-plane displacement $\tilde\nabla\tilde\bu$ and define the membrane strain $\bepsilon$ as its symmetric part. We similarly define the bending curvature $\bchi$ and shear strain $\bgamma$.

# In[14]:


t_gu = dot(P_plane.T, t_grad(u))
eps = sym(t_gu)
kappa = sym(dot(P_plane.T, t_grad(cross(e3, theta))))
gamma = t_grad(dot(u, e3)) - dot(P_plane.T, cross(e3, theta))

eps_ = replace(eps, {v: v_})
kappa_ = replace(kappa, {v: v_})
gamma_ = replace(gamma, {v: v_})


# ### Stress measures
# 
# We then define the corresponding stress measure using linear elastic constitutive laws.
# 
# 

# In[15]:


def plane_stress_elasticity(e):
    return lmbda_ps * tr(e) * Identity(2) + 2 * mu * e

N = thick * plane_stress_elasticity(eps)
M = thick ** 3 / 12 * plane_stress_elasticity(kappa)
Q = mu * thick * gamma


# ### Drilling rotation stabilization
# 
# A classical problem in shell models involving 6 degrees of freedom (3D rotation) is the absence of any constraint on the drilling rotation 
# $\theta_3=\btheta\cdot\be_3$. However, this degree of freedom is necessary to tackle non-smooth junctions
# between planar shell facets which have a different normal vector. In our
# implementation, we propose to add an additional quadratic energy penalizing the drilling strain, as commonly done for elastic shells.
# The drilling strain is obtained from the skew symmetric in-plane component of the transformation gradient and the drilling rotation: 
# \begin{equation}
# \varpi = \dfrac{1}{2}(u_{1,2}-u_{2,1})+\theta_3
# \end{equation}
# 
# We consider an additional drilling contribution to the work of deformation given by:
# 
# \begin{equation}
# w_\text{def,drilling} = E h^3 \varpi^2
# \end{equation}
# where the drilling stiffness $Eh^3$ is the usually recommended value in the literature.

# In[16]:


drilling_strain = (t_gu[0, 1] - t_gu[1, 0]) / 2 - dot(theta, e3)
drilling_strain_ = replace(drilling_strain, {v: v_})
drilling_stress = E * thick ** 3 * drilling_strain


# ### Boundary conditions and resolution
# We then apply fixed boundary conditions on both ends and finally form the corresponding bilinear form and linear right-hand side form before solving the corresponding system.

# In[17]:


bc = [
    DirichletBC(V.sub(0), Constant((0.0,) * 3), facets, 1),
    DirichletBC(V.sub(0), Constant((0.0,) * 3), facets, 2),
]

Wdef = (
    inner(N, eps_)
    + inner(M, kappa_)
    + dot(Q, gamma_)
    + drilling_stress * drilling_strain_
) * dx
a = derivative(Wdef, v, dv)
Wext = dot(f, u_) * dx

solve(a == Wext, v, bc)


# We then output the displacement and rotation fields to a XDMF file.

# In[18]:


u = v.sub(0, True)
u.rename("Displacement", "u")
theta = v.sub(1, True)
theta.rename("Rotation", "theta")
ffile.write(u, 0)
ffile.write(theta, 0)
ffile.close()


# which yields the following deformed configuration (2000 times amplification):
# 
# <center><img src="deformed_configuration.png" width="800"></center>
# 
# 
# ## References
# 
# [CAM03] Campello, E. M. B., Pimenta, P. M., & Wriggers, P. (2003). A triangular finite shell element based on a fully nonlinear shell formulation. Computational Mechanics, 31(6), 505-518.

# In[ ]:





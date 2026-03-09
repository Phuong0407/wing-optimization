import ufl

def hstack(vecs):
  return ufl.as_matrix([[vi[i] for i in range(len(vi))] for vi in vecs]).T

def tangent_projection(e1,e2):
  return hstack([e1, e2])

def tangential_gradient(w, P_plane):
  return ufl.dot(ufl.grad(w), P_plane)

def membrane_strain(u, P_plane):
  t_gu = ufl.dot(P_plane.T, tangential_gradient(u, P_plane))
  return ufl.sym(t_gu), t_gu

def bending_strain(theta, e3, P_plane):
  beta = ufl.cross(e3, theta)
  return ufl.sym(
    ufl.dot(P_plane.T, tangential_gradient(beta, P_plane))
  )

def shear_strain(u, theta, e3, P_plane):
  beta = ufl.cross(e3, theta)
  return (
    tangential_gradient(ufl.dot(u, e3), P_plane)
    - ufl.dot(P_plane.T, beta)
  )

def drilling_strain(t_gu, theta, e3):
  return (t_gu[0, 1] - t_gu[1, 0]) / 2 + ufl.dot(theta, e3)

def shell_strains(u, theta, e1, e2, e3):
  P_plane   = tangent_projection(e1, e2)
  eps, t_gu = membrane_strain(u, P_plane)
  kappa     = bending_strain(theta, e3, P_plane)
  gamma     = shear_strain(u, theta, e3, P_plane)
  eps_d     = drilling_strain(t_gu, theta, e3)
  
  return eps, kappa, gamma, eps_d

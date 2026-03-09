import ufl
from helper import tangent_projection

def tangential_gradient(w, P):
    return ufl.dot(ufl.grad(w), P)

def membrane_strain(u, P):
    t_gu = ufl.dot(P.T, tangential_gradient(u, P))
    return ufl.sym(t_gu), t_gu

def bending_strain(theta, e3, P):
    beta = ufl.cross(e3, theta)
    return ufl.sym(ufl.dot(P.T, tangential_gradient(beta, P)))

def shear_strain(u, theta, e3, P):
    beta = ufl.cross(e3, theta)
    return tangential_gradient(ufl.dot(u, e3), P) - ufl.dot(P.T, beta)

def compute_drilling_strain(t_gu, theta, e3):
    return (t_gu[0, 1] - t_gu[1, 0]) / 2 + ufl.dot(theta, e3)

def shell_strains(u, theta, e1, e2, e3):
    P               = tangent_projection(e1, e2)
    eps, t_gu       = membrane_strain(u, P)
    kappa           = bending_strain(theta, e3, P)
    gamma           = shear_strain(u, theta, e3, P)
    drilling_strain = compute_drilling_strain(t_gu, theta, e3)
    return eps, kappa, gamma, drilling_strain

def shell_strains_test(eps, kappa, gamma, drilling_strain, v, v_):
    eps_             = ufl.derivative(eps,           v, v_)
    kappa_           = ufl.derivative(kappa,         v, v_)
    gamma_           = ufl.derivative(gamma,         v, v_)
    drilling_strain_ = ufl.replace(drilling_strain, {v: v_})
    return eps_, kappa_, gamma_, drilling_strain_

def shell_strains_from_model(model):
    u           = model.u
    theta       = model.theta
    e1, e2, e3  = model.e1, model.e2, model.e3
    v           = model.v
    v_          = model.v_

    eps, kappa, gamma, drilling_strain = shell_strains(u, theta, e1, e2, e3)
    eps_, kappa_, gamma_, drilling_strain_ = shell_strains_test(eps, kappa, gamma, drilling_strain, v, v_)
    return eps, kappa, gamma, drilling_strain, eps_, kappa_, gamma_, drilling_strain_

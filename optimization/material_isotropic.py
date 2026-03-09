from types import SimpleNamespace
from dolfinx import fem
from helper import vprint, local_frame, tangent_projection, to_voigt
from shell_kinematics import membrane_strain, bending_strain
import numpy as np
import ufl



def isotropic_material(thickness, young, poisson, domain):
    h        = fem.Constant(domain, float(thickness))
    E        = fem.Constant(domain, float(young))
    nu       = fem.Constant(domain, float(poisson))
    lmbda    = E * nu / (1 + nu) / (1 - 2 * nu)
    mu       = E / 2 / (1 + nu)
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)
    return SimpleNamespace(
            h=h,
            E=E,
            nu=nu,
            lmbda=lmbda,
            mu=mu,
            lmbda_ps=lmbda_ps,
            kind="isotropic",
        )




def von_mises_iso_cells(domain, v_sol, mat, cell_indices, label=""):
    assert mat.kind == "isotropic"
    if len(cell_indices) == 0:
        vprint(f"  [{label}] NO CELLS — SKIPPING VON MISES ANALYSIS.")
        return 0.0

    u_h, theta_h = ufl.split(v_sol)
    e1, e2, e3 = local_frame(domain)
    P = tangent_projection(e1, e2)
    eps_h, _ = membrane_strain(u_h, P)
    kappa_h = bending_strain(theta_h, e3, P)

    DG0 = fem.functionspace(domain, ("DG", 0, (3,)))
    eps_fn = fem.Function(DG0)
    kap_fn = fem.Function(DG0)
    eps_fn.interpolate(fem.Expression(to_voigt(eps_h),   DG0.element.interpolation_points))
    kap_fn.interpolate(fem.Expression(to_voigt(kappa_h), DG0.element.interpolation_points))

    eps0_vals  = eps_fn.x.array.reshape(-1, 3)[cell_indices]
    kappa_vals = kap_fn.x.array.reshape(-1, 3)[cell_indices]

    E_val  = float(mat.E.value)
    nu_val = float(mat.nu.value)
    h_val  = float(mat.h.value)
    lps    = E_val * nu_val / (1 - nu_val**2)
    mu_val = E_val / (2 * (1 + nu_val))

    def constitutive(e_v):
        s11 = (lps + 2*mu_val) * e_v[:, 0] + lps * e_v[:, 1]
        s22 =  lps             * e_v[:, 0] + (lps + 2*mu_val) * e_v[:, 1]
        s12 =  mu_val          * e_v[:, 2]
        return s11, s22, s12

    e_mid = eps0_vals
    e_top = eps0_vals + (h_val / 2.0) * kappa_vals
    e_bot = eps0_vals - (h_val / 2.0) * kappa_vals

    vm_list = []
    for e_v, fiber in [(e_mid, "mid"), (e_top, "top"), (e_bot, "bot")]:
        s11, s22, s12 = constitutive(e_v)
        vm = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)
        vm_max = vm.max()
        vm_list.append(vm_max)
        vprint(f"  [{label}]  fiber={fiber}  max sigma_vm = {vm_max/1e6:.2f} MPa")

    return max(vm_list)
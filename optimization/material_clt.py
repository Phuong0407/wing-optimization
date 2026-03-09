import numpy as np
from helper import vprint
from types import SimpleNamespace
import ufl

def _Q_ply(E1, E2, G12, nu12):
    nu21 = nu12 * E2 / E1
    d = 1 - nu12 * nu21
    return np.array([
        [ E1/d,      nu12*E2/d, 0   ],
        [ nu12*E2/d, E2/d,      0   ],
        [ 0,         0,         G12 ],
    ])

def _Qbar_ply(Q, angle):
    a = np.radians(angle)
    s = np.sin(a)
    c = np.cos(a)
    T = np.array([
        [ c**2,  s**2,   2*c*s       ],
        [ s**2,  c**2,  -2*c*s       ],
        [-c*s,   c*s,    c**2 - s**2 ],
    ])
    R = np.diag([1.0, 1.0, 2.0])
    Rinv = np.diag([1.0, 1.0, 0.5])
    return np.linalg.inv(T) @ Q @ R @ T @ Rinv

def compute_ABD(layup_angles, t_ply, E1, E2, G12, nu12):
    Q = _Q_ply(E1, E2, G12, nu12)
    H = t_ply * len(layup_angles)
    z = -H / 2.0
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for angle in layup_angles:
        Qb     = _Qbar_ply(Q, angle)
        z0, z1 = z, z + t_ply
        A += Qb * (z1 - z0)
        B += Qb * (z1**2 - z0**2) / 2.0
        D += Qb * (z1**3 - z0**3) / 3.0
        z  = z1
    return A, B, D, H

def clt_composite(layup_angles, t_ply, E1, E2, G12, nu12, G13=None, G23=None, kappa_s=5/6, label="CLT", verbose=False):
    if G13 is None: G13 = G12
    if G23 is None: G23 = G12 * 0.5

    A_np, B_np, D_np, H = compute_ABD(layup_angles, t_ply, E1, E2, G12, nu12)

    max_B = np.abs(B_np).max()
    if verbose:
        vprint(f"[{label}] Layup   : {layup_angles}")
        vprint(f"[{label}] H       : {H*1E3:.2f} mm")
        vprint(f"[{label}] max|B|  : {max_B:.2e}  "
            f"{'SYMMETRIC' if max_B < 1E-6 * A_np.max() else 'NON-SYMMETRIC'}")
        vprint(f"[{label}] A11     : {A_np[0,0]/1E6:.2f} MPa·m")
        vprint(f"[{label}] D11     : {D_np[0,0]:.4f} N·m^2")

    As_np = kappa_s * H * np.array([
        [G13, 0.0],
        [0.0, G23],
    ])

    # EFFECTIVE IN-PLANE SHEAR FOR DRILLING STABILISATION
    G_eff = float(A_np[2, 2]) / H

    return SimpleNamespace(
        kind   = "clt",
        H      = H,
        A_np   = A_np, B_np = B_np, D_np = D_np, As_np = As_np,
        G_eff  = G_eff,
        A_ufl  = ufl.as_tensor(A_np),
        B_ufl  = ufl.as_tensor(B_np),
        D_ufl  = ufl.as_tensor(D_np),
        As_ufl = ufl.as_tensor(As_np),
    )





def tsai_wu(sigma_mat, strengths):
    s1, s2, s6 = sigma_mat
    Xt = strengths["Xt"]
    Xc = strengths["Xc"]
    Yt = strengths["Yt"]
    Yc = strengths["Yc"]
    S = strengths["S"]
    F1 = 1/Xt - 1/Xc
    F2 = 1/Yt - 1/Yc
    F11 = 1/(Xt*Xc)
    F22 = 1/(Yt*Yc)
    F66 = 1/S**2
    F12 = -0.5 / np.sqrt(Xt * Xc * Yt * Yc)
    return (F1*s1 + F2*s2 + F11*s1**2 + F22*s2**2 + F66*s6**2 + 2*F12*s1*s2)

def hashin(sigma_mat, strengths):
    s1, s2, s6 = sigma_mat
    Xt, Xc     = strengths["Xt"], strengths["Xc"]
    Yt, Yc     = strengths["Yt"], strengths["Yc"]
    SL         = strengths["S"]
    ST         = strengths.get("ST", Yc / 2.0)
    out = {}
    if s1 >= 0:
        out["fiber_t"]  = (s1/Xt)**2 + (s6/SL)**2
    else:
        out["fiber_c"]  = (s1/Xc)**2
    if s2 >= 0:
        out["matrix_t"] = (s2/Yt)**2 + (s6/SL)**2
    else:
        out["matrix_c"] = ((s2/(2*ST))**2 + (Yc/(2*ST))**2 * (s2/Yc) + (s6/SL)**2)
    return out
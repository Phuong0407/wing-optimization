import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re
import os

MAIN_SCRIPT = "wingfiber.py"

# =============================================================================
# EXACT AERODYNAMIC CONSTANTS (Derived from Report)
# =============================================================================
V_cruise = 250.0        # m/s (from CFD freestream conditions)
rho = 0.7361            # kg/m^3 (ISA at 5000m)
q = 0.5 * rho * V_cruise**2

# Geometry
c_root = 1.0            # m
c_tip = 0.5             # m
b_half = 3.0            # m
S_half = b_half * (c_root + c_tip) / 2.0  # 2.25 m^2
S_wing = S_half * 2.0                     # 4.5 m^2 (Full planform area)
AR_geom = (b_half * 2.0)**2 / S_wing      # 8.0
taper = c_tip / c_root                    # 0.5

# OpenFOAM CFD Loads (on the half-wing)
F_x_half = 1133.076     # N (Drag)
F_z_half = 17824.76     # N (Lift)

# Calculate Exact Rigid Coefficients
C_L_rigid = F_z_half / (q * S_half)
C_D_total_rigid = F_x_half / (q * S_half)

def calculate_planform_constants(AR, taper):
    """Solves Prandtl's Monoplane Equation and returns lift slopes and drag constants."""
    a0 = 2 * np.pi  
    N = 20          
    n_vals = np.arange(1, 2*N, 2)
    theta = np.linspace(np.pi/(4*N), np.pi/2, N)
    
    c_root_over_b = 2.0 / (AR * (1.0 + taper))
    c_over_b = c_root_over_b * (1.0 - (1.0 - taper) * np.cos(theta))
    
    Matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            n = n_vals[j]
            Matrix[i, j] = np.sin(n * theta[i]) * (np.sin(theta[i]) + n * a0 * c_over_b[i] / 4.0)

    RHS_A = (a0 * c_over_b / 4.0) * 1.0 * np.sin(theta)
    A_coeffs = np.linalg.solve(Matrix, RHS_A)

    RHS_B = (a0 * c_over_b / 4.0) * np.cos(theta) * np.sin(theta)
    B_coeffs = np.linalg.solve(Matrix, RHS_B)

    a1, b1 = A_coeffs[0], B_coeffs[0]
    
    u_inv, v_sum, w_sum = 0.0, 0.0, 0.0
    for i in range(N):
        n = n_vals[i]
        an_a1 = A_coeffs[i] / a1
        bn_term = B_coeffs[i] - A_coeffs[i] * (b1 / a1)
        
        u_inv += n * (an_a1)**2
        if n > 1:
            v_sum += n * an_a1 * bn_term
            w_sum += n * (bn_term)**2

    u = 1.0 / u_inv
    v = 2.0 * v_sum
    w = np.pi * AR * w_sum
    return a1, b1, u, v, w

# Baseline Rigid Wing Analysis to isolate C_D0
a1_rigid, _, u_rigid, _, _ = calculate_planform_constants(AR_geom, taper)
C_L_alpha_rigid = a1_rigid * np.pi * AR_geom

# Extract effective cruise angle of attack & exact parasitic drag
alpha_root = C_L_rigid / C_L_alpha_rigid
C_Di_rigid = (C_L_rigid**2) / (np.pi * AR_geom * u_rigid)
C_D0 = C_D_total_rigid - C_Di_rigid

L_rigid_total = C_L_rigid * q * S_wing

print(f"--- Rigid Wing Aerodynamics (Derived from CFD) ---")
print(f"C_L Rigid  : {C_L_rigid:.4f}")
print(f"C_D Total  : {C_D_total_rigid:.5f}")
print(f"C_D0 Exact : {C_D0:.5f} (Parasitic)")
print(f"C_Di Rigid : {C_Di_rigid:.5f} (Induced)")
print(f"Eff. Alpha : {np.degrees(alpha_root):.2f} degrees\n")

# =============================================================================
# SWEEP INITIALIZATION
# =============================================================================
eta_values = np.linspace(0.0, 0.9, 20)

max_displacements, max_rotations = [], []
tip_bendings, center_bendings, ea_bendings, tip_angles = [], [], [], []
b_projs, delta_lifts, modified_CLs, modified_CDs = [], [], [], []
induced_drag_coeffs, e_factors, finesses = [], [], [] # <-- NEW: added finesses array

print("=== Starting Eta Sweep ===")

for eta in eta_values:
    eta_str = f"{eta:.3f}"
    print(f"Running simulation for eta = {eta_str}...")
    
    result = subprocess.run(["python3", MAIN_SCRIPT, eta_str], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running eta={eta_str}.")
        break

    disp_match = re.search(r"Max displacement.*:\s*([0-9\.eE\+\-]+)", result.stdout)
    rot_match = re.search(r"Max rotation.*:\s*([0-9\.eE\+\-]+)", result.stdout)
    bend_match = re.search(r"Mean Tip Bending.*:\s*([0-9\.eE\+\-]+)", result.stdout) 
    angle_match = re.search(r"Tip Bending Angle.*:\s*([0-9\.eE\+\-]+)", result.stdout)
    center_bend_match = re.search(r"Max Center Bending.*:\s*([0-9\.eE\+\-]+)", result.stdout)
    ea_bend_match = re.search(r"Max EA Bending.*:\s*([0-9\.eE\+\-]+)", result.stdout)
    
    if disp_match and angle_match:
        disp_val = float(disp_match.group(1))
        rot_val = float(rot_match.group(1))
        
        max_displacements.append(disp_val)
        max_rotations.append(rot_val)
        tip_bendings.append(abs(float(bend_match.group(1)))) 
        center_bendings.append(abs(float(center_bend_match.group(1))))
        ea_bendings.append(abs(float(ea_bend_match.group(1))))
        tip_angles.append(abs(float(angle_match.group(1)))) 
        
        csv_file = f"ea_data_eta_{eta_str}.csv"
        
        if os.path.exists(csv_file):
            data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
            y_coords = data[:, 0]
            bending_angles = data[:, 1]
            
            # 1. Bending Penalty: Calculate Projected Span
            dy = np.diff(y_coords)
            cos_gamma = np.cos(bending_angles[:-1])
            b_half_proj = np.sum(cos_gamma * dy)
            b_proj = 2.0 * b_half_proj
            AR_bent = (b_proj**2) / S_wing 
            
            # 2. Re-calculate LLT constants for the BENT Aspect Ratio
            a1_bent, b1_bent, u_bent, v_bent, w_bent = calculate_planform_constants(AR_bent, taper)
            C_L_alpha_bent = a1_bent * np.pi * AR_bent
            C_L_theta_bent = b1_bent * np.pi * AR_bent
            
            # 3. Twist Penalty
            theta_t = -abs(rot_val) 
            
            # 4. Calculate Aeroelastic Deformed Lift
            C_L_def = (C_L_alpha_bent * alpha_root) + (C_L_theta_bent * theta_t)
            L_def_total = C_L_def * q * S_wing
            Delta_L = L_def_total - L_rigid_total
            
            # 5. Calculate Modified Total Drag & Induced Drag Coefficient
            C_Di_def = (C_L_def**2 / (np.pi * AR_bent * u_bent)) + (v_bent * C_L_def * theta_t) + (w_bent * theta_t**2)
            C_D_total = C_D0 + C_Di_def
            e_def = (C_L_def**2) / (np.pi * AR_bent * C_Di_def)
            
            # 6. Calculate Finesse (L/D ratio)
            finesse = (C_L_def + C_L_rigid) / C_D_total # <-- NEW: Calculate L/D
            
            b_projs.append(b_proj)
            delta_lifts.append(Delta_L)
            modified_CLs.append(C_L_def + C_L_rigid)
            modified_CDs.append(C_D_total)
            induced_drag_coeffs.append(C_Di_def) 
            e_factors.append(e_def)       
            finesses.append(finesse)      # <-- NEW: Store L/D
            
            os.remove(csv_file)
        else:
            delta_lifts.append(np.nan)
            modified_CLs.append(np.nan)
            modified_CDs.append(np.nan)
            b_projs.append(np.nan)
            induced_drag_coeffs.append(np.nan)
            e_factors.append(np.nan)
            finesses.append(np.nan)       # <-- NEW: Handle missing data

print("\n=== Sweep Complete! Generating Plots ===")

# =============================================================================
# PLOTTING RESULTS (Now a 4x3 Grid)
# =============================================================================
fig, axes = plt.subplots(4, 3, figsize=(18, 20)) # Increased height for 4 rows

# --- ROW 1, PLOT 1: Displacements ---
axes[0, 0].plot(eta_values, max_displacements, 'b-o', label='Total Max Displacement')
axes[0, 0].plot(eta_values, tip_bendings, 'c--s', label='Mean Tip Bending')
axes[0, 0].plot(eta_values, center_bendings, 'm-^', label='Max Center Bending (50%)') 
axes[0, 0].plot(eta_values, ea_bendings, 'y-d', label='Max EA Bending (25%)') 
axes[0, 0].set_title("Displacements vs $\eta$")
axes[0, 0].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[0, 0].set_ylabel("Displacement (m)")
axes[0, 0].grid(True, linestyle='--')
axes[0, 0].legend(fontsize=8) 

# --- ROW 1, PLOT 2: Rotation ---
axes[0, 1].plot(eta_values, max_rotations, 'r-o')
axes[0, 1].set_title("Max Torsional Rotation vs $\eta$")
axes[0, 1].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[0, 1].set_ylabel("Max Rotation (rad)")
axes[0, 1].grid(True, linestyle='--')

# --- ROW 1, PLOT 3: Trade-off (Pareto) with ETA Annotations ---
axes[0, 2].plot(max_rotations, max_displacements, 'g-s')
axes[0, 2].set_title("Trade-off: Displacement vs Torsion")
axes[0, 2].set_xlabel("Max Rotation (rad)")
axes[0, 2].set_ylabel("Total Max Displacement (m)")
axes[0, 2].grid(True, linestyle='--')

for i, txt in enumerate(eta_values):
    if i % 2 == 0:  
        axes[0, 2].annotate(f"$\eta$={txt:.2f}", 
                            (max_rotations[i], max_displacements[i]), 
                            textcoords="offset points", 
                            xytext=(0, 6), 
                            ha='center', 
                            fontsize=8,
                            color='darkgreen')

# --- ROW 2, PLOT 1: Projected Span vs Eta ---
axes[1, 0].plot(eta_values, b_projs, 'k-p', linewidth=2)
axes[1, 0].set_title("Bending Penalty: Projected Span vs $\eta$")
axes[1, 0].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[1, 0].set_ylabel("Projected Span $b_{proj}$ (m)")
axes[1, 0].grid(True, linestyle='--')

# --- ROW 2, PLOT 2: Tip Bending Angle ---
axes[1, 1].plot(eta_values, tip_angles, color='purple', marker='v', linewidth=2)
axes[1, 1].set_title("Tip Bending Angle vs $\eta$")
axes[1, 1].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[1, 1].set_ylabel("Tip Bending Angle (rad)")
axes[1, 1].grid(True, linestyle='--')

# --- ROW 2, PLOT 3: Total Lift Coefficient ---
axes[1, 2].plot(eta_values, modified_CLs, color='dodgerblue', marker='s', linewidth=2)
axes[1, 2].set_title("Aeroelastic Lift Coefficient ($C_L$) vs $\eta$")
axes[1, 2].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[1, 2].set_ylabel("Total $C_L$")
axes[1, 2].grid(True, linestyle='--')

# --- ROW 3, PLOT 1: Additional Lift (Delta L) ---
axes[2, 0].plot(eta_values, delta_lifts, color='firebrick', marker='X', linewidth=2)
axes[2, 0].axhline(0, color='black', linestyle=':', linewidth=1.5) 
axes[2, 0].set_title("Lift Generation Change ($\Delta L_{full wing}$) vs $\eta$")
axes[2, 0].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[2, 0].set_ylabel("$\Delta L$ (Newtons)")
axes[2, 0].grid(True, linestyle='--')

# --- ROW 3, PLOT 2: Aeroelastic Drag Polar (CL vs CD) ---
scatter = axes[2, 1].scatter(modified_CDs, modified_CLs, c=eta_values, cmap='viridis', s=80, edgecolor='k', zorder=5)
axes[2, 1].plot(modified_CDs, modified_CLs, color='gray', linestyle='--', linewidth=1, zorder=4)
axes[2, 1].set_title("Aeroelastic Drag Polar ($C_L$ vs $C_D$)")
axes[2, 1].set_xlabel("Total Drag Coefficient $C_D$ ($C_{D0} + C_{D_i}$)")
axes[2, 1].set_ylabel("Total Lift Coefficient $C_L$")
axes[2, 1].grid(True, linestyle='--')
cbar = fig.colorbar(scatter, ax=axes[2, 1])
cbar.set_label('Fiber Fraction $\eta$ ($0^\circ$)')

# --- ROW 3, PLOT 3: INDUCED DRAG COEFFICIENT & EFFICIENCY ---
color1 = 'darkorange'
axes[2, 2].plot(eta_values, induced_drag_coeffs, color=color1, marker='D', linewidth=2, label='LLT $C_{D_i}$')
axes[2, 2].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[2, 2].set_ylabel("Induced Drag Coefficient $C_{D_i}$", color=color1)
axes[2, 2].tick_params(axis='y', labelcolor=color1)
axes[2, 2].set_title("Aerodynamic Penalty (Analytical LLT)")
axes[2, 2].grid(True, linestyle='--')

ax2 = axes[2, 2].twinx()  
color2 = 'forestgreen'
ax2.plot(eta_values, e_factors, color=color2, marker='X', linestyle=':', linewidth=2, label='Oswald Efficiency $e$')
ax2.set_ylabel("Equivalent Span Efficiency $e$", color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# --- ROW 4, PLOT 1: NEW FINESSE PLOT (L/D) ---
axes[3, 0].plot(eta_values, finesses, color='mediumvioletred', marker='*', markersize=10, linewidth=2)
axes[3, 0].set_title("Aerodynamic Finesse ($L/D$) vs $\eta$")
axes[3, 0].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[3, 0].set_ylabel("Finesse ($C_L / C_D$)")
axes[3, 0].grid(True, linestyle='--')

# Hide the unused subplots on the last row so it looks clean
axes[3, 1].axis('off')
axes[3, 2].axis('off')

plt.tight_layout()
plt.savefig("eta_sweep_results_Aeroelastic_Final.png", dpi=300)
print("Plot saved successfully as 'eta_sweep_results_Aeroelastic_Final.png'!")
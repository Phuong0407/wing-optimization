import subprocess
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

# 1. Define the power law function
def power_law(x, a, b):
    return a * (x ** b)

# Helper function to calculate R^2
def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# 2. Define the range of thicknesses to test
thicknesses = np.linspace(4e-3, 1e-2, 10)

# Lists to store the parsed results
max_disp_list = []
max_rot_list = []
max_disp_mag_list = []
valid_thicknesses = []

print("Starting parametric study...")

# 3. Loop through each thickness and execute wingallt.py
for t in thicknesses:
    print(f"Running simulation for thick = {t:.4e} m...")
    
    result = subprocess.run(
        [sys.executable, 'wingallt.py', str(t)], 
        capture_output=True, 
        text=True
    )
    
    output = result.stdout
    
    disp_match = re.search(r"Max displacement\s*:\s*([0-9\.eE+-]+)\s*m", output)
    rot_match  = re.search(r"Max rotation\s*:\s*([0-9\.eE+-]+)\s*rad", output)
    mag_match  = re.search(r"Max displacement magnitude\s*:\s*([0-9\.eE+-]+)\s*m", output)
    
    if disp_match and rot_match and mag_match:
        max_disp_list.append(float(disp_match.group(1)))
        max_rot_list.append(float(rot_match.group(1)))
        max_disp_mag_list.append(float(mag_match.group(1)))
        valid_thicknesses.append(t)
    else:
        print(f"  -> Failed to parse output for t={t}. Check if the solver converged.")

print("\nSimulations complete. Calculating regressions...\n")

# Convert lists to numpy arrays
x = np.array(valid_thicknesses)
y_disp = np.array(max_disp_list)
y_rot = np.array(max_rot_list)
y_mag = np.array(max_disp_mag_list)

# 4. Compute Regressions
# Linear Regressions
reg_lin_disp = linregress(x, y_disp)
reg_lin_rot  = linregress(x, y_rot)
reg_lin_mag  = linregress(x, y_mag)

# Power Law Regressions
# We provide an initial guess (p0) to help the optimizer converge: a=1e-5, b=-3
popt_disp, _ = curve_fit(power_law, x, y_disp, p0=[1e-5, -3])
popt_rot, _  = curve_fit(power_law, x, y_rot, p0=[1e-5, -3])
popt_mag, _  = curve_fit(power_law, x, y_mag, p0=[1e-5, -3])

# Calculate R^2 for Power Law
r2_pow_disp = calc_r2(y_disp, power_law(x, *popt_disp))
r2_pow_rot  = calc_r2(y_rot, power_law(x, *popt_rot))
r2_pow_mag  = calc_r2(y_mag, power_law(x, *popt_mag))

print("--- Power Law Results (y = a * x^b) ---")
print(f"Max Displacement : a = {popt_disp[0]:.4e}, b = {popt_disp[1]:.4f} (R^2 = {r2_pow_disp:.4f})")
print(f"Max Rotation     : a = {popt_rot[0]:.4e},  b = {popt_rot[1]:.4f} (R^2 = {r2_pow_rot:.4f})")
print(f"Max Disp. Mag.   : a = {popt_mag[0]:.4e}, b = {popt_mag[1]:.4f} (R^2 = {r2_pow_mag:.4f})")

# 5. Plot the results
fig, axs = plt.subplots(3, 1, figsize=(8, 14), sharex=True)

def plot_comparisons(ax, x_data, y_data, reg_lin, popt_pow, r2_pow, color, ylabel, title):
    # Smooth X for a cleaner power law curve
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    
    # Raw Data
    ax.plot(x_data, y_data, 'o', color=color, markersize=7, label="Simulation Data")
    
    # Linear Fit
    y_lin = reg_lin.slope * x_smooth + reg_lin.intercept
    ax.plot(x_smooth, y_lin, '--', color='gray', linewidth=1.5, label=f"Linear Fit ($R^2$={reg_lin.rvalue**2:.2f})")
    
    # Power Law Fit
    y_pow = power_law(x_smooth, *popt_pow)
    ax.plot(x_smooth, y_pow, '-', color='black', linewidth=2, label=f"Power Law ($R^2$={r2_pow:.4f})")
    
    # Text box for Power Law equation
    text_str = f"Power Law: $y = {popt_pow[0]:.2e} \cdot t^{{{popt_pow[1]:.2f}}}$"
    ax.text(0.5, 0.85, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower left')

# Generate Subplots
plot_comparisons(axs[0], x, y_disp, reg_lin_disp, popt_disp, r2_pow_disp, 'blue', 'Max Displacement (m)', 'Max Displacement')
plot_comparisons(axs[1], x, y_rot, reg_lin_rot, popt_rot, r2_pow_rot, 'red', 'Max Rotation (rad)', 'Max Rotation')
plot_comparisons(axs[2], x, y_mag, reg_lin_mag, popt_mag, r2_pow_mag, 'green', 'Max Disp. Magnitude (m)', 'Max Displacement Magnitude')

axs[2].set_xlabel('Thickness (m)')

plt.tight_layout()
plt.savefig("thickness_study_with_power_law.png", dpi=300)
plt.show()
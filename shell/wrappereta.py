import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re

# Name of your main FEniCS script
MAIN_SCRIPT = "wingfiber.py" # <-- Change this if your file has a different name

eta_values = np.linspace(0.0, 0.9, 20)
max_displacements = []
max_rotations = []


print("=== Starting Eta Sweep ===")

for eta in eta_values:
    print(f"Running simulation for eta = {eta:.2f}...")
    
    # Run the main script and pass 'eta' as an argument
    # equivalent to typing: python3 wingfiber.py 0.6
    result = subprocess.run(
        ["python3", MAIN_SCRIPT, str(eta)], 
        capture_output=True, 
        text=True
    )
    
    # Check for crashes
    if result.returncode != 0:
        print(f"Error running eta={eta}. Check your main script!")
        print(result.stderr)
        break

    # Search the terminal output for the printed numbers using Regex
    disp_match = re.search(r"Max displacement.*:\s*([0-9\.eE\+\-]+)", result.stdout)
    rot_match = re.search(r"Max rotation.*:\s*([0-9\.eE\+\-]+)", result.stdout)
    
    if disp_match and rot_match:
        disp_val = float(disp_match.group(1))
        rot_val = float(rot_match.group(1))
        
        max_displacements.append(disp_val)
        max_rotations.append(rot_val)
        print(f"  -> Extracted: Disp = {disp_val:.4e} m | Rot = {rot_val:.4e} rad")
    else:
        print(f"  -> Could not parse output for eta={eta}. Did the print statements change?")
        print("Raw output:", result.stdout)

print("\n=== Sweep Complete! Generating Plots ===")

# =============================================================================
# PLOTTING RESULTS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Displacement vs Eta
axes[0].plot(eta_values, max_displacements, 'b-o', linewidth=2)
axes[0].set_title("Max Displacement vs $\eta$")
axes[0].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[0].set_ylabel("Max Displacement (m)")
axes[0].grid(True, linestyle='--', alpha=0.7)

# Plot 2: Rotation vs Eta
axes[1].plot(eta_values, max_rotations, 'r-o', linewidth=2)
axes[1].set_title("Max Rotation vs $\eta$")
axes[1].set_xlabel("Fiber Fraction $\eta$ ($0^\circ$)")
axes[1].set_ylabel("Max Rotation (rad)")
axes[1].grid(True, linestyle='--', alpha=0.7)

# Plot 3: Pareto Curve (Displacement vs Rotation)
axes[2].plot(max_rotations, max_displacements, 'g-s', linewidth=2)
for i, eta in enumerate(eta_values):
    axes[2].annotate(f"$\eta$={eta:.1f}", (max_rotations[i], max_displacements[i]), 
                     textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
axes[2].set_title("Trade-off: Displacement vs Rotation")
axes[2].set_xlabel("Max Rotation (rad)")
axes[2].set_ylabel("Max Displacement (m)")
axes[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("eta_sweep_results.png", dpi=300)
print("Plot saved successfully as 'eta_sweep_results.png'!")
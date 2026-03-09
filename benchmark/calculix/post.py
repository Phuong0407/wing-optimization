import numpy as np

displacements = []
reading = False

with open("wing_calculix.dat", "r") as f:
    for line in f:
        # Tim dong header cua displacement block
        if "displacements (vx,vy,vz)" in line:
            reading = True
            continue
        if reading:
            parts = line.split()
            if len(parts) == 4:
                try:
                    node_id = int(parts[0])
                    ux = float(parts[1])
                    uy = float(parts[2])
                    uz = float(parts[3])
                    displacements.append([ux, uy, uz])
                except ValueError:
                    reading = False

displacements = np.array(displacements)
mag = np.linalg.norm(displacements, axis=1)

print(f"Nodes read               : {len(displacements)}")
print(f"Max displacement (mag)   : {mag.max():.6e} m")
print(f"Max Ux                   : {np.abs(displacements[:,0]).max():.6e} m")
print(f"Max Uy                   : {np.abs(displacements[:,1]).max():.6e} m")
print(f"Max Uz                   : {np.abs(displacements[:,2]).max():.6e} m")
print(f"Node with max disp       : {np.argmax(mag) + 1}")

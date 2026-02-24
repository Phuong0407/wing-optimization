"""
export_to_calculix.py
=====================
Exports the FEM mesh (from FEMMesh.xdmf) and the mapped aerodynamic
traction (from MappedTraction.xdmf) into CalculiX .inp format blocks.

Run this script BEFORE running CalculiX.

Usage:
    python export_to_calculix.py

Outputs:
    wing_ccx_nodes.inp     -- *NODE block
    wing_ccx_elements.inp  -- *ELEMENT block (S3 triangles)
    wing_ccx_bc.inp        -- *NSET + *BOUNDARY block (clamped root)
    wing_ccx_load.inp      -- *CLOAD block (nodal forces from traction)
    wing_benchmark_full.inp -- complete assembled CalculiX input file

Then run:
    ccx wing_benchmark_full
"""

import meshio
import numpy as np
import os

# ---------------------------------------------------------------
# Paths  (adjust relative paths as needed)
# ---------------------------------------------------------------
FEM_MESH_FILE     = "../data/CAD/FEMMesh.xdmf"
TRACTION_FILE     = "../data/FORCE/MappedTraction.xdmf"
OUTPUT_DIR        = "."

ROOT_TAG          = 11        # gmsh physical tag for the clamped root
MATERIAL_NAME     = "STEEL"
THICKNESS         = 6e-3      # m  (must match wing.py)
E_MODULUS         = 210e9     # Pa
POISSON_RATIO     = 0.3

# ---------------------------------------------------------------
# Load meshes
# ---------------------------------------------------------------
print("Reading FEM mesh ...")
fem_mesh = meshio.read(FEM_MESH_FILE)
pts = fem_mesh.points                          # (N, 3) metres
tri = fem_mesh.cells_dict["triangle"]          # (M, 3) zero-indexed

print(f"  Nodes    : {pts.shape[0]}")
print(f"  Elements : {tri.shape[0]}")

print("Reading mapped traction ...")
trac_mesh = meshio.read(TRACTION_FILE)
traction  = trac_mesh.point_data["traction"]   # (N, 3) Pa
assert traction.shape[0] == pts.shape[0], \
    "Traction and mesh node counts do not match! Re-run TractionMapping.py"

# ---------------------------------------------------------------
# Identify root nodes (physical tag == ROOT_TAG)
# ---------------------------------------------------------------
phys = fem_mesh.cell_data_dict["gmsh:physical"]["triangle"]
root_elems    = np.where(phys == ROOT_TAG)[0]
root_node_ids = np.unique(tri[root_elems].flatten())   # zero-indexed
print(f"  Root nodes (tag {ROOT_TAG}): {len(root_node_ids)}")

# ---------------------------------------------------------------
# Compute nodal areas → nodal forces  (traction [Pa] × area [m²] = force [N])
# ---------------------------------------------------------------
def compute_nodal_areas(pts, tri):
    """Distribute element area equally to the 3 corner nodes."""
    nodal_area = np.zeros(len(pts))
    for t in tri:
        v0, v1, v2 = pts[t[0]], pts[t[1]], pts[t[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        nodal_area[t] += area / 3.0
    return nodal_area

print("Computing nodal areas ...")
nodal_area  = compute_nodal_areas(pts, tri)
nodal_force = traction * nodal_area[:, np.newaxis]   # (N, 3) Newtons
total_force = nodal_force.sum(axis=0)
print(f"  Total applied force: Fx={total_force[0]:.3e} N, "
      f"Fy={total_force[1]:.3e} N, Fz={total_force[2]:.3e} N")

# ---------------------------------------------------------------
# Write *NODE block
# ---------------------------------------------------------------
node_file = os.path.join(OUTPUT_DIR, "wing_ccx_nodes.inp")
with open(node_file, "w") as f:
    f.write("*NODE\n")
    for i, (x, y, z) in enumerate(pts):
        f.write(f"{i+1}, {x:.8e}, {y:.8e}, {z:.8e}\n")
print(f"Written: {node_file}")

# ---------------------------------------------------------------
# Write *ELEMENT block  (S3 = 3-node linear triangle)
# ---------------------------------------------------------------
elem_file = os.path.join(OUTPUT_DIR, "wing_ccx_elements.inp")
with open(elem_file, "w") as f:
    f.write("*ELEMENT, TYPE=S3, ELSET=WING_SURFACE\n")
    for i, (n1, n2, n3) in enumerate(tri):
        f.write(f"{i+1}, {n1+1}, {n2+1}, {n3+1}\n")
print(f"Written: {elem_file}")

# ---------------------------------------------------------------
# Write *NSET + *BOUNDARY block
# ---------------------------------------------------------------
bc_file = os.path.join(OUTPUT_DIR, "wing_ccx_bc.inp")
ccx_ids = root_node_ids + 1   # CalculiX uses 1-based indexing
with open(bc_file, "w") as f:
    f.write("*NSET, NSET=ROOT_NODES\n")
    for j in range(0, len(ccx_ids), 8):
        f.write(", ".join(map(str, ccx_ids[j:j+8])) + "\n")
    f.write("*BOUNDARY\n")
    f.write("ROOT_NODES, 1, 6, 0.0\n")   # clamp all 6 DOF
print(f"Written: {bc_file}")

# ---------------------------------------------------------------
# Write *CLOAD block
# ---------------------------------------------------------------
load_file = os.path.join(OUTPUT_DIR, "wing_ccx_load.inp")
with open(load_file, "w") as f:
    f.write("*CLOAD\n")
    for i, (fx, fy, fz) in enumerate(nodal_force):
        nid = i + 1
        if abs(fx) > 1e-14:
            f.write(f"{nid}, 1, {fx:.8e}\n")
        if abs(fy) > 1e-14:
            f.write(f"{nid}, 2, {fy:.8e}\n")
        if abs(fz) > 1e-14:
            f.write(f"{nid}, 3, {fz:.8e}\n")
print(f"Written: {load_file}")

# ---------------------------------------------------------------
# Assemble the full .inp file
# ---------------------------------------------------------------
def read_block(fname):
    with open(fname) as f:
        return f.read()

full_inp = f"""\
**
** ============================================================
** CalculiX Full Benchmark File  (auto-generated)
** Wing shell — linear static analysis
** ============================================================
**
*MATERIAL, NAME={MATERIAL_NAME}
*ELASTIC
{E_MODULUS:.3e}, {POISSON_RATIO}
*DENSITY
7850.0
**
*SHELL SECTION, ELSET=WING_SURFACE, MATERIAL={MATERIAL_NAME}
{THICKNESS:.4e}
**
{read_block(node_file)}
{read_block(elem_file)}
{read_block(bc_file)}
**
*STEP, NLGEOM=NO
*STATIC
**
{read_block(load_file)}
**
*NODE FILE
U, RF
*EL FILE
S, E, SF
*NODE PRINT, NSET=ALL_NODES
U
*EL PRINT, ELSET=WING_SURFACE
S
**
*END STEP
"""

full_file = os.path.join(OUTPUT_DIR, "wing_benchmark_full.inp")
with open(full_file, "w") as f:
    f.write(full_inp)
print(f"\nAssembled complete input: {full_file}")
print("\nRun with:  ccx wing_benchmark_full")
print("View with: cgx -c wing_benchmark_full.frd  (or ParaView)")

# ---------------------------------------------------------------
# Sanity check: print displacement scale estimate
# ---------------------------------------------------------------
max_traction = np.linalg.norm(traction, axis=1).max()
L = np.ptp(pts[:, 0])   # approximate span length
print(f"\nQuick estimate:")
print(f"  Max traction magnitude : {max_traction:.2f} Pa")
print(f"  Approximate span L     : {L:.3f} m")
print(f"  Bending stiffness D    : {E_MODULUS * THICKNESS**3 / (12*(1-POISSON_RATIO**2)):.3e} N·m")
print(f"  Rough tip deflection   : ~{max_traction * L**4 / (E_MODULUS * THICKNESS**3 / 12) * 1e3:.2f} mm")

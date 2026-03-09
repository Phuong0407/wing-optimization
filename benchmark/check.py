# WING lumped nodal force (same CSV as Cast3M)
import LocalFrame
import MeshImport
from ShellKinematics import shell_strains
from IsotropicShell import IsotropicShell
from dolfinx import fem, io
import basix
import ufl
import numpy as np
from mpi4py import MPI
import dolfinx.fem.petsc
from petsc4py import PETSc
from scipy.spatial import cKDTree
from pathlib import Path

# ── MESH ─────────────────────────────────────────────────────────────────────
MESHFILE = "wing.msh"
DOMAIN, CELL_TAGS, FACET_TAGS = MeshImport.load_mesh(MESHFILE)
GDIM = DOMAIN.geometry.dim
TDIM = DOMAIN.topology.dim
FDIM = TDIM - 1

# Check cell tag values present
import numpy as np
print("Cell tag values:", np.unique(CELL_TAGS.values))
print("Facet tag values:", np.unique(FACET_TAGS.values))

# Area per cell tag
for tag_id in np.unique(CELL_TAGS.values):
    cells = CELL_TAGS.find(tag_id)
    marker = fem.Function(fem.functionspace(DOMAIN, ("DG", 0)))
    marker.x.array[:] = 0.0
    marker.x.array[cells] = 1.0
    area_tag = fem.assemble_scalar(fem.form(marker * ufl.dx))
    print(f"  Tag {tag_id}: area = {area_tag:.6f} m²")


import meshio
import numpy as np

CASTEMMesh  = meshio.read("wing.med", file_format="med")
FENICSXMesh = meshio.read("wing.msh")

CASTEMPoint  = CASTEMMesh.points
FENICSXPoint = FENICSXMesh.points * 1e-3

CASTEMPointSorted  = CASTEMPoint[np.lexsort(CASTEMPoint.T)]
FENICSXPointSorted = FENICSXPoint[np.lexsort(FENICSXPoint.T)]

diff = np.abs(CASTEMPointSorted - FENICSXPointSorted)

print("==================== CHECKING IDENTICAL MESH ====================")
print(f"Cast3M nodes: {len(CASTEMPoint)}")
print(f"Gmsh   nodes: {len(FENICSXPoint)}")
print(f"\nMax diff X : {diff[:,0].max():.2e} m")
print(f"Max diff Y : {diff[:,1].max():.2e} m")
print(f"Max diff Z : {diff[:,2].max():.2e} m")
print(f"Max diff global : {diff.max():.2e} m")

if diff.max() < 1e-10:
  print("\nIdentical node coordinates")
else:
  print("\nNonidentical node coordinates")
  bad = np.where(diff.max(axis=1) > 1e-10)[0]
  print(f"  {len(bad)} Mesh is distored when imported to castem")
  for i in bad[:5]:
    print(f"  Node {i}: Cast3M {CASTEMPointSorted[i]} vs Gmsh {FENICSXPointSorted[i]}")
print("==================== CHECKING IDENTICAL MESH ====================")

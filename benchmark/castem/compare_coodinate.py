import meshio
import numpy as np

# Đọc cả hai mesh
m_castem = meshio.read("wing_castem.med", file_format="med")
m_gmsh   = meshio.read("../../data/CAD/wing.msh")

pts_castem = m_castem.points
pts_gmsh   = m_gmsh.points * 1e-3  # mm -> m

print(f"Cast3M nodes: {len(pts_castem)}")
print(f"Gmsh   nodes: {len(pts_gmsh)}")

# So sánh từng node - cần sort trước vì thứ tự có thể khác nhau
pts_castem_sorted = pts_castem[np.lexsort(pts_castem.T)]
pts_gmsh_sorted   = pts_gmsh[np.lexsort(pts_gmsh.T)]

diff = np.abs(pts_castem_sorted - pts_gmsh_sorted)

print(f"\nMax diff X : {diff[:,0].max():.2e} m")
print(f"Max diff Y : {diff[:,1].max():.2e} m")
print(f"Max diff Z : {diff[:,2].max():.2e} m")
print(f"Max diff global : {diff.max():.2e} m")

if diff.max() < 1e-10:
    print("\n✓ Tọa độ khớp hoàn toàn!")
else:
    print("\n✗ Có sự khác biệt!")
    # In ra các nodes khác nhau
    bad = np.where(diff.max(axis=1) > 1e-10)[0]
    print(f"  {len(bad)} nodes khác nhau")
    for i in bad[:5]:
        print(f"  Node {i}: Cast3M {pts_castem_sorted[i]} vs Gmsh {pts_gmsh_sorted[i]}")
import meshio
import numpy as np

# Đọc file .msh từ Gmsh
msh = meshio.read("wingTRI.msh")

# Lấy triangle cells
tri_cells = msh.get_cells_type("triangle")

# Nếu là mesh 2D nằm trong mặt phẳng 3D (z = 0),
# FEniCS legacy thường cần bỏ tọa độ z

# Tạo meshio mesh chỉ chứa triangle
mesh = meshio.Mesh(
    points=msh.points,
    cells=[("triangle", tri_cells)],
)

print(msh.points.shape)

# Ghi ra XDMF (format ổn định nhất cho FEniCS)
meshio.write(
    "wingTRI.xdmf",
    mesh,
    file_format="xdmf",
    data_format="HDF",
)

# -------------------------
# Đọc vào FEniCS
# -------------------------

from dolfin import *

mesh = Mesh()
with XDMFFile("wingTRI.xdmf") as infile:
    infile.read(mesh)

# Xuất lại để kiểm tra
with XDMFFile("check_mesh.xdmf") as f:
    f.write(mesh)

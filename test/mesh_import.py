import gmsh
import numpy as np

filename = "I_beam"
# I-beam profile
wb = 0.2                # bottom flange width
wt = 0.3                # top flange width
h = 0.5                 # web height
# Arch geometry
theta = np.pi/6         # arch opening half-angle
L = 10.                  # chord length
R = L/2/np.sin(theta)   # arch radius
f = R-L/2/np.tan(theta) # arch rise

gmsh.initialize()
geom = gmsh.model.geo



lcar = 0.1 # characteristic mesh size density (will not be used)
bottom_points = [geom.addPoint(0, -wb/2, -h/2, lcar),
                 geom.addPoint(0, 0, -h/2, lcar),
                 geom.addPoint(0, wb/2, -h/2, lcar)]
top_points = [geom.addPoint(0, -wt/2, h/2, lcar),
              geom.addPoint(0, 0, h/2, lcar),
              geom.addPoint(0, wt/2, h/2, lcar)]
bottom_flange = [geom.addLine(bottom_points[0], bottom_points[1]),
                 geom.addLine(bottom_points[1], bottom_points[2])]
web = [geom.addLine(bottom_points[1], top_points[1])]
top_flange = [geom.addLine(top_points[0], top_points[1]),
              geom.addLine(top_points[1], top_points[2])]
start_section = bottom_flange + web + top_flange

dimTags = [(1, l) for l in start_section]
geom.rotate(dimTags, 0, 0, 0, 0, 1, 0, -theta)

end_bottom_flange = []
end_top_flange = []
end_web = []
surfaces = []
for ini, end in zip([bottom_flange, web, top_flange],
                    [end_bottom_flange, end_web, end_top_flange]):
    for l in ini:
        outDimTags = geom.revolve([(1, l)], L/2, 0, -(R-f), 0, 1, 0, 2*theta,
                                 numElements=[50], heights=[1.0])
        end.append(outDimTags[0][1])
        surfaces.append(outDimTags[1][1])
        geom.synchronize()
end_section = end_bottom_flange + end_web + end_top_flange

for f in bottom_flange + top_flange + end_bottom_flange + end_top_flange:
    geom.mesh.setTransfiniteCurve(f, 6)
for w in web + end_web:
    geom.mesh.setTransfiniteCurve(w, 11)

gmsh.model.addPhysicalGroup(2, surfaces, 1)
gmsh.model.addPhysicalGroup(1, start_section, 1)
gmsh.model.addPhysicalGroup(1, end_section, 2)

gmsh.model.mesh.generate(dim=2)
gmsh.write(filename + ".msh")

gmsh.finalize()


import meshio

# from J. Dokken website
def create_mesh(mesh, cell_type):
    cells = np.vstack([cell.data for cell in mesh.cells if cell.type==cell_type])
    data = np.hstack([mesh.cell_data_dict["gmsh:physical"][key]
                           for key in mesh.cell_data_dict["gmsh:physical"].keys() if key==cell_type])
    mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells},
                       cell_data={"name_to_read": [data]})
    return mesh

mesh = meshio.read(filename + ".msh")

shell_mesh = create_mesh(mesh, "triangle")
line_mesh = create_mesh(mesh, "line")
meshio.write(filename + ".xdmf", shell_mesh)
meshio.write(filename + "_facet_region.xdmf", line_mesh)

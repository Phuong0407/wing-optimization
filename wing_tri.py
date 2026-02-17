from dolfin import *

mesh = Mesh()
with XDMFFile("wingQUAD.xdmf") as infile:
    infile.read(mesh)

plot(mesh)
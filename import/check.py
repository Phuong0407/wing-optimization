import vtk
import numpy as np
import meshio
from vtk.util.numpy_support import vtk_to_numpy

# Reading OpenFOAM data in the form of VTK file
# Provide the data file with extension .vtp
FOAMFILE = "../data/FOAM/wing.vtp"
FOAMReader = vtk.vtkXMLPolyDataReader()
FOAMReader.SetFileName(FOAMFILE)
FOAMReader.Update()
poly = FOAMReader.GetOutput()

# The mesh from OpenFOAM is done using SnappyHexMesh
# so it contains few to no triangles.
# Therefore, a triangulation is needed
triangulation = vtk.vtkTriangleFilter()
triangulation.SetInputData(poly)
triangulation.Update()
poly = triangulation.GetOutput()

# The number of 
# Therefore, the dataset is taken by normal filter to match the number
normal_filter = vtk.vtkPolyDataNormals()
normal_filter.SetInputData(poly)
normal_filter.ComputePointNormalsOn()
normal_filter.ComputeCellNormalsOff()
normal_filter.AutoOrientNormalsOn()
normal_filter.ConsistencyOn()
normal_filter.SplittingOff()
normal_filter.Update()
poly = normal_filter.GetOutput()

points  = vtk_to_numpy(poly.GetPoints().GetData())
p       = vtk_to_numpy(poly.GetPointData().GetArray("p"))
wss     = vtk_to_numpy(poly.GetPointData().GetArray("wallShearStress"))
normals = vtk_to_numpy(poly.GetPointData().GetArray("Normals"))

# Check what is vtkCompositeIndex
A = poly.GetCellData().GetNumberOfArrays()
for i in range(A):
    print(poly.GetCellData().GetArrayName(i))

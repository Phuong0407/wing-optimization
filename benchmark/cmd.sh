#!/bin/bash

gmsh wing.geo -2 -o wing.msh -format msh2

python3 wing.py

python3 Mesh2STL.py
castem25 CASTEMMeshImport.dgibi
python3 CheckIdenticalMesh.py
# if yes, then proceed further
python3 TractionImport2CASTEM.py
castem25 wing.dgibi

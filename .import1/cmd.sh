#!/bin/bash

gmsh wing.geo -2 -o wing.msh -format msh2

python3 WingFEMMeshExporter.py
python3 FOAMDataExtractor.py
python3 TractionMapping.py
python3 validate.py
python3 diag.py

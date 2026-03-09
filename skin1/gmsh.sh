#!/bin/bash

# gmsh wing.geo -2 -o wing.msh -format msh2
# gmsh 2ndwing.geo -0

# python3 Wing2ndConverter.py
# python3 2ndTractionMapping.py

python3 CalculixGenerator.py
ccx wing_calculix
python3 post*

python3 shell.py
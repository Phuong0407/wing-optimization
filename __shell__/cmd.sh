#!/bin/bash

gmsh wing.geo -2 -o wing.msh -format msh2

python3 wing.py

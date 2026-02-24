#!/bin/bash

python3 FOAMDataExtractor.py
python3 WingFEMMeshExporter.py
python3 TractionMapping.py

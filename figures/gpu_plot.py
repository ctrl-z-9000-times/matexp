#!/usr/bin/python3
"""
Make a CSV table of the GPU speed measurements
"""
from pathlib import Path
import argparse
import numpy as np
import csv

measurements = []
for file in sorted(Path("gpu_data").iterdir()):
    method = file.name
    with open(file, 'rt') as f:
        text = f.read()

measurements.sort()

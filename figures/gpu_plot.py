#!/usr/bin/python3
"""
Make a CSV table of the GPU speed measurements
"""
from pathlib import Path
import argparse
import numpy as np
import csv

fieldnames = None
measurements = []
for file in sorted(Path("gpu_data").iterdir()):
    model = file.stem
    with open(file, 'rt') as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        else:
            assert fieldnames == reader.fieldnames
        for row in reader:
            data_tuple = (model,) + tuple(row[f] for f in fieldnames)
            measurements.append(data_tuple)

measurements.sort()

with open("gpu_table.csv", 'wt') as f:
    fieldnames.insert(0, "model")
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    for row in measurements:
        writer.writerow(row)

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
    model = file.stem
    with open(file, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            measurements.append((model, row["batch1000"], row["throughput"]))

measurements.sort()

with open("gpu_table.csv", 'wt') as f:
    writer = csv.DictWriter(f, fieldnames=["model", "batch1000", "throughput"])
    writer.writeheader()
    for (model, batch1000, throughput) in measurements:
        writer.writerow({"model": model, "batch1000": batch1000, "throughput": throughput})

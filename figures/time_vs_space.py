#!/usr/bin/python3

from pathlib import Path
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

for file in Path("complexity_data/").glob("*.csv"):
    model = file.stem
    speed = []
    size = []
    with open(file, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            speed.append(float(row["speed"]))
            size.append(float(row["size"]))
        plt.scatter(size, speed, label=model)
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")
plt.xlabel('Space')
plt.ylabel('Time')
plt.legend()
plt.show()


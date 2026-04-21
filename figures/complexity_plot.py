#!/usr/bin/env python
from matexp import main, LinearInput, LogarithmicInput
from pathlib import Path
import argparse
import csv
import matexp
import matplotlib.pyplot as plt
import numpy as np
import os

plt.figure('Speed vs Accuracy', figsize=(7.5, 7.5))

for file in Path("complexity_data").glob("*.csv"):
    name = file.stem
    data = []
    with open(file, 'rt') as f:
        for row in csv.DictReader(f):
            data.append((float(row["error"]), float(row["speed"])))
    data.sort()
    error, speed = zip(*data)

    plt.semilogx(error, speed, marker='o', label=name)

plt.title('Speed vs Accuracy')
plt.ylabel('Time to Integrate, per Instance per Time Step\nNanoseconds')
plt.xlabel('Error Parameter')
plt.ylim(bottom=0.0)
plt.legend()

plt.savefig("complexity.png", dpi=600, bbox_inches='tight')

if not os.environ['NOSHOW']: plt.show()

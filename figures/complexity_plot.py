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

min_table_size = np.inf

for file in Path("complexity_data").glob("*.csv"):
    name = file.stem
    data = []
    with open(file, 'rt') as f:
        for row in csv.DictReader(f):
            data.append((float(row["error"]), float(row["speed"]), float(row["size"])))
    data.sort()
    error, speed, size = zip(*data)
    plt.semilogx(error, speed, marker='o', label=name)
    
    min_table_size = min(min_table_size, min(size))
print("Smallest table in dataset:", min_table_size, "bytes")

# plt.title('Speed vs Accuracy')
plt.ylabel('Time (ns)')
plt.xlabel('Accuracy')
plt.ylim(bottom=0.0)
plt.legend()

plt.gca().spines[['right', 'top']].set_visible(False) # Hide the top & right borders

plt.savefig("complexity.png", dpi=600, bbox_inches='tight')

if not os.environ.get('NOSHOW', ''): plt.show()

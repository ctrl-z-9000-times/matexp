#!/usr/bin/env python
from matexp import main, LinearInput, LogarithmicInput
from pathlib import Path
import argparse
import csv
import matexp
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import numpy as np
import os

plt.figure('Speed vs Accuracy', figsize=(7.5, 7.5))

min_table_size = np.inf

all_data = []
for file in Path("complexity_data").glob("*.csv"):
    name = file.stem
    data = []
    with open(file, 'rt') as f:
        for row in csv.DictReader(f):
            data.append((float(row["error"]), float(row["speed"]), float(row["size"])))
    data.sort()
    error, speed, size = zip(*data)
    all_data.append([name, error, speed])
    min_table_size = min(min_table_size, min(size))
print("Smallest table in dataset:", min_table_size, "bytes")

order = [
        'AMPA_13state',
        'NMDA_10state',
        'Nav11_6state',
        'Kv11_4state',
        'Kv11_6state',
        'Kv11_11state',
        'Kv11_13state']
all_data = sorted(all_data, key=lambda x: order.index(x[0]))

display_names = {
        'AMPA_13state': 'AMPA',
        'NMDA_10state': 'NMDA',
        'Nav11_6state': 'Na$_{v}$1.1',
        'Kv11_4state': 'K$_{v}$1.1 (4 states)',
        'Kv11_6state': 'K$_{v}$1.1 (6 states)',
        'Kv11_11state': 'K$_{v}$1.1 (11 states)',
        'Kv11_13state': 'K$_{v}$1.1 (13 states)',
}
plt.rcParams.update({'mathtext.default':  'regular' })

linestyle = [
        (0, (.7, 1.6)), 
        (0, (3.6, 1.8)), 
        'solid', 
        (0, (7, 1.8)), 
        (0, (8, 1.6, .1, 1.6)),
        (0, (8, 1.6, .1, 1.6, .1, 1.6)),
        (0, (8, 1.6, .1, 1.6, .1, 1.6, .1, 1.6, .1, 1.6)),
]

for index, (name, error, speed) in enumerate(all_data):
    color = cmc.batlow(index / 8)
    speed = np.array(speed) * 1e-6 # Convert from ns to ms
    plt.semilogx(error, speed,
                linestyle=linestyle[index], linewidth=2, dash_capstyle='round',
                label=display_names[name], color=color)

# plt.title('Speed vs Accuracy')
plt.ylabel('Time to advance (ms)')
plt.xlabel('Accuracy')
plt.gca().set_yscale("log")
plt.gca().xaxis.minorticks_off()
plt.legend(handlelength=4)
plt.gca().spines[['right', 'top']].set_visible(False) # Hide the top & right borders
plt.savefig("complexity.png", dpi=600, bbox_inches='tight')

if not os.environ.get('NOSHOW', ''): plt.show()

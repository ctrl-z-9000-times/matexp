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

cm = 1/2.54 # Unit conversion
fontsize = 8.
plt.rcParams.update({'font.size': fontsize})
plt.figure('Speed vs Accuracy', figsize=(8.5*cm, 12*cm), dpi=300)

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
        'Kv11_4state': 'K$_{v}$1.1 (4 States)',
        'Kv11_6state': 'K$_{v}$1.1 (6 States)',
        'Kv11_11state': 'K$_{v}$1.1 (11 States)',
        'Kv11_13state': 'K$_{v}$1.1 (13 States)',
}
plt.rcParams.update({'mathtext.default':  'regular' })

for index, (name, error, speed) in enumerate(all_data):
    color = cmc.batlow(index / 7)
    speed = np.array(speed) * 1e-6 # Convert from ns to ms
    plt.semilogx(error, speed, linewidth=2, color=color)
    markers = ['o', '^', 's', 'd', 'P', 'X', '*']
    plt.plot([error[0]], [speed[0]], color=color, marker=markers[index],
            label=display_names[name],
            markersize=8, markeredgecolor='black', markeredgewidth=.5)

# plt.title('Speed vs Accuracy')
plt.ylabel('Wall-Clock Time (ms)')
plt.xlabel('Accuracy')
plt.gca().set_yscale("log")
plt.gca().xaxis.minorticks_off()
plt.gca().xaxis.set_ticks([10**-n for n in range(8, 0, -1)])
plt.legend(loc='lower left')
plt.gca().spines[['right', 'top']].set_visible(False) # Hide the top & right borders
plt.savefig("complexity.png", bbox_inches='tight', pad_inches=0.)

if not os.environ.get('NOSHOW', ''): plt.show()

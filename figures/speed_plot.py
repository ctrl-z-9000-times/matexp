#!/usr/bin/python3
"""
Make a bar chart of speed measurements
"""
from pathlib import Path
import os
import argparse
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("TIME_STEP", type=float)
parser.add_argument("CELLS", type=int)
args = parser.parse_args()

# Load the data
speed_data = {}
mechanism_names = set()
for file in sorted(Path("speed_data").iterdir()):
    method = file.name
    with open(file, 'rt') as f:
        text = f.read()
    text = text.split('[[END OF BENCHMARK]]')[1]
    lines = [x.strip() for x in text.split('\n')]
    for x in lines:
        if not x.startswith('state-'):
            continue
        if x.startswith('state-update'):
            continue
        x = x[len('state-'):].split()
        mechanism = x[0]
        elapsed_sec = float(x[2])
        elapsed_ms = elapsed_sec * 1e3
        elapsed_ns = elapsed_sec * 1e9
        simulation_steps = 1 / args.TIME_STEP
        run_speed =  elapsed_ms / args.CELLS / simulation_steps
        speed_data.setdefault(method, {})[mechanism] = run_speed
        mechanism_names.add(mechanism)

# Sort the data
method_names = ["approx", "sparse", "matexp"]
mechanism_names = [
        'AMPA',
        'NMDA',
        'na11a',
        'Kv11_4',
        'Kv11_6',
        'Kv11_11',
        'Kv11_13',]
display_names = {
    'AMPA': 'AMPA',
    'NMDA': 'NMDA',
    'na11a': 'Na$_{v}$1.1',
    'Kv11_4': 'K$_{v}$1.1\n4 States',
    'Kv11_6': 'K$_{v}$1.1\n6 States',
    'Kv11_11': 'K$_{v}$1.1\n11 States',
    'Kv11_13': 'K$_{v}$1.1\n13 States',
}
for method, mech_speed in speed_data.items():
    speed_data[method] = [speed for mech, speed in sorted(mech_speed.items())]
speed_data = sorted(speed_data.items(),
                    key=lambda method_data: method_names.index(method_data[0]))

# 
x = np.arange(len(mechanism_names))
width = 1 / (len(speed_data) + 1)  # the width of the bars
multiplier = 0

cm = 1/2.54 # Unit conversion
fontsize = 8.
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(layout='constrained', figsize=(8.5*cm, 8.5*cm), dpi=300)

for index, (method, mech_speed) in enumerate(speed_data):
    offset = width * multiplier
    if method == 'approx': label = r"AME Method"
    if method == 'sparse': label = r"bE Method"
    if method == 'matexp': label = r"ME Method"
    rects = ax.bar(x + offset, mech_speed, width, label=label, 
        color=cmc.batlow(.1 + index / (len(speed_data))))
    multiplier += 1

# 
ax.set_xticks(x + width * 1.5, [display_names.get(name, name) for name in mechanism_names])
ax.tick_params(axis='x', length=0, labelrotation=50)
ax.set_ylabel('Wall-Clock Time (ms)')
ax.set_yscale("log")
ax.set_ylim(1e-5, 1)
ax.legend(loc=[.60, .82])
plt.gca().spines[['right', 'top']].set_visible(False) # Hide the top & right borders
plt.savefig("speed_plot.png", bbox_inches='tight', pad_inches=0)
if not os.environ.get('NOSHOW', ''): plt.show()

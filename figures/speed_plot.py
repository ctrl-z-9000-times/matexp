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
    'na11a': 'Nav1.1',
    'Kv11_11': 'Kv1.1\n11 states',
    'Kv11_13': 'Kv1.1\n13 states',
    'Kv11_4': 'Kv1.1\n4 states',
    'Kv11_6': 'Kv1.1\n6 states',
}
for method, mech_speed in speed_data.items():
    speed_data[method] = [speed for mech, speed in sorted(mech_speed.items())]
speed_data = sorted(speed_data.items(),
                    key=lambda method_data: method_names.index(method_data[0]))

# 
x = np.arange(len(mechanism_names))
width = 1 / (len(speed_data) + 1)  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(7.5, 7.5))

for index, (method, mech_speed) in enumerate(speed_data):
    offset = width * multiplier
    if method == 'approx': label = r"$\it{ame}$ method"
    if method == 'sparse': label = r"$\it{bE}$ method"
    if method == 'matexp': label = r"$\it{me}$ method"
    rects = ax.bar(x + offset, mech_speed, width, label=label, 
        color=cmc.batlow(index / (len(speed_data)-.5)))
    multiplier += 1

# 
ax.set_xticks(x + width * 1.5, [display_names.get(name, name) for name in mechanism_names])
ax.tick_params(axis='x', length=0)
ax.set_ylabel('Time to advance (ms)')
ax.set_yscale("log")
ax.set_ylim(1e-5, None)
ax.legend()
plt.gca().spines[['right', 'top']].set_visible(False) # Hide the top & right borders
plt.savefig("speed_plot.png", dpi=600, bbox_inches='tight')
if not os.environ.get('NOSHOW', ''): plt.show()

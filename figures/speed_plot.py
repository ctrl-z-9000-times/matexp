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
        elapsed_ns = elapsed_sec * 1e9
        simulation_steps = 1 / args.TIME_STEP
        run_speed =  elapsed_ns / args.CELLS / simulation_steps
        speed_data.setdefault(method, {})[mechanism] = run_speed
        mechanism_names.add(mechanism)

# Sort the data
method_names = ["sparse", "matexp", "approx32", "approx64"]
mechanism_names = sorted(mechanism_names)
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
    rects = ax.bar(x + offset, mech_speed, width, label=method, 
        color=cmc.batlow(index / (len(speed_data)-.5)))
    multiplier += 1

# 
ax.set_xticks(x + width * 1.5, mechanism_names)
ax.tick_params(axis='x', length=0)
ax.set_ylabel('Time to advance (ns)')
ax.set_yscale("log")
# ax.set_ylim(0, 250)
ax.legend()
plt.savefig("speed_plot.png", dpi=600, bbox_inches='tight')
if not os.environ.get('NOSHOW', ''): plt.show()

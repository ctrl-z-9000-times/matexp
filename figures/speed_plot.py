#!/usr/bin/python3
"""
Make a bar chart of speed measurements
"""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("TIME_STEP", type=float)
parser.add_argument("CELLS", type=int)
args = parser.parse_args()

speed_data = []
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
        run_speed = float(x[2]) / args.CELLS / args.TIME_STEP
        speed_data.append((mechanism, method, run_speed))

speed_data.sort()
mechanisms = sorted(set(sample[0] for sample in speed_data))

fig, ax = plt.subplots(layout='constrained')

label_locations = np.arange(len(mechanisms))
bar_width = 0.25
method_names = ["sparse", "matexp", "approx"]
method_colors = ["r", "g", "b"]
for mech, method, run_speed in speed_data:
    x = method_names.index(method)
    offset = mechanisms.index(mech) + bar_width * x
    ax.bar(offset, run_speed, bar_width, color=method_colors[x])
ax.set_title('Real time to advance one simulated time step, per instance')
ax.set_ylabel('Nanoseconds')
ax.set_yscale("log")
ax.set_xticks(label_locations + bar_width, mechanisms)
ax.legend(labels=method_names)
plt.show()

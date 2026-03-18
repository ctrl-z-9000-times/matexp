#!/usr/bin/env python
"""
Plot the data in "ap_traces/"
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

traces = {
    "matexp": {},
    "sparse": {},
    "approx32": {},
    "approx64": {},
    "approx2": {},
}

# Load the AP trace data.
for file in Path('ap_traces').iterdir():
    with open(file, 'rb') as f:
        data = pickle.load(f)
    if file.name.count("_") == 1:
        method, time_step = file.name.split("_")
        traces[method][time_step] = data
    else:
        method, time_step, accuracy = file.name.split("_")
        traces["approx2"][accuracy] = data

# Sort by dt.
for method, data in traces.items():
    traces[method] = dict(sorted(data.items(), key=lambda pair: float(pair[0])))

# Select one of the traces.
for method, data in traces.items():
    for time_step, (t, v_soma, v_proximal, v_distal) in data.items():
        data[time_step] = (t, v_soma)

# Setup the figure.
fig = plt.figure("Menon et al. (2009)")
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
grid = gs.subplots(sharex='all', sharey='all')
titles = [
    "Matrix Exponential Method",
    "Backwards Euler Method",
    "Approximate Matrix Exponential Method vs Time Step",
    "Approximate Matrix Exponential Method vs Accuracy",]
methods = ["matexp", "sparse", "approx64", "approx2"]

t_min, t_max = (0, 100)

for row, col, index in [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)]:
    axes = grid[row, col]
    title = titles[index]
    method = methods[index]
    # 
    axes.set_title(title)
    axes.text(t_min, 60, chr(ord("A") + index), ha='left', va='top')
    for time_step, (t, v) in traces[method].items():
        label = f"max error = {error}" if method == "approx2" else f"Δt = {time_step}"
        axes.plot(t, v, label=label)
    axes.set_xlabel("time (ms)")
    axes.set_xlim(xmin=100, xmax=150)
    axes.set_yticks([])
    axes.legend()

fig.savefig("ap_demo.png", dpi=600, bbox_inches='tight')
plt.show()

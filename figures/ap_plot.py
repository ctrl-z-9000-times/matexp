#!/usr/bin/env python
"""
Plot the data in "ap_traces/"
"""

from pathlib import Path
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pickle

traces = {
    "matexp": {},
    "sparse": {},
    "approx32": {},
    "approx64": {},
    "accuracy": {},
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
        traces["accuracy"][accuracy] = data

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
methods = ["matexp", "sparse", "approx64", "accuracy"]

t_min, t_max = (125, 130)

for row, col, index in [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)]:
    axes = grid[row, col]
    title = titles[index]
    method = methods[index]
    # 
    axes.text(t_min+.4, 15, chr(ord("A") + index), ha='left', va='top')
    num_traces = len(traces[method])
    for trace_index, (value, (t, v)) in enumerate(traces[method].items()):
        label = f"max error = {value}" if index == 3 else f"Δt = {value}"
        axes.plot(t, v, label=label, color=cmc.batlow(trace_index / num_traces))
    axes.set_xlabel("time (ms)")
    axes.set_xlim(xmin=t_min, xmax=t_max)
    # axes.set_yticks([])
    axes.legend()

fig.savefig("ap_demo.png", dpi=600, bbox_inches='tight')
plt.show()

#!/usr/bin/env python
"""
Plot the data in "ap_data/"
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
for file in Path('ap_data').iterdir():
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

# Setup the figure.
fig = plt.figure("AP Demo", figsize=(7.5, 7.5))
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
grid = gs.subplots(sharex='all', sharey='all')
titles = [
    "Matrix Exponential Method",
    "Backwards Euler Method",
    "Approximate Matrix Exponential Method vs Time Step",
    "Approximate Matrix Exponential Method vs Accuracy",]
methods = ["matexp", "sparse", "approx32", "accuracy"]

t_min, t_max = (3, 3.6)

# Offest the time by 100 ms. (stimulus starts after 100 ms delay) 
for method, data in traces.items():
    for value, (t, v) in data.items():
        t -= 100

# Find the exact time of AP peak.
peak_time = {}
for method, data in traces.items():
    peak_time[method] = {}
    for value, (t, v) in data.items():
        idx_start = next(i for i, x in enumerate(t) if x >= t_min)
        idx_stop  = next(i for i, x in enumerate(t) if x >= t_max)
        idx_peak  = idx_start + np.argmax(v[idx_start:idx_stop+1])
        peak_time[method][value] = tp = t[idx_peak]
        print(method, value, tp)

for row, col, index in [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)]:
    axes = grid[row, col]
    title = titles[index]
    method = methods[index]
    # 
    axes.text(t_min+.1, 15, chr(ord("A") + index), ha='left', va='top',
              fontsize="large", weight="bold")
    num_traces = len(traces[method])
    for trace_index, (value, (t, v)) in enumerate(traces[method].items()):
        assert all(np.isfinite(t))
        assert all(np.isfinite(v))
        if index in [0, 1, 2]:
            label = "Δt = %g"%float(value)
        elif index == 3:
            label = f"max error = {value}"
        # axes.plot(t, v, label=label)
        axes.plot(t, v, label=label, color=cmc.batlow(trace_index / num_traces))
    # axes.set_yticks([])
    # axes.grid(which="major", axis='both', linestyle='solid', linewidth=1)
    axes.set_xlabel("time (ms)")
    axes.set_xlim(xmin=t_min, xmax=t_max)
    axes.legend()

fig.savefig("ap_demo.png", dpi=600, bbox_inches='tight')
plt.show()

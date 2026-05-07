#!/usr/bin/env python
"""
Plot the data in "ap_data/"
"""

from pathlib import Path
import os
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pickle

traces = {
    "matexp": {},
    "sparse": {},
    "approx": {},
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
# grid = gs.subplots(sharex='all', sharey='all') # DEBUG ONLY (tick labels collide)
grid = gs.subplots() # RELEASE
titles = [
    "Matrix Exponential Method",
    "Backwards Euler Method",
    "Approximate Matrix Exponential Method vs Time Step",
    "Approximate Matrix Exponential Method vs Accuracy",]
methods = ["matexp", "sparse", "approx", "accuracy"]

t_min, t_max = (3, 4)

# Offset the time by 100 ms. (stimulus starts after 100 ms delay) 
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
    # Sub-figure labels (A, B, C, D)
    axes.text(t_min+.06, 27, chr(ord("A") + index), ha='left', va='top',
              size=16, weight="bold")
    # 
    num_traces = len(traces[method])
    for trace_index, (value, (t, v)) in enumerate(traces[method].items()):
        assert all(np.isfinite(t))
        assert all(np.isfinite(v))
        if index in [0, 1, 2]:
            dt = float(value)
            label = "Δt = %g"%dt
        elif index == 3:
            accuracy = float(value)
            label = f"accuracy = {value}"
        linewidth = 2
        if index == 3 and accuracy <= .001:
            linewidth = 1
        axes.plot(t, v, label=label, color=cmc.batlow(trace_index / 4), linewidth=linewidth)
    # 
    axes.set_xlim(xmin=t_min, xmax=t_max)
    # Y-Axis labels & ticks
    axes.yaxis.set_tick_params(direction="in")
    if col == 0:
        axes.set_ylabel("Membrane potential (mV)")
        axes.set_yticks([-60, -40, -20, 0, 20])
    elif col == 1:
        axes.set_yticks([-60, -40, -20, 0, 20], labels=[""]*5)
    # X-Axis labels & ticks
    axes.xaxis.set_tick_params(direction="in")
    if row == 0:
        axes.set_xticks([3.0, 3.2, 3.4, 3.6, 3.8, 4.0], labels=[""]*6)
    elif row == 1:
        axes.set_xlabel("time (ms)")
        if col == 0:
            axes.set_xticks([3.0, 3.2, 3.4, 3.6, 3.8])
        else:
            axes.set_xticks([3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
    # Legend
    if index == 3:
        axes.legend(loc='center right')
    else:
        axes.legend(loc='lower right')
    # Hide the top & right borders
    if index == 0:
        axes.spines[['top']].set_visible(False)
    elif index == 1:
        axes.spines[['right', 'top']].set_visible(False)
    elif index == 3:
        axes.spines[['right']].set_visible(False)

fig.savefig("ap_demo.png", dpi=600, bbox_inches='tight')
if not os.environ.get('NOSHOW', ''): plt.show()

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

# Offset the time by 100 ms. (stimulus starts after 100 ms delay) 
for method, data in traces.items():
    for value, (t, v) in data.items():
        t -= 100

t_min, t_max = (3, 4)

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

subplots = []
methods = ["matexp", "sparse", "approx", "accuracy"]

fontsize = 8.
plt.rcParams.update({'font.size': fontsize})

for index, method in enumerate(methods):

    # Setup the figure.
    cm = 1/2.54 # Unit conversion
    name = 'ap_demo_%c.jpg'%chr(ord('A') + index)
    fig = plt.figure(name, figsize=(8.5*cm, 5*cm), dpi=300)
    axes = fig.add_axes([0, 0, 1, 1])
    subplots.append(fig)

    num_traces = len(traces[method])
    for trace_index, (value, (t, v)) in enumerate(traces[method].items()):
        assert all(np.isfinite(t))
        assert all(np.isfinite(v))
        if index in [0, 1, 2]:
            dt = float(value)
            label = "Δt = %g"%dt
        elif index == 3:
            accuracy = float(value)
            label = f"Error Target = {value}"
        linestyle = 'dashed' if trace_index == 1 else 'solid'
        linewidth = 2
        if trace_index == 1 and index == 3:
            linestyle = 'dotted'
            linewidth = 4
        axes.plot(t, v, label=label, linestyle=linestyle, linewidth=linewidth,
            color=cmc.batlow(trace_index / (4)))
    # 
    axes.set_xlim(xmin=t_min, xmax=t_max)
    axes.set_ylim(ymin=-70, ymax=35)
    # Y-Axis labels & ticks
    axes.yaxis.set_tick_params(direction="in")
    axes.set_ylabel("Membrane Potential (mV)")
    axes.set_yticks([-60, -40, -20, 0, 20])
    # X-Axis labels & ticks
    axes.xaxis.set_tick_params(direction="in")
    axes.set_xlabel("Time (ms)")
    axes.set_xticks([3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
    # Legend
    if index == 3:
        axes.legend(loc='upper right', handlelength=3)
    else:
        axes.legend(loc='lower right', handlelength=3)
    # Hide the top & right borders
    axes.spines[['right', 'top']].set_visible(False)

    fig.savefig(name, bbox_inches='tight', pad_inches=0.)
if not os.environ.get('NOSHOW', ''): plt.show()

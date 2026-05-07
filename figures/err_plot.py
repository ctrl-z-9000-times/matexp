#!/usr/bin/python3
"""
Compare the final state data of the solver methods in "err_data/"
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cmcrameri.cm as cmc

import sys
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("DATA_DIR", type=Path)
args = parser.parse_args()

assert args.DATA_DIR.is_dir()

final_state = {}

all_seeds = set()

for file in Path(args.DATA_DIR).iterdir():
    method, time_step = file.name.split("_")
    time_step = float(time_step)
    with open(file, 'rb') as f:
        seed, mech_state = pickle.load(f)
    all_seeds.add(seed)
    final_state.setdefault(method, {})
    final_state[method][time_step] = mech_state
assert len(all_seeds) == 1

methods = sorted(final_state.keys(), reverse=True)

time_steps = set()
for method, time_step_data in final_state.items():
    time_steps.update(time_step_data.keys())
time_steps = sorted(time_steps)

# Fill in the exact solution for every time step.
try:
    data = final_state["matexp"][1.0]
except KeyError:
    pass
else:
    for time_step in time_steps:
        if time_step not in final_state["matexp"]:
            final_state["matexp"][time_step] = data

# Sort by dt
for method, dt_data in final_state.items():
    final_state[method] = dict(sorted(dt_data.items()))

# Calculate error for each mechanism in each scenario
traces = {}
mechanisms = set()
for method in methods:
    if method == "matexp":
        continue
    traces[method] = {}
    for dt, mech_state in final_state[method].items():
        if dt not in final_state["matexp"]:
            raise ValueError(f"missing matexp data for dt={dt}")
        for mech, state in mech_state.items():
            exact = final_state["matexp"][dt][mech]
            max_err = np.max(np.abs(exact - state))
            traces[method].setdefault(mech, ([], []))
            traces[method][mech][0].append(dt)
            traces[method][mech][1].append(max_err)

# Find the min/max error for each method at each time step.
error_bounds = {}
for method, mech_data in traces.items():
    error_bounds[method] = {}
    for mech, (dt_data, err_data) in mech_data.items():
        for dt, err in zip(dt_data, err_data):
            if dt not in error_bounds[method]:
                error_bounds[method][dt] = [err, err]
            else:
                error_bounds[method][dt][0] = min(err, error_bounds[method][dt][0])
                error_bounds[method][dt][1] = max(err, error_bounds[method][dt][1])

# Sanity check
for dt, (min_err, max_err) in error_bounds["approx"].items():
    assert np.all(max_err <= 0.001)

# Setup the figure
plt.figure("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Δt (ms)")
plt.xlim(min(time_steps), max(time_steps))
plt.ylim(1e-4, .3)

# Draw max error for each method
for method, method_data in error_bounds.items():
    dt_data = []
    max_err_data = []
    for dt, (min_err, max_err) in sorted(method_data.items()):
        dt_data.append(dt)
        max_err_data.append(max_err)
    color = cmc.batlow(.0 if method == 'sparse' else .5)
    plt.loglog(dt_data, max_err_data, label=f'{method} maximum accuracy',
               linewidth=3, color=color, zorder=100)

# Draw every accuracy trace (very lightly)
for method, mech_data in traces.items():
    color = cmc.batlow(.25 if method == 'sparse' else .75)
    label = f'{method} accuracy'
    for mech, (dt, err) in mech_data.items():
        marker = '*' if len(dt) == 1 else None
        plt.loglog(dt, err, marker=marker, color=color, label=label, linewidth=.5)
        label = None

plt.legend()
plt.gca().spines[['right', 'top']].set_visible(False) # Hide the top & right borders
plt.savefig(args.DATA_DIR.name + ".png", dpi=600, bbox_inches='tight')
if not os.environ.get('NOSHOW', ''): plt.show()

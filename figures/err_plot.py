#!/usr/bin/python3
"""
Compare the final state data of the solver methods in "err_data/"
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("DATA_DIR", type=Path)
parser.add_argument("MECHANISMS", nargs="*", type=str)
args = parser.parse_args()

assert args.DATA_DIR.is_dir()

final_state = {}

all_seeds = set()

for file in Path(args.DATA_DIR).iterdir():
    method, time_step = file.name.split("_")
    time_step = float(time_step) * 1000 # Convert to microseconds
    with open(file, 'rb') as f:
        seed, mech_state = pickle.load(f)
    all_seeds.add(seed)
    final_state.setdefault(method, {})
    final_state[method][time_step] = mech_state
assert len(all_seeds) == 1

methods = list(final_state.keys())

time_steps = set()
for method, time_step_data in final_state.items():
    time_steps.update(time_step_data.keys())

# Fill in the exact solution for every time step.
try:
    data = final_state["matexp"][1000.0]
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

plt.figure("Accuracy Comparison")
for method, mech_data in traces.items():
    for mech, (dt, err) in mech_data.items():
        if args.MECHANISMS and mech not in args.MECHANISMS:
            continue
        plt.loglog(dt, err, label=f"{method}: {mech}")

plt.ylabel("error")
plt.xlabel("Δt (μs)")
plt.legend()

plt.savefig(args.DATA_DIR.name + ".png", dpi=600, bbox_inches='tight')
plt.show()

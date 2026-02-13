#!/usr/bin/python3
"""
Compare the final state data of the solver methods in "err_data/"
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

all_seeds = set()

final_state = {
    "matexp": {},
    "sparse": {},
    "approx": {},
}

for file in Path('err_data').iterdir():
    method, time_step = file.name.split("_")
    time_step = float(time_step)
    with open(file, 'rb') as f:
        seed, mech_state = pickle.load(f)
    all_seeds.add(seed)
    final_state[method][time_step] = mech_state
assert len(all_seeds) == 1

# Sort by dt
for method, dt_data in final_state.items():
    final_state[method] = dict(sorted(dt_data.items()))

# Calculate error for each mechanism in each scenario
traces = {}
mechanisms = set()
for method in ["sparse", "approx"]:
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

plt.figure("Accuracy Measurement")
plt.title("Accuracy vs. Time Step")
for method, mech_data in traces.items():
    for mech, (dt, err) in mech_data.items():
        plt.loglog(dt, err, label=f"{method}: {mech}")

plt.ylabel("error")
plt.xlabel("Δt (μs)")
plt.legend()

plt.show()

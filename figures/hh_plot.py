#!/usr/bin/python3
"""
Plot the data in "hh_traces/"
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

traces = {
    "matexp": {},
    "sparse": {},
    "approx": {},
    "approx2": {},
}

for file in Path('hh_traces').iterdir():
    with open(file, 'rb') as f:
        t, v = pickle.load(f)
    if file.name.count("_") == 1:
        method, time_step = file.name.split("_")
        traces[method][time_step] = (t, v)
    else:
        method, time_step, accuracy = file.name.split("_")
        traces["approx2"][accuracy] = (t, v)
# Sort by dt
for method, data in traces.items():
    traces[method] = dict(sorted(data.items(), key=lambda pair: float(pair[0])))

plt.figure("Hodgkin Huxley Demonstration")

def setup_plot():
    plt.ylabel("potential (mV)")
    plt.legend()

plt.subplot(4, 1, 1)
plt.title("Matrix Exponential Method")
for time_step, (t, v) in traces["matexp"].items():
    plt.plot(t, v, label=f"Δt = {time_step}")
setup_plot()

plt.subplot(4, 1, 2)
plt.title("Backwards Euler Method")
for time_step, (t, v) in traces["sparse"].items():
    plt.plot(t, v, label=f"Δt = {time_step}")
setup_plot()

plt.subplot(4, 1, 3)
plt.title("Approximate Matrix Exponential Method vs Time Step")
for time_step, (t, v) in traces["approx"].items():
    plt.plot(t, v, label=f"Δt = {time_step}")
setup_plot()

plt.subplot(4, 1, 4)
plt.title("Approximate Matrix Exponential Method vs Accuracy")
for error, (t, v) in traces["approx2"].items():
    plt.plot(t, v, label=f"max error = {error}")
setup_plot()

plt.xlabel("time (ms)")

plt.show()

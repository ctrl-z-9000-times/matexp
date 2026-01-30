#!/usr/bin/python
"""
Simulate a standard HH model with different solver methods and time-steps.
"""
from pathlib import Path
import argparse
import numpy as np
import os
import pickle
import utils

args = argparse.ArgumentParser()
args.add_argument("METHOD", type=str, choices=["matexp", "approx", "sparse"])
args.add_argument("TIME_STEP", type=float)
args.add_argument("ACCURACY", type=float, nargs='?', default=None)
args.add_argument("-v", "--verbose", action="store_true", help="plot voltage trace immediately")
args = args.parse_args()

neuron = utils.load(["hh_markov.mod"], args.METHOD, zero_conductance=False, dt=args.TIME_STEP, c=6.3, error=args.ACCURACY)
from neuron import n
from neuron.units import ms, mV

soma = n.Section()
soma.nseg = 1
soma.insert("hh_markov")
ic = n.IClamp(soma(0.5))
ic.delay = 0
ic.dur   = 50 * ms
ic.amp   = 10
v_trace  = n.Vector().record(soma(0.5)._ref_v)
t_trace  = n.Vector().record(n._ref_t)

n.dt = args.TIME_STEP
n.secondorder = 1
n.finitialize(-65 * mV)
n.continuerun(50 * ms)
assert n.dt == args.TIME_STEP

v = np.array(v_trace.as_numpy())
t = np.array(t_trace.as_numpy())

# Dump the trace to a file
data_dir = Path("hh_traces")
data_file = f"{args.METHOD}_{1000 * args.TIME_STEP}"
if args.ACCURACY:
    data_file += f"_{args.ACCURACY}"
os.makedirs(data_dir, exist_ok=True)
with open(data_dir.joinpath(data_file), 'wb') as f:
    pickle.dump((t, v), f)

if args.verbose:
    import matplotlib.pyplot as plt
    plt.plot(t, v)
    plt.show()

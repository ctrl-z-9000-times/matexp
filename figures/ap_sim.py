#!/usr/bin/env python
from pathlib import Path
import argparse
import os
import pickle
import requests
import subprocess
import time
import utils
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("METHOD", type=str, choices=["matexp", "approx32", "approx64", "sparse"])
parser.add_argument("TIME_STEP", type=float)
parser.add_argument("ACCURACY", type=float, nargs='?', default=None)
parser.add_argument("-v", "--verbose", action='store_true')
args = parser.parse_args()

# Gather up the MOD files.
mod_dir = Path(__file__).parent.parent / "mod"
mod_files = [
    mod_dir / "Nav11_6state.mod",
    mod_dir / "Nav14_6state.mod",
    mod_dir / "Nav16_6state.mod",
    mod_dir / "Kv11_6state.mod",
    mod_dir / "Kv12_6state.mod",
    mod_dir / "Kv16_6state.mod",
    mod_dir / "Kv21_11state.mod",
    mod_dir / "Kv22_6state.mod",
]
out_dir = utils.copy_mod_files(mod_files)
utils.set_solver(out_dir, args.METHOD)

# Run the matexp program if necessary
if args.METHOD.startswith("approx"):
    cmd = ["matexp", "-v", "-v", "--time_step", str(args.TIME_STEP), "--temperature", "37"]
    if args.ACCURACY:
        cmd.extend(["-e", str(args.ACCURACY)])
    if args.METHOD.endswith("32"):
        cmd.extend(["-f", "32"])
    cmd.extend(["--input", "v", "-70", "40"])
    for in_path in mod_files:
        print("RUN CMD: ", ' '.join(str(x) for x in cmd))
        subprocess.run(cmd + [in_path, out_dir], check=True)

# from neuron import n
neuron = utils.build_models(out_dir, True)
n = neuron.n

# Setup the NEURON simulation
n.dt = args.TIME_STEP
n.celsius = 37

# Setup the test model
soma = n.Section()
soma.L = soma.diam = 12.6157
soma.nseg = 1
soma.Ra = 100
soma.cm = 1
# Insert mechanisms
soma.insert("pas")
soma.insert("na11a")
soma.insert("na14a")
soma.insert("na16a")
soma.insert("Kv11_6")
soma.insert("Kv12_6")
soma.insert("Kv16_6")
soma.insert("Kv21_11")
soma.insert("Kv22_6")

for seg in soma:
    seg.pas.g *= 1
    seg.na11a.gbar *= 1
    seg.na14a.gbar *= 1
    seg.na16a.gbar *= 5
    seg.Kv11_6.gbar *= .1
    seg.Kv12_6.gbar *= .1
    seg.Kv16_6.gbar *= .1
    seg.Kv21_11.gbar *= .1
    seg.Kv22_6.gbar *= .1

ic = n.IClamp(soma(0.5))
ic.amp = 0.01
ic.dur = 800
ic.delay = 100

# Measure voltage waveform.
time_trace  = n.Vector().record(n._ref_t)
soma_trace  = n.Vector().record(soma(0.5)._ref_v)
kv21_trace  = n.Vector().record(soma(0.5).Kv21_11._ref_g)

# Setup simulation parameters.
n.dt = args.TIME_STEP
n.celsius = 37
n.secondorder = 1
n.finitialize()

t_stop = 1000 # Do a full run.
if True: t_stop = 115 # Only run the first AP.

while n.t < t_stop:
    n.fadvance()

traces = (time_trace.as_numpy(),
         soma_trace.as_numpy())

# Dump the trace to a file
data_dir = Path(__file__).parent / "ap_data"
data_file = f"{args.METHOD}_{args.TIME_STEP}"
if args.ACCURACY:
    data_file += f"_{args.ACCURACY}"
os.makedirs(data_dir, exist_ok=True)
with open(data_dir.joinpath(data_file), 'wb') as f:
    pickle.dump(traces, f)

if args.verbose:
    import matplotlib.pyplot as plt
    plt.plot(traces[0], traces[1], label='voltage')
    plt.plot(traces[0], kv21_trace, label='kv21.g')
    plt.legend()
    plt.show()

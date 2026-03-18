#!/usr/bin/env python
from pathlib import Path
import argparse
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
args = parser.parse_args()

# Go to the directory containing this file.
repo = Path(__file__).parent
import os
os.chdir(repo)

# Download the modeldb project.
MenonEtAl2009 = repo / 'MenonEtAl2009'
if not MenonEtAl2009.exists():
    ascension_number = 222716
    model_url = f"https://modeldb.science/download/{ascension_number}"
    print('Downloading:', model_url)
    myfile = requests.get(model_url)
    zip_filename = Path(f'{ascension_number}.zip')
    with open(zip_filename, 'wb') as f:
        f.write(myfile.content)
    #
    print("Unzipping:", zip_filename)
    zipfile.ZipFile(zip_filename).extractall()
    zip_filename.unlink()

os.chdir(MenonEtAl2009)

mod_files = [
    "ih.mod",
    "kadist.mod",
    "kaprox.mod",
    "kdr.mod",
    "leak.mod",
    "nafast.mod",
    "naslow.mod",
    "spines.mod",
    "synampa.mod",
    "vms.mod"]

out_dir = utils.copy_mod_files(mod_files)
utils.set_solver(out_dir, args.METHOD)

# Run the matexp program if necessary
if args.METHOD.startswith("approx"):
    cmd = ["matexp", "-v", "-v", "-t", str(args.TIME_STEP), "-c", "35"]
    if args.ACCURACY:
        cmd.extend(["-e", str(args.ACCURACY)])
    if args.METHOD.endswith("32"):
        cmd.extend(["-f", "32"])
    cmd.extend(["--input", "v", "-70", "30"])
    for in_path in ["nafast.mod", "naslow.mod"]:
        subprocess.run(cmd + [in_path, out_dir], check=True)

neuron = utils.build_models(out_dir, True)
n = neuron.n
n.xopen("ri06_runAPtrain.hoc")

# Add a current clamp on the soma, BC original experiment's stimulus has a 10 second delay.
ic = n.IClamp(n.somaA(0.5))
ic.amp = 0.13
ic.dur = 800
ic.delay = 100

# Measure voltage waveform.
time_trace  = n.Vector().record(n._ref_t)
soma_trace  = n.Vector().record(n.somaA(0.5)._ref_v)
proximal_trace = n.Vector().record(n.dendA5_011111111111(0.5)._ref_v)
distal_trace   = n.Vector().record(n.dendA5_011111111111111(0.5)._ref_v)

# Setup simulation parameters.
n.cvode.active(0)
n.dt = args.TIME_STEP
n.secondorder = 1
n.finitialize()

while n.t < 1000:
    n.fadvance()

traces = (time_trace.as_numpy(),
    soma_trace.as_numpy(),
    proximal_trace.as_numpy(),
    distal_trace.as_numpy())

# Dump the trace to a file
data_dir = repo / "ap_traces"
data_file = f"{args.METHOD}_{1000 * args.TIME_STEP}"
if args.ACCURACY:
    data_file += f"_{args.ACCURACY}"
os.makedirs(data_dir, exist_ok=True)
with open(data_dir.joinpath(data_file), 'wb') as f:
    pickle.dump(traces, f)

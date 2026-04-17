#!/usr/bin/python3
"""
Simulate models in isolation with random initial state and constant random inputs.

Arguments control the integration method and time step. The program err_plot.py
compares results across methods and time steps to asses their accuracy.
"""
from pathlib import Path
from pprint import pprint
import argparse
import numpy as np
import os
import pickle
import sys
import utils

parser = argparse.ArgumentParser()
parser.add_argument("OUT_FILE", type=Path)
parser.add_argument("SEED", type=int, default=None)
parser.add_argument("METHOD", type=str, choices=["matexp", "approx32", "approx64", "sparse", "pwd"])
parser.add_argument("TIME_STEP", type=float)
parser.add_argument("CELLS", type=int)
parser.add_argument("--error", type=float, default=1e-3)
parser.add_argument("MODELS", type=str, choices=["all", "dedup"])
args = parser.parse_args()

# Compile and load the MOD files into NEURON
mod_dir = Path(__file__).parent.parent.joinpath("mod")
if args.MODELS == "all":
    mod_files = utils.all_mod_files()
elif args.MODELS == "dedup":
    mod_files = utils.dedup_mod_files()
if args.METHOD == "pwd":
    import neuron
else:
    neuron = utils.load(mod_files, args.METHOD, dt=args.TIME_STEP, c=37, error=args.error)
mechanisms = utils.mechanism_names(mod_files) # This function depends on neuron being loaded first
n = neuron.n
from neuron.units import ms, mV, µm
from neuron import hoc
print("Loaded mechanisms:")
pprint(mechanisms)
del mechanisms['presyn']

import matexp
v_inp = matexp.LinearInput('v', -100, 100)
c_inp = matexp.LogarithmicInput('C', 0, 10)
all_inputs = [v_inp, c_inp]

# Generate psuedorandom initial states
inital_state = {}
rng = np.random.default_rng(args.SEED)
for mech, states in sorted(mechanisms.items()):
    data = rng.uniform(0, 1, [args.CELLS, len(states)])
    data /= data.sum(axis=1, keepdims=True)
    inital_state[mech] = data

# Generate psuedorandom inputs
initial_inputs = {}
for input_spec in sorted(all_inputs, key=lambda inp: inp.name):
    input_spec.scale = 1.
    input_spec.set_num_buckets(1)
    bucket_values = rng.uniform(0, input_spec.num_buckets, args.CELLS)
    input_values = input_spec.get_input_value(bucket_values)
    initial_inputs[input_spec.name] = input_values

# Setup the NEURON simulation
n.dt = args.TIME_STEP
n.celsius = 37

# Setup the test model
cells = []
presyn = []
for cell_idx in range(args.CELLS):
    soma = n.Section()
    soma.L = soma.diam = 12.6157 * µm
    soma.nseg = 1
    soma.Ra = 100
    soma.cm = 1
    presyn_inst = n.presyn(soma(.5))
    presyn.append(presyn_inst)
    cells.append(soma)
# Insert mechanisms
point_processes = {}
pproc_instances = {}
for m_name, states in mechanisms.items():
    m_class = getattr(n, m_name)
    pproc = isinstance(m_class, hoc.HocObject) # Determine if point process or distributed mechanism
    if pproc:
        point_processes[m_name] = states
        pproc_instances[m_name] = []
    for cell_idx, soma in enumerate(cells):
        if not pproc:
            soma.insert(m_class)
        else:
            m_inst = m_class(soma(.5))
            # Connect receptors to presyn model for neurotransmitter concentration
            n.setpointer(presyn[cell_idx]._ref_C, "C", m_inst)
            pproc_instances[m_name].append(m_inst)
# 
n.finitialize()
# Assign initial mechanism states
for m_name, states in mechanisms.items():
    m_init = inital_state[m_name]
    for cell_idx, soma in enumerate(cells):
        for seg in soma:
            if m_name in point_processes:
                m_inst = pproc_instances[m_name][cell_idx]
            else:
                m_inst = getattr(seg, m_name)
            for s_idx, s_name in enumerate(states):
                value = float(m_init[cell_idx, s_idx])
                setattr(m_inst, s_name, value)
            # Zero conductances
            for g in ["gbar", "gmax", "gnabar", "gkbar"]:
                if hasattr(m_inst, g):
                    setattr(m_inst, g, 0.)
            # Assign initial inputs
            setattr(seg, 'v', initial_inputs['v'][cell_idx])
for cell_idx, presyn_inst in enumerate(presyn):
    setattr(presyn_inst, 'C', initial_inputs['C'][cell_idx])

# Run the experiment
for step in range(round(1 / args.TIME_STEP)):
    n.fadvance()

# Collect the final state of all model instances
final_state = {}
for m_name, states in mechanisms.items():
    m_class = getattr(n, m_name)
    pproc = isinstance(m_class, hoc.HocObject)
    data = np.zeros([args.CELLS, len(states)])
    for cell_idx, soma in enumerate(cells):
        if pproc:
            m_inst = pproc_instances[m_name][cell_idx]
        else:
            for seg in soma:
                m_inst = getattr(seg, m_name)
        for s_idx, s_name in enumerate(states):
            value = getattr(m_inst, s_name)
            data[cell_idx, s_idx] = value
    final_state[m_name] = data

# Dump final state to file
data_file = Path(args.OUT_FILE)
os.makedirs(data_file.parent, exist_ok=True)
with open(data_file, 'wb') as f:
    pickle.dump((args.SEED, final_state), f)

print("[[END OF BENCHMARK]]", file=sys.stderr)

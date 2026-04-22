#!/usr/bin/env python
"""
Experiment with the logarithmic transform and its offset parameter.
"""
from matexp import LinearInput, LogarithmicInput
from matexp.lti_model import LTI_Model
from matexp.optimizer import Optimizer
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
root_dir = Path(__file__).parent.parent

# Experimental setup.
min_scale   = 1e-9
num_scales  = 50
if os.environ.get('RUNFAST', ''):
    num_scales  = 5
error_arg   = 0.001 # Unused by this experiment.
time_step   = 0.025
temperature = 37
verbose     = 2
v_input     = LinearInput("v", -100, 100)
g_input     = LogarithmicInput('C', 0, 10)
all_inputs  = [v_input, g_input]

# AMPA
model_file  = root_dir / "mod/AMPA_13state.mod"
model = LTI_Model(model_file, all_inputs, time_step, temperature)
ampa = Optimizer(model, error_arg, np.float64, "host", (verbose >= 2))

# NMDA
model_file  = root_dir / "mod/NMDA_10state.mod"
model = LTI_Model(model_file, all_inputs, time_step, temperature)
nmda = Optimizer(model, error_arg, np.float64, "host", (verbose >= 2))
assert nmda.model.input1.name == 'C'
assert nmda.model.input2.name == 'v'

fig = plt.figure(f"Log Offset {model.name}", figsize=(7.5, 4))
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
grid = gs.subplots(sharex='all', sharey='all')

if True:
    axes = grid[0, 0]
    num_buckets = [20]
    for polynomial in (5, 4, 3, 2, 1, 0):
        scales, errors = ampa._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-', marker='o', markerfacecolor='none', label=f'{polynomial} degree polynomial')
    axes.set_title(f"Error vs Polynomial\n{num_buckets[0]} input partitions")
    axes.set_xlabel("ϵ, the logarithmic offset")
    axes.set_ylabel("RMS of residual error")
    axes.legend()

if True:
    axes = grid[0, 1]
    polynomial = 3
    for num_buckets in ([160], [80], [40], [20], [10], [5]):
        scales, errors = ampa._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-', marker='o', markerfacecolor='none', label=f'{num_buckets[0]} input partitions')
    axes.set_title(f"Error vs Partitions\n{polynomial} degree polynomial")
    axes.set_xlabel("ϵ, the logarithmic offset")
    axes.legend()

if True:
    axes = grid[1, 0]
    num_buckets = [10, 500]
    for polynomial in ["v^3+v^2+v+1+C+C^2+C^3+v*C", "v^3+v^2+v+1", "1+C+C^2+C^3"]:
        scales, errors = nmda._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-', marker='o', markerfacecolor='none',
                    label=f'{polynomial} degree polynomial')
    axes.set_title(f"Error vs Polynomial\n{num_buckets[0]} input partitions")
    axes.set_xlabel("ϵ, the logarithmic offset")
    axes.set_ylabel("RMS of residual error")
    axes.legend()

if True:
    axes = grid[1, 1]
    polynomial = "v^3+v^2+v+1+C+C^2+C^3+v*C"
    for num_buckets in ([10, 500], [5, 500], [10, 250], [5, 250]):
        scales, errors = nmda._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-', marker='o', markerfacecolor='none',
                    label=f'{num_buckets[0]} input partitions')
    axes.set_title(f"Error vs Partitions\n{polynomial} degree polynomial")
    axes.set_xlabel("ϵ, the logarithmic offset")
    axes.legend()

fig.savefig("log_offset.png", dpi=600, bbox_inches='tight')
if not os.environ.get('NOSHOW', ''): plt.show()

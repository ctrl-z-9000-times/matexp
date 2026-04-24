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
import cmcrameri.cm as cmc
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
g_input     = LogarithmicInput('C', 0, 100)
all_inputs  = [v_input, g_input]

# AMPA
model_file  = root_dir / "mod/AMPA_13state.mod"
model = LTI_Model(model_file, all_inputs, time_step, temperature)
ampa = Optimizer(model, error_arg, np.float64, "host", (verbose >= 2))

# NMDA
def make_nmda():
    model_file  = root_dir / "mod/NMDA_10state.mod"
    model = LTI_Model(model_file, all_inputs, time_step, temperature)
    nmda = Optimizer(model, error_arg, np.float64, "host", (verbose >= 2))
    assert nmda.model.input1.name == 'C'
    assert nmda.model.input2.name == 'v'
    return nmda

fig = plt.figure(f"Log Offset {model.name}", figsize=(7.5, 7.5))
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
grid = gs.subplots(sharex='all', sharey='all')

for axes in grid.flat:
    axes.spines[['right', 'top']].set_visible(False) # Hide the top & right borders
    axes.xaxis.set_tick_params(direction="in")
    axes.yaxis.set_tick_params(direction="in")

if True:
    axes = grid[0, 0]
    num_buckets = [20]
    for index, polynomial in enumerate((5, 4, 3, 2, 1, 0)):
        scales, errors = ampa._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-', label=f'{polynomial} degree polynomial',
                    color=cmc.batlow(index / 5))
    # axes.set_title(f"Error vs Polynomial\n{num_buckets[0]} input partitions")
    axes.set_ylabel("Residual error")
    axes.legend()

if True:
    axes = grid[0, 1]
    polynomial = 3
    for index, num_buckets in enumerate(([160], [80], [40], [20], [10], [5])):
        scales, errors = ampa._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-', label=f'{num_buckets[0]} input partitions',
                    color=cmc.batlow(index / 5))
    # axes.set_title(f"Error vs Partitions\n{polynomial} degree polynomial")
    axes.legend()

del ampa

if True:
    axes = grid[1, 0]
    num_buckets = [10, 500]
    for index, polynomial in enumerate(["v^3+v^2+v+1+C+C^2+C^3+v*C", "v^3+v^2+v+1", "1+C+C^2+C^3"]):
        nmda = make_nmda()
        scales, errors = nmda._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-',
                    label=f'{polynomial} degree polynomial',
                    color=cmc.batlow(index / 2))
    axes.set_xlabel("Logarithmic offset, ϵ")
    axes.set_ylabel("Residual error")
    axes.legend()

if True:
    axes = grid[1, 1]
    polynomial = "v^3+v^2+v+1+C+C^2+C^3+v*C"
    for index, num_buckets in enumerate(([10, 500], [5, 500], [10, 250], [5, 250])):
        nmda = make_nmda()
        scales, errors = nmda._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-',
                    label=f'{num_buckets[0]} input partitions',
                    color=cmc.batlow(index / 3))
    axes.set_xlabel("Logarithmic offset, ϵ")
    axes.legend()

fig.savefig("log_offset.png", dpi=600, bbox_inches='tight')
if not os.environ.get('NOSHOW', ''): plt.show()

"""
Experiment with the logarithmic transform and its offset parameter.
"""
from matexp import LinearInput, LogarithmicInput
from matexp.lti_model import LTI_Model
from matexp.optimizer import Optimizer
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
root_dir = Path(__file__).parent.parent

# Experimental setup.
min_scale   = 1e-9
num_scales  = 100
if os.environ.get('RUNFAST', ''):
    num_scales  = 5
error_arg   = 0.001 # Unused by this experiment.
time_step   = 0.100
temperature = 37
verbose     = 2
v_input     = LinearInput("v", -100, 100)
g_input     = LogarithmicInput('C', 0, 1e3)
all_inputs  = [v_input, g_input]
model_file  = root_dir / "mod/AMPA_13state.mod"
# model_file  = root_dir / "mod/NMDA_10state.mod"

# 
model = LTI_Model(model_file, all_inputs, time_step, temperature)
optimizer = Optimizer(model, error_arg, np.float64, "host", (verbose >= 2))
fig = plt.figure(f"Log Offset {model.name}", figsize=(7.5, 4))
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
grid = gs.subplots(sharey='all')

if True:
    axes = grid[0]
    num_buckets = [20]
    for polynomial in (5, 4, 3, 2, 1, 0):
        scales, errors = optimizer._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-', marker='o', markerfacecolor='none', label=f'{polynomial} degree polynomial')
    axes.set_title(f"Error vs Polynomial\n{num_buckets[0]} input partitions")
    axes.set_xlabel("ϵ, the logarithmic offset")
    axes.set_ylabel("RMS of residual error")
    axes.legend()

if True:
    axes = grid[1]
    polynomial = 3
    for num_buckets in ([160], [80], [40], [20], [10], [5]):
        scales, errors = optimizer._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        axes.loglog(scales, errors, linestyle='-', marker='o', markerfacecolor='none', label=f'{num_buckets[0]} input partitions')
    axes.set_title(f"Error vs Partitions\n{polynomial} degree polynomial")
    axes.set_xlabel("ϵ, the logarithmic offset")
    axes.legend()

fig.savefig("log_offset.png", dpi=600, bbox_inches='tight')
if not os.environ.get('NOSHOW', ''): plt.show()

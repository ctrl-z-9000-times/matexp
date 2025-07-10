"""
Experiment with the logarithmic transform and its offset parameter.
"""
from matexp import LinearInput, LogarithmicInput
from matexp.lti_model import LTI_Model
from matexp.optimizer import Optimizer
import matplotlib.pyplot as plt
import numpy as np

# Experimental setup.
min_scale   = 1e-9
num_scales  = 50
error_arg   = 0.001 # Unused by this experiment.
time_step   = 0.1
temperature = 37
verbose     = 2
v_input     = LinearInput("v", -100, 100)
g_input     = LogarithmicInput('C', 0, 1e3)
all_inputs  = [v_input, g_input]
model_file  = "mod/src/ampa13.mod"
# model_file  = "mod/src/NMDA.mod"

# 
model = LTI_Model(model_file, all_inputs, time_step, temperature)
optimizer = Optimizer(model, error_arg, np.float64, "host", (verbose >= 2))
plt.figure("Log Transform")
plt.suptitle(f"Logarithmic Transform Analysis for {model.name}")

if True:
    plt.subplot(1, 2, 1)
    num_buckets = [20]
    for polynomial in (5, 4, 3, 2, 1, 0):
        scales, errors = optimizer._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        plt.loglog(scales, errors, linestyle='-', marker='o', markerfacecolor='none', label=f'{polynomial} degree polynomial')
    plt.title(f"Error vs Polynomial\n{num_buckets[0]} input partitions")
    plt.xlabel("ϵ, the logarithmic offset")
    plt.ylabel("RMS of residual error")
    plt.legend()

if True:
    plt.subplot(1, 2, 2)
    polynomial = 3
    for num_buckets in ([160], [80], [40], [20], [10], [5]):
        scales, errors = optimizer._eval_log_scale(num_buckets, polynomial, min_scale, num_scales)
        plt.loglog(scales, errors, linestyle='-', marker='o', markerfacecolor='none', label=f'{num_buckets[0]} input partitions')
    plt.title(f"Error vs Partitions\n{polynomial} degree polynomial")
    plt.xlabel("ϵ, the logarithmic offset")
    plt.ylabel("RMS of residual error")
    plt.legend()

plt.show()

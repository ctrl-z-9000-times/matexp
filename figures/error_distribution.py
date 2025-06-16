"""
Measure and analyse the approximation error.
This treats every cell of the matrix independently.
Compare matexp's internal error model with empirical measurements.
"""
from matexp import main, LinearInput, LogarithmicInput
from matexp.approx import MatrixSamples
from matexp.convolve import autoconvolve
from matexp.lti_model import LTI_Model
import matplotlib.pyplot as plt
import numpy as np

# Experimental setup.
error_arg   = 0.001
time_step   = 0.1
temperature = 37
verbose     = 2
v_input     = LinearInput("v", -100, 100)
g_input     = LogarithmicInput('C', 0, 1e3)
all_inputs  = [v_input, g_input]
model_file  = "mod/src/Balbi2017/Nav11.mod"
# model_file  = "mod/src/ampa13.mod"
# model_file  = "mod/src/NMDA.mod"

# Run the matexp program.
parameters      = main(model_file, all_inputs, time_step, temperature, error_arg, np.float64, "host", verbose=verbose)
num_states      = parameters.model.num_states
error_samples   = parameters.approx.sample_error()
model_dist      = parameters.approx.error_distribution(error_samples)

# Get fresh samples to work with.
parameters.approx.samples = MatrixSamples(parameters.model, verbose=verbose)
parameters.approx._ensure_enough_exact_samples(100)

# Measure the error.
error_samples   = parameters.approx.sample_error()
error_data      = np.abs(error_samples)
print()
print("Approximation error summary")
print("samples:     ", len(error_data))
print("min:         ", np.min(error_data))
print("mean:        ", np.mean(error_data))
print("median:      ", np.median(error_data))
print("max:         ", np.max(error_data))
print("RMS:         ", np.mean(error_data ** 2) ** .5)
print("50%:         ", np.percentile(error_data, 50))
print("99%:         ", np.percentile(error_data, 99))
print()

# Visualize the approximation errors.
plt.figure("Approximation Errors")
plt.subplot(1, 2, 1)
plt.title("Histogram of Approximation Errors")
plt.hist(error_samples, bins=500)
plt.xlabel('error')
plt.ylabel('samples')
plt.subplot(1, 2, 2)
plt.title("Absolute Value of Errors, with Logarithmic Scaling")
plt.hist(error_data, bins=250, log=True)
plt.xlabel('absolute error')
plt.ylabel('samples')

# Estimate the sample's error distribution.
error_range     = (0.0, np.max(error_data))
error_bins      = np.linspace(*error_range, 2**13+1)
error_hist, error_bins = np.histogram(error_data, bins=error_bins)
error_hist      = error_hist / len(error_data)
# Estimate the accumulation of errors.
instances       = round(num_states * 1000 / time_step)
error_hist, error_range = autoconvolve(error_hist, error_range, instances, epsilon=1e-14)
error_bins      = np.linspace(*error_range, len(error_hist)+1)
bin_width       = error_bins[1] - error_bins[0]
bin_centers     = 0.5 * (error_bins[:-1] + error_bins[1:])
print("Model-state error summary")
print("bins:        ", len(error_hist))
print("bin-width:   ", bin_width)
print("mean:        ", np.average(bin_centers, weights=error_hist))
print("50%:         ", model_dist.ppf(.50))
print("99%:         ", model_dist.ppf(.99))
print()

# Visualize the model-state errors.
plt.figure("State Errors")
plt.title("State Error Distribution")
plt.plot(bin_centers, np.diff(model_dist.cdf(error_bins)), 'k', label='Model distribution')
plt.plot(bin_centers, error_hist, 'r', label='Sample errors')
plt.xlabel('absolute error')
plt.ylabel('probability density')
plt.legend()
plt.show()

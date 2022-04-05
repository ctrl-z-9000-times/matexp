"""
Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.

For more information see:
    Exact digital simulation of time-invariant linear systems with applications
    to neuronal modeling. Rotter S, Diesmann M (1999).
    https://doi.org/10.1007/s004220050570
"""

from .inputs import LinearInput, LogarithmicInput
from .nmodl_compiler import NMODL_Compiler
from .optimizer import Optimize1D
import numpy as np
import time

__all__ = ('main_1D', 'LinearInput', 'LogarithmicInput')

def main_1D(nmodl_filename, input1, time_step, temperature,
            accuracy, float_size, target,
            outfile=False, plot=False, benchmark=False,):
    if   float_size == 32: float_dtype = np.float32
    elif float_size == 64: float_dtype = np.float64
    else: raise ValueError(f'Invalid argument "float_size", expected 32 or 64, got "{float_size}"')
    # 
    nmodl_info = NMODL_Compiler(nmodl_filename, [input1], temperature)
    _check_is_LTI_1D(nmodl_info.derivative, nmodl_info.num_states, input1)
    optimizer = Optimize1D(nmodl_info, time_step, accuracy, float_dtype, target)
    if plot:
        optimizer.approx.plot(nmodl_info.name)
    if benchmark:
        mult = _estimate_multiplies(len(nmodl_info.state_names), optimizer.order,
                                    isinstance(input1, LogarithmicInput), nmodl_info.conserve_sum)
        mean, std = optimizer.benchmark
        print("Number of Buckets:", optimizer.num_buckets)
        print("Polynomial Order: ", optimizer.order)
        print("Table size:", round(optimizer.approx.table.nbytes / 1000), "kB")
        print("Multiplies:", mult, '(estimated)')
        print(f"Nanoseconds per instance per advance: {round(mean, 1)} +/- {round(std, 2)}")
    if outfile:
        optimizer.backend.write(outfile)
        if outfile != optimizer.backend.filename:
            print(f'Output written to: "{optimizer.backend.filename}"')
    initial_state = optimizer.exact.initial_state(nmodl_info.conserve_sum)
    initial_state = dict(zip(nmodl_info.state_names, initial_state))
    return (initial_state, optimizer.backend.load())

def _check_is_LTI_1D(derivative_function, num_states, input1):
    for trial in range(3):
        input1_value = np.random.uniform(input1.minimum, input1.maximum)
        state1       = np.random.uniform(0.0, 1.0, size=num_states)
        state2       = state1 * 2.0
        d1 = derivative_function(input1_value, *state1)
        d2 = derivative_function(input1_value, *state2)
        for s1, s2 in zip(d1, d2):
            assert abs(s1 - s2 / 2.0) < 1e-12, "Non-Linear system detected!"

def _measure_speed(f, num_states, input1, conserve_sum, float_dtype, target):
    num_instances = 10 * 1000
    num_repetions = 1000
    # 
    if target == 'host':
        xp = np
    elif target == 'cuda':
        import cupy
        xp = cupy
    # Generate valid initial states.
    state = [xp.array(xp.random.uniform(size=num_instances), dtype=float_dtype)
                for x in range(num_states)]
    if conserve_sum is not None:
        conserve_sum = float(conserve_sum)
        sum_states = xp.zeros(num_instances)
        for data in state:
            sum_states = sum_states + data
        correction_factor = conserve_sum / sum_states
        for data in state:
            data *= correction_factor
    # 
    input_indicies = xp.arange(num_instances, dtype=np.int32)
    elapsed_times = np.empty(num_repetions)
    for trial in range(num_repetions):
        inp = xp.array(xp.random.uniform(input1.minimum, input1.maximum,
                                         num_instances), dtype=float_dtype)
        _clear_cache(xp)
        if target == 'cuda':
            cupy.cuda.runtime.deviceSynchronize()
        time.sleep(0)
        start_time = time.time_ns()
        f(num_instances, inp, input_indicies, *state)
        if target == 'cuda':
            cupy.cuda.runtime.deviceSynchronize()
        elapsed_times[trial] = time.time_ns() - start_time
    elapsed_times /= num_instances
    return (np.mean(elapsed_times), np.std(elapsed_times))

def _clear_cache(array_module):
    big_data = array_module.zeros(int(2e6))
    for _ in range(3):
        big_data += 1.0

def _estimate_multiplies(num_states, order, log, conserve_sum):
    matrix_size = num_states ** 2
    x = 0
    if log:
        x += 3 # Compute log2. This is a guess, I have no idea how it's actually implemented.
    x += 1 # Scale the input value into an index.
    x += 1 # Compute the offset into the table.
    x += order - 1 # Compute the terms of the polynomial basis.
    x += matrix_size * order # Evaluate the polynomial approximation.
    x += matrix_size # Compute the dot product.
    if conserve_sum is not None:
        x += num_states + 1 # Conserve statement.
    return x

if __name__ == '__main__':
    import __main__

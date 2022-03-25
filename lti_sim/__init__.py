"""
Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.

For more information see:
    Exact digital simulation of time-invariant linear systems with applications
    to neuronal modeling. Rotter S, Diesmann M (1999).
    https://doi.org/10.1007/s004220050570
"""

from .inputs import LinearInput, LogarithmicInput
from .nmodl_compiler import NMODL_Compiler
from .tables import Exact1D, Approx1D
from .optimizer import Optimize1D
from .codegen import Codegen1D
import numpy as np

__all__ = ('main_1D', 'LinearInput', 'LogarithmicInput')

def main_1D(nmodl_filename, input1, time_step, temperature, order, accuracy,
            plot=False, benchmark=False, outfile=False):
    parser = NMODL_Compiler(nmodl_filename)
    if len(parser.inputs) != 1 or input1.name != parser.inputs[0]:
        expected_inputs = ' & '.join(parser.inputs)
        raise ValueError(f'Invalid inputs, expected {expected_inputs} got {input1.name}')
    deriv = parser.get_derivative_function(temperature)
    _check_is_LTI_1D(deriv, parser.state_names, input1)
    optimizer = Optimize1D(time_step, deriv, parser.state_names, input1, order, accuracy)
    approx = optimizer.approx
    if plot: approx.plot(parser.name)
    backend = Codegen1D(parser.name, approx, parser.conserve_sum)
    if benchmark:
        table_size = approx.table.nbytes / 1000
        print("Table size:", round(table_size), "kB")
        mult = _estimate_multiplies(len(parser.state_names), order,
                                    isinstance(input1, LogarithmicInput), parser.conserve_sum)
        print("Multiplies:", mult, 'x')
        mean, std = _measure_speed(backend.load(), parser.state_names, input1, parser.conserve_sum)
        print(f"Nanoseconds per instance per advance: {round(mean, 1)} +/- {round(std, 2)}")
    if outfile:
        backend.write(outfile)
        if outfile != backend.filename:
            print(f'Output written to: "{backend.filename}"')
    else:
        initial_state = approx.exact.initial_state(parser.conserve_sum)
        initial_state = dict(zip(parser.state_names, initial_state))
        return (initial_state, backend.load())


def _check_is_LTI_1D(deriv, state_names, input1):
    for trial in range(3):
        input1_value = np.random.uniform(input1.minimum, input1.maximum)
        state1       = np.random.uniform(0.0, 1.0, size=len(state_names))
        state2       = state1 * 2.0
        d1 = deriv(input1_value, *state1)
        d2 = deriv(input1_value, *state2)
        for s1, s2 in zip(d1, d2):
            assert abs(s1 - s2 / 2.0) < 1e-12, "Non-Linear system detected!"


def _measure_speed(f, state_names, input1, conserve_sum):
    import cupy
    import time
    num_instances = 50 * 1000 # Less than 20k instances yields unreliable results.
    input_indicies = cupy.arange(num_instances, dtype=np.int32)
    # Generate valid inital states.
    state = {x: cupy.random.uniform(size=num_instances, dtype=np.float32)
                for x in state_names}
    if conserve_sum is not None:
        conserve_sum = float(conserve_sum)
        sum_states = cupy.zeros(num_instances)
        for data in state.values():
            sum_states = sum_states + data
        correction_factor = conserve_sum / sum_states
        for data in state.values():
            data *= correction_factor
    # 
    num_repetions = 10000 # Less than 10k trials yields unreliable results.
    elapsed_times = np.empty(num_repetions)
    inp_size = 10e6 / 4 # Generate extra random numbers to clear the GPU's data cache.
    for trial in range(num_repetions):
        inp = cupy.random.uniform(input1.minimum, input1.maximum - 1e-9,
                                  max(int(inp_size), num_instances), dtype=np.float32)
        cupy.cuda.runtime.deviceSynchronize()
        start_time = time.time_ns()
        f(num_instances, inp, input_indicies, **state)
        cupy.cuda.runtime.deviceSynchronize()
        elapsed_times[trial] = time.time_ns() - start_time
    elapsed_times /= num_instances
    return (np.mean(elapsed_times), np.std(elapsed_times))


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

"""
Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.

For more information see:
    Exact digital simulation of time-invariant linear systems with applications
    to neuronal modeling. Rotter S, Diesmann M (1999).
    https://doi.org/10.1007/s004220050570
"""

from .inputs import LinearInput, LogarithmicInput
from .lti_model import LTI_Model
from .optimizer import Optimize1D, Optimize2D
import numpy as np
import time

__all__ = ('main', 'LinearInput', 'LogarithmicInput')

def main(nmodl_filename, inputs, time_step, temperature,
         accuracy, float_dtype, target,
         outfile=False, verbose=False, plot=False,):
    # 
    model = LTI_Model(nmodl_filename, inputs, time_step, temperature)
    # 
    if model.num_inputs == 1:
        optimizer = Optimize1D(model, accuracy, float_dtype, target)
    elif model.num_inputs == 2:
        optimizer = Optimize2D(model, accuracy, float_dtype, target)
    else:
        raise NotImplementedError(f'too many inputs.')

    if outfile:
        optimizer.backend.write(outfile)
        if outfile != optimizer.backend.filename:
            print(f'Output written to: "{optimizer.backend.filename}"')
    if verbose:
        optimizer.approx.print_summary()
        mean, std = optimizer.benchmark
        print(f"Run speed:   {round(mean, 1)} +/- {round(std, 2)}", "ns/Δt")
    if plot:
        optimizer.approx.plot(model.name)
    return (model.get_initial_state(), optimizer.backend.load())

def _measure_speed(f, num_states, inputs, conserve_sum, float_dtype, target):
    num_instances = 10 * 1000
    num_repetions = 1000
    # 
    if target == 'host':
        xp = np
    elif target == 'cuda':
        import cupy
        xp = cupy
        start_event = cupy.cuda.Event()
        end_event   = cupy.cuda.Event()
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
        input_arrays = []
        for inp in inputs:
            input_arrays.append(inp.random(num_instances, float_dtype, xp))
            input_arrays.append(input_indicies)
        _clear_cache(xp)
        if target == 'cuda':
            start_event.record()
            f(num_instances, *input_arrays, *state)
            end_event.record()
            end_event.synchronize()
            elapsed_times[trial] = 1e6 * cupy.cuda.get_elapsed_time(start_event, end_event)
        elif target == 'host':
            start_time = time.thread_time_ns()
            f(num_instances, *input_arrays, *state)
            elapsed_times[trial] = time.thread_time_ns() - start_time
    elapsed_times /= num_instances
    return (np.mean(elapsed_times), np.std(elapsed_times))

def _clear_cache(array_module):
    big_data = array_module.zeros(int(2e6))
    for _ in range(3):
        big_data += 1.0

if __name__ == '__main__':
    import __main__

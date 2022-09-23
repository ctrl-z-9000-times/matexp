"""
Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.

Usage: python -m lti_sim

For more information see:
    Exact digital simulation of time-invariant linear systems with applications
    to neuronal modeling. Rotter S, Diesmann M (1999).
    https://doi.org/10.1007/s004220050570
"""

# Written by David McDougall, 2022

from .inputs import LinearInput, LogarithmicInput
from .lti_model import LTI_Model
from .optimizer import Optimize1D, Optimize2D
from . import cache
import numpy as np
import os
import time

__all__ = ('main', 'LinearInput', 'LogarithmicInput')

def main(nmodl_filename, inputs, time_step, temperature,
         error, float_dtype, target,
         outfile=None, verbose=False, plot=False, use_cache=True, load=False):
    outfile = _get_output_filename(outfile, nmodl_filename, target, verbose)
    # Check the cache for a saved solution.
    arguments = (inputs, time_step, temperature, error, float_dtype, target)
    if use_cache and not plot and not load:
        source_code = cache.read(nmodl_filename, arguments)
        if source_code:
            if verbose:
                print("Using cached output.")
            with open(outfile, 'wt') as f:
                f.write(source_code)
            return
    # Read and process the NMODL file.
    model = LTI_Model(nmodl_filename, inputs, time_step, temperature)
    if   model.num_inputs == 1: OptimizerClass = Optimize1D
    elif model.num_inputs == 2: OptimizerClass = Optimize2D
    else: raise NotImplementedError('too many inputs.')
    optimized = OptimizerClass(model, error, float_dtype, target, (verbose >= 2)).best
    source_code = optimized.backend.source_code
    # 
    if verbose or plot:
        print(str(optimized.approx) +
              f"Run speed:    {round(optimized.runtime)} ns/Δt")
    if plot:
        optimized.approx.plot(model.name)
    if use_cache:
        cache.write(nmodl_filename, arguments, source_code)
    with open(outfile, 'wt') as f:
        f.write(source_code)
    if load:
        return optimized.backend.load()

def _get_output_filename(outfile, nmodl_filename, target, verbose):
    """ Clean the output filename and generate a default one if none was given. """
    generate_filename = not outfile
    if generate_filename:
        nmodl_filename = os.path.basename(nmodl_filename)
        nmodl_filename = os.path.splitext(nmodl_filename)[0]
        if target == 'cuda':
            outfile = nmodl_filename + '.cu'
        elif target == 'host':
            outfile = nmodl_filename + '.cpp'
    outfile = os.path.abspath(outfile)
    if generate_filename or verbose:
        print(f'Output will be written to: "{outfile}"')
    return outfile

def _measure_speed(f, num_states, inputs, conserve_sum, float_dtype, target):
    num_instances = 10 * 1000
    num_repetions = 200
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
        _clear_CPU_cache(xp)
        time.sleep(0) # Try to avoid task switching while running.
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
    return np.min(elapsed_times) / num_instances

def _clear_CPU_cache(array_module):
    # Read and then write back 32MB of data. Assuming that the CPU is using a
    # least-recently-used replacement policy, touching every piece of data once
    # should be sufficient to put it into the cache.
    big_data = array_module.empty(int(32e6 / 8), dtype=np.int64)
    big_data += 1

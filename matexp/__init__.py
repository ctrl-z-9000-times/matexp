"""
Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.

Usage: python -m matexp

For more information see:
    Exact digital simulation of time-invariant linear systems with applications
    to neuronal modeling. Rotter S, Diesmann M (1999).
    https://doi.org/10.1007/s004220050570
"""

# Written by David McDougall, 2022-2025

from .approx import Approx1D, Approx2D, MatrixSamples
from .codegen import Codegen
from .inputs import LinearInput, LogarithmicInput
from .lti_model import LTI_Model
from .optimizer import Optimize1D, Optimize2D
from pathlib import Path
import concurrent.futures
import numpy as np
import os
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

__all__ = ('main', 'LinearInput', 'LogarithmicInput')

_num_threads = len(os.sched_getaffinity(0))
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=_num_threads)

def main(nmodl_filename, inputs, time_step, temperature,
         error, float_dtype, target,
         outfile=None, verbose=False):
    # Read and process the NMODL file.
    model = LTI_Model(nmodl_filename, inputs, time_step, temperature)
    if   model.num_inputs == 1: OptimizerClass = Optimize1D
    elif model.num_inputs == 2: OptimizerClass = Optimize2D
    else: raise NotImplementedError('too many inputs.')
    optimizer = OptimizerClass(model, error, float_dtype, target, (verbose >= 2))
    optimizer.run()
    optimized = optimizer.best

    if verbose:
        print(optimized)

    if outfile:
        _save_model(model, optimized.backend.get_nmodl_text(), outfile)

    return optimized

def main_manual(nmodl_filename, inputs, time_step, temperature,
            polynomial, float_dtype, target,
            outfile, verbose=False):
    model = LTI_Model(nmodl_filename, inputs, time_step, temperature)
    samples = MatrixSamples(model, (verbose >= 2))
    if   model.num_inputs == 1: ApproxClass = Approx1D
    elif model.num_inputs == 2: ApproxClass = Approx2D
    else: raise NotImplementedError('too many inputs.')
    approx = ApproxClass(samples, polynomial)
    codegen = Codegen(approx, float_dtype, target)
    if verbose:
        print(str(approx).strip())
        residual = approx.measure_residual_error()
        print("Residual error: %.3g"%residual)
    _save_model(model, codegen.get_nmodl_text(), outfile)

def _save_model(model, nmodl_text, outfile):
    outfile = Path(outfile).resolve()
    if outfile.is_dir():
        outfile = outfile / Path(model.nmodl_filename).name
    assert outfile != model.nmodl_filename, "operation would overwrite input file"
    with open(outfile, 'wt') as f:
        f.write(nmodl_text)

def _initial_state(array_module, num_states, conserve_sum, num_instances, float_dtype):
    """ Generate valid initial states, for testing and benchmarks. """
    state = [array_module.random.uniform(size=num_instances) for x in range(num_states)]
    state = [array_module.array(x, dtype=float_dtype) for x in state]
    if conserve_sum is not None:
        conserve_sum = float(conserve_sum)
        sum_states = array_module.zeros(num_instances, dtype=float_dtype)
        for array in state:
            sum_states = sum_states + array
        correction_factor = conserve_sum / sum_states
        for array in state:
            array *= correction_factor
    return state

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
    # 
    state = _initial_state(xp, num_states, conserve_sum, num_instances, np.float64)
    # 
    input_indicies = xp.arange(num_instances, dtype=np.int32)
    elapsed_times = np.empty(num_repetions)
    for trial in range(num_repetions):
        input_arrays = []
        for inp in inputs:
            input_arrays.append(inp.random(num_instances, np.float64, xp))
            input_arrays.append(input_indicies)
        _clear_CPU_cache(xp)
        time.sleep(0) # Try to avoid task switching while running.
        os.sched_yield()
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
    big_data += 1
    big_data += 1

#!/usr/bin/python3
"""
"""
from pathlib import Path
from pprint import pprint
import argparse
import cupy
import matexp
import numpy as np
import itertools
# 
parser = argparse.ArgumentParser()
parser.add_argument("MOD_FILE", type=Path)
args = parser.parse_args()
# 
time_step       = .025
float_dtype     = np.float32
voltage_input   = matexp.LinearInput('v', -100, 100)
glutamate_input = matexp.LogarithmicInput('C', 0, 10)
parameters = matexp.main(args.MOD_FILE, [voltage_input, glutamate_input],
    time_step=time_step,
    temperature=37,
    error=1e-4,
    target='cuda',
    float_dtype=float_dtype,
    verbose=2)
model_name  = parameters.model.name
inputs      = parameters.model.inputs
num_states  = parameters.model.num_states
cuda_fn     = parameters.backend.load()
# 
def flush_cache():
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    props = cupy.cuda.runtime.getDeviceProperties(0)
    l2 = props['l2CacheSize']
    big_data = cupy.empty(l2, dtype=np.int64) # 8x multiplier
    big_data += 1
# 
def measure_speed(num_instances, continuous):
    # Setup random initial state.
    state = matexp._initial_state(cupy, num_states,
        conserve_sum=1.0,
        num_instances=num_instances,
        float_dtype=float_dtype)
    # Cycle through several different input values.
    input_arrays = [[] for _ in range(20)]
    for input_set in input_arrays:
        for inp in inputs:
            input_set.append(inp.random(num_instances, float_dtype, cupy))
            input_set.append(cupy.arange(num_instances, dtype=np.int32))
    input_iter = itertools.cycle(input_arrays)
    # 
    def advance():
        inputs = next(input_iter)
        cuda_fn(num_instances, *inputs, *state)
    # 
    start_event = cupy.cuda.Event()
    end_event   = cupy.cuda.Event()
    # 
    if continuous:
        num_steps = round(1 / time_step)
        # Warmup
        for _ in range(5):
            advance()
    else:
        num_steps = 1
        flush_cache() # Cold start
    # Perform the measurement.
    start_event.record()
    for t in range(num_steps):
        advance()
    end_event.record()
    end_event.synchronize()
    # 
    elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
    return elapsed_ms

ten_k = measure_speed(10000, False)
print("10k", ten_k)

elapsed_ms      = 0
num_instances   = 1000
multiplier      = .01
prev_direction  = False
while True:
    elapsed_ms = measure_speed(num_instances, True)
    if direction := elapsed_ms < 1:
        num_instances *= 1 + multiplier
    else:
        num_instances *= 1 - multiplier
    if direction != prev_direction:
        multiplier *= .9
        prev_direction = direction
    num_instances = round(num_instances)
    print(elapsed_ms, multiplier, num_instances)
    if multiplier < .001: # ~20 iterations
        break

# 
data_file = Path("gpu_data").joinpath(model_name)
with open(data_file, 'wt') as file:
    print(ten_k, file=file)
    print(num_instances, file=file)

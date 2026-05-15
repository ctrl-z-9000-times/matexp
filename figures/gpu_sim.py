#!/usr/bin/python3
from pathlib import Path
from pprint import pprint
import argparse
import cupy
import matexp
import numpy as np
import itertools
import sys

sys.modules['__main__'] = sys.modules['matexp']
# 
parser = argparse.ArgumentParser()
parser.add_argument("MOD_FILE", type=Path)
args = parser.parse_args()
# 
time_step       = .025
voltage_input   = matexp.LinearInput('v', -100, 100)
glutamate_input = matexp.LogarithmicInput('C', 0, 10)
parameters = matexp.main(args.MOD_FILE, [voltage_input, glutamate_input],
    time_step=time_step,
    temperature=37,
    error=1e-3,
    target='cuda',
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
    big_data += 1
    big_data += 1
# 
def measure_speed(num_instances, continuous):
    # Setup random initial state.
    state = matexp._initial_state(cupy, num_states,
        conserve_sum=1.0,
        num_instances=num_instances)
    # Cycle through several different input values.
    input_arrays = [[] for _ in range(100)]
    for input_set in input_arrays:
        for inp in inputs:
            input_set.append(inp.random(num_instances, np.float64, cupy))
            input_set.append(cupy.arange(num_instances, dtype=np.int32))
    input_iter = iter(input_arrays)
    # 
    def advance():
        inputs = next(input_iter)
        cuda_fn(num_instances, *inputs, *state)
    # 
    if continuous:
        warmup_state = matexp._initial_state(cupy, num_states, conserve_sum=1.0, num_instances=10000)
        warmup_inputs = []
        for inp in inputs:
            warmup_inputs.append(inp.random(10000, np.float64, cupy))
            warmup_inputs.append(cupy.arange(10000, dtype=np.int32))
    # 
    start_event = cupy.cuda.Event()
    end_event   = cupy.cuda.Event()
    # Always cold start
    flush_cache()
    # 
    if continuous:
        num_steps = round(1 / time_step)
        cuda_fn(10000, *warmup_inputs, *warmup_state)
        for _ in range(10):
            advance()
    else:
        num_steps = 1
    # Perform the measurement.
    start_event.record()
    for t in range(num_steps):
        advance()
    end_event.record()
    end_event.synchronize()
    # 
    elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
    return elapsed_ms
# 
batch_data = []
for batch_size in [1, 10, 100, 1000, 10000, 100000]:
    batch_samples = []
    for _ in range(200):
        batch_samples.append(measure_speed(batch_size, False))
    batch_time = min(batch_samples)
    batch_data.append((batch_size, batch_time))
    print(f"batch {batch_size} x {batch_time} ms")
batch_size, batch_time = zip(*batch_data)
# 
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

print(f"throughput {num_instances} x")

# 
data_file = Path("gpu_data").joinpath(model_name).with_suffix('.csv')
with open(data_file, 'wt') as file:
    header = list(f'batch{x}' for x in batch_size) + ["throughput"]
    row    = list(str(x) for x in batch_time) + [str(num_instances)]
    print(','.join(header), file=file)
    print(','.join(row), file=file)

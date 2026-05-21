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
parser.add_argument("--target", type=str, choices=["host", "cuda"], default="cuda")
args = parser.parse_args()
# 
time_step       = .025
voltage_input   = matexp.LinearInput('v', -100, 100)
glutamate_input = matexp.LogarithmicInput('C', 0, 10)
parameters = matexp.main(args.MOD_FILE, [voltage_input, glutamate_input],
    time_step=time_step,
    temperature=37,
    error=1e-3,
    target=args.target,
    verbose=2)
model_name  = parameters.model.name
# 
data_path = Path("gpu_data").joinpath(model_name).with_suffix('.csv')
data_file = open(data_path, 'wt')
print("warmups,instances,elapsed_ms")
print("warmups,instances,elapsed_ms", file=data_file)
# 
num_warmups   =  100 * 1000
num_instances = 1000 * 1000
num_repetitions = 200
elapsed_ns = matexp.measure_speed(parameters.approx, args.target,
            num_warmups=num_warmups, num_instances=num_instances, num_repetitions=num_repetitions)
# 
elapsed_ms = elapsed_ns / 1e6
print(f"{num_warmups},{num_instances},{elapsed_ms}")
print(f"{num_warmups},{num_instances},{elapsed_ms}", file=data_file)
data_file.close()

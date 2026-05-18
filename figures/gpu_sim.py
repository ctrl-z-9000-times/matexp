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
# 
data_path = Path("gpu_data").joinpath(model_name).with_suffix('.csv')
data_file = open(data_path, 'wt')
print("warmups,instances,elapsed_ns")
print("warmups,instances,elapsed_ns", file=data_file)
for num_warmups in [0, 1000, 10000, 100000]:
    for num_instances in [1, 1000, 10000, 100000, 1000000]:
        elapsed_ns = matexp.measure_speed(parameters.approx, 'cuda',
                    num_warmups=num_warmups, num_instances=num_instances)
        print(f"{num_warmups},{num_instances},{elapsed_ns}")
        print(f"{num_warmups},{num_instances},{elapsed_ns}", file=data_file)
data_file.close()

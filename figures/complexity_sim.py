#!/usr/bin/env python
"""
Measure the tradeoff between simulation speed versus accuracy.
"""

from matexp import main, LinearInput, LogarithmicInput
from pathlib import Path
import argparse
import csv
import matexp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.modules['__main__'] = sys.modules['matexp']

def speed_vs_accuracy(file, errors, time_step=0.025, temperature=37.0):
    output = Path("complexity_data").joinpath(Path(file).name).with_suffix(".csv")
    if not output.exists():
        with open(output, 'wt') as f:
            f.write("error,speed,size,multiplications\n")
    for error in errors:
        approx = build_approximation(file, time_step, temperature, error)
        speed = measure_speed(approx)
        size = table_size(approx)
        mult = num_multiplications(approx)
        with open(output, 'at') as f:
            f.write(f"{error},{speed},{size},{mult}\n")

def build_approximation(file, time_step, temperature, error):
    voltage_input = matexp.LinearInput('v', -100, 100)
    glu_input     = matexp.LogarithmicInput('C', 0, 10)
    inputs        = [voltage_input, glu_input]
    return matexp.main(file, inputs, time_step, error=error, temperature=temperature,
                        target='host', verbose=2)

def measure_speed(parameters):
    fn = parameters.backend.load()
    inputs = parameters.model.inputs
    num_states = parameters.model.num_states
    return matexp._measure_speed(fn, num_states, inputs, conserve_sum = 1.0,
                                target = 'host')

def table_size(parameters):
    return parameters.approx.table.nbytes

def num_multiplications(parameters):
    return parameters.approx._estimate_multiplies()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('FILE', type=Path)
parser.add_argument('ERROR', type=float, nargs='+')
args = parser.parse_args()
speed_vs_accuracy(args.FILE, args.ERROR)

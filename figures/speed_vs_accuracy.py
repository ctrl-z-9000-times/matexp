"""
Measure the tradeoff between simulation speed versus accuracy.

This program runs in two phases:
    Phase 1: Data Collection
    Phase 2: Data Visualization

Phase 1:
Run this program with a list of NMODL files to be processed.
This will output a CSV file named after each model into the current working directory.

Phase 2:
Run this program with the "--plot" argument and a list of CSV files.
"""

from matexp import main, LinearInput, LogarithmicInput
from pathlib import Path
import argparse
import csv
import matexp
import matplotlib.pyplot as plt
import numpy as np

def speed_vs_accuracy(file, time_step=0.1, temperature=37.0):
    # samples = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    # samples = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    samples = [1e-2, 1e-3, 1e-4, 1e-5]
    output = Path(".").joinpath(Path(file).name).with_suffix(".csv")
    if not output.exists():
        with open(output, 'wt') as f:
            f.write("error,speed\n")
    for error in samples:
        speed = measure_speed(build_approximation(file, time_step, temperature, error))
        with open(output, 'at') as f:
            f.write(f"{error},{speed}\n")

def build_approximation(file, time_step, temperature, error):
    voltage_input = matexp.LinearInput('v', -100, 100)
    glu_input     = matexp.LogarithmicInput('C', 0, 1e3)
    inputs        = [voltage_input, glu_input]
    return matexp.main(file, inputs, time_step, error=error, temperature=temperature,
            float_dtype=np.float64, target='host', verbose=2)

def measure_speed(parameters):
    fn = parameters.backend.load()
    inputs = parameters.model.inputs
    num_states = parameters.model.num_states
    return matexp._measure_speed(fn, num_states, inputs, conserve_sum = 1.0,
                                  float_dtype = np.float64, target = 'host')

def plot(files):
    plt.figure('Speed vs Accuracy')
    for file in files:
        file = Path(file)
        name = file.stem
        data = []
        with open(file, 'rt') as f:
            for row in csv.DictReader(f):
                data.append((float(row["error"]), float(row["speed"])))
        data.sort()
        error, speed = zip(*data)
        plt.semilogx(error, speed, marker='o', label=name)
    plt.title('Speed vs Accuracy')
    plt.ylabel('Time to Integrate, per Instance per Time Step\nNanoseconds')
    plt.xlabel('Error Parameter')
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('FILE', type=Path, nargs='+')
    args = parser.parse_args()
    if args.plot:
        plot(args.FILE)
    else:
        for file in args.FILE:
            speed_vs_accuracy(file)

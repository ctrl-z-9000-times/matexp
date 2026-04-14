from matexp import main_manual, LinearInput, LogarithmicInput
import argparse
import numpy as np
import re

parser = argparse.ArgumentParser(prog='matexp-manual',
        description='Solves Markov models for NEURON mechanisms')
parser.add_argument('nmodl_filename',
        metavar='INPUT_PATH',
        help="path of unsolved NMODL file")
parser.add_argument('output', type=str, metavar='OUTPUT_PATH',
        help="path for solved NMODL file")
parser.add_argument('-v', '--verbose', action='count', default=0,
        help="print diagnostic information, give twice for trace mode")
sim = parser.add_argument_group('simulation parameters')
sim.add_argument('-dt', '--time_step', type=float, default=.025,
        help="milliseconds, default: 0.025")
sim.add_argument('-t', '--temperature', type=float, default=37.0,
        help="degrees celsius, default: 37")
sim.add_argument('-p', '--polynomial', type=str, required=True,
        help="polynomial form, ex: v^2+v+1")
inputs = parser.add_argument_group('input specification')
inputs.add_argument('-i', '--input', action='append', default=[],
        nargs=4, metavar=('NAME', 'MIN', 'MAX', 'BINS'),
        help="input name, bounds, and number of paritions")
inputs.add_argument('--log', nargs=2, action='append', default=[],
        metavar=('INPUT', 'SCALE'),
        help="scale input logarithmically, for chemical concentrations")
computer = parser.add_argument_group('computer specification')
computer.add_argument('--target', choices=['host','cuda'], default='host',
        help="default: host")
computer.add_argument('-f', '--float', choices=['32','64'], default='64',
        help="default: 64")
args = parser.parse_args()

if   args.float == '32': float_dtype = np.float32
elif args.float == '64': float_dtype = np.float64

# Create the input data structures.
inputs = {}
log_scales = {name: float(scale) for name, scale in args.log}
for (name, minimum, maximum, bins) in args.input:
    if name in log_scales:
        inputs[name] = inp = LogarithmicInput(name, minimum, maximum)
        inp.set_num_buckets(bins, log_scales[name])
    else:
        inputs[name] = inp = LinearInput(name, minimum, maximum)
        inp.set_num_buckets(bins)
# 
for name in log_scales:
    if name not in inputs:
        parser.error(f'Argument "--log {name}" does not match any input name.')

main_manual(args.nmodl_filename, list(inputs.values()), args.time_step, args.temperature,
        args.polynomial, target=args.target, float_dtype=float_dtype,
        outfile=args.output, verbose=args.verbose)

_placeholder = lambda: None # Symbol for the CLI script to import and call.


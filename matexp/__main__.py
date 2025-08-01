from matexp import main, LinearInput, LogarithmicInput
import argparse
import numpy as np

parser = argparse.ArgumentParser(prog='matexp',
        description="Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.",)
parser.add_argument('nmodl_filename',
        metavar='INPUT_PATH',
        help="input filename for the unsolved NMODL file")
parser.add_argument('output', type=str, metavar='OUTPUT_PATH',
        help="output filename for the solution")
parser.add_argument('-v', '--verbose', action='count', default=0,
        help="print diagnostic information, give twice for trace mode")
sim = parser.add_argument_group('simulation parameters')
sim.add_argument('-t', '--time_step', type=float, required=True,
        help="milliseconds")
sim.add_argument('-c', '--celsius', type=float, default=37.0,
        help="default: 37°")
sim.add_argument('-e', '--error', type=float, default=1e-4,
        help="maximum error per time step. default: 10^-4")
inputs = parser.add_argument_group('input specification')
inputs.add_argument('-i', '--input', action='append', default=[],
        nargs=3, metavar=('NAME', 'MIN', 'MAX'),
        help="")
inputs.add_argument('--log', nargs='?', action='append', default=[],
        metavar='INPUT',
        help="scale input logarithmically, for chemical concentrations")
computer = parser.add_argument_group('computer specification')
computer.add_argument('--target', choices=['host','cuda'], default='host',
        help="default: host")
computer.add_argument('-f', '--float', choices=['32','64'], default='64',
        help="default: 64")
args = parser.parse_args()

if   args.float == '32': float_dtype = np.float32
elif args.float == '64': float_dtype = np.float64

# Gather & organize all information about the inputs.
inputs = {}
for (name, minimum, maximum) in args.input:
    inputs[name] = [LinearInput, (name, minimum, maximum)]
for name in args.log:
    if name is None:
        if len(inputs) == 1:
            name = next(iter(inputs))
        else:
            parser.error(f'Argument "--log" must specify which input it refers to.')
    elif name not in inputs:
        parser.error(f'Argument "--log {name}" does not match any input name.')
    inputs[name][0] = LogarithmicInput
# Create the input data structures.
inputs = [input_type(*args) for (input_type, args) in inputs.values()]

main(args.nmodl_filename, inputs, args.time_step, args.celsius,
     error=args.error, target=args.target, float_dtype=float_dtype,
     outfile=args.output, verbose=args.verbose)

_placeholder = lambda: None # Symbol for the CLI script to import and call.


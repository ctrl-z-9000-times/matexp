from lti_sim import main_1D
from lti_sim.inputs import LinearInput, LogarithmicInput
import argparse

parser = argparse.ArgumentParser(prog='lti',
        description="Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.",)
parser.add_argument('nmodl_filename',
        metavar='NMODL_FILE',
        help="")
parser.add_argument('-t', '--time_step', type=float, required=True,
        help="")
parser.add_argument('-c', '--celsius', type=float, default=37.0,
        help="")
parser.add_argument('-a', '--accuracy', type=float, default=1e-6,
        help="")
parser.add_argument('-i', '--input', action='append', required=True,
        nargs=3, metavar=('NAME', 'MIN', 'MAX'),
        help="")
parser.add_argument('--logarithmic', nargs='?', action='append', default=[],
        metavar='INPUT',
        help="")
parser.add_argument('--initial', nargs=2, action='append', default=[],
        metavar=('INPUT', 'VALUE'),
        help="")
parser.add_argument('--plot', action='store_true',
        help="")
parser.add_argument('-o', '--output', type=str, default=True,
        metavar='FILE',
        help="")
parser.add_argument('--order', type=int, default=4,
        help="")
parser.add_argument('--benchmark', action='store_true',
        help="")
args = parser.parse_args()

# Gather & organize all information about the inputs.
inputs = {}
for (name, minimum, maximum) in args.input:
    inputs[name] = [LinearInput, [name, minimum, maximum, None]]
for name in args.logarithmic:
    if name is None:
        if len(inputs) == 1:
            name = next(iter(inputs))
        else:
            parser.error(f'Argument --logarithmic must specify which input it refers to.')
    elif name not in inputs:
        parser.error(f'Argument "--logarithmic {name}" does not match any input name.')
    inputs[name][0] = LogarithmicInput
for name, initial_value in args.initial:
    if name not in inputs:
        parser.error(f'Argument "--initial {name}" does not match any input name.')
    inputs[name][1][3] = float(initial_value)
# Make the input data structures.
inputs = [input_type(*args) for (input_type, args) in inputs.values()]
# Run main function of program.
if len(inputs) == 1:
    main_1D(args.nmodl_filename, inputs[0],
            args.time_step, args.celsius, args.order, args.accuracy,
            plot=args.plot, benchmark=args.benchmark, outfile=args.output)
else:
    raise NotImplementedError(f'Too many inputs.')

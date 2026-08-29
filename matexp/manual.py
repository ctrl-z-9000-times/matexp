from matexp import main_manual, LinearInput, LogarithmicInput
import argparse
import sys

parser = argparse.ArgumentParser(prog='ame-manual',
        description='Solves Markov kinetic models for NEURON mechanisms')
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
        help="degrees Celsius, default: 37")
sim.add_argument('-p', '--polynomial', type=str, required=True,
        help="polynomial form, ex: v^2+v+1")
sim.add_argument('-s', '--samples', type=int, default=100,
        help="minimum sample count safety factor")
inputs = parser.add_argument_group('input specification')
inputs.add_argument('-i', '--input', action='append', default=[],
        nargs=4, metavar=('NAME', 'MIN', 'MAX', 'BINS'),
        help="input name, bounds, and number of partitions")
inputs.add_argument('--log', nargs=2, action='append', default=[],
        metavar=('INPUT', 'SCALE'),
        help="scale input logarithmically, for chemical concentrations")
computer = parser.add_argument_group('computer specification')
computer.add_argument('--target', choices=['host','cuda'], default='host',
        help="default: host")

if __name__.endswith('__main__') or __name__ == 'matexp.manual':
    args = parser.parse_args()

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
            args.polynomial, args.samples, target=args.target,
            outfile=args.output, verbose=args.verbose)

    _placeholder = lambda: None # Symbol for the CLI script to import and call.


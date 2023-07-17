# matexp

This program solves systems of differential equations for the NEURON simulator
using the matrix-exponential method of integration. This is a new method of
integration. The solution is faster and more accurate than NEURONs built
in "sparse" solver. This method is only applicable to systems which are linear
and time-invariant, such as Markov kinetic models. This method is also limited
to systems with one or two inputs.

This program uses the
[NMODL file format](https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/nmodl.html)
(".mod" files). 
The input kinetic model is an NMODL file, and the solution is written to a new NMODL file.

For more information about how this program works see [DETAILS.md](./DETAILS.md).


### Installation

Prerequisites:
* Compiler for the target system.
    + CPU requires `g++`
    + Cuda requires the `cupy` python package.

```
$ pip install matexp
```


### Usage

```
$ matexp --help
usage: matexp [-h] [-v] [--plot] -t TIME_STEP [-c CELSIUS] [-e ERROR]
              [-i NAME MIN MAX] [--log [INPUT]] [--target {host,cuda}]
              [-f {32,64}]
              INPUT_PATH OUTPUT_PATH

positional arguments:
  INPUT_PATH            input filename for the unsolved NMODL file
  OUTPUT_PATH           output filename for the solution

options:
  -h, --help            show this help message and exit
  -v, --verbose         print diagnostic information, give twice for trace mode
  --plot                show the propagator matrix

simulation parameters:
  -t TIME_STEP, --time_step TIME_STEP
  -c CELSIUS, --celsius CELSIUS
                        default: 37°
  -e ERROR, --error ERROR
                        maximum error per time step. default: 10^-4

input specification:
  -i NAME MIN MAX, --input NAME MIN MAX
  --log [INPUT]         scale input logarithmically, for chemical concentrations

computer specification:
  --target {host,cuda}  default: host
  -f {32,64}, --float {32,64}
                        default: 64

```


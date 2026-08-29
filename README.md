# Approximate Matrix Exponential

The `ame` program solves systems of differential equations for the NEURON simulator
using the approximate matrix exponential method of integration. This is a new
method of integration. The solution is faster and more accurate than NEURONs
built in "sparse" solver. This method is only applicable to systems which are
linear and time-invariant, such as Markov kinetic models. This method is also
limited to systems with one or two inputs.

This program uses the
[NMODL file format](https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/nmodl.html)
(".mod" files). 
The input kinetic model is an NMODL file, and the solution is written to a new NMODL file.


## Installation

Prerequisites:
* Linux
* The `g++` compiler

```
$ pip install matexp
```


## Usage

```
$ ame --help
usage: ame [-h] [-v] [-dt TIME_STEP] [-t TEMPERATURE] [-e ERROR]
           [-i NAME MIN MAX] [--log [INPUT]] [--target {host,cuda}]
           INPUT_PATH OUTPUT_PATH

Solves Markov kinetic models for NEURON mechanisms

positional arguments:
  INPUT_PATH            input path for unsolved NMODL file
  OUTPUT_PATH           output path for solved NMODL file

options:
  -h, --help            show this help message and exit
  -v, --verbose         print diagnostic information, give twice for trace mode

simulation parameters:
  -dt TIME_STEP, --time_step TIME_STEP
                        milliseconds, default: 0.025
  -t TEMPERATURE, --temperature TEMPERATURE
                        degrees celsius, default: 37
  -e ERROR, --error ERROR
                        maximum absolute error per millisecond. default: 10^-3

input specification:
  -i NAME MIN MAX, --input NAME MIN MAX
  --log [INPUT]         scale input logarithmically, for chemical concentrations

computer specification:
  --target {host,cuda}  default: host
```

### Manually Specifying Approximations

```
$ ame-manual --help
usage: ame-manual [-h] [-v] [-dt TIME_STEP] [-t TEMPERATURE] -p POLYNOMIAL
                  [-s SAMPLES] [-i NAME MIN MAX BINS] [--log INPUT SCALE]
                  [--target {host,cuda}]
                  INPUT_PATH OUTPUT_PATH

Solves Markov kinetic models for NEURON mechanisms

positional arguments:
  INPUT_PATH            path of unsolved NMODL file
  OUTPUT_PATH           path for solved NMODL file

options:
  -h, --help            show this help message and exit
  -v, --verbose         print diagnostic information, give twice for trace mode

simulation parameters:
  -dt TIME_STEP, --time_step TIME_STEP
                        milliseconds, default: 0.025
  -t TEMPERATURE, --temperature TEMPERATURE
                        degrees celsius, default: 37
  -p POLYNOMIAL, --polynomial POLYNOMIAL
                        polynomial form, ex: v^2+v+1
  -s SAMPLES, --samples SAMPLES
                        minimum sample count safety factor

input specification:
  -i NAME MIN MAX BINS, --input NAME MIN MAX BINS
                        input name, bounds, and number of partitions
  --log INPUT SCALE     scale input logarithmically, for chemical concentrations

computer specification:
  --target {host,cuda}  default: host
```

# The Modified Matrix Exponential Method

Written by David McDougall, 2023


## Abstract

This document introduces a new method for solving a certain class of equations
which commonly occur in science and engineering. The accompanying
program, "matexp", implements it in the context of neuroscience simulations
where it is useful for simulating kinetic models such as of ion channels. The
new method is an extension of the matrix exponential method, which finds exact
solutions for systems of ordinary differential equations which are linear and
time-invariant. The new method builds upon the prior art by allowing the linear
coefficients to change over time, with the caveat that they vary much slower
than the time step of numeric integration. The new method is approximate but
has an arbitrary level of accuracy and can be significantly faster to compute
than existing methods.


## Introduction

Linear Time-Invariant systems have exact solutions, using the matrix-exponential
function. The result is a "propagator" matrix which advances the state of the
system by a fixed time step. To advance the state simply multiply the
propagator matrix and the current state vector. For more information see
(Rotter and Diesmann, 1999). However computing the matrix exponential can be
difficult. And what's worse: the matrix is a function of the inputs to the
system so naively it needs to be computed at run time. The matexp program
solves this problem by computing the solution for every possible input ahead of
time and storing them in a specialized look up table.


## Methods

The matexp program computes the propagator matrix for every possible input
value. It then reduces all of those exact solutions into an approximation which
is optimized for both speed and accuracy.

The approximation is structured as a piecewise-polynomial:
1. The input space is divided into uniformly spaced bins.
2. Each bin contains a polynomial approximation of the matrix.  
   All polynomials have the same degree, just with different coefficients.

The optimization proceeds as follows:

1. Start with an initial configuration, which consists of a polynomial degree
and a number of input bins.  

2. Determine the number of input bins which yields the target accuracy.  
   The accuracy is directly proportional to the number of input bins.

3. Measure the speed performance of the configuration by running a benchmark.

4. Experiment with different polynomials to find the fastest configuration which
meets the target accuracy.  
Use a simple hill-climbing procedure to find the first local maxima of
performance.

At run time the integration proceeds as follows:
1. The input values are transformed into the index of their corresponding bin.
2. The polynomials stored in that bin are loaded from memory.
3. The location of the inputs within the bin are scaled into the range [0, 1] for numeric stability.
4. Compute the propagator matrix by evaluating the polynomials with the scaled input values.
5. Advance the state vector by multiplying it by the propagator matrix.


## Results

### The matexp program

The matexp program implements the modified matrix exponential method for
[the NEURON simulator](https://www.neuron.yale.edu/neuron/).
The NEURON simulator uses the NMODL file format for the user
to describe their kinetic models. The matexp program reads the user's NMODL
file, solves the equations in it, and then generates a new NMODL file for the
NEURON simulator to use. The matexp program embeds the "C" source code for the
solver and the pre-computed look-up table inside of a "VERBATIM" block.

The matexp program itself is written in python.
It is hosted on [PyPI](https://pypi.org/project/matexp/)
and can be installed via `pip install matexp`.  
The source code is available on github.com at
<https://github.com/ctrl-z-9000-times/matexp>
under the MIT license.


### Comparison with the Backward Euler Method

The prior state of the art method for solving this type of problem is the
Backward Euler method. The NEURON simulator's "sparse" solver method is in fact
an alias for the Backward Euler method. I chose three kinetic models to
evaluate and compare the accuracy and performance characteristics of the matrix
exponential and the backward Euler methods. The models are of a voltage gated
sodium channel (Nav11), an AMPA receptor, and an NMDA receptor.

| Model         | Inputs | Number of States | ModelDB Accession |
|---------------|:------:|:----------------:|:-------------------:|
| Nav11 Channel | voltage            |  6 | [230137](http://modeldb.yale.edu/230137) |
| AMPA Receptor | glutamate          | 13 | [266925](http://modeldb.yale.edu/266925) |
| NMDA Receptor | voltage, glutamate | 10 | [50207](http://modeldb.yale.edu/50207) |

To see how the length of the integration time step affects the accuracy,
I measured the accuracy of both methods with a variety of different time step
lengths. Since the matrix exponential method allows for an arbitrary level of
accuracy, for this test I set the maximum allowable error to one part in ten
thousand (1e-4) which is the default value for the matexp program. I simulated
one thousand instances of each of the three kinetic models for ten seconds,
using all combinations of methods and time steps. Each instance started in a
random valid state and the inputs were a series of random numbers. The accuracy
was quantified as the root-mean-square (RMS) of the error of the final state of
each instance, as measured across the entire population of instances. The exact
solutions were found using the matexp program with the maximum error set to one
part per billion (1e-9) except for the NMDA receptor which used a maximum error
of one part per million (1e-6) for speed performance reasons.

![Figure 1: Accuracy vs Time Step](backward_euler/Accuracy_Comparison.png)

Notice that the backwards Euler and matrix exponential methods have opposite
reactions to changes in the length of the time step. The backwards Euler method
becomes less accurate as it integrates over longer time steps, but the matrix
exponential method actually becomes more accurate as the time step increases.
This is because the matrix exponential method introduces a roughly constant
amount of error on each time step, regardless of the length of the time step,
and so a few long time steps are more accurate than many short time steps.

---

To measure the run time speed performance I generated ten thousand instances of
each model and simulated them for two hundred iterations on my CPU. The fastest
iteration was considered representative of the potential speed of the method
and the run times of the other 199 iterations were discarded. Many iterations
were interrupted by CPU task switches which invalidated those measurements.

![Figure 2: Real Time to Integrate](backward_euler/Speed_Comparison.png)

---

Although the modified matrix exponential method allows for arbitrary levels of
accuracy, there is a trade-off between accuracy and speed. The accuracy of the
method is proportional to the size of the pre-computed look-up table, and
loading large amounts of data from memory is slow. Ameliorate this issue by
loading the look-up table into the CPU's data cache and then processing all
instances of the model in large batches. The following graph was generated
using ten thousand instances of each model.

![Figure 3: Simulation Speed vs Accuracy](backward_euler/Speed-Accuracy_Trade-off.png)

Notice that the Nav11 channel and AMPA receptor have a roughly linear trade-off
between speed and accuracy, but the NMDA receptor has a quadratic relationship
between speed and accuracy. This is because the internal look-up table has one
dimension for each input, and because the NMDA receptor has two inputs the
look-up table is two dimensional and scales quadratically. This is why the
matexp program cannot handle models with three or more inputs.


## References

* Exact digital simulation of time-invariant linear systems with applications
  to neuronal modeling.  
  S. Rotter, M. Diesmann (1999)  
  https://doi.org/10.1007/s004220050570

* How to expand NEURON's library of mechanisms.  
  The NEURON Book  
  N. T. Carnevale, M. L. Hines (2006)  
  https://doi.org/10.1017/CBO9780511541612.010

* "MATEXP" A general purpose digital computer program for solving ordinary
  differential equations by the matrix exponential method.  
  S. J. Ball, R. K. Adams (1967)  
  https://doi.org/10.2172/4147077


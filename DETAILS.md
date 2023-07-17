# matexp

Written by David McDougall, 2023

### Abstract

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


### Introduction

Linear Time-Invariant systems have exact solutions, using the matrix-exponential
function. The result is a "propagator" matrix which advances the state of the
system by a fixed time step. To advance the state simply multiply the
propagator matrix and the current state vector. For more information see
(Rotter and Diesmann, 1999). However computing the matrix exponential can be
difficult. And what's worse: the matrix is a function of the inputs to the
system so naively it needs to be computed at run time. The matexp program
solves this problem by computing the solution for every possible input ahead of
time and storing them in a specialized look up table.


### Methods

The matexp program computes the propagator matrix for every possible input
value. It then reduces all of those exact solutions into an approximation which
is optimized for both speed and accuracy.

The approximation is structured as a piecewise-polynomial:
1. The input space is divided into evenly spaced bins.
2. Each bin contains a polynomial approximation of the function.  
   All polynomial have the same degree, just with different coefficients.

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


### References

* Exact digital simulation of time-invariant linear systems with applications
  to neuronal modeling.  
  S. Rotter, M. Diesmann (1999).  
  https://doi.org/10.1007/s004220050570

* How to expand NEURON's library of mechanisms.  
  The NEURON Book  
  N. T. Carnevale, M. L. Hines (2006)  
  https://doi.org/10.1017/CBO9780511541612.010

* "MATEXP" A general purpose digital computer program for solving ordinary
  differential equations by the matrix exponential method.  
  S. J. Ball, R. K. Adams (1967)  
  https://doi.org/10.2172/4147077


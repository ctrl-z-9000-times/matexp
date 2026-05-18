from .approx import MatrixSamples, Approx, Approx1D, Approx2D
from .codegen import Codegen
from .inputs import LinearInput, LogarithmicInput
from .polynomial import PolynomialForm
import collections.abc
import math
import numpy as np

class Parameters:
    def __init__(self, optimizer, num_buckets, polynomial, verbose=False):
        self.optimizer  = optimizer
        self.model      = optimizer.model
        self.verbose    = bool(verbose)
        if isinstance(num_buckets, collections.abc.Iterable):
            self.num_buckets = tuple(round(x) for x in num_buckets)
        else:
            self.num_buckets = (round(num_buckets),)
        self.set_num_buckets()
        # Make num_buckets1 & num_buckets2 aliases.
        for inp_idx, buckets in enumerate(self.num_buckets):
            setattr(self, f'num_buckets{inp_idx+1}', buckets)
        self.polynomial = PolynomialForm(self.model.inputs, polynomial)
        self.order      = self.polynomial.degree
        if self.verbose:
            print(f'Trying polynomial ({self.polynomial}) bins {self.num_buckets}')
        if   self.model.num_inputs == 1: ApproxClass = Approx1D
        elif self.model.num_inputs == 2: ApproxClass = Approx2D
        self.approx = ApproxClass(self.optimizer.samples, self.polynomial)
        self.num_samples = len(self.optimizer.samples)
        # Measure the accuracy of these parameters.
        self.rmse       = self.approx.rmse
        self.error      = self.approx.measure_residual_error()
        if self.verbose:
            status = 'PASS' if self.error <= self.optimizer.max_error else 'FAIL'
            print(f'Error: {self.error}  \t{status}\n')

    def __str__(self):
        s = str(self.approx)
        s += f"# Samples:    {self.num_samples}\n"
        s += f"Error:        {self.error}\n"
        if hasattr(self, "runtime"):
            s += f"Run speed:    {round(self.runtime, 2)}  ns/Δt\n"
        return s

    def set_num_buckets(self):
        """ Fixup shared data structures. """
        for inp, buckets in zip(self.model.inputs, self.num_buckets):
            inp.set_num_buckets(buckets)

    def benchmark(self):
        if hasattr(self, 'runtime'):
            return
        if self.verbose: print(f"Benchmarking polynomial ({self.polynomial}) bins {self.num_buckets}")
        self.set_num_buckets()
        self.backend = Codegen(
                self.approx,
                self.optimizer.target)
        from . import _measure_speed # Import just before using to avoid circular imports.
        self.runtime = _measure_speed(
                self.backend.load(),
                self.model.num_states,
                self.model.inputs,
                self.model.conserve_sum,
                self.backend.target)
        self.table_size = self.approx.table.nbytes
        if self.verbose: print(f"Result: {round(self.runtime, 3)} ns/Δt\n")

class Optimizer:
    def __init__(self, model, max_error, target, verbose=False):
        self.model          = model
        self.max_error      = float(max_error)
        model.target_error  = self.max_error
        self.target         = target
        self.verbose        = bool(verbose)
        self.samples        = MatrixSamples(self.model, self.verbose)
        self.best           = None # This attribute will store the optimized parameters.
        assert 0.0 < self.max_error
        if self.verbose: print()

    def init_num_buckets(self):
        raise TypeError('abstract method called')

    def init_polynomial(self):
        raise TypeError('abstract method called')

    def run(self):
        # Initial parameters, starting point for iterative search.
        num_buckets = self.init_num_buckets()
        polynomial  = self.init_polynomial()
        # Run the optimization routines.
        self._optimize_log_scale(num_buckets, polynomial)
        self._optimize_polynomial(num_buckets, polynomial)
        # Re-make the final product using all available samples.
        if self.best.num_samples < len(self.samples):
            if self.verbose: print('Remaking best approximation with more samples ...\n')
            self.best = Parameters(self, self.best.num_buckets, self.best.polynomial)
            self.best.benchmark()

    def _optimize_log_scale(self, num_buckets: [int], polynomial):
        if not any(isinstance(inp, LogarithmicInput) for inp in self.model.inputs):
            return
        if self.verbose: print('Optimizing logarithmic scale ...')
        scales, errors = self._eval_log_scale(num_buckets, polynomial, 1e-9, 50)
        # Use the global minima of RMSE.
        argmin     = np.argmin(errors)
        min_error  = errors[argmin]
        best_scale = scales[argmin]
        # Check if the error is still decreasing at the edges of the sample space.
        assert min_error < errors[0] and min_error < errors[-1], "failed to find log scale"
        # Apply the new log scale.
        for inp in self.model.inputs:
            if isinstance(inp, LogarithmicInput):
                inp.set_scale(best_scale)
        if self.verbose: print(f'Optimial logarithmic scale = {best_scale}\n')

    def _eval_log_scale(self, num_buckets, polynomial, min_scale, num_scales):
        # Find the logarithmic input.
        log_inp = [inp for inp in self.model.inputs if isinstance(inp, LogarithmicInput)]
        assert len(log_inp) == 1, "multiple logarithmic inputs not supported"
        log_inp = log_inp[0]

        # Initialize the input's num_buckets and scale parameters.
        for inp, buckets in zip(self.model.inputs, num_buckets):
            if isinstance(inp, LinearInput):
                inp.set_num_buckets(buckets)
            elif isinstance(inp, LogarithmicInput):
                inp.set_num_buckets(buckets, scale=min_scale)

        # Search the range [min_scale, log_inp.maximum] at exponentially increasing intervals.
        search_space = log_inp.sample_space(num_scales + 1)[1:] # Do not try scale=zero

        # Collect all of the samples before building any polynomials.
        for scale in search_space:
            log_inp.set_scale(scale)
            Approx(self.samples, polynomial)._ensure_enough_exact_samples()

        if   self.model.num_inputs == 1: ApproxClass = Approx1D
        elif self.model.num_inputs == 2: ApproxClass = Approx2D

        # Measure the error associated with each scale parameter.
        if self.verbose: print(f'Evaluating {num_scales} scales', end='', flush=True)
        rss_error = []
        # absmax_error = []
        for scale in search_space:
            log_inp.set_scale(scale)
            approx = ApproxClass(self.samples, polynomial)
            rss_error.append(approx.rmse)
            # absmax_error.append(approx.measure_residual_error())
            if self.verbose: print('.', end='', flush=True)
        if self.verbose: print()
        return search_space, rss_error

    def _optimize_polynomial(self, num_buckets, polynomial):
        cursor = self._optimize_num_buckets(num_buckets, polynomial)
        cursor.benchmark()
        results = {cursor.polynomial: cursor}
        while True:
            queue = cursor.polynomial.suggest_add() + cursor.polynomial.suggest_remove()
            for polynomial in queue:
                if polynomial in results:
                    continue
                try:
                    new = self._optimize_num_buckets(cursor.num_buckets, polynomial, cursor.runtime)
                except RuntimeError as error_message:
                    if self.verbose: print(f'Aborting polynomial ({polynomial}) {error_message}')
                    continue
                new.benchmark()
                results[new.polynomial] = new
            self.best = min(results.values(), key=lambda x: x.runtime)
            if cursor is self.best:
                self.best.set_num_buckets()
                return
            else:
                cursor = self.best
                if self.verbose:
                    print(f'New best: polynomial ({self.best.polynomial}) bins {self.best.num_buckets}\n')

class Optimize1D(Optimizer):
    def init_num_buckets(self):
        return [20]

    def init_polynomial(self):
        return 3

    def _optimize_num_buckets(self, num_buckets, polynomial, max_runtime=None):
        (num_buckets,) = num_buckets
        cursor = Parameters(self, num_buckets, polynomial, self.verbose)
        min_buckets = 1
        max_buckets = None
        while cursor.error > self.max_error:
            # Terminate early if it's slower than max_runtime.
            if max_runtime is not None: # and cursor.num_buckets1 > 1000:
                cursor.benchmark()
                if cursor.runtime > max_runtime:
                    if self.verbose: print(f'Aborting Polynomial ({cursor.polynomial}) runs too slow.\n')
                    return cursor # It's ok to return invalid results BC they won't be used.
            # 
            min_buckets = cursor.num_buckets1 + 1
            # Heuristics to guess new num_buckets.
            delta = .5
            increase = lambda x: max(x + 1, x * (1 + delta))
            new = Parameters(self, increase(cursor.num_buckets1), polynomial, self.verbose)
            # Check that the error is decreasing monotonically.
            if new.error < cursor.error:
                cursor = new
            else:
                # Remake the cursor with more samples and recheck.
                cursor = Parameters(self, cursor.num_buckets, cursor.polynomial, self.verbose)
                if new.error < cursor.error:
                    cursor = new
                else:
                    raise RuntimeError("Failed to reach target accuracy (check model stability).")
        max_buckets = cursor.num_buckets1
        max_buckets_parameters = cursor
        # Bisect search the range min_buckets to max_buckets.
        while min_buckets < max_buckets:
            midpoint = int((min_buckets + max_buckets) / 2)
            cursor = Parameters(self, midpoint, polynomial, self.verbose)
            if cursor.error > self.max_error:
                min_buckets = cursor.num_buckets1 + 1
            else:
                max_buckets = cursor.num_buckets1
                max_buckets_parameters = cursor
        return max_buckets_parameters

class Optimize2D(Optimizer):
    def init_num_buckets(self):
        return [20, 20]

    def init_polynomial(self):
        return [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [3, 0], [0, 3]]

    def _optimize_num_buckets(self, num_buckets, polynomial, max_runtime=None):
        cache = {}
        def make_parameters(num_buckets):
            num_buckets = tuple(num_buckets)
            if num_buckets not in cache:
                cache[num_buckets] = Parameters(self, num_buckets, polynomial, self.verbose)
            return cache[num_buckets]
        def remake_parameters(num_buckets):
            num_buckets = tuple(num_buckets)
            del cache[num_buckets]
            return make_parameters(num_buckets)
        # Increase the number of input partitions until it exceeds the target accuracy.
        # Then reduce the increment and re-run from the last failed parameters, until the
        # increment reaches one.
        num_buckets = [1, 1]
        cursor = make_parameters(num_buckets) 
        if self.verbose: print(f'Starting cursor {cursor.polynomial} bins {cursor.num_buckets}')
        delta = .5
        increase = lambda x: max(x + 1, round(x * (1 + delta)))
        iteration = 0
        while True:
            # Terminate early if it's already slower than max_runtime.
            if iteration == 0 and max_runtime is not None and np.prod(cursor.num_buckets) > 1000:
                cursor.benchmark()
                if cursor.runtime > max_runtime:
                    if self.verbose: print(f'Aborting Polynomial ({cursor.polynomial}), runs too slow.\n')
                    return cursor # It's ok to return invalid results BC they won't be used.
            # Try increasing num_buckets in both dimensions in isolation.
            b1, b2 = cursor.num_buckets
            step_size = max(increase(b1) - b1, increase(b2) - b2)
            A = make_parameters([increase(b1), b2])
            B = make_parameters([b1, increase(b2)])
            # Take whichever experiment yielded better results.
            new = min([A, B], key=lambda p: p.error)
            # Check that the error is decreasing monotonically.
            if new.error >= cursor.error:
                # Remake the both parameters with more samples and recheck
                cursor = remake_parameters(cursor.num_buckets)
                new    = remake_parameters(new.num_buckets)
                if new.error >= cursor.error:
                    raise RuntimeError("Failed to reach target accuracy (check model stability).")
            # Advance the cursor
            if new.error > self.max_error:
                cursor = new
                if self.verbose: print(f'New cursor {cursor.polynomial} bins {cursor.num_buckets}')
            elif step_size <= 1 or iteration >= 10:
                return new
            else:
                delta /= 2
                iteration += 1
                if self.verbose: print(f'Iteration {iteration}, reduce delta to {delta}')

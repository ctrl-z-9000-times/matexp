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
                self.optimizer.float_dtype,
                self.optimizer.target)
        from . import _measure_speed # Import just before using to avoid circular imports.
        self.runtime = _measure_speed(
                self.backend.load(),
                self.model.num_states,
                self.model.inputs,
                self.model.conserve_sum,
                self.optimizer.float_dtype,
                self.optimizer.target)
        self.table_size = self.approx.table.nbytes
        if self.verbose: print(f"Result: {round(self.runtime, 3)} ns/Δt\n")

class Optimizer:
    def __init__(self, model, max_error, float_dtype, target, verbose=False):
        self.model          = model
        self.max_error      = float(max_error)
        model.target_error  = self.max_error
        self.float_dtype    = float_dtype
        self.target         = target
        self.verbose        = bool(verbose)
        self.samples        = MatrixSamples(self.model, self.verbose)
        self.best           = None # This attribute will store the optimized parameters.
        assert 0.0 < self.max_error < 1.0
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
        scales, errors = self._eval_log_scale(num_buckets, polynomial, 1e-9, 100)
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
        if self.verbose: print(f'Evaluating {num_scales} scales ...')
        errors = []
        for scale in search_space:
            log_inp.set_scale(scale)
            approx = ApproxClass(self.samples, polynomial)
            errors.append(approx.rmse)
        return search_space, errors

    def _optimize_polynomial(self, num_buckets, polynomial):
        self.best = self._optimize_num_buckets(num_buckets, polynomial)
        self.best.benchmark()
        experiments = {self.best.polynomial}
        # Try removing terms from the polynomial.
        experiment_queue = self.best.polynomial.suggest_remove()
        while experiment_queue:
            polynomial = experiment_queue.pop()
            if polynomial in experiments:
                continue
            experiments.add(polynomial)
            try:
                new = self._optimize_num_buckets(self.best.num_buckets, polynomial, self.best.runtime)
            except RuntimeError as error_message:
                if self.verbose: print(f'Aborting polynomial ({polynomial}) {error_message}')
                continue
            new.benchmark()
            if new.runtime < self.best.runtime:
                self.best = new
                if self.verbose: print(f'New best: polynomial ({self.best.polynomial}) bins {self.best.num_buckets}\n')
                experiment_queue = self.best.polynomial.suggest_remove()
        # Try adding more terms to the polynomial.
        experiment_queue = self.best.polynomial.suggest_add()
        while experiment_queue:
            polynomial = experiment_queue.pop()
            if polynomial in experiments:
                continue
            experiments.add(polynomial)
            new = self._optimize_num_buckets(self.best.num_buckets, polynomial, self.best.runtime)
            new.benchmark()
            if new.runtime < self.best.runtime:
                self.best = new
                if self.verbose: print(f'New best: polynomial ({self.best.polynomial}) bins {self.best.num_buckets}\n')
                experiment_queue = self.best.polynomial.suggest_add()
        self.best.set_num_buckets()

class Optimize1D(Optimizer):
    def init_num_buckets(self):
        return [20]

    def init_polynomial(self):
        return 3

    def _optimize_num_buckets(self, num_buckets, polynomial, max_runtime=None):
        (num_buckets,) = num_buckets
        cursor = Parameters(self, num_buckets, polynomial, self.verbose)
        min_buckets = 1
        # Quickly increase the num_buckets until it exceeds the target accuracy.
        while cursor.error > self.max_error:
            # Terminate early if it's slower than max_runtime.
            if max_runtime is not None: # and cursor.num_buckets1 > 1000:
                cursor.benchmark()
                if cursor.runtime > max_runtime:
                    if self.verbose: print(f'Aborting Polynomial ({cursor.polynomial}) runs too slow.\n')
                    return cursor # It's ok to return invalid results BC they won't be used.
            min_buckets = cursor.num_buckets1
            # Heuristics to guess new num_buckets.
            orders_of_magnitude = math.log(cursor.error / self.max_error, 10)
            pct_incr = max(1.5, 1.7 ** orders_of_magnitude)
            num_buckets = num_buckets * pct_incr
            new = Parameters(self, num_buckets, polynomial, self.verbose)
            # Check that the error is decreasing monotonically.
            if new.error < cursor.error:
                cursor = new
            else:
                raise RuntimeError("Failed to reach target accuracy.")
        # Slowly reduce the num_buckets until it fails to meet the target accuracy.
        while True:
            num_buckets *= 0.9
            if num_buckets <= min_buckets:
                break
            new = Parameters(self, num_buckets, polynomial, self.verbose)
            if new.error > self.max_error:
                break
            else:
                cursor = new
        return cursor

class Optimize2D(Optimizer):
    def init_num_buckets(self):
        return [20, 20]

    def init_polynomial(self):
        return [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [3, 0], [0, 3]]

    def _optimize_num_buckets(self, num_buckets, polynomial, max_runtime=None):
        cursor = Parameters(self, num_buckets, polynomial, self.verbose)
        # Quickly increase the num_buckets until it exceeds the target accuracy.
        increase = lambda x: x * 1.50
        while cursor.error > self.max_error:
            # Terminate early if it's already slower than max_runtime.
            if max_runtime is not None and np.product(cursor.num_buckets) > 1000:
                cursor.benchmark()
                if cursor.runtime > max_runtime:
                    if self.verbose: print(f'Aborting Polynomial ({cursor.polynomial}), runs too slow.\n')
                    return cursor # It's ok to return invalid results BC they won't be used.
            # Try increasing num_buckets in both dimensions in isolation.
            if self.verbose: print(f'Increasing {self.model.input1.name} bins:')
            A = Parameters(self, [increase(cursor.num_buckets1), cursor.num_buckets2], polynomial, self.verbose)
            if self.verbose: print(f'Increasing {self.model.input2.name} bins:')
            B = Parameters(self, [cursor.num_buckets1, increase(cursor.num_buckets2)], polynomial, self.verbose)
            # Take whichever experiment yielded better results. If they both
            # performed about the same then take both modifications.
            pct_diff = 2 * (A.error - B.error) / (A.error + B.error)
            thresh   = .25
            if pct_diff < -thresh:
                new = A
                if self.verbose: print(f'Taking increased {self.model.input1.name} bins.\n')
            elif pct_diff > thresh:
                new = B
                if self.verbose: print(f'Taking increased {self.model.input2.name} bins.\n')
            else:
                if self.verbose: print('Taking the increases in both dimensions:')
                new = Parameters(self, [increase(cursor.num_buckets1), increase(cursor.num_buckets2)],
                                polynomial, self.verbose)
            # Check that the error is decreasing monotonically.
            if new.error < cursor.error:
                cursor = new
            else:
                raise RuntimeError("Failed to reach target accuracy.")
        # Slowly reduce the num_buckets until it fails to meet the target accuracy.
        decrease = lambda x: x * .90
        while True:
            # Try decreasing num_buckets in both dimensions in isolation.
            if self.verbose: print(f'Decreasing {self.model.input1.name} bins.')
            A = Parameters(self, [decrease(cursor.num_buckets1), cursor.num_buckets2], polynomial, self.verbose)
            if self.verbose: print(f'Decreasing {self.model.input2.name} bins.')
            B = Parameters(self, [cursor.num_buckets1, decrease(cursor.num_buckets2)], polynomial, self.verbose)
            new = min(A, B, key=lambda p: p.error)
            if new.error > self.max_error:
                break
            else:
                cursor = new
        return cursor

from .approx import MatrixSamples, Approx1D, Approx2D
from .codegen import Codegen
from .inputs import LinearInput, LogarithmicInput
from .polynomial import PolynomialForm
import lti_sim
import collections.abc
import math
import numpy as np

class Parameters:
    def __init__(self, optimizer, num_buckets, polynomial):
        self.optimizer  = optimizer
        self.model      = optimizer.model
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
        if   self.model.num_inputs == 1: ApproxClass = Approx1D
        elif self.model.num_inputs == 2: ApproxClass = Approx2D
        self.approx = ApproxClass(self.optimizer.samples, self.polynomial)
        # Measure the accuracy of these parameters.
        self.rmse       = self.approx.rmse
        self.error      = self.approx.measure_error()
        self.max_rmse   = np.max(self.rmse)
        self.max_error  = np.max(self.error)

    def __str__(self):
        s = str(self.approx)
        try:
            s += f"Run speed:     {round(self.runtime, 2)}  ns/Δt\n"
        except AttributeError: pass
        s += f"# Samples:    {len(self.optimizer.samples)}\n"
        s += f"RMS Error:    {self.max_rmse}\n"
        if self.max_error > self.optimizer.accuracy:
            status = '- failed!'
        else:
            status = ''
        s += f"Max |Error|:  {self.max_error}  {status}\n"
        return s

    def set_num_buckets(self):
        """ Fixup shared data structures. """
        for inp, buckets in zip(self.model.inputs, self.num_buckets):
            inp.set_num_buckets(buckets)

    def benchmark(self):
        if hasattr(self, 'runtime'):
            return
        self.set_num_buckets()
        self.backend = Codegen(
                self.approx,
                self.optimizer.float_dtype,
                self.optimizer.target)
        self.runtime = lti_sim._measure_speed(
                self.backend.load(),
                self.model.num_states,
                self.model.inputs,
                self.model.conserve_sum,
                self.optimizer.float_dtype,
                self.optimizer.target)
        self.table_size = self.approx.table.nbytes

class Optimizer:
    def __init__(self, model, accuracy, float_dtype, target):
        self.model          = model
        self.accuracy       = float(accuracy)
        self.float_dtype    = float_dtype
        self.target         = target
        self.samples        = MatrixSamples(self.model)
        self.best           = None # This attribute will store the optimized parameters.
        assert 0.0 < self.accuracy < 1.0

class Optimize1D(Optimizer):
    def __init__(self, model, accuracy, float_dtype, target):
        super().__init__(model, accuracy, float_dtype, target)
        self.input1 = self.model.input1
        # Initial parameters, starting point for iterative search.
        num_buckets = 10
        polynomial  =  3
        # Run the optimization routines.
        if isinstance(self.input1, LogarithmicInput):
            self._optimize_log_scale(num_buckets, polynomial)
        self._optimize_polynomial(num_buckets, polynomial)

    def _optimize_log_scale(self, num_buckets, order):
        self.input1.set_num_buckets(num_buckets, scale=1.0)
        cursor = Parameters(self, num_buckets, order)
        while np.argmax(cursor.rmse) == 0:
            self.input1.set_num_buckets(num_buckets, self.input1.scale / 10)
            cursor = Parameters(self, num_buckets, order)

    def _optimize_polynomial(self, num_buckets, order):
        cursor = self._optimize_num_buckets(num_buckets, order)
        cursor.benchmark()
        experiments = [cursor]
        # Decrease the order until it slows down as a result.
        low_cursor = cursor
        while low_cursor.order - 1 >= 0:
            new = self._optimize_num_buckets(low_cursor.num_buckets1, low_cursor.order - 1,
                                             max_runtime = low_cursor.runtime)
            new.benchmark()
            experiments.append(new)
            if new.runtime > low_cursor.runtime:
                break
            else:
                low_cursor = new
        local_minima_found = (len(experiments) >= 3)
        if not local_minima_found:
            # Increase the order until it slows down as a result.
            high_cursor = cursor
            while True:
                new = self._optimize_num_buckets(high_cursor.num_buckets1, high_cursor.order + 1,
                                                 max_runtime = high_cursor.runtime)
                new.benchmark()
                experiments.append(new)
                if new.runtime > high_cursor.runtime:
                    break
                else:
                    high_cursor = new
        # 
        self.best = min(experiments, key=lambda parameters: parameters.runtime)
        self.best.set_num_buckets()

    def _optimize_num_buckets(self, num_buckets, order, max_runtime=None):
        cursor = Parameters(self, num_buckets, order)
        min_buckets = 1
        # Quickly increase the num_buckets until it exceeds the target accuracy.
        while cursor.max_error > self.accuracy:
            # Terminate early if it's slower than max_runtime.
            if max_runtime is not None and cursor.num_buckets1 > 1000:
                cursor.benchmark()
                if cursor.runtime > max_runtime:
                    return cursor # It's ok to return invalid results BC they won't be used.
            min_buckets = cursor.num_buckets1
            # Heuristics to guess new num_buckets.
            orders_of_magnitude = math.log(cursor.max_error / self.accuracy, 10)
            pct_incr = max(1.5, 1.7 ** orders_of_magnitude)
            num_buckets = num_buckets * pct_incr
            new = Parameters(self, num_buckets, order)
            if new.max_rmse < cursor.max_rmse:
                cursor = new
            else:
                raise RuntimeError("Failed to reach target accuracy.")
        # Slowly reduce the num_buckets until it fails to meet the target accuracy.
        while True:
            num_buckets *= 0.9
            if num_buckets <= min_buckets:
                break
            new = Parameters(self, num_buckets, order)
            if new.max_error > self.accuracy:
                break
            else:
                cursor = new
        return cursor

class Optimize2D(Optimizer):
    def __init__(self, model, accuracy, float_dtype, target):
        super().__init__(model, accuracy, float_dtype, target)
        if any(isinstance(inp, LogarithmicInput) for inp in self.model.inputs):
            self._optimize_log_scale()
        self._optimize_polynomial()

    def _optimize_log_scale(self):
        # Use a very small and simple approximation for optimizing the log
        # scale, since the goal is not to solve the problem but rather the goal
        # is to identify the complicated areas of the function and scale them
        # away from the edge of the view.
        num_buckets = [10, 10]
        polynomial  = [[0, 0], [1, 0], [0, 1], [2, 0], [0, 2], [3, 0], [0, 3],]
        # Initialize the input's num_buckets and scale parameters.
        for inp, buckets in zip(self.model.inputs, num_buckets):
            if   isinstance(inp, LinearInput):      inp.set_num_buckets(buckets)
            elif isinstance(inp, LogarithmicInput): inp.set_num_buckets(buckets, scale=1.0)
        # Reduce the scale parameter until the buckets containing zero no longer
        # have the largest errors.
        done = False
        while not done:
            cursor      = Parameters(self, num_buckets, polynomial)
            error_dim0  = np.max(cursor.rmse, axis=1)
            error_dim1  = np.max(cursor.rmse, axis=0)
            done = True
            for inp, errors in zip(self.model.inputs, (error_dim0, error_dim1)):
                if not isinstance(inp, LogarithmicInput):
                    continue
                if np.argmax(errors) == 0:
                    inp.set_num_buckets(inp.num_buckets, inp.scale / 10)
                    done = False

    def _optimize_polynomial(self):
        # Start with a medium sized polynomial.
        self.best = self._optimize_num_buckets([
                [0, 0],
                [1, 0], [0, 1],
                [2, 0], [1, 1], [0, 2],
                [3, 0], [0, 3]])
        self.best.benchmark()
        experiments = {self.best.polynomial}
        # Try removing terms from the polynomial.
        experiment_queue = self.best.polynomial.suggest_remove()
        while experiment_queue:
            polynomial = experiment_queue.pop()
            if polynomial in experiments:
                continue
            experiments.add(polynomial)
            new = self._optimize_num_buckets(polynomial, self.best.runtime)
            new.benchmark()
            if new.runtime < self.best.runtime:
                self.best = new
                experiment_queue = self.best.polynomial.suggest_remove()
        # Try adding more terms to the polynomial.
        experiment_queue = self.best.polynomial.suggest_add()
        while experiment_queue:
            polynomial = experiment_queue.pop()
            if polynomial in experiments:
                continue
            experiments.add(polynomial)
            new = self._optimize_num_buckets(polynomial, self.best.runtime)
            new.benchmark()
            if new.runtime < self.best.runtime:
                self.best = new
                experiment_queue = self.best.polynomial.suggest_add()
        self.best.set_num_buckets()

    def _optimize_num_buckets(self, polynomial, max_runtime=None):
        cursor = Parameters(self, [10, 10], polynomial)
        # Slowly increase the num_buckets until it exceeds the target accuracy.
        increase = lambda x: max(x + 5, x * 1.10)
        while cursor.max_error > self.accuracy:
            # Terminate early if it's already slower than max_runtime.
            if max_runtime is not None and np.product(cursor.num_buckets) > 1000:
                cursor.benchmark()
                if cursor.runtime > max_runtime:
                    return cursor # It's ok to return invalid results BC they won't be used.
            # Try increasing num_buckets in both dimensions and take which ever
            # experiment yields better results.
            A = Parameters(self, [increase(cursor.num_buckets1), cursor.num_buckets2], polynomial)
            B = Parameters(self, [cursor.num_buckets1, increase(cursor.num_buckets2)], polynomial)
            new = min(A, B, key=lambda p: p.max_rmse)
            if new.max_rmse < cursor.max_rmse:
                cursor = new
            else:
                raise RuntimeError("Failed to reach target accuracy.")
        return cursor

from .codegen import Codegen1D
from .inputs import LinearInput, LogarithmicInput
from .tables import Exact1D, Approx1D
import lti_sim
import math
import numpy as np

class Optimize1D:
    def __init__(self, nmodl_info, time_step, accuracy, float_dtype, target):
        self.nmodl_info     = nmodl_info
        self.time_step      = float(time_step)
        self.accuracy       = float(accuracy)
        self.float_dtype    = float_dtype
        self.target         = target
        assert 0.0 < self.accuracy < 1.0
        # Initial parameters, starting point for iterative search.
        self.num_buckets = 10
        self.order       = 4
        # Initialize the table of exact matrix data.
        if isinstance(self.nmodl_info.input1, LinearInput):
            self.nmodl_info.input1.set_num_buckets(self.num_buckets)
            self._setup_exact()
        elif isinstance(self.nmodl_info.input1, LogarithmicInput):
            self.log_scale = 1.0
            self.nmodl_info.input1.set_num_buckets(self.num_buckets, self.log_scale)
            self._setup_exact()
            self._optimize_log_scale()
        self._optimize_polynomial_order() # Run the main optimization routine.
        self.nmodl_info.input1.set_num_buckets(self.num_buckets) # Fixup shared data structures.

    def _setup_exact(self):
        self.exact = Exact1D(self.time_step,
                             self.nmodl_info.derivative,
                             self.nmodl_info.state_names,
                             self.nmodl_info.input1)

    def _optimize_log_scale(self):
        approx = Approx1D(self.exact, order=self.order)
        error  = approx.measure_error()
        while np.argmax(error) == 0:
            self.log_scale /= 10.0
            self.nmodl_info.input1.set_num_buckets(self.num_buckets, self.log_scale)
            self._setup_exact()
            approx = Approx1D(self.exact, order=self.order)
            error  = approx.measure_error()

    def _optimize_polynomial_order(self):
        cursor = self._optimize_num_buckets(self.num_buckets, self.order)
        cursor.benchmark()
        experiments = [cursor]
        # Decrease the order until it slows down as a result.
        low_cursor = cursor
        while low_cursor.order - 1 >= 0:
            new = self._optimize_num_buckets(low_cursor.num_buckets, low_cursor.order - 1,
                                             max_runtime = low_cursor.runtime)
            new.benchmark()
            experiments.append(new)
            if new.runtime > low_cursor.runtime:
                break
            else:
                low_cursor = new
        local_minima_found = (len(experiments) >= 3)
        # Increase the order until it slows down as a result.
        if not local_minima_found:
            high_cursor = cursor
            while True:
                new = self._optimize_num_buckets(high_cursor.num_buckets, high_cursor.order + 1,
                                                 max_runtime = high_cursor.runtime)
                new.benchmark()
                experiments.append(new)
                if new.runtime > high_cursor.runtime:
                    break
                else:
                    high_cursor = new
        # Save the results.
        fastest = min(experiments, key=lambda parameters: parameters.runtime)
        self.order          = fastest.order
        self.num_buckets    = fastest.num_buckets
        self.approx         = fastest.approx
        self.backend        = fastest.backend
        self.benchmark      = (fastest.runtime, fastest.runtime_std)

    def _optimize_num_buckets(self, num_buckets, order, max_runtime=None):
        cursor = Parameters1D(self, num_buckets, order)
        min_buckets = 1
        # Quickly increase the num_buckets until it exceeds the target accuracy.
        while cursor.max_error > self.accuracy:
            # Terminate early if it exceeds max_runtime. It's ok to return
            # invalid results BC they won't be used.
            if max_runtime is not None and cursor.num_buckets > 1000:
                cursor.benchmark()
                if cursor.runtime > max_runtime:
                    return cursor
            min_buckets = cursor.num_buckets
            orders_of_magnitude = math.log(cursor.max_error / self.accuracy, 10)
            pct_incr = max(1.5, 1.7 ** orders_of_magnitude)
            num_buckets = num_buckets * pct_incr
            new = Parameters1D(self, num_buckets, order)
            if new.max_error < cursor.max_error:
                cursor = new
            else:
                raise RuntimeError("Failed to reach target accuracy.")
        # Slowly reduce the num_buckets until it fails to meet the target accuracy.
        while True:
            num_buckets *= 0.9
            if num_buckets <= min_buckets:
                break
            new = Parameters1D(self, num_buckets, order)
            if new.max_error > self.accuracy:
                break
            else:
                cursor = new
        return cursor

class Parameters1D:
    def __init__(self, optimizer, num_buckets, order):
        self.optimizer      = optimizer
        self.nmodl_info     = optimizer.nmodl_info
        self.num_buckets    = round(num_buckets)
        self.order          = int(order)
        self.nmodl_info.input1.set_num_buckets(self.num_buckets)
        self.approx         = Approx1D(self.optimizer.exact, order=self.order)
        self.error          = self.approx.measure_error()
        self.max_error      = np.max(self.error)

    def print(self):
        print("Buckets: ", self.num_buckets)
        print("Order:   ", self.order)
        print("Accuracy:", self.max_error)
        try:
            print("Runtime: ", round(self.runtime, 2), '+/-', round(self.runtime_std, 2))
            print("Table sz:", round(self.table_size / 1000), 'kbytes')
        except AttributeError: pass
        print(flush=True)

    def benchmark(self):
        if hasattr(self, 'runtime'): return
        self.nmodl_info.input1.set_num_buckets(self.num_buckets)
        self.backend = Codegen1D(
                self.nmodl_info.name,
                self.approx,
                self.nmodl_info.conserve_sum,
                self.optimizer.float_dtype,
                self.optimizer.target)
        mean, std = lti_sim._measure_speed(
                self.backend.load(),
                self.nmodl_info.num_states,
                self.nmodl_info.input1,
                self.nmodl_info.conserve_sum,
                self.optimizer.float_dtype,
                self.optimizer.target)
        self.runtime = mean
        self.runtime_std = std
        self.table_size = self.approx.table.nbytes

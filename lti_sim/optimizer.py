from .inputs import LinearInput, LogarithmicInput
from .tables import Exact1D, Approx1D
import math
import numpy as np

class Optimize1D:
    def __init__(self, time_step, deriv, state_names, input1, order, accuracy):
        self.time_step      = time_step
        self.deriv          = deriv
        self.state_names    = state_names
        self.input1         = input1
        self.order          = int(order)
        self.accuracy       = float(accuracy)
        assert 0.0 < self.accuracy < 1.0
        # Initial parameters, starting point for iterative search.
        self.num_buckets = 10
        # Initialize the table of exact matrix data.
        if isinstance(self.input1, LinearInput):
            self.input1.set_num_buckets(self.num_buckets)
            self._setup_exact()
        elif isinstance(self.input1, LogarithmicInput):
            self.log_scale = 1.0
            self.input1.set_num_buckets(self.num_buckets, self.log_scale)
            self._setup_exact()
            self._optimize_log_scale()
        # Run the main optimization routine.
        self.approx = self._optimize_num_buckets(self.num_buckets, self.order)

    def _setup_exact(self):
        self.exact = Exact1D(self.time_step, self.deriv, self.state_names, self.input1)

    def _optimize_log_scale(self):
        approx = Approx1D(self.exact, order=self.order)
        error  = approx.measure_error()
        while np.argmax(error) == 0:
            self.log_scale /= 10.0
            self.input1.set_num_buckets(self.num_buckets, self.log_scale)
            self._setup_exact()
            approx = Approx1D(self.exact, order=self.order)
            error  = approx.measure_error()

    def _optimize_num_buckets(self, num_buckets, order):
        cursor = Parameters1D(self, num_buckets, order)
        min_buckets = 0
        # Quickly increase the num_buckets until it exceeds the target accuracy.
        while cursor.max_error >= self.accuracy:
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
            if new.max_error >= self.accuracy:
                break
            else:
                cursor = new
        self.input1.set_num_buckets(cursor.num_buckets) # Fixup the shared data structure.
        return cursor.approx

class Parameters1D:
    def __init__(self, optimizer, num_buckets, order):
        self.num_buckets    = round(num_buckets)
        self.order          = int(order)
        optimizer.input1.set_num_buckets(self.num_buckets)
        self.approx         = Approx1D(optimizer.exact, order=self.order)
        self.error          = self.approx.measure_error()
        self.max_error      = np.max(self.approx.measure_error())

    def print(self):
        print("Buckets: ", self.num_buckets)
        print("Order:   ", self.order)
        print("Accuracy:", self.max_error)
        print()

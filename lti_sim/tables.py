from numpy.polynomial.polynomial import Polynomial
import numpy as np
import scipy.linalg

from .inputs import Input, LinearInput, LogarithmicInput

class Exact1D:
    def __init__(self, time_step, derivative, state_names, input1, num_samples=1000):
        self.time_step      = float(time_step)
        self.derivative     = derivative
        self.state_names    = state_names
        self.num_states     = len(self.state_names)
        self.input1         = input1
        assert isinstance(self.input1, Input)
        self.set_num_samples(num_samples)

    def set_num_samples(self, num_samples):
        self.num_samples    = int(num_samples)
        self.input1_values  = self.input1.sample_space(self.num_samples)
        self.data_samples   = np.empty([self.num_samples, self.num_states, self.num_states])
        for input1_index, input1_value in enumerate(self.input1_values):
            self.data_samples[input1_index, :, :] = self.make_matrix(input1_value)

    def make_matrix(self, input1, time_step=None):
        input1 = float(input1)
        if time_step is None:
            time_step = self.time_step
        A = np.empty([self.num_states, self.num_states])
        for col in range(self.num_states):
            state = [float(x == col) for x in range(self.num_states)]
            A[:, col] = self.derivative(input1, *state)
        matrix = scipy.linalg.expm(A * time_step)
        for col in range(self.num_states):
            matrix[:, col] *= 1.0 / sum(matrix[:, col].flat)
        return matrix

    def get_samples_in_bucket(self, bucket_index):
        index1_scalar   = (self.num_samples - 1) / self.input1.num_buckets
        index1_lower    = int(bucket_index) * index1_scalar
        index1_upper    = index1_lower + index1_scalar
        sample_slice1   = slice(round(index1_lower), round(index1_upper) + 1)
        sample_input1   = self.input1_values[sample_slice1]
        sample_data     = self.data_samples[sample_slice1,:,:]
        return (sample_input1, sample_data)

    def initial_state(self, conserve_sum=None):
        if conserve_sum is None:
            return np.zeros(self.num_states)
        else:
            conserve_sum = float(conserve_sum)
            valid_state  = np.full(self.num_states, conserve_sum / self.num_states)
            time_step    = 3600e3 # 1 hour in milliseconds.
            matrix       = self.make_matrix(self.input1.initial, time_step)
            return matrix.dot(valid_state)

class Approx1D:
    def __init__(self, exact, order):
        self.exact = exact
        assert isinstance(self.exact, Exact1D)
        self.input1         = self.exact.input1
        self.state_names    = self.exact.state_names
        self.num_states     = self.exact.num_states
        self.time_step      = self.exact.time_step
        self.order          = int(order)
        assert self.order >= 0
        self._make_table()

    def _make_table(self):
        # First make sure there are enough samples of exact data.
        samples_per_bucket = self.exact.num_samples / self.input1.num_buckets
        if samples_per_bucket < 10 + self.order + 1:
            self.exact.set_num_samples(self.input1.num_buckets * 50)
        # 
        self.table = np.empty([self.input1.num_buckets, self.num_states, self.num_states, self.order + 1])
        for bucket_index in range(self.input1.num_buckets):
            input1_values, exact_data = self.exact.get_samples_in_bucket(bucket_index)
            # Scale the inputs into the range [0,1].
            input1_locations = []
            for value in input1_values:
                _, loc = self.input1.get_bucket_location(value, bucket_index)
                input1_locations.append(loc)
            # Make an approximation for each entry in the matrix.
            for row in range(self.num_states):
                for col in range(self.num_states):
                    data_entry = exact_data[:, row, col]
                    poly = Polynomial.fit(input1_locations, data_entry, self.order, domain=[])
                    self.table[bucket_index, row, col, :] = poly.coef

    def approximate_matrix(self, input1):
        if input1 == self.input1.maximum:
            bucket_index    = self.input1.num_buckets - 1
            bucket_location = 1.0
        else:
            bucket_index, bucket_location = self.input1.get_bucket_location(input1)
        basis = np.array([bucket_location ** power for power in range(self.order + 1)])
        approx_matrix = np.empty([self.num_states, self.num_states])
        for row in range(self.num_states):
            for col in range(self.num_states):
                coef = self.table[bucket_index, row, col, :]
                approx_matrix[row, col] = coef.dot(basis)
        return approx_matrix

    def measure_error(self):
        max_err = np.zeros(self.input1.num_buckets)
        for bucket in range(self.input1.num_buckets):
            for input1_value, exact_matrix in zip(*self.exact.get_samples_in_bucket(bucket)):
                approx_matrix   = self.approximate_matrix(input1_value)
                approx_matrix  -= exact_matrix
                np.abs(approx_matrix, out=approx_matrix)
                max_err[bucket] = max(max_err[bucket], np.max(approx_matrix))
        return max_err

    def plot(self, name=""):
        import matplotlib.pyplot as plt
        input1_values = self.exact.input1_values
        exact  = self.exact.data_samples
        approx = np.empty([len(input1_values), self.num_states, self.num_states])
        for index, value in enumerate(input1_values):
            approx[index, :, :] = self.approximate_matrix(value)
        plt.figure(name + " Transfer Function, dt = %g"%self.time_step)
        for row_idx, row in enumerate(self.state_names):
            for col_idx, col in enumerate(self.state_names):
                plt.subplot(self.num_states, self.num_states, row_idx*self.num_states + col_idx + 1)
                plt.title(col + " -> " + row)
                if isinstance(self.input1, LinearInput):
                    plt.plot(input1_values, exact[:, row_idx, col_idx], color='k')
                    plt.plot(input1_values, approx[:, row_idx, col_idx], color='r')
                elif isinstance(self.input1, LogarithmicInput):
                    plt.semilogx(input1_values, exact[:, row_idx, col_idx], color='k')
                    plt.semilogx(input1_values, approx[:, row_idx, col_idx], color='r')
                if self.num_states < 10: # Otherwise there is not enough room on the figure.
                    plt.xlabel(self.input1.name, labelpad=1.0)
                for input1_value in self.input1.sample_space(self.input1.num_buckets + 1):
                    plt.axvline(input1_value)
        x = .05
        plt.subplots_adjust(left=x, bottom=x, right=1-x, top=1-x, wspace=0.6, hspace=1.0)
        plt.show()

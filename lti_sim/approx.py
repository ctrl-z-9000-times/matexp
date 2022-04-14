from .inputs import LinearInput, LogarithmicInput
from .polynomial import PolynomialForm
import math
import numpy as np

class MatrixSamples:
    def __init__(self, model):
        self.model   = model
        self.inputs  = [np.empty(0) for _ in range(model.num_inputs)]
        self.samples = np.empty((0, model.num_states, model.num_states))

    def _get_bucket_shape(self):
        return tuple(inp.num_buckets for inp in self.model.inputs)

    def _get_bucket_indices(self):
        return tuple(np.array(inp.get_bucket_value(data), dtype=np.int32)
                    for inp, data in zip(self.model.inputs, self.inputs))

    def _get_bucket_flat_indices(self):
        bucket_shape   = self._get_bucket_shape()
        bucket_indices = self._get_bucket_indices()
        return np.ravel_multi_index(bucket_indices, bucket_shape)

    def _count_samples_per_bucket(self):
        bucket_shape   = self._get_bucket_shape()
        num_samples    = np.zeros(bucket_shape, dtype=np.int32)
        bucket_indices = self._get_bucket_indices()
        for idx in zip(*bucket_indices):
            num_samples[idx] += 1
        return num_samples

    def sample(self, minimum_samples_per_bucket):
        assert isinstance(minimum_samples_per_bucket, int) and minimum_samples_per_bucket >= 0
        num_samples = self._count_samples_per_bucket()
        # Determine how many samples to add to each bucket.
        num_samples = minimum_samples_per_bucket - num_samples
        np.maximum(num_samples, 0, out=num_samples)
        bucket_indices = np.nonzero(num_samples)
        num_samples    = num_samples[bucket_indices]
        if len(bucket_indices[0]) == 0:
            return # All buckets already have enough data.
        # Generate input values for the new samples.
        cum_samples = np.cumsum(num_samples)
        total_new   = cum_samples[-1]
        sample_buckets = [np.random.uniform(size=total_new) for _ in self.model.inputs]
        for bucket_idx, num, end in zip(zip(*bucket_indices), num_samples, cum_samples):
            for values, idx in zip(sample_buckets, bucket_idx):
                values[end-num:end] += idx
        sample_inputs = [inp.get_input_value(values) for inp, values in zip(self.model.inputs, sample_buckets)]
        # Sample the matrix.
        sample_matrices = []
        for input_value in zip(*sample_inputs):
            matrix = self.model.make_matrix(input_value)
            sample_matrices.append(np.expand_dims(matrix, 0))
        # Add the new samples to the existing data arrays.
        for dim in range(self.model.num_inputs):
            self.inputs[dim] = np.concatenate((sample_inputs[dim], self.inputs[dim]))
        sample_matrices.append(self.samples)
        self.samples = np.concatenate(sample_matrices)

    def sort(self):
        if len(self.samples) == 0: return
        flat_indices = self._get_bucket_flat_indices()
        sort_order   = np.argsort(flat_indices)
        # Apply the new sorted order to the samples.
        for dim in range(self.model.num_inputs):
            self.inputs[dim] = self.inputs[dim][sort_order]
        self.samples = self.samples[sort_order, :, :]

    def __iter__(self):
        self.sort()
        flat_indices = self._get_bucket_flat_indices()
        bucket_shape = self._get_bucket_shape()
        num_buckets  = np.product(bucket_shape)
        slice_bounds = np.nonzero(np.diff(flat_indices, prepend=-1, append=num_buckets))[0]
        inputs  = [[] for _ in range(self.model.num_inputs)]
        samples = []
        for start, end in zip(slice_bounds[:-1], slice_bounds[1:]):
            for dim in range(self.model.num_inputs):
                inputs[dim].append(self.inputs[dim][start : end])
            samples.append(self.samples[start : end, :, :])
        bucket_indices = np.ndindex(bucket_shape)
        return zip(bucket_indices, zip(*inputs), samples)

class Approx:
    """ Abstract base class. """
    def __init__(self, samples, polynomial):
        self.samples        = samples
        self.model          = samples.model
        self.state_names    = self.model.state_names
        self.num_states     = self.model.num_states
        self.polynomial     = PolynomialForm(self.model.inputs, polynomial)
        self.num_terms      = len(self.polynomial)

    def _ensure_enough_exact_samples(self):
        ninp = self.model.num_inputs
        if ninp == 1:
            div = 10
        elif ninp == 2:
            div = 7
        safety_factor = 3
        # order = max(sum(term) + 1 for term in self.polynomial)
        orig = [inp.num_buckets for inp in self.model.inputs]
        for inp in self.model.inputs:
            inp.set_num_buckets(inp.num_buckets * div)
        self.samples.sample(math.ceil(safety_factor * self.polynomial.order / (div ** ninp)))
        for inp, q in zip(self.model.inputs, orig):
            inp.set_num_buckets(q)

    def print_summary(self):
        for inp in self.model.inputs:
            print(f"# {inp.name} bins:".ljust(12), inp.num_buckets)
            if isinstance(inp, LogarithmicInput):
                print(f"{inp.name} log scale:".ljust(12), inp.scale)
        print("Polynomial: ", self.polynomial)
        print("Table size: ", round(self.table.nbytes / 1000), "kB")
        print("Multiplies: ", self._estimate_multiplies(), '(estimated)')

    def _estimate_multiplies(self):
        matrix_size = self.num_states ** 2
        x = 0
        for inp in self.model.inputs:
            if isinstance(inp, LogarithmicInput):
                x += 3 # Compute log2. This is a guess, I have no idea how it's actually implemented.
            x += 1 # Scale the input value into an index.
            x += 1 # Compute the offset into the table.
        x += self.polynomial.num_var_terms # Compute the terms of the polynomial basis.
        x += matrix_size * self.polynomial.num_var_terms # Evaluate the polynomial approximation.
        x += matrix_size # Compute the dot product.
        if self.model.conserve_sum is not None:
            x += self.num_states + 1 # Conserve statement.
        return x

class Approx1D(Approx):
    def __init__(self, samples, order):
        super().__init__(samples, [[power] for power in range(int(order) + 1)])
        self.input1 = self.model.input1
        self._make_table()

    def _make_table(self):
        self._ensure_enough_exact_samples()
        self.table = np.empty([self.input1.num_buckets, self.num_states, self.num_states, self.num_terms])
        for (bucket_index,), (input_values,), exact_data  in self.samples:
            # Scale the inputs into the range [0,1].
            input1_locations = self.input1.get_bucket_value(input_values) - bucket_index
            # Make an approximation for each entry in the matrix.
            A = np.empty([len(input1_locations), self.num_terms])
            A[:, 0] = 1.0
            for power in range(1, self.num_terms):
                A[:, power] = input1_locations ** power
            B = exact_data.reshape(-1, self.num_states**2)
            coef = np.linalg.lstsq(A, B, rcond=None)[0]
            coef = coef.reshape(self.num_terms, self.num_states, self.num_states).transpose(1,2,0)
            self.table[bucket_index, :, :, :] = coef

    def approximate_matrix(self, input1):
        bucket_index, bucket_location = self.input1.get_bucket_location(input1)
        basis = np.array([bucket_location ** power for power in range(self.num_terms)])
        approx_matrix = np.empty([self.num_states, self.num_states])
        for row in range(self.num_states):
            for col in range(self.num_states):
                coef = self.table[bucket_index, row, col, :]
                approx_matrix[row, col] = coef.dot(basis)
        return approx_matrix

    def measure_error(self):
        max_err = np.zeros(self.input1.num_buckets)
        for (bucket,), (input_values,), exact_data  in self.samples:
            for input1_value, exact_matrix in zip(input_values, exact_data):
                approx_matrix   = self.approximate_matrix(input1_value)
                approx_matrix  -= exact_matrix
                np.abs(approx_matrix, out=approx_matrix)
                max_err[bucket] = max(max_err[bucket], np.max(approx_matrix))
        return max_err

    def plot(self, name=""):
        import matplotlib.pyplot as plt
        input1_values = self.samples.inputs[0]
        exact  = self.samples.samples
        approx = np.empty([len(input1_values), self.num_states, self.num_states])
        for index, value in enumerate(input1_values):
            approx[index, :, :] = self.approximate_matrix(value)
        fig_title = name + " Transfer Function, Δt = %g ms"%self.model.time_step
        plt.figure(fig_title)
        if self.num_states < 10: # Otherwise there is not enough room on the figure.
            plt.suptitle(fig_title)
        # TODO: If there are many plots (>100) then only show the x&y axis
        # labels on the bottom & left side of page.
        for row_idx, row in enumerate(self.model.state_names):
            for col_idx, col in enumerate(self.model.state_names):
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
                    # TODO: Should there be a ylabel? "Percent" or "Fraction"
                if self.input1.num_buckets < 100: # Otherwise the plots become unreadable.
                    # TODO: Consider making an option to control whether the vertical lines are shown?
                    for input1_value in self.input1.sample_space(self.input1.num_buckets + 1):
                        plt.axvline(input1_value)
        x = .05
        plt.subplots_adjust(left=x, bottom=x, right=1-x, top=1-x, wspace=0.6, hspace=1.0)
        plt.show()

class Approx2D(Approx):
    def __init__(self, samples, polynomial):
        super().__init__(samples, polynomial)
        self.input1 = self.model.input1
        self.input2 = self.model.input2
        self._make_table()

    def _make_table(self):
        self._ensure_enough_exact_samples()
        self.table = np.empty([self.input1.num_buckets, self.input2.num_buckets, self.num_states, self.num_states, self.num_terms])
        for (bucket_index1, bucket_index2,), (input1_values, input2_values,), exact_data  in self.samples:
            # Scale the inputs into the range [0,1].
            input1_locations = self.input1.get_bucket_value(input1_values) - bucket_index1
            input2_locations = self.input2.get_bucket_value(input2_values) - bucket_index2
            # Make an approximation for each entry in the matrix.
            A = np.empty([len(input1_values), self.num_terms])
            for term, (power1, power2) in enumerate(self.polynomial.terms):
                A[:, term] = (input1_locations ** power1) * (input2_locations ** power2)
            B = exact_data.reshape(-1, self.num_states**2)
            coef = np.linalg.lstsq(A, B, rcond=None)[0]
            coef = coef.reshape(self.num_terms, self.num_states, self.num_states).transpose(1,2,0)
            self.table[bucket_index1, bucket_index2, :, :, :] = coef

    def approximate_matrix(self, input1, input2):
        bucket1_index, bucket1_location = self.input1.get_bucket_location(input1)
        bucket2_index, bucket2_location = self.input2.get_bucket_location(input2)
        basis = np.empty(self.num_terms)
        for term, (power1, power2) in enumerate(self.polynomial.terms):
            basis[term] = (bucket1_location ** power1) * (bucket2_location ** power2)
        approx_matrix = np.empty([self.num_states, self.num_states])
        for row in range(self.num_states):
            for col in range(self.num_states):
                coef = self.table[bucket1_index, bucket2_index, row, col, :]
                approx_matrix[row, col] = coef.dot(basis)
        return approx_matrix

    def measure_error(self):
        max_err = np.zeros([self.input1.num_buckets, self.input2.num_buckets])
        for (bucket1, bucket2,), (input1_values, input2_values,), exact_data  in self.samples:
            for inp1, inp2, mat in zip(input1_values, input2_values, exact_data):
                approx_matrix   = self.approximate_matrix(inp1, inp2)
                approx_matrix  -= mat
                np.abs(approx_matrix, out=approx_matrix)
                max_err[bucket1, bucket2] = max(max_err[bucket1, bucket2], np.max(approx_matrix))
        return max_err

    def plot(self, name=""):
        import matplotlib.pyplot as plt
        fig_title = name + " Transfer Function, Δt = %g ms"%self.model.time_step
        plt.figure(fig_title)
        if self.num_states < 10: # Otherwise there is not enough room on the figure.
            plt.suptitle(fig_title)
        for row_idx, row in enumerate(self.model.state_names):
            for col_idx, col in enumerate(self.model.state_names):
                ax = plt.subplot(self.num_states, self.num_states, row_idx*self.num_states + col_idx + 1)
                plt.title(col + " -> " + row)
                tfer_exact = self.data_samples[:, :, row_idx, col_idx]
                im = ax.pcolormesh(self.input1_values, self.input2_values, tfer_exact.T)
                plt.colorbar(im)
                ax.set_xlim(0.0001, self.input1.maximum)
                ax.set_ylim(self.input2.minimum, self.input2.maximum)
                if isinstance(self.input1, LogarithmicInput):
                    ax.set_xscale('log')
                if isinstance(self.input2, LogarithmicInput):
                    ax.set_yscale('log')
                # TODO: put labels at the bottom and left edge.
                # plt.xlabel()
                # plt.ylabel()
        x = .05
        plt.subplots_adjust(left=x, bottom=x, right=1-x, top=1-x, wspace=0.6, hspace=1.0)
        plt.show()

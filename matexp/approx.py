from .inputs import LinearInput, LogarithmicInput
from .polynomial import PolynomialForm
import math
import numpy as np
import scipy.stats

class MatrixSamples:
    def __init__(self, model, verbose=False):
        self.model   = model
        self.verbose = bool(verbose)
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
        # Make parallel arrays: bucket_indices & num_samples.
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
        inputs = [inp.get_input_value(values) for inp, values in zip(self.model.inputs, sample_buckets)]
        # Sample the matrix.
        if self.verbose: print(f'Collecting {total_new} matrix samples ... ', end='', flush=True)
        matrices = self.model.make_matrix(inputs)
        # Append the new samples to the existing data arrays.
        for dim in range(self.model.num_inputs):
            self.inputs[dim] = np.concatenate((self.inputs[dim], inputs[dim]))
        self.samples = np.concatenate([self.samples, matrices])
        if self.verbose: print('done')

    def sort(self):
        if len(self.samples) == 0: return
        flat_indices = self._get_bucket_flat_indices()
        sort_order   = np.argsort(flat_indices)
        # Apply the new sorted order to the samples.
        for dim in range(self.model.num_inputs):
            self.inputs[dim] = self.inputs[dim][sort_order]
        self.samples = self.samples[sort_order, :, :]

    def __iter__(self):
        """ Yields triples of (bucket_indices, input_values, samples) """
        self.sort()
        flat_indices = self._get_bucket_flat_indices()
        bucket_shape = self._get_bucket_shape()
        num_buckets  = np.prod(bucket_shape)
        slice_bounds = np.nonzero(np.diff(flat_indices, prepend=-1, append=num_buckets))[0]
        inputs  = [[] for _ in range(self.model.num_inputs)]
        samples = []
        for start, end in zip(slice_bounds[:-1], slice_bounds[1:]):
            for dim in range(self.model.num_inputs):
                inputs[dim].append(self.inputs[dim][start : end])
            samples.append(self.samples[start : end, :, :])
        bucket_indices = np.ndindex(bucket_shape)
        return zip(bucket_indices, zip(*inputs), samples)

    def __len__(self):
        return len(self.samples)

class Approx:
    """ Abstract base class. """
    def __init__(self, samples, polynomial):
        self.samples        = samples
        self.model          = samples.model
        self.state_names    = self.model.state_names
        self.num_states     = self.model.num_states
        self.polynomial     = PolynomialForm(self.model.inputs, polynomial)
        self.num_terms      = len(self.polynomial)
        self.num_buckets    = tuple(inp.num_buckets for inp in self.model.inputs)

    def set_num_buckets(self):
        """
        Set num_buckets to this approximation's values, needed because the
        inputs data structures are shared by all approximations.
        """
        for inp, num_buckets in zip(self.model.inputs, self.num_buckets):
            inp.set_num_buckets(num_buckets)

    def _ensure_enough_exact_samples(self, safety_factor=100):
        samples_per_bucket = safety_factor * self.polynomial.num_terms
        # Divide the input space into many more buckets to ensure that the
        # samples are uniformly spaced within each bucket.
        subdivisions = math.ceil(samples_per_bucket ** (1 / self.model.num_inputs))
        for inp in self.model.inputs:
            inp.set_num_buckets(inp.num_buckets * subdivisions)
        # 
        self.samples.sample(1)
        # Restore the original bucket dimensions.
        self.set_num_buckets()

    def measure_error(self):
        self.samples = MatrixSamples(self.model, self.samples.verbose)
        self._ensure_enough_exact_samples()
        return self.measure_residual_error()

    def measure_residual_error(self):
        max_abs_error = 0
        self.set_num_buckets()
        for (bucket_indices, input_values, samples) in self.samples:
            for input_vector, exact in zip(zip(*input_values), samples):
                approx = self.approximate_matrix(*input_vector)
                # Increase the timestep to 1 ms
                approx = np.linalg.matrix_power(approx, round(1 / self.model.time_step))
                exact  = np.linalg.matrix_power(exact, round(1 / self.model.time_step))
                # Find the max-abs-diff between the approx and exact samples.
                max_abs_error += np.max(np.abs(approx - exact))
        return max_abs_error

    def __str__(self):
        s = ''
        self.set_num_buckets()
        for inp in self.model.inputs:
            s += f'{inp.name} # bins:'.ljust(14) + str(inp.num_buckets) + '\n'
            if isinstance(inp, LogarithmicInput):
                s += f'{inp.name} log scale:'.ljust(14) + str(inp.scale) + '\n'
        s += (f'Polynomial:   {self.polynomial}\n'
              f'Table size:   {round(self.table.nbytes / 1000)} kB\n'
              f'Multiplies:   {self._estimate_multiplies()} (estimated)\n')
        return s

    def _estimate_multiplies(self):
        matrix_size = self.num_states ** 2
        x = 0
        for inp in self.model.inputs:
            if isinstance(inp, LogarithmicInput):
                x += 4 # log2
            x += 1 # Scale the input value into an index.
            x += 1 # Compute the offset into the table.
        x += self.polynomial.num_var_terms # Compute the terms of the polynomial basis.
        x += matrix_size * self.polynomial.num_var_terms # Evaluate the polynomial approximation.
        x += matrix_size # Compute the dot product.
        if self.model.conserve_sum is not None:
            x += self.num_states + 1 # Conserve statement.
        return x

class Approx1D(Approx):
    def __init__(self, samples, polynomial):
        super().__init__(samples, polynomial)
        self.input1 = self.model.input1
        self._ensure_enough_exact_samples()
        self._make_table()

    def _polynomial_basis(self, input1_locations, num_terms):
        # Make an approximation for each entry in the matrix.
        A = np.empty([len(input1_locations), num_terms])
        A[:, 0] = 1.0
        for power in range(1, num_terms):
            A[:, power] = input1_locations ** power
        return A

    def _make_table(self):
        self.table = np.empty([self.input1.num_buckets, self.num_states, self.num_states, self.num_terms])
        rss_sum = 0
        for (bucket_index,), (input_values,), exact_data in self.samples:
            # Scale the inputs into the range [0,1].
            input1_locations = self.input1.get_bucket_value(input_values) - bucket_index
            A = self._polynomial_basis(input1_locations, self.num_terms)
            B = exact_data.reshape(-1, self.num_states**2)
            coef, rss = np.linalg.lstsq(A, B, rcond=None)[:2]
            coef = coef.reshape(self.num_terms, self.num_states, self.num_states).transpose(1,2,0)
            self.table[bucket_index, :, :, :] = coef
            rss_sum += np.sum(rss)
        self.rmse = (rss_sum / self.num_states**2 / len(self.samples)) ** .5

    def approximate_matrix(self, input1):
        self.set_num_buckets()
        bucket_index, bucket_location = self.input1.get_bucket_location(input1)
        basis = np.array([bucket_location ** power for power in range(self.num_terms)])
        coef  = self.table[bucket_index].reshape(-1, self.num_terms)
        return coef.dot(basis).reshape(self.num_states, self.num_states)

class Approx2D(Approx):
    def __init__(self, samples, polynomial):
        super().__init__(samples, polynomial)
        self.input1 = self.model.input1
        self.input2 = self.model.input2
        self._ensure_enough_exact_samples()
        self._make_table()

    def _make_table(self):
        self.table = np.empty([self.input1.num_buckets, self.input2.num_buckets,
                                self.num_states, self.num_states, self.num_terms])
        rss_sum = 0
        for (bucket_index1, bucket_index2), (input1_values, input2_values), exact_data in self.samples:
            # Scale the inputs into the range [0,1].
            input1_locations = self.input1.get_bucket_value(input1_values) - bucket_index1
            input2_locations = self.input2.get_bucket_value(input2_values) - bucket_index2
            # Make an approximation for each entry in the matrix.
            A = np.empty([len(input1_values), self.num_terms])
            for term, (power1, power2) in enumerate(self.polynomial.terms):
                A[:, term] = (input1_locations ** power1) * (input2_locations ** power2)
            B = exact_data.reshape(-1, self.num_states**2)
            coef, rss = np.linalg.lstsq(A, B, rcond=None)[:2]
            coef = coef.reshape(self.num_terms, self.num_states, self.num_states).transpose(1,2,0)
            self.table[bucket_index1, bucket_index2, :, :, :] = coef
            rss_sum += np.sum(rss)
        self.rmse = (rss_sum / self.num_states**2 / len(self.samples)) ** .5

    def approximate_matrix(self, input1, input2):
        self.set_num_buckets()
        bucket1_index, bucket1_location = self.input1._get_bucket_location_array(input1)
        bucket2_index, bucket2_location = self.input2._get_bucket_location_array(input2)
        basis = np.empty(self.num_terms)
        for term, (power1, power2) in enumerate(self.polynomial.terms):
            basis[term] = (bucket1_location ** power1) * (bucket2_location ** power2)
        coef = self.table[bucket1_index, bucket2_index].reshape(-1, self.num_terms)
        return coef.dot(basis).reshape(self.num_states, self.num_states)

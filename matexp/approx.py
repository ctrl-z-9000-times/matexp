from .inputs import LinearInput, LogarithmicInput
from .polynomial import PolynomialForm
from multiprocessing.shared_memory import SharedMemory
from itertools import pairwise, repeat
import math
import numpy as np

class MatrixSamples:
    def __init__(self, model, verbose=False):
        self.model   = model
        self.verbose = bool(verbose)
        self.inputs  = [np.empty(0) for _ in range(model.num_inputs)]
        self.samples = np.empty((0, model.num_states, model.num_states))
        self.inputs_sm  = [None for _ in range(model.num_inputs)]
        self.samples_sm = None
        self._allocated = 0

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
        # Save the results to shared memory.
        old_samples = len(self)
        total_samples = old_samples + total_new
        num_inputs = self.model.num_inputs
        num_states = self.model.num_states
        samples_shape = (total_samples, num_states, num_states)
        if total_samples > self._allocated:
            # Reallocate into larger arrays.
            inputs_tmp = [buf.copy() for buf in self.inputs]
            samples_tmp = self.samples.copy()
            self._free_sm()
            new_alloc = total_samples * 2
            self.inputs_sm = []
            self.inputs = []
            for dim in range(num_inputs):
                sm = SharedMemory(f'matexp_inputs_{dim}', True, 8 * new_alloc)
                buf = np.ndarray((total_samples,), dtype=np.float64, buffer=sm.buf)
                self.inputs_sm.append(sm)
                self.inputs.append(buf)
                buf[:old_samples] = inputs_tmp[dim]
                buf[old_samples:] = inputs[dim]
            samples_bytes = 8 * new_alloc * num_states * num_states
            self.samples_sm = SharedMemory('matexp_samples', True, samples_bytes)
            self.samples = np.ndarray(samples_shape, dtype=np.float64, buffer=self.samples_sm.buf)
            self.samples[:old_samples, :, :] = samples_tmp
            self.samples[old_samples:, :, :] = matrices
        else:
            # Append to the existing arrays.
            self.inputs = []
            for sm, new_data in zip(self.inputs_sm, inputs):
                buf = np.ndarray((total_samples,), dtype=np.float64, buffer=sm.buf)
                buf[old_samples:] = new_data
                self.inputs.append(buf)
            self.samples = np.ndarray(samples_shape, dtype=np.float64, buffer=self.samples_sm.buf)
            self.samples[old_samples:, :, :] = matrices
        if self.verbose: print('done')

    def sort(self):
        if len(self.samples) == 0: return
        flat_indices = self._get_bucket_flat_indices()
        sort_order   = np.argsort(flat_indices)
        # Apply the new sorted order to the samples.
        for dim in range(self.model.num_inputs):
            self.inputs[dim][:] = self.inputs[dim][sort_order]
        self.samples[:, :, :] = self.samples[sort_order, :, :]

    def __iter__(self):
        """ Yields pairs of (bucket_indices, data_range) """
        self.sort()
        flat_indices = self._get_bucket_flat_indices()
        bucket_shape = self._get_bucket_shape()
        num_buckets  = np.prod(bucket_shape)
        slice_bounds = np.nonzero(np.diff(flat_indices, prepend=-1, append=num_buckets))[0]
        bucket_indices = np.ndindex(bucket_shape)
        return zip(bucket_indices, pairwise(slice_bounds))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _get_sm_weakref(num_inputs):
        inputs_sm = []
        for dim in range(num_inputs):
            inputs_sm.append(SharedMemory(f'matexp_inputs_{dim}', False))
        samples_sm = SharedMemory('matexp_samples', False)
        return (inputs_sm, samples_sm)

    def _free_sm(self):
        for sm in self.inputs_sm:
            if sm is not None:
                sm.close()
                sm.unlink()
        self.inputs_sm = [None for _ in self.inputs_sm]
        if self.samples_sm is not None:
            self.samples_sm.close()
            self.samples_sm.unlink()
            self.samples_sm = None

    def __del__(self):
        self._free_sm()

_table_name_autoinc = 0

class Approx:
    """ Abstract base class. """
    def __init__(self, samples, polynomial):
        self.table_sm       = None
        self.samples        = samples
        self.model          = samples.model
        self.state_names    = self.model.state_names
        self.num_states     = self.model.num_states
        self.polynomial     = PolynomialForm(self.model.inputs, polynomial)
        self.num_terms      = len(self.polynomial)
        self.num_buckets    = tuple(inp.num_buckets for inp in self.model.inputs)

    def _alloc_table(self):
        global _table_name_autoinc
        table_shape = self.num_buckets + (self.num_states, self.num_states, self.num_terms)
        self.table_name = f"matexp_approx_{_table_name_autoinc}"
        self.table_sm  = SharedMemory(self.table_name, True, 8 * np.prod(table_shape))
        self.table = np.ndarray(table_shape, dtype=np.float64, buffer=self.table_sm.buf)
        _table_name_autoinc += 1

    def _make_table(self):
        from . import _thread_pool
        args = zip(repeat(self.table_name),
                    *(repeat(inp) for inp in self.model.inputs),
                    repeat(self.model.num_states),
                    repeat(self.polynomial),
                    repeat(len(self.samples)),
                    iter(self.samples))
        rss_sum = sum(map(self._table_kernel, args)) # Single threaded
        # rss_sum = sum(_thread_pool.map(self._table_kernel, list(args), chunksize=1)) # Multithreaded
        self.rmse = (rss_sum / self.num_states**2 / len(self.samples)) ** .5

    def __del__(self):
        if self.table_sm is not None:
            self.table_sm.close()
            self.table_sm.unlink()

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
        self.set_num_buckets()
        self.samples = MatrixSamples(self.model, self.samples.verbose)
        self._ensure_enough_exact_samples()
        return self.measure_residual_error()

    def measure_residual_error(self):
        self.set_num_buckets()
        power = round(1 / self.model.time_step)
        from . import _thread_pool # Lazy import to avoid circular dependency.
        args = zip(repeat(self.table_name),
                    repeat(power),
                    repeat(self.model.inputs),
                    repeat(self.polynomial),
                    repeat(self.num_states),
                    repeat(len(self.samples)),
                    self.samples)
        # return max(map(self._error_kernel, args)) # Single threaded
        return max(_thread_pool.map(self._error_kernel, list(args), chunksize=1)) # Multithreaded

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
        self._alloc_table()
        self._make_table()

    @staticmethod
    def _polynomial_basis(input1_locations, num_terms):
        # Make an approximation for each entry in the matrix.
        A = np.empty([len(input1_locations), num_terms])
        A[:, 0] = 1.
        for power in range(1, num_terms):
            A[:, power] = input1_locations ** power
        return A

    @staticmethod
    def _table_kernel(args):
        # Unpack the arguments.
        table_name, input1, num_states, polynomial, num_samples, ((bucket_index,), data_range) = args
        num_terms = polynomial.num_terms
        (inputs_sm,), samples_sm = MatrixSamples._get_sm_weakref(1)
        table_sm        = SharedMemory(table_name, False)
        inputs_shape    = (num_samples,)
        samples_shape   = (num_samples, num_states, num_states)
        table_shape     = (input1.num_buckets, num_states, num_states, num_terms)
        inputs_buf      = np.ndarray(inputs_shape, dtype=np.float64, buffer=inputs_sm.buf)
        samples_buf     = np.ndarray(samples_shape, dtype=np.float64, buffer=samples_sm.buf)
        table_buf       = np.ndarray(table_shape, dtype=np.float64, buffer=table_sm.buf)
        # Slice out the current chunk of data.
        inputs_buf  = inputs_buf[data_range[0] : data_range[1]]
        samples_buf = samples_buf[data_range[0] : data_range[1]]
        # Scale the inputs into the range [0,1].
        input1_locations = input1.get_bucket_value(inputs_buf) - bucket_index
        # 
        A = Approx1D._polynomial_basis(input1_locations, num_terms)
        B = samples_buf.reshape(-1, num_states**2)
        coef, rss = np.linalg.lstsq(A, B, rcond=None)[:2]
        coef = coef.reshape(num_terms, num_states, num_states).transpose(1,2,0)
        table_buf[bucket_index, :, :, :] = coef
        return np.sum(rss)

    def approximate_matrix(self, input1):
        num_samples = len(input1)
        self.set_num_buckets()
        bucket_index, bucket_location = self.input1._get_bucket_location_array(input1)
        basis = np.array([bucket_location ** power for power in range(self.num_terms)])
        basis = basis.T.reshape(num_samples, 1, 1, self.num_terms)
        coef  = self.table[bucket_index]
        return np.sum(coef * basis, axis = -1)

    @staticmethod
    def _error_kernel(args):
        # Unpack the arguments.
        (table_name, power, (input1,), polynomial, num_states,
                num_samples, ((bucket_index,), data_range)) = args
        num_terms = polynomial.num_terms
        # Access the shared memory.
        (inputs_sm,), samples_sm = MatrixSamples._get_sm_weakref(1)
        table_sm        = SharedMemory(table_name, False)
        inputs_shape    = (num_samples,)
        samples_shape   = (num_samples, num_states, num_states)
        table_shape     = (input1.num_buckets, num_states, num_states, num_terms)
        inputs_buf      = np.ndarray(inputs_shape, dtype=np.float64, buffer=inputs_sm.buf)
        samples_buf     = np.ndarray(samples_shape, dtype=np.float64, buffer=samples_sm.buf)
        table_buf       = np.ndarray(table_shape, dtype=np.float64, buffer=table_sm.buf)
        # Slice out one chunk of data.
        num_samples = data_range[1] - data_range[0]
        inputs_buf  = inputs_buf[data_range[0] : data_range[1]]
        exact       = samples_buf[data_range[0] : data_range[1]]
        # Evaluate the approximation.
        bucket_index, bucket_location = input1._get_bucket_location_array(inputs_buf)
        basis  = np.array([bucket_location ** power for power in range(num_terms)])
        basis  = basis.T.reshape(num_samples, 1, 1, num_terms)
        coef   = table_buf[bucket_index]
        approx = np.sum(coef * basis, axis = -1)
        # Increase the timestep to 1 ms
        approx = np.linalg.matrix_power(approx, power)
        exact  = np.linalg.matrix_power(exact, power)
        return np.max(np.abs(approx - exact))

class Approx2D(Approx):
    def __init__(self, samples, polynomial):
        super().__init__(samples, polynomial)
        self.input1 = self.model.input1
        self.input2 = self.model.input2
        self._ensure_enough_exact_samples()
        self._alloc_table()
        self._make_table()

    @staticmethod
    def _polynomial_basis(input1_locations, input2_locations, polynomial):
        # Make an approximation for each entry in the matrix.
        A = np.empty([len(input1_locations), polynomial.num_terms])
        for term, (power1, power2) in enumerate(polynomial.terms):
            A[:, term] = (input1_locations ** power1) * (input2_locations ** power2)
        return A

    @staticmethod
    def _table_kernel(args):
        # Unpack the arguments.
        (table_name, input1, input2, num_states, polynomial,
                num_samples, ((bucket_index1, bucket_index2,), data_range)) = args
        num_terms = polynomial.num_terms
        # Setup the shared memory.
        (input1_sm, input2_sm), samples_sm = MatrixSamples._get_sm_weakref(2)
        table_sm        = SharedMemory(table_name, False)
        inputs_shape    = (num_samples,)
        samples_shape   = (num_samples, num_states, num_states)
        table_shape     = (input1.num_buckets, input2.num_buckets, num_states, num_states, num_terms)
        input1_buf      = np.ndarray(inputs_shape, dtype=np.float64, buffer=input1_sm.buf)
        input2_buf      = np.ndarray(inputs_shape, dtype=np.float64, buffer=input2_sm.buf)
        samples_buf     = np.ndarray(samples_shape, dtype=np.float64, buffer=samples_sm.buf)
        table_buf       = np.ndarray(table_shape, dtype=np.float64, buffer=table_sm.buf)
        # Slice out the current bucket's samples.
        num_samples = data_range[1] - data_range[0]
        input1_buf  = input1_buf[data_range[0] : data_range[1]]
        input2_buf  = input2_buf[data_range[0] : data_range[1]]
        samples_buf = samples_buf[data_range[0] : data_range[1]]
        # Scale the inputs into the range [0,1].
        input1_locations = input1.get_bucket_value(input1_buf) - bucket_index1
        input2_locations = input2.get_bucket_value(input2_buf) - bucket_index2
        #
        A = Approx2D._polynomial_basis(input1_locations, input2_locations, polynomial)
        B = samples_buf.reshape(-1, num_states**2)
        coef, rss = np.linalg.lstsq(A, B, rcond=None)[:2]
        coef = coef.reshape(num_terms, num_states, num_states).transpose(1,2,0)
        table_buf[bucket_index1, bucket_index2, :, :, :] = coef
        return np.sum(rss)

    def approximate_matrix(self, input1, input2):
        assert len(input1.shape) == 1 and input1.shape == input2.shape
        num_samples = len(input1)
        self.set_num_buckets()
        bucket1_index, bucket1_location = self.input1._get_bucket_location_array(input1)
        bucket2_index, bucket2_location = self.input2._get_bucket_location_array(input2)
        basis = np.empty([self.num_terms, num_samples])
        for term, (power1, power2) in enumerate(self.polynomial.terms):
            basis[term] = (bucket1_location ** power1) * (bucket2_location ** power2)
        basis = basis.T.reshape(num_samples, 1, 1, self.num_terms)
        coef = self.table[bucket1_index, bucket2_index]
        return np.sum(coef * basis, axis = -1)

    @staticmethod
    def _error_kernel(args):
        # Unpack the arguments.
        (table_name, power, (input1, input2), polynomial, num_states,
                num_samples, ((bucket_index1, bucket_index2), data_range)) = args
        num_terms = polynomial.num_terms
        # Access the shared memory.
        (input1_sm, input2_sm), samples_sm = MatrixSamples._get_sm_weakref(2)
        table_sm        = SharedMemory(table_name, False)
        inputs_shape    = (num_samples,)
        samples_shape   = (num_samples, num_states, num_states)
        table_shape     = (input1.num_buckets, input2.num_buckets, num_states, num_states, num_terms)
        input1_buf      = np.ndarray(inputs_shape, dtype=np.float64, buffer=input1_sm.buf)
        input2_buf      = np.ndarray(inputs_shape, dtype=np.float64, buffer=input2_sm.buf)
        samples_buf     = np.ndarray(samples_shape, dtype=np.float64, buffer=samples_sm.buf)
        table_buf       = np.ndarray(table_shape, dtype=np.float64, buffer=table_sm.buf)
        # Slice out one chunk of data.
        num_samples = data_range[1] - data_range[0]
        input1_buf  = input1_buf[data_range[0] : data_range[1]]
        input2_buf  = input2_buf[data_range[0] : data_range[1]]
        exact       = samples_buf[data_range[0] : data_range[1]]
        # Evaluate the approximation.
        input1_index, input1_location = input1._get_bucket_location_array(input1_buf)
        input2_index, input2_location = input2._get_bucket_location_array(input2_buf)
        basis = Approx2D._polynomial_basis(input1_location, input2_location, polynomial)
        basis  = basis.reshape(num_samples, 1, 1, num_terms)
        coef   = table_buf[bucket_index1, bucket_index2, :, :, :]
        approx = np.sum(coef * basis, axis = -1)
        # Increase the timestep to 1 ms
        approx = np.linalg.matrix_power(approx, power)
        exact  = np.linalg.matrix_power(exact, power)
        return np.max(np.abs(approx - exact))

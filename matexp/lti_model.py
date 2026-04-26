from .nmodl_compiler import NMODL_Compiler
import itertools
import numpy as np
import scipy.linalg

class LTI_Model(NMODL_Compiler):
    """ Specialization of NMODL_Compiler for Linear & Time-Invariant models. """
    def __init__(self, nmodl_filename, inputs, time_step, temperature):
        super().__init__(nmodl_filename, inputs, temperature)
        self.time_step = float(time_step)
        assert self.time_step > 0.0
        self._check_is_LTI()

    def _check_is_LTI(self):
        for trial in range(3):
            inputs = [np.random.uniform(inp.minimum, inp.maximum) for inp in self.inputs]
            state1 = np.random.uniform(0.0, 1.0, size=self.num_states)
            state2 = state1 * 2.0
            d1 = self.derivative(*inputs, *state1)
            d2 = self.derivative(*inputs, *state2)
            for s1, s2 in zip(d1, d2):
                assert abs(s1 - s2 / 2.0) < 1e-12, "Non-linear system detected!"

    def make_deriv_matrix(self, inputs):
        # Cleanup the arguments.
        inputs = np.array(inputs, dtype=float)
        assert (inputs.ndim == 2) and (inputs.shape[0] == self.num_inputs)
        num_samples = inputs.shape[1]
        for dim, input_data in enumerate(self.inputs):
            assert np.all(input_data.minimum <= inputs[dim, :])
            assert np.all(input_data.maximum >= inputs[dim, :])
        # 
        A = np.empty([num_samples, self.num_states, self.num_states])
        # Break up the input into chunks for multithreading.
        from . import _num_threads, _thread_pool # Lazy import to avoid circular dependency.
        num_chunks = _num_threads * 3
        boundaries = [num_samples * i // num_chunks for i in range(num_chunks + 1)]
        input_slices = [slice(*pair) for pair in itertools.pairwise(boundaries)]
        # 
        def compute_chunk(input_slice):
            chunk_size = input_slice.stop - input_slice.start
            chunk_inputs = inputs[:, input_slice]
            for col in range(self.num_states):
                state = np.zeros([self.num_states, chunk_size])
                state[col, :] = 1
                A[input_slice, :, col] = np.transpose(self.derivative(*chunk_inputs, *state))
        _thread_pool.map(compute_chunk, input_slices, chunksize=1)
        return A

    def make_matrix(self, inputs, time_step=None):
        deriv_matrix = self.make_deriv_matrix(inputs)
        num_samples = deriv_matrix.shape[0]
        if time_step is None:
            time_step = self.time_step
        propagator_matrix = np.empty_like(deriv_matrix)
        from . import _num_threads, _thread_pool # Lazy import to avoid circular dependency.
        num_chunks = _num_threads * 3
        boundaries = [i * num_samples // num_chunks for i in range(num_chunks + 1)]
        input_slices = [slice(*pair) for pair in itertools.pairwise(boundaries)]
        def compute_chunk(input_slice):
            deriv_matrix[input_slice] *= time_step
            propagator_matrix[input_slice] = scipy.linalg.expm(deriv_matrix[input_slice])
        #
        _thread_pool.map(compute_chunk, input_slices, chunksize=1)
        return propagator_matrix

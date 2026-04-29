from .nmodl_compiler import NMODL_Compiler
from multiprocessing.shared_memory import SharedMemory
from itertools import pairwise, repeat
import dill
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
        # Setup shared memory buffers.
        inputs_shape = (self.num_inputs, num_samples)
        deriv_shape = (num_samples, self.num_states, self.num_states)
        inputs_sm = SharedMemory('_matexp_inputs', True, inputs.nbytes)
        deriv_sm = SharedMemory('_matexp_deriv', True, 8 * num_samples * self.num_states * self.num_states)
        inputs_sm_buf = np.ndarray(inputs_shape, dtype=np.float64, buffer=inputs_sm.buf)
        inputs_sm_buf[:,:] = inputs
        try:
            # Break up the input into chunks for multithreading.
            from . import _num_threads, _thread_pool # Lazy import to avoid circular dependency.
            num_chunks = _num_threads * 3
            boundaries = [num_samples * i // num_chunks for i in range(num_chunks + 1)]
            input_slices = [slice(*pair) for pair in pairwise(boundaries)]
            # 
            derivative = dill.dumps(self.derivative)
            args = (repeat(derivative), repeat(self.num_inputs), repeat(self.num_states), repeat(num_samples), input_slices)
            for _ in _thread_pool.map(self._compute_deriv, zip(*args), chunksize=1): pass
            # for _ in map(self._compute_deriv, zip(*args)): pass
            return np.ndarray(deriv_shape, dtype=np.float64, buffer=deriv_sm.buf).copy(), deriv_sm
        finally:
            inputs_sm.close()
            inputs_sm.unlink()
            # deriv_sm.unlink() # returned instead of freed

    @staticmethod
    def _compute_deriv(args):
        derivative, num_inputs, num_states, num_samples, input_slice = args
        derivative = dill.loads(derivative)
        inputs_shape = (num_inputs, num_samples)
        deriv_shape = (num_samples, num_states, num_states)
        inputs_sm = SharedMemory('_matexp_inputs', False)
        deriv_sm = SharedMemory('_matexp_deriv', False)
        inputs = np.ndarray(inputs_shape, dtype=np.float64, buffer=inputs_sm.buf)
        deriv = np.ndarray(deriv_shape, dtype=np.float64, buffer=deriv_sm.buf)
        chunk_size = input_slice.stop - input_slice.start
        chunk_inputs = inputs[:, input_slice]
        state = np.empty([num_states, chunk_size])
        for col in range(num_states):
            state.fill(0.)
            state[col, :] = 1.
            deriv[input_slice, :, col] = np.transpose(derivative(*chunk_inputs, *state))

    def make_matrix(self, inputs, time_step=None):
        """
        Argument inputs is 2D array with shape [N-INPUTS, N-SAMPLES]
        """
        deriv_matrix, deriv_sm = self.make_deriv_matrix(inputs)
        try:
            num_samples = deriv_matrix.shape[0]
            if time_step is None:
                time_step = self.time_step
            propagator_matrix = np.empty_like(deriv_matrix)
            from . import _num_threads, _thread_pool # Lazy import to avoid circular dependency.
            num_chunks = _num_threads * 3
            boundaries = [i * num_samples // num_chunks for i in range(num_chunks + 1)]
            input_slices = [slice(*pair) for pair in pairwise(boundaries)]
            #
            args = (repeat(time_step), repeat(deriv_matrix.shape), input_slices)
            for _ in _thread_pool.map(self._compute_expm, zip(*args), chunksize=1): pass
            # for _ in map(self._compute_expm, zip(*args)): pass
            return np.ndarray(deriv_matrix.shape, dtype=np.float64, buffer=deriv_sm.buf).copy()
        finally:
            deriv_sm.close()
            deriv_sm.unlink()

    @staticmethod
    def _compute_expm(args):
        time_step, deriv_shape, input_slice = args
        deriv_sm = SharedMemory('_matexp_deriv', False)
        deriv_matrix = np.ndarray(deriv_shape, dtype=np.float64, buffer=deriv_sm.buf)
        deriv_chunk = deriv_matrix[input_slice, :, :]
        deriv_chunk *= time_step
        deriv_matrix[input_slice, :, :] = scipy.linalg.expm(deriv_chunk)

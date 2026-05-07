from .nmodl_compiler import NMODL_Compiler
from multiprocessing.shared_memory import SharedMemory
from itertools import pairwise, repeat
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
        # Lazy import to avoid circular dependency.
        from . import _num_threads, _thread_pool, _initialize_thread_pool
        if _thread_pool is None:
            _thread_pool = _initialize_thread_pool(self, False)
        # Setup shared memory buffers.
        inputs_shape = (self.num_inputs, num_samples)
        deriv_shape = (num_samples, self.num_states, self.num_states)
        inputs_sm = SharedMemory('matexp_deriv_inputs', True, inputs.nbytes)
        deriv_sm = SharedMemory('matexp_deriv_matrix', True, 8 * num_samples * self.num_states * self.num_states)
        inputs_buf = np.ndarray(inputs_shape, dtype=np.float64, buffer=inputs_sm.buf)
        inputs_buf[:,:] = inputs
        try:
            # Break up the input into chunks for multithreading.
            num_chunks = _num_threads * 2
            boundaries = [num_samples * i // num_chunks for i in range(num_chunks + 1)]
            input_slices = [slice(*pair) for pair in pairwise(boundaries)]
            # 
            args = (repeat(self.num_inputs),
                    repeat(self.num_states),
                    repeat(num_samples),
                    input_slices)
            for _ in _thread_pool.map(self._compute_deriv, zip(*args), chunksize=1): pass
            # for _ in map(self._compute_deriv, zip(*args)): pass
            return np.ndarray(deriv_shape, dtype=np.float64, buffer=deriv_sm.buf).copy(), deriv_sm
        finally:
            inputs_sm.close()
            inputs_sm.unlink()
            # deriv_sm.unlink() # returned instead of freed

    @staticmethod
    def _compute_deriv(args):
        num_inputs, num_states, num_samples, input_slice = args
        from . import _derivative
        inputs_shape = (num_inputs, num_samples)
        deriv_shape = (num_samples, num_states, num_states)
        inputs_sm = SharedMemory('matexp_deriv_inputs', False)
        deriv_sm = SharedMemory('matexp_deriv_matrix', False)
        inputs = np.ndarray(inputs_shape, dtype=np.float64, buffer=inputs_sm.buf)
        deriv = np.ndarray(deriv_shape, dtype=np.float64, buffer=deriv_sm.buf)
        chunk_size = input_slice.stop - input_slice.start
        chunk_inputs = inputs[:, input_slice]
        state = np.empty([num_states, chunk_size])
        for col in range(num_states):
            state.fill(0.)
            state[col, :] = 1.
            deriv[input_slice, :, col] = np.transpose(_derivative(*chunk_inputs, *state))
        inputs_sm.close()
        deriv_sm.close()

    def make_matrix(self, inputs, time_step=None):
        """
        Argument inputs is 2D array with shape [N-INPUTS, N-SAMPLES]
        """
        # *_sm are shared memory handles
        # *_buf are numpy arrays
        deriv_buf, deriv_sm = self.make_deriv_matrix(inputs)
        try:
            num_samples = deriv_buf.shape[0]
            if time_step is None:
                time_step = self.time_step
            propagator_matrix = np.empty_like(deriv_buf)
            from . import _num_threads, _thread_pool # Lazy import to avoid circular dependency.
            num_chunks = _num_threads * 2
            boundaries = [i * num_samples // num_chunks for i in range(num_chunks + 1)]
            input_slices = [slice(*pair) for pair in pairwise(boundaries)]
            #
            args = (repeat(time_step), repeat(deriv_buf.shape), input_slices)
            for _ in _thread_pool.map(self._compute_expm, zip(*args), chunksize=1): pass
            # for _ in map(self._compute_expm, zip(*args)): pass
            return np.ndarray(deriv_buf.shape, dtype=np.float64, buffer=deriv_sm.buf).copy()
        finally:
            deriv_sm.close()
            deriv_sm.unlink()

    @staticmethod
    def _compute_expm(args):
        time_step, deriv_shape, input_slice = args
        deriv_sm = SharedMemory('matexp_deriv_matrix', False)
        deriv_buf = np.ndarray(deriv_shape, dtype=np.float64, buffer=deriv_sm.buf)
        deriv_slice = deriv_buf[input_slice, :, :]
        deriv_slice *= time_step
        deriv_buf[input_slice, :, :] = scipy.linalg.expm(deriv_slice)
        deriv_sm.close()

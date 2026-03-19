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

    def make_matrix(self, inputs, time_step=None):
        # Cleanup the arguments.
        inputs = np.array(inputs, dtype=float)
        assert (inputs.ndim == 2) and (inputs.shape[0] == self.num_inputs)
        num_samples = inputs.shape[1]
        for dim, input_data in enumerate(self.inputs):
            assert np.all(input_data.minimum <= inputs[dim, :])
            assert np.all(input_data.maximum >= inputs[dim, :])
        # 
        if time_step is None:
            time_step = self.time_step
        # Break up the input into chunks for multithreading.
        num_chunks = max(1, num_samples // 1000)
        num_chunks = 16
        boundaries = [i * num_samples // num_chunks for i in range(num_chunks + 1)]
        input_slices = [slice(*pair) for pair in itertools.pairwise(boundaries)]
        # 
        def compute_chunk(input_slice):
            chunk_size = input_slice.stop - input_slice.start
            A = np.empty([chunk_size, self.num_states, self.num_states])
            chunk_inputs = inputs[:, input_slice]
            for col in range(self.num_states):
                state = np.zeros([self.num_states, chunk_size])
                state[col, :] = 1
                A[:, :, col] = np.transpose(self.derivative(*chunk_inputs, *state))
            return A * time_step
        from . import thread_pool # Lazy import to avoid circular dependency.
        chunk_results = list(thread_pool.map(compute_chunk, input_slices))
        # Scipy expm is already multithreaded internally.
        matrices = scipy.linalg.expm(np.concatenate(chunk_results))
        return matrices

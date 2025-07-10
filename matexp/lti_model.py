from .nmodl_compiler import NMODL_Compiler
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
        for dim, input_data in enumerate(self.inputs):
            assert np.all(input_data.minimum <= inputs[dim, :])
            assert np.all(input_data.maximum >= inputs[dim, :])
        # 
        if time_step is None:
            time_step = self.time_step
        # 
        num_samples = inputs.shape[-1]
        A = np.empty([num_samples, self.num_states, self.num_states])
        for sample in range(num_samples):
            for col in range(self.num_states):
                state = [float(x == col) for x in range(self.num_states)]
                A[sample, :, col] = self.derivative(*inputs[:, sample], *state)
        return scipy.linalg.expm(A * time_step)

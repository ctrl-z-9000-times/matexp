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
        inputs = [float(input_value) for input_value in inputs]
        assert len(inputs) == len(self.inputs)
        for input_value, input_data in zip(inputs, self.inputs):
            assert input_data.minimum <= input_value <= input_data.maximum
        if time_step is None:
            time_step = self.time_step
        A = np.empty([self.num_states, self.num_states])
        for col in range(self.num_states):
            state = [float(x == col) for x in range(self.num_states)]
            A[:, col] = self.derivative(*inputs, *state)
        matrix = scipy.linalg.expm(A * time_step)
        for col in range(self.num_states):
            matrix[:, col] *= 1.0 / sum(matrix[:, col].flat)
        return matrix

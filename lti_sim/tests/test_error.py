from lti_sim import LinearInput, LogarithmicInput
from lti_sim.approx import MatrixSamples, Approx1D, Approx2D
from lti_sim.lti_model import LTI_Model
import math
import os
import pytest

test_dir = os.path.dirname(__file__)

maximum_under_estimation_factor = 5

def test_1D():
    nmodl_file = os.path.join(test_dir, "Nav11.mod")
    v = LinearInput('v', -100, 100)
    v.set_num_buckets(50)
    m = LTI_Model(nmodl_file, [v], 0.1, 37.0)
    x = MatrixSamples(m)
    a = Approx1D(x, 2)
    est_err = a.measure_error()
    print('default num samples', len(x.samples))
    x.sample(math.ceil(20 * len(x.samples) / v.num_buckets))
    print('testing num samples', len(x.samples))
    true_err = a.measure_error()
    print("True Error           \t| Percent Underestimation")
    for e1, e2 in zip(est_err, true_err):
        print(e2, '\t| ', round(100 * (e1 - e2) / e2, 2))
    assert all(est_err <= true_err), "sanity check"
    assert all(est_err >= true_err / maximum_under_estimation_factor)

def test_2D():
    nmodl_file = os.path.join(test_dir, "NMDA.mod")
    v = LinearInput('v', -100, 100)
    v.set_num_buckets(5)
    C = LogarithmicInput('C', 0, 100)
    C.set_num_buckets(5, scale=.001)
    m = LTI_Model(nmodl_file, [C, v], 0.1, 37.0)
    x = MatrixSamples(m)
    a = Approx2D(x, [[0, 0], [1, 0], [0, 1], [2, 0], [0, 2], [3, 0], [0, 3], [5, 0], [0, 5]])
    est_err = a.measure_error().reshape(-1)
    print('default num samples', len(x.samples))
    x.sample(math.ceil(3 * len(x.samples) / v.num_buckets))
    print('testing num samples', len(x.samples))
    true_err = a.measure_error().reshape(-1)
    print("True Error           \t| Percent Underestimation")
    for e1, e2 in zip(est_err, true_err):
        print(e2, '\t| ', round(100 * (e1 - e2) / e2, 2))
    assert all(est_err <= true_err), "sanity check"
    assert all(est_err >= true_err / maximum_under_estimation_factor)

import argparse
import ctypes
import lti_sim
import matplotlib.pyplot as plt
import numpy as np
import os.path
import random
import subprocess
import tempfile

nav11_input = lti_sim.LinearInput('v', -100, 100)
ampa_input  = lti_sim.LogarithmicInput('C', 0, 1e3)

py_dir    = os.path.dirname(lti_sim.__file__)
nav11_mod = os.path.join(py_dir, "tests", "Nav11.mod")
ampa_mod  = os.path.join(py_dir, "tests", "ampa13.mod")

lti_kwargs = {'temperature': 37.0, 'float_size': 64, 'target': 'host'}
speed_kwargs = {'conserve_sum': 1.0, 'float_dtype': np.float64, 'target': 'host'}

def load_cpp(filename, input1, num_states, TIME_STEP, opt_level):
    """ Compile one of the Backward Euler C++ files and link it into python. """
    dirname  = os.path.abspath(os.path.dirname(__file__))
    src_file = os.path.join(dirname, filename)
    so_file  = tempfile.NamedTemporaryFile(suffix='.so', delete=False)
    so_file.close()
    eigen = os.path.join(dirname, "eigen")
    subprocess.run(["g++", src_file, "-o", so_file.name,
                    f"-DTIME_STEP={TIME_STEP}",
                    "-I"+eigen, "-shared", "-fPIC", f"-O{opt_level}"],
                    check=True)
    fn = ctypes.CDLL(so_file.name).advance_state
    argtypes = [ctypes.c_int]
    argtypes.append(np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C'))
    argtypes.append(np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, flags='C'))
    for _ in range(num_states):
        argtypes.append(np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C'))
    fn.argtypes = argtypes
    return fn

def measure_accuracy(input1, num_states, functions, time_steps,
                    num_instances = 1000,
                    run_time = 10e6,):
    """
    Measure the accuracy at different time-steps.
    Returns the RMS Error of the states at every time step.
    """
    num_functions = len(functions)
    assert len(time_steps) == num_functions
    assert all(isinstance(ts, int) and ts > 0 for ts in time_steps)
    assert all(ts % time_steps[0] == 0 for ts in time_steps)
    assert all(ts <= run_time for ts in time_steps)
    # Generate valid initial state data.
    initial_state = [np.random.uniform(size=num_instances) for _ in range(num_states)]
    sum_x = np.zeros(num_instances)
    for x in initial_state:
        sum_x += x
    for x in initial_state:
        x /= sum_x
    # Duplicate the initial_state data for every time step.
    state = [initial_state]
    for _ in range(num_functions - 1):
        state.append([np.array(x, copy=True) for x in initial_state])
    # Generate random inputs. Will hold the input constant for the duration of the test.
    inputs = np.random.uniform(input1.minimum, input1.maximum, size=num_instances)
    input_idx = np.array(np.arange(num_instances), dtype=np.int32)
    # 
    t = [0] * num_functions
    def advance(idx):
        functions[idx](num_instances, inputs, input_idx, *state[idx])
        t[idx] += time_steps[idx]
    # 
    sqr_err_accum  = [0.0] * num_functions
    num_points     = [0] * num_functions
    num_points[0] += 1 # Don't div zero.
    def compare(idx):
        for a, b in zip(state[0], state[idx]):
            sqr_err_accum[idx] += np.sum((a - b) ** 2)
            num_points[idx] += num_instances
    # 
    while t[0] < run_time:
        advance(0)
        for idx in range(1, num_functions):
            if t[idx] <  t[0]: advance(idx)
            if t[idx] == t[0]: compare(idx)
    # 
    return [(sum_sqr / num) ** 0.5 for sum_sqr, num in zip(sqr_err_accum, num_points)]

def plot_accuracy_vs_timestep():
    time_steps_ns = [100, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    time_steps_ms = [ts/1000/1000 for ts in time_steps_ns]
    nav11_exact = lti_sim.main_1D(nav11_mod, nav11_input, time_steps_ms[0], accuracy=1e-12, **lti_kwargs)[1]
    ampa_exact  = lti_sim.main_1D(ampa_mod,  ampa_input,  time_steps_ms[0], accuracy=1e-12, **lti_kwargs)[1]
    # Nav11 Backward Euler
    nav11_be_fn = [load_cpp("Nav11.cpp", nav11_input, 6, ts, 1) for ts in time_steps_ms[1:]]
    nav11_be_fn.insert(0, nav11_exact)
    nav11_be_err = measure_accuracy(nav11_input, 6, nav11_be_fn, time_steps_ns)
    # Nav11 Matrix Exponential
    nav11_me_fn = [lti_sim.main_1D(nav11_mod, nav11_input, ts, accuracy=1e-6, **lti_kwargs)[1] for ts in time_steps_ms[1:]]
    nav11_me_fn.insert(0, nav11_exact)
    nav11_me_err = measure_accuracy(nav11_input, 6, nav11_me_fn, time_steps_ns)
    # AMPA Receptor Backward Euler
    ampa_be_fn = [load_cpp("ampa13.cpp", ampa_input, 13, ts, 1) for ts in time_steps_ms[1:]]
    ampa_be_fn.insert(0, ampa_exact)
    ampa_be_err = measure_accuracy(ampa_input, 13, ampa_be_fn, time_steps_ns)
    # AMPA Receptor Matrix Exponential
    ampa_me_fn = [lti_sim.main_1D(ampa_mod, ampa_input, ts, accuracy=1e-6, **lti_kwargs)[1] for ts in time_steps_ms[1:]]
    ampa_me_fn.insert(0, ampa_exact)
    ampa_me_err = measure_accuracy(ampa_input, 13, ampa_me_fn, time_steps_ns)
    # 
    plt.figure('Accuracy Comparison')
    plt.title('Accuracy vs Time Step')
    plt.ylabel('RMS Error')
    plt.xlabel('Time Step, Milliseconds')
    plt.loglog(time_steps_ms[1:], nav11_be_err[1:], 'r', label='Nav11,\nBackward Euler')
    plt.loglog(time_steps_ms[1:], ampa_be_err[1:],  'b', label='AMPA Receptor,\nBackward Euler')
    plt.loglog(time_steps_ms[1:], nav11_me_err[1:], 'firebrick',  marker='s', label='Nav11,\nMatrix Exponential')
    plt.loglog(time_steps_ms[1:], ampa_me_err[1:],  'mediumblue', marker='s', label='AMPA Receptor,\nMatrix Exponential')
    plt.grid(axis='y')
    plt.legend()
    plt.show()

def plot_speed():
    # Nav11 Backward Euler
    fn = load_cpp("Nav11.cpp", nav11_input, 6, 0.1, opt_level=1)
    nav11_be = lti_sim._measure_speed(fn, 6, nav11_input, **speed_kwargs)[0]
    # AMPA Receptor Backward Euler
    fn = load_cpp("ampa13.cpp", ampa_input, 13, 0.1, opt_level=1)
    ampa_be = lti_sim._measure_speed(fn, 13, ampa_input, **speed_kwargs)[0]
    # Nav11 Matrix Exponential
    fn = lti_sim.main_1D(nav11_mod, nav11_input, 0.1, accuracy=1e-6, **lti_kwargs)[1]
    nav11_me = lti_sim._measure_speed(fn, 6, nav11_input, **speed_kwargs)[0]
    # AMPA Receptor Matrix Exponential
    fn = lti_sim.main_1D(ampa_mod, ampa_input, 0.1, accuracy=1e-6, **lti_kwargs)[1]
    ampa_me = lti_sim._measure_speed(fn, 13, ampa_input, **speed_kwargs)[0]
    # 
    print("Nav11 Speed Comparison:")
    print('BE', nav11_be)
    print('ME', nav11_me)
    print("AMPA Receptor Speed Comparison:")
    print('BE', ampa_be)
    print('ME', ampa_me)
    print()
    # 
    plt.figure('Speed Comparison')
    plt.title('Real Time to Integrate, per Instance per Time Step')
    x = np.arange(2)
    width = 1/3
    plt.bar(x-width/2, [nav11_be, ampa_be],
        width=width,
        label='Backward Euler')
    plt.bar(x+width/2, [nav11_me, ampa_me],
        width=width,
        label='Matrix Exponential,\nMaximum Error: 1e-6')
    plt.ylabel('Nanoseconds')
    plt.xticks(x, ["Nav11\n6 States", "AMPA Receptor\n13 States"])
    plt.legend()
    plt.show()

def plot_speed_vs_accuracy():
    max_err = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    nav11_speed = []
    for x in max_err:
        fn = lti_sim.main_1D(nav11_mod, nav11_input, 0.1, accuracy=x, **lti_kwargs)[1]
        nav11_speed.append(lti_sim._measure_speed(fn, 6, nav11_input, **speed_kwargs)[0])
    ampa_speed = []
    for x in max_err:
        fn = lti_sim.main_1D(ampa_mod, ampa_input, 0.1, accuracy=x, **lti_kwargs)[1]
        ampa_speed.append(lti_sim._measure_speed(fn, 13, ampa_input, **speed_kwargs)[0])
    # 
    plt.figure('Speed/Accuracy Trade-off')
    plt.title('Simulation Speed vs Accuracy')
    plt.ylabel('Real Time to Integrate, per Instance per Time Step\nNanoseconds')
    plt.xlabel('Maximum Absolute Error')
    plt.semilogx(max_err, nav11_speed, label='Nav11, 6 States')
    plt.semilogx(max_err, ampa_speed,  label='AMPA Receptor, 13 States')
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.show()

parser = argparse.ArgumentParser(prog='backward_euler', description=
        """Compare the Backward Euler method of integration with the Matrix Exponential method. """)
parser.add_argument('--accuracy', action='store_true')
parser.add_argument('--speed', action='store_true')
parser.add_argument('--tradeoff', action='store_true')
args = parser.parse_args()

if args.accuracy:   plot_accuracy_vs_timestep()
if args.speed:      plot_speed()
if args.tradeoff:   plot_speed_vs_accuracy()

import matexp.convolve
import math
import numpy as np
import pytest
import scipy.stats
import time

def make_signal():
    bins = 2**12
    signal = np.zeros(bins)
    signal[40:90] = np.random.uniform(0.0, 2.0, 50)
    signal[90:3990] = np.exp(-np.linspace(1, 20, 3900))
    signal *= 1.0 / np.sum(signal)
    signal_range = (1.23e-6, 4.56e-2)
    # signal_range = (1.23, 4.56)
    return (signal, signal_range)

def bin_centers(data_range, data):
    bins_edges = np.linspace(*data_range, len(data) + 1)
    return 0.5 * (bins_edges[:-1] + bins_edges[1:])

def ilerp(value, start, end):
    """ Inverse of linear interpolation """
    assert start <= end
    return (value - start) / (end - start)

def test_lerp():
    assert matexp.convolve._lerp(0, -5, 5) == pytest.approx(-5)
    assert matexp.convolve._lerp(.1, -5, 5) == pytest.approx(-4)
    assert matexp.convolve._lerp(.5, -5, 5) == pytest.approx(0)
    assert matexp.convolve._lerp(.75, -5, 5) == pytest.approx(2.5)
    assert matexp.convolve._lerp(1, -5, 5) == pytest.approx(5)

def test_autoconvolve():
    # 
    instances = 60
    signal, signal_range = make_signal()

    # 
    matexp_result, matexp_range = matexp.convolve.autoconvolve(signal, signal_range, instances)

    # Compare with a scipy based implementation.
    scipy_result = signal
    for _ in range(instances - 1):
        scipy_result = scipy.signal.convolve(scipy_result, signal)
    scipy_range  = (instances * signal_range[0], instances * signal_range[1])

    # Debugging
    print("signal len", len(signal),        "\trange", signal_range)
    print("scipy  len", len(scipy_result),  "\trange", scipy_range)
    print("matexp len", len(matexp_result), "\trange", matexp_range)
    if False:
        import matplotlib.pyplot as plt
        plt.plot(bin_centers(signal_range, signal), signal, label="signal")
        plt.plot(bin_centers(matexp_range, matexp_result), matexp_result, label="matexp")
        plt.plot(bin_centers(scipy_range,  scipy_result), scipy_result, label="scipy")
        plt.legend()
        plt.show()

    # Check data range and number of output elements.
    matexp_bin_width = (matexp_range[1] - matexp_range[0]) / len(matexp_result)
    scipy_bin_width  = (scipy_range[1]  - scipy_range[0])  / len(scipy_result)
    assert matexp_bin_width == pytest.approx(scipy_bin_width)

    # Locate matexp's result within scipy's result.
    roi_start = round(len(scipy_result) * ilerp(matexp_range[0], *scipy_range))
    roi_end   = round(len(scipy_result) * ilerp(matexp_range[1], *scipy_range))
    scipy_roi = scipy_result[roi_start:roi_end]

    # Compare the two signals.
    atol = 1e-12
    assert np.all(np.abs(matexp_result - scipy_roi) < atol)

    # Check the truncated portions of the signal.
    assert np.all(np.abs(scipy_result[:roi_start]) < atol)
    assert np.all(np.abs(scipy_result[roi_end:]) < atol)

def measure_speed():
    # Experimental setup
    np.random.seed(42)
    instances = round(6 * 1000 / 0.1)
    instances = 60000
    signal, signal_range = make_signal()
    print("Signal length:", len(signal))
    print("Signal instances:", instances)

    # Perform the measurement
    time.sleep(0)
    start_time = time.time()
    matexp_result, matexp_range = matexp.convolve.autoconvolve(signal, signal_range, instances)
    end_time = time.time()
    print("Elapsed time:", end_time - start_time, 'seconds')
    np.random.seed()

if __name__ == "__main__":
    measure_speed()
    test_autoconvolve()

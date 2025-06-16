import numpy as np
import math

def autoconvolve(signal, signal_range, instances, epsilon=1e-12):
    """
    Convolve a signal with itself N times.

    Argument signal is a 1D signal array.

    Argument signal_range is a pair of floating point numbers, inclusive bound
             of the data range.

    Argument instances is number of signal to convolve together.
    """
    # Clean the arguments.
    signal = np.asarray(signal, dtype=np.float64)
    instances = round(instances)
    if instances <= 0:
        raise ValueError("argument instances must be a positive integer")
    signal_range = tuple(float(bound) for bound in signal_range)
    if len(signal_range) != 2:
        raise ValueError("argument signal_range must be a pair of floating point numbers")
    # Normalize the signal to sum to one, for numerical stability.
    signal *= 1.0 / np.sum(signal)
    # 
    iterations  = instances - 1
    out_len     = len(signal) + (len(signal) - 1) * iterations
    fft_len     = 2 ** math.ceil(math.log2(out_len))
    truncate    = min(2 ** 15, fft_len)
    while truncate <= fft_len:
        signal_fft  = np.fft.rfft(signal, truncate)
        result_fft  = signal_fft ** instances
        result      = np.fft.irfft(result_fft)[:out_len]
        loss        = np.max(result[int(.90 * len(result)):])
        if loss <= epsilon or truncate == fft_len:
            break
        else:
            truncate = min(fft_len, truncate * 2)
    # Positions the values along the signal's length
    result_range = (instances * signal_range[0], instances * signal_range[1])
    truncate     = len(result) / out_len
    result_range = (result_range[0], _lerp(truncate, *result_range))
    return (result, result_range)

def _lerp(fraction, start, end):
    return start + fraction * (end - start)

"""
These classes transforms the inputs between real world values and indexes which
are suitable for use with a look up table.
"""

import numpy as np

class Input:
    """ Abstract base class. """
    def __init__(self, name, minimum, maximum, initial=None):
        self.name       = str(name)
        self.minimum    = float(minimum)
        self.maximum    = float(maximum)
        self.range      = self.maximum - self.minimum
        assert self.minimum < self.maximum
        if initial is None:
            self.initial = self.minimum
        else:
            self.initial = float(initial)
            assert self.minimum <= self.initial < self.maximum

    def set_num_buckets(self, num_buckets):
        self.num_buckets = int(num_buckets)
        assert self.num_buckets >= 1

    def get_bucket_value(self, input_value):
        raise NotImplementedError(type(self))

    def get_input_value(self, bucket_value):
        raise NotImplementedError(type(self))

    def get_bucket_location(self, input_value, bucket_index=None):
        """ Returns pair of (bucket_index, location_within_bucket) """
        input_value = float(input_value)
        location = self.get_bucket_value(input_value)
        if bucket_index is None:
            bucket_index = int(location)
        else:
            bucket_index = int(bucket_index)
        return (bucket_index, location - bucket_index)

    def sample_space(self, number):
        """
        Note: this returns the endpoint, which technically should be excluded
        from the input space.
        """
        number = int(number)
        samples = np.linspace(0, self.num_buckets, number, endpoint=True)
        for sample_index, bucket_location in enumerate(samples):
            samples[sample_index] = self.get_input_value(bucket_location)
        # Fix any numeric instability.
        samples[0]  = self.minimum
        samples[-1] = self.maximum
        return samples

class LinearInput(Input):
    """ """
    def set_num_buckets(self, num_buckets):
        super().set_num_buckets(num_buckets)
        self.bucket_frq     = self.num_buckets / self.range
        self.bucket_width   = self.range / self.num_buckets

    def get_bucket_value(self, input_value):
        return (input_value - self.minimum) * self.bucket_frq

    def get_input_value(self, bucket_value):
        return self.minimum + bucket_value * self.bucket_width

class LogarithmicInput(Input):
    """ """
    def __init__(self, name, minimum, maximum, initial=None):
        super().__init__(name, minimum, maximum, initial)
        assert self.minimum == 0.0, 'Logarithmic inputs must have minimum value of 0.'

    def set_num_buckets(self, num_buckets, scale=None):
        super().set_num_buckets(num_buckets)
        if scale is None:
            assert hasattr(self, "scale")
        else:
            self.scale      = float(scale)
        self.log2_minimum   = np.log2(self.minimum + self.scale)
        self.log2_maximum   = np.log2(self.maximum + self.scale)
        self.log2_range     = self.log2_maximum - self.log2_minimum
        self.bucket_frq     = self.num_buckets / self.log2_range
        self.bucket_width   = self.log2_range / self.num_buckets

    def get_bucket_value(self, input_value):
        log2_value = np.log2(input_value + self.scale)
        return (log2_value - self.log2_minimum) * self.bucket_frq

    def get_input_value(self, bucket_value):
        log2_value = self.log2_minimum + bucket_value * self.bucket_width
        return 2.0 ** log2_value - self.scale

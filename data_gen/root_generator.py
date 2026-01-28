import numpy as np
from data_gen.noise_generator import _gaussian_cauchy_mixture_noise


def cause_func_chooser(noise_type):
    if 'Mixture' in noise_type:
        return mixture_cause(noise_type)
    else:
        if noise_type == 'Gaussian':
            return gaussian_cause()
        elif noise_type == 'Cauchy':
            return cauchy_cause()
        elif noise_type == 'Uniform':
            return uniform_cause()
        else:
            raise ValueError('Noise type not supported')


def mixture_cause(noise_type):
    def f(n):
        return _gaussian_cauchy_mixture_noise(n, noise_type).reshape(-1)

    return f


def cauchy_cause():
    return lambda x: np.random.standard_cauchy(size=x).reshape(-1)


def gaussian_cause():
    return lambda x: np.random.randn(x, 1)[:, 0]


def uniform_cause():
    return lambda x: np.random.uniform(-3, 3, size=x).reshape(-1)

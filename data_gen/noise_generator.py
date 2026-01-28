import numpy as np


def noise_func_chooser(noise_type):
    if 'GaussianCauchyMixture' in noise_type:
        return gaussian_cauchy_mixture_noise(noise_type)
    elif 'T' in noise_type:
        return t_dist_noise(noise_type)
    elif 'GaussianMixture' in noise_type:
        return gaussian_mixture_noise(noise_type)
    else:
        if noise_type == 'Gaussian':
            return gaussian_noise()
        elif noise_type == 'Cauchy':
            return cauchy_noise()
        else:
            raise ValueError('Noise type not supported')


def gaussian_noise():
    return lambda x: np.random.randn(x).reshape(x, 1)


def cauchy_noise():
    return lambda x: np.random.standard_cauchy(size=(x, 1))


def _gaussian_cauchy_mixture_noise(n, ratio):
    """
    example: Mixture8, percent = 8
    Gaussian : Cauchy = percent : (10 - percent)
    """
    gaussian_samples = np.random.normal(loc=0, scale=1, size=int(n * ratio))
    cauchy_samples = np.random.standard_cauchy(size=int(n - int(n * ratio)))
    # 将两个数组合并，并打乱顺序
    samples = np.concatenate((gaussian_samples, cauchy_samples))
    np.random.shuffle(samples)
    return samples.reshape(n, 1)


def gaussian_cauchy_mixture_noise(noise_type):
    ratio = int(noise_type.strip('GaussianCauchyMixture')) / 10

    def f(n):
        return _gaussian_cauchy_mixture_noise(n, ratio)

    return f


def t_dist_noise(noise_type):
    degree_freedom = int(noise_type.strip('T'))

    def f(n):
        return _t_dist_noise(n, degree_freedom)

    return f


def _t_dist_noise(n, degree_freedom):
    samples = np.random.standard_t(df=degree_freedom, size=(n, 1))
    return samples.reshape(n, 1)


def gaussian_mixture_noise(noise_type):
    mixture_num = int(noise_type.strip('GaussianMixture'))

    def f(n):
        return _gaussian_mixture_noise(n, mixture_num)

    return f


def _gaussian_mixture_noise(n, mixture_num):
    weights = np.random.rand(mixture_num)
    weights /= np.sum(weights)
    means = np.random.uniform(low=-5, high=5, size=mixture_num)
    stds = np.random.uniform(low=0.1, high=2.0, size=mixture_num)

    samples = np.zeros(n)

    for i in range(mixture_num):
        samples += weights[i] * np.random.normal(loc=means[i], scale=stds[i], size=n)
    return samples.reshape(n, 1)

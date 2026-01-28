from collections import Counter
import numpy as np
from scipy import stats
from scipy.stats import norm
from stat_test.init import *
from utils.tools import dict2namespace
from scipy.stats import chi2
from abc import abstractmethod, ABC


class EnsembleBase(Test_Base, ABC):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, uit_l_kwargs,
                 cit_l_kwargs, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        from stat_test.test_chooser import _stat_test_chooser
        self.t_chooser = _stat_test_chooser
        self.save_cache = False
        self.p_combination = None
        self.uit_instance = {test_name: self.t_chooser(self.data, test_kwargs, test_name) for test_name, test_kwargs in
                             uit_l_kwargs.items()}
        self.cit_instance = {test_name: self.t_chooser(self.data, test_kwargs, test_name) for test_name, test_kwargs in
                             cit_l_kwargs.items()}

    def ensemble_save_cache_key(self, cache_key, vote_p):
        if cache_key not in self.pvalue_cache.keys():
            self.pvalue_cache[cache_key] = {}
        if self.method not in self.pvalue_cache[cache_key].keys():
            self.pvalue_cache[cache_key][self.method] = {}
        vote_p[self.method].update(self.pvalue_cache[cache_key][self.method])
        self.pvalue_cache[cache_key].update(vote_p)

    @abstractmethod
    def ensemble_method(self, **kwargs):
        pass

    def vote_cache_management(self, X, Y, condition_set):
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        p = self.ensemble_method(X, Y, cache_key, condition_set)
        self.save_to_local_cache()
        return p


class VotingTest(EnsembleBase):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, uit_l_kwargs,
                 cit_l_kwargs, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, uit_l_kwargs,
                         cit_l_kwargs, **kwargs)
        self.method = 'vt'

    def vote(self, p_dict):
        vote = {k: v > self.alpha for k, v in p_dict.items()}
        vote_counts = Counter(vote.values())
        p = float(vote_counts.most_common(1)[0][0])  # p of voting is either 0 or 1
        return p

    def ensemble_method(self, x_idx: int, y_idx: int, cache_key, condition_set=None):
        p_dict = {k: v(x_idx, y_idx, condition_set) for k, v in self.uit_instance.items()} if condition_set is None \
            else {k: v(x_idx, y_idx, condition_set) for k, v in self.cit_instance.items()}
        p = self.vote(p_dict)
        p_dict[self.method] = {'p_vote': p, 'alpha': self.alpha}

        return p

    def __call__(self, X: int, Y: int, condition_set: list = None, **kwargs):
        return self.vote_cache_management(X, Y, condition_set)


class PValueEnsemble(EnsembleBase):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds,
                 p_combination, uit_l_kwargs, cit_l_kwargs, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, uit_l_kwargs,
                         cit_l_kwargs, **kwargs)
        self.p_combination = p_combination
        self.epsilon = 1e-5
        self.method = 'p_combination'

    def ensemble_method(self, x_idx: int, y_idx: int, cache_key, condition_set=None):
        p_dict = {k: v(x_idx, y_idx, condition_set) for k, v in self.uit_instance.items()} if condition_set is None \
            else {k: v(x_idx, y_idx, condition_set) for k, v in self.cit_instance.items()}
        if self.p_combination == 'cauchy':
            p = self.cauchy_combination(list(p_dict.values()))
        elif self.p_combination == 'fisher':
            p = self.fisher_combination(list(p_dict.values()))
        elif self.p_combination == 'stouffer':
            p = self.stouffer_combination(list(p_dict.values()))
        else:
            logging.error('p value combination must be "cauchy", "stouffer" or "fisher".')
            raise NotImplementedError
        p_dict[self.method] = {self.p_combination: p}
        self.ensemble_save_cache_key(cache_key, p_dict)
        return p

    def cauchy_combination(self, p_values):
        p_values = np.array(p_values)

        if np.any((p_values < 0) | (p_values > 1)):
            raise ValueError("All p value should in [0, 1].")

        p_values_clip = np.clip(p_values, self.epsilon, 1 - self.epsilon)
        weights = np.ones_like(p_values_clip) / len(p_values_clip)

        # Cauchy Transformation: tan((0.5 - p_i) * π)
        transformed = np.tan((0.5 - p_values_clip) * np.pi)
        # Calculate T Statistics
        T = np.sum(weights * transformed)
        # Calculate P Value
        p_comb = 0.5 - np.arctan(T) / np.pi
        return p_comb

    def fisher_combination(self, p_values):
        p_values = np.array(p_values)

        if np.any((p_values < 0) | (p_values > 1)):
            raise ValueError("All p value should in [0, 1].")

        p_values_clip = np.clip(p_values, self.epsilon, 1 - self.epsilon)

        k = len(p_values_clip)
        X2 = -2 * np.sum(np.log(p_values_clip))
        p_comb = chi2.sf(X2, df=2 * k)

        return p_comb

    def stouffer_combination(self, p_values):
        p_values = np.asarray(p_values)
        if np.any((p_values < 0) | (p_values > 1)):
            raise ValueError("All p value should in [0, 1].")

        p_values_clip = np.clip(p_values, self.epsilon, 1 - self.epsilon)
        z_scores = norm.ppf(1 - p_values_clip)

        weights = np.ones_like(p_values_clip)  # equal weights
        weighted_z = weights * z_scores
        combined_z = np.sum(weighted_z) / np.sqrt(np.sum(weights ** 2))

        p_comb = norm.sf(combined_z)  # survival function

        return p_comb

    def __call__(self, X: int, Y: int, condition_set: list = None, **kwargs):
        return self.vote_cache_management(X, Y, condition_set)


class EValueEnsemble(EnsembleBase):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, func,
                 uit_l_kwargs, cit_l_kwargs, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, uit_l_kwargs,
                         cit_l_kwargs, **kwargs)
        self.method = 'e_combination'
        self.func = func
        self.epsilon = 1e-5

    def ensemble_method(self, x_idx: int, y_idx: int, cache_key, condition_set=None):
        p_dict = {k: v(x_idx, y_idx, condition_set) for k, v in self.uit_instance.items()} if condition_set is None \
            else {k: v(x_idx, y_idx, condition_set) for k, v in self.cit_instance.items()}
        e_dict = {k: 1 / np.clip(p, self.epsilon, 1 - self.epsilon) for k, p in p_dict.items()}
        if self.func == 'product':
            e = np.prod(list(e_dict.values()))
        elif self.func == 'mean':
            e = np.mean(list(e_dict.values()))
        else:
            logging.error('e value combination must be "product", "mean".')
            raise NotImplementedError
        p_dict[self.method] = {self.func: e}
        return e

    def __call__(self, X: int, Y: int, condition_set: list = None, **kwargs):
        return self.vote_cache_management(X, Y, condition_set)

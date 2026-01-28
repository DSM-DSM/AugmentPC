import numpy as np
import json

import pandas as pd
from scipy.stats import skew
from scipy import stats
from math import sqrt, exp


class TesterAuxiliary:
    def __init__(self, alpha):
        self.container = {'FN': [], 'FP': []}
        self.alpha = alpha

    def detect(self, p_h0, p_refer, p_combination, t_res, X, Y, conditional_set):
        l = len(conditional_set) if conditional_set is not None else 0
        credibility = np.mean(t_res > self.alpha)
        X, Y = int(X), int(Y)
        if l > 0:
            conditional_set = tuple(map(int, conditional_set))
        else:
            conditional_set = ()
        detect_info = {
            'p_h0': p_h0,
            'p_refer': p_refer,
            'p_combination': p_combination,
            'test_problems': (X, Y, conditional_set),
            'k': len(conditional_set),
        }
        if p_combination < self.alpha < p_h0:
            # H0: p_h0 > α (X ⊥ Y|S) but p_combination < α --> FP
            self.container['FP'].append(detect_info)
        elif p_combination > self.alpha > p_h0:
            # H0: p_h0 < α (X not ⊥ Y|S) but p_combination > α --> FN
            self.container['FN'].append(detect_info)

    def save(self, filepath):
        """
        Save self.container as JSON file。

        :param filepath:
        """

        with open(filepath, 'w') as f:
            json.dump(self.container, f, indent=4)

    def summarize(self):
        FN, FP = self.container['FN'], self.container['FP']
        df_fp = pd.DataFrame(data=FP).drop_duplicates(subset=['test_problems'], keep='first')
        df_fn = pd.DataFrame(data=FN).drop_duplicates(subset=['test_problems'], keep='first')
        fn_summary, fp_summary = [], []

        if len(df_fp) > 0:
            group_df_fp = df_fp.groupby('k')
            for key, group in group_df_fp:
                fp_summary.append({
                    'k': key,
                    'p_refer_negative_num': int(np.sum(group['p_refer'] > self.alpha)),
                    'p_combination_positive_num': len(group),
                })

        if len(df_fn) > 0:
            group_df_fn = df_fn.groupby('k')
            for key, group in group_df_fn:
                fn_summary.append({
                    'k': key,
                    'p_refer_positive_num': int(np.sum(group['p_refer'] < self.alpha)),
                    'p_combination_negative_num': len(group),
                })
        return fn_summary, fp_summary


class SBoostTester(object):
    def __init__(self, tester_list, alpha=0.05, threshold=0.5, p_combination=None, gt_graph=None, ref_tester=None,
                 norm=1, epsilon=1e-5, weight_method='inv_prob_diff'):
        self.tester_list = tester_list
        self.repeat = len(tester_list)
        self.alpha = alpha
        self.threshold = threshold
        self.p_combination = p_combination
        self.gt_graph = gt_graph
        self.ref_tester = ref_tester
        self.norm = norm
        self.auxiliary = TesterAuxiliary(alpha, threshold)
        self.epsilon = epsilon
        self.weight_method = weight_method
        self.power_function = lambda cond_set_len: (epsilon - self.alpha) * exp(cond_set_len) + self.alpha
        self.loc, self.scale = self.repeat * (1 - alpha), self.repeat * alpha * (1 - alpha)

    def cauchy_combination(self, p_values, X, Y, conditional_set):
        d = len(p_values)

        p_values_clip = np.clip(p_values, self.epsilon, 1 - self.epsilon)
        # 权重默认等权重
        if self.ref_tester is not None:
            p_refer = np.clip(self.ref_tester(X, Y, conditional_set), self.epsilon, 1 - self.epsilon)
            dist = np.abs(p_values - p_refer) ** self.norm
            weights = self.imbalance_weight(dist)
        else:
            weights = np.ones(d) / d
        # 检查权重合法性
        assert np.isclose(np.sum(weights), 1), "Sum of weights is not 1."
        assert np.all(weights >= 0), "weights should be non-negative."

        # 柯西变换：tan((0.5 - p_i) * π)
        transformed = np.tan((0.5 - p_values_clip) * np.pi)
        # 计算检验统计量T
        T = np.sum(weights * transformed)
        # 计算p值
        p_value = 0.5 - np.arctan(T) / np.pi
        return p_value

    def cauchy_stable_combination(self, p_refer, p_cauchy, t_res, X, Y, conditional_set):
        credibility = np.mean(t_res > self.alpha)
        # l = len(conditional_set) if conditional_set is not None else 0
        if p_refer < self.alpha < p_cauchy:
            # H0: p_refer < α (X not ⊥ Y|S) but p_combination > α --> may FN
            return p_refer
        elif p_refer > self.alpha > p_cauchy:
            # H0: p_refer > α (X ⊥ Y|S) but p_combination < α --> may FP
            return p_cauchy
        else:
            return p_cauchy

    def cauchy_stable_combination_plus(self, p_refer, p_cauchy, t_res, X, Y, conditional_set):
        # How can we add l to cauchy combination methods?
        # Rules1: When l is 0, we are incline to reject H0
        # Rules2: When l is small, we are incline to reject H0
        # Rules3: When l is big, we incline to accept H0

        if p_refer < self.alpha < p_cauchy:
            # H0: p_refer < α (X not ⊥ Y|S) but p_combination > α --> may FN
            return p_refer
        elif p_refer > self.alpha > p_cauchy:
            # H0: p_refer > α (X ⊥ Y|S) but p_combination < α --> may FP
            return p_cauchy
        elif p_refer > self.alpha and p_cauchy > self.alpha:
            l = len(conditional_set) if conditional_set is not None else 0
            if l > 0:
                return p_cauchy
            else:
                # Consider the power of unconditional independence test
                y_obs = np.sum(t_res > self.alpha)
                p_binominal = 1 - stats.norm.cdf(y_obs, loc=self.loc, scale=self.scale)
                if p_binominal < self.alpha:
                    return p_cauchy
                else:
                    return 0
        else:
            return p_cauchy

    def combination(self, t_res, X, Y, conditional_set, p_refer):
        if self.p_combination == 'vote':
            p = int(np.mean(t_res > self.alpha) > self.threshold)
            return p
        elif self.p_combination == 'cauchy':
            p = self.cauchy_combination(t_res, X, Y, conditional_set)
            return p
        elif self.p_combination == 'cauchy_stable':
            p_cauchy = self.cauchy_combination(t_res, X, Y, conditional_set)
            p = self.cauchy_stable_combination(p_refer, p_cauchy, t_res, X, Y, conditional_set)
            return p
        elif self.p_combination == 'cauchy_stable_plus':
            p_cauchy = self.cauchy_combination(t_res, X, Y, conditional_set)
            p = self.cauchy_stable_combination_plus(p_refer, p_cauchy, t_res, X, Y, conditional_set)
            return p
        else:
            raise ValueError(f"Unknown p_combination: {self.p_combination}")

    def __call__(self, X, Y, conditional_set=None, *args, **kwargs):
        import networkx as nx
        p_refer = float(np.clip(self.ref_tester(X, Y, conditional_set), self.epsilon, 1 - self.epsilon))
        t_res = np.array([tester(X, Y, conditional_set) for tester in self.tester_list])
        if conditional_set is not None:
            p_h0 = float(nx.is_d_separator(self.gt_graph, {X}, {Y}, set(conditional_set)))
        else:
            p_h0 = float(nx.is_d_separator(self.gt_graph, {X}, {Y}, set()))
        if self.p_combination == 'original' and len(t_res) < 2:
            p_combination = t_res[0]
        else:
            p_combination = float(self.combination(t_res, X, Y, conditional_set, p_refer))

        self.auxiliary.detect(p_h0, p_refer, p_combination, t_res, X, Y, conditional_set)
        return p_combination

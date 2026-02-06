import math
import numpy as np
from scipy.optimize import minimize_scalar
from utils.tools import *


class MultiTestBase(object):
    def __init__(self, alpha, epsilon=1e-5):
        """
        Reference: E-Values, Multiple Test and Beyond
        :param alpha:
        """
        self.alpha = alpha
        self.epsilon = epsilon

    def procedure(self, p_list, t_func, e_func, t_range):
        # Find numerical Supremum
        threshold = self.func_supremum(p_list, t_func, t_range)
        reject = (p_list <= threshold).tolist()

        # Calculate e values
        if threshold > 0:
            e_values = [e_func(indicator, threshold) for indicator in reject]
        else:
            e_values = [math.inf if indicator else 0 for indicator in reject]
        return reject, e_values, threshold

    def BHProcedure(self, p_list):
        """

        :param p_list:
        :return:

        mtb = MultiTestBase(0.05)
        p_list = [0.001, 0.008, 0.039, 0.041, 0.5, 0.9, 0.003]
        reject, e_values, threshold = mtb.BHProcedure(p_list)
        """
        p_list = np.array(p_list)

        # R(t) represents the number of p_values less than 't'
        # Iterate R(t) to calculate T_BH and T_BC is the fastest Algorithm which has the time complexity of O(n).
        r_t_range = [int(np.sum(p_list <= v)) for v in np.unique(np.sort(p_list))]
        t_func = lambda r_t: len(p_list) * np.sort(p_list)[r_t - 1] / r_t
        e_func = lambda indicator, threshold: float(int(indicator) / threshold)
        return self.procedure(p_list, t_func, e_func, r_t_range)

    def BCProcedure(self, p_list):
        """

        :param p_list:
        :return:

        mtb = MultiTestBase(0.05)
        p_list = [0.001, 0.008, 0.039, 0.041, 0.5, 0.9, 0.003]
        reject, e_values, threshold = mtb.BCProcedure(p_list)
        """
        p_list = np.array(p_list)

        r_t_range = [int(np.sum(p_list <= v)) for v in np.unique(np.sort(p_list)) if v < 0.5]
        t_func = lambda r_t: (1 + np.sum(p_list >= (1 - np.sort(p_list)[r_t - 1]))) / r_t
        e_func = lambda indicator, threshold: float(
            (len(p_list) * int(indicator)) / (1 + np.sum(p_list >= (1 - threshold))))
        return self.procedure(p_list, t_func, e_func, r_t_range)

    def StoreyProcedure(self, p_list, lbd):
        """

        :param p_list:
        :param lbd:
        :return:

        mtb = MultiTestBase(0.05)
        reject, e_values, threshold = mtb.StoreyProcedure(p_list, lbd=0.3)
        """
        p_list = np.array(p_list)

        r_t_range = [int(np.sum(p_list <= v)) for v in np.unique(np.sort(p_list)) if v < lbd]
        pi_lambda = (1 + len(p_list) - np.sum(p_list <= lbd)) / ((1 - lbd) * len(p_list))
        t_func = lambda r_t: (len(p_list) * pi_lambda * np.sort(p_list)[r_t - 1]) / r_t
        e_func = lambda indicator, threshold: float(int(indicator) / (threshold * pi_lambda))
        return self.procedure(p_list, t_func, e_func, r_t_range)

    def eBHProcedure(self, e_values):
        """
        BH Procedure based on e values
        """
        n = len(e_values)
        descending_idx = np.argsort(e_values)[::-1]
        # Descending Order
        sorted_e_values = np.array(e_values)[descending_idx]

        # Calculate threshold
        f = lambda x: n / (x * self.alpha)
        e_threshold = np.array([f(i) for i in range(1, n + 1)])

        try:
            k_hat = np.where(sorted_e_values >= e_threshold)[0][-1]
        except IndexError:
            k_hat = -1
        reject_index = descending_idx[:k_hat + 1]
        reject = np.array([True if i in reject_index else False for i in range(n)])

        # For Validation
        # df = pd.DataFrame({'e_values': e_values, 'threshold': e_threshold, 'reject': reject})
        return reject

    def func_supremum(self, p_list, func, discrete_range):
        supremum_candidate = np.array([func(v) for v in discrete_range])
        supremum_candidate_index = np.where(supremum_candidate <= self.alpha)[0]
        if len(supremum_candidate_index) > 0:
            # In case there are 'Tie' in p values list.
            supremum = np.sort(p_list)[discrete_range[supremum_candidate_index[-1]] - 1]
        else:
            supremum = 0
        return supremum

    def func_inverse(self, func, y, range):
        """
        使用二分法求解f(x)=y，返回x的近似值。
        前提：f在[low, high]上单调连续，且y在f(low)和f(high)之间。
        """
        low, high = range
        while high - low > self.epsilon:
            mid = (low + high) / 2
            if func(mid) < y:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def e_values_boost(self, e_values):
        e_values = e_values.reshape(e_values.shape[0], -1)
        n, p = e_values.shape
        T = lambda x: n / np.ceil(n / x) * (x >= 1).astype(int)
        exp = lambda r, bk: np.mean(T(e_values[:, r] * self.alpha * b_k_min))

        b = []
        for i in range(p):
            b_k_min = 1
            b_k_max = 20
            while exp(i, b_k_max) <= self.alpha:
                b_k_min = b_k_max
                b_k_max += 20
            while b_k_max - b_k_min > self.epsilon:
                if exp(i, (b_k_min + b_k_max) / 2) < self.alpha:
                    b_k_min = (b_k_min + b_k_max) / 2
                else:
                    b_k_max = (b_k_min + b_k_max) / 2

            b.append(b_k_min)
        return np.array(b)

    def lab_top_k(self, e_values_boosted, y, top_k):
        n = len(y)
        label = np.zeros(n)

        indices = np.argsort(e_values_boosted)[-top_k:]
        label[indices] = 1

        reject = self.eBHProcedure(e_values_boosted)
        label = reject * label
        return label


class HybridMultiTest(MultiTestBase):
    def __init__(self, alpha, epsilon=1e-5, **kwargs):
        super(HybridMultiTest, self).__init__(alpha, epsilon)

    def HybAdaProcedure(self, p_list):
        n = len(p_list)
        mtb = MultiTestBase(self.alpha / (1 + self.alpha))
        reject_bh, e_values_bh, threshold_bh = mtb.BHProcedure(p_list)
        reject_bc, e_values_bc, threshold_bc = mtb.BCProcedure(p_list)

        weight_bh, weight_bc = [], []
        max_t_bh_j = 0
        for i in range(n):
            p_list_tilde_sub_i = tilde_substitute_i(p_list, i)
            _, _, t_bh_i = mtb.BHProcedure(p_list_tilde_sub_i)
            if t_bh_i > max_t_bh_j:
                max_t_bh_j = t_bh_i

            sum_of_indicator = 0
            for j in range(n):
                if j != i:
                    p_list_sub_j_i = substitute_i(p_list, j)
                    p_list_sub_j_i[i] = 0
                    _, _, t_bc_j_i = mtb.BCProcedure(p_list_sub_j_i)
                    sum_of_indicator += int(p_list[j] >= (1 - t_bc_j_i))
            weight_bh_i = t_bh_i / (t_bh_i + (1 + sum_of_indicator) / n)
            weight_bh.append(weight_bh_i)

        for i in range(n):
            sum_of_indicator = 0
            for j in range(n):
                if j != i:
                    sum_of_indicator += int(p_list[j] >= (1 - threshold_bc))
            weight_bc_i = (1 + sum_of_indicator) / (max_t_bh_j + (1 + sum_of_indicator) / n) / n
            weight_bc.append(weight_bc_i)

        weight_bh = np.array(weight_bh)
        weight_bc = np.array(weight_bc)

        weight_bh = np.clip(weight_bh, self.epsilon**2, 1 - self.epsilon**2)
        weight_bc = np.clip(weight_bc, self.epsilon**2, 1 - self.epsilon**2)
        weight_bh = weight_bh / (weight_bh + weight_bc)
        weight_bc = weight_bc / (weight_bh + weight_bc)
        e_values = weight_bh * e_values_bh + weight_bc * e_values_bc
        reject = self.eBHProcedure(e_values)
        return np.array(reject), e_values

    def FastHybAdaProcedure(self, p_list):
        n = len(p_list)
        mtb = MultiTestBase(self.alpha / (1 + self.alpha))
        reject_bh, e_values_bh, threshold_bh = mtb.BHProcedure(p_list)
        reject_bc, e_values_bc, threshold_bc = mtb.BCProcedure(p_list)

        weight_bh, weight_bc = [], []
        max_t_bh_j = 0
        for i in range(n):
            p_list_sub_i = tilde_substitute_i(p_list, i)
            _, _, t_bh_i = mtb.BHProcedure(p_list_sub_i)
            if t_bh_i > max_t_bh_j:
                max_t_bh_j = t_bh_i
            sum_of_indicator = 0
            for j in range(n):
                if j != i:
                    p_list_sub_j = substitute_i(p_list, j)
                    _, _, t_hc_j = mtb.BCProcedure(p_list_sub_j)
                    sum_of_indicator += int(p_list[j] >= (1 - t_hc_j))
            sum_of_indicator = 0

            weight_bh_i = t_bh_i / (t_bh_i + (1 + sum_of_indicator) / n)
            weight_bh.append(weight_bh_i)

        for i in range(n):
            sum_of_indicator = 0
            for j in range(n):
                if j != i:
                    sum_of_indicator += int(p_list[j] >= (1 - threshold_bc))
            weight_bc_i = (1 + sum_of_indicator) / (max_t_bh_j + (1 + sum_of_indicator) / n) / n
            weight_bc.append(weight_bc_i)

        weight_bh = np.array(weight_bh)
        weight_bc = np.array(weight_bc)

        e_values = weight_bh * e_values_bh + weight_bc * e_values_bc
        reject = self.eBHProcedure(e_values)
        return np.array(reject), e_values

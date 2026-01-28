from stat_test.init import *


class RecurrentRateUIT(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, rr_n_sim=100, lp=1, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha)
        self.n, self.p = data.shape
        self.N = int(self.n * (self.n - 1) / 2)
        self.An = np.full(self.N, np.nan)
        self.Bn = np.full(self.N, np.nan)
        self.Cn = np.full(self.N, np.nan)
        self.n_sim = rr_n_sim
        dist_metric_dic = {
            1: 'cityblock',
            2: 'euclidean',
            0: 'chebyshev'
        }
        self.dist_metric = dist_metric_dic[lp]
        self.method = 'rr'

    def tran_input2matrix(self, x):
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import norm

        mat = np.triu(squareform(
            pdist(x.reshape(-1, 1), metric=self.dist_metric))).reshape(1, -1)  # Pairwise distances for x
        mat = mat[mat != 0]
        mat = norm.cdf(mat, loc=np.mean(mat), scale=np.std(mat))  # Normalize using CDF
        mat_order = np.sort(mat)  # Flatten and sort Z by row
        return mat_order

    def _calculate_ir2(self, mat):
        l = len(mat)
        ir = 1 - (1 / (self.N * self.N)) * np.sum(np.arange(1, 2 * l + 1, 2) * mat)
        return ir

    def calculate_ir2(self, x):
        mat_order = self.tran_input2matrix(x)
        ir = self._calculate_ir2(mat_order)
        return ir, mat_order

    def calculate_cross_ir2(self, Z, T):
        An = np.full(self.N, np.nan)
        Bn = np.full(self.N, np.nan)
        Cn = np.full(self.N, np.nan)

        for i in range(self.N):
            An[i] = np.mean(
                (1 - (1 / 2) * (np.abs(Z[i] - Z) + Z[i] + Z)) *
                (1 - (1 / 2) * (np.abs(T[i] - T) + T[i] + T))
            )
            Bn[i] = np.mean(1 - (1 / 2) * (np.abs(T[i] - T) + T[i] + T))
            Cn[i] = np.mean(1 - (1 / 2) * (np.abs(Z[i] - Z) + Z[i] + Z))

        # Alternative implementation using broadcasting
        # However, this method is far slower.
        # Z_abs_diff = np.abs(Z[:, None] - Z)  # Precompute abs(Z[i] - Z[j])
        # T_abs_diff = np.abs(T[:, None] - T)  # Precompute abs(T[i] - T[j])
        # Z_part = 1 - 0.5 * (Z_abs_diff + Z[:, None] + Z)
        # T_part = 1 - 0.5 * (T_abs_diff + T[:, None] + T)
        # An = np.mean(Z_part * T_part, axis=1)
        # Bn = np.mean(T_part, axis=1)
        # Cn = np.mean(Z_part, axis=1)

        return np.mean(An), np.mean(Bn * Cn)

    def permutation(self, x, T, iry2):
        t = np.full(self.n_sim, np.nan)

        for j in range(self.n_sim):
            # Permute x
            x_perm = x[np.random.permutation(self.n)]
            irx2_perm, Z_perm = self.calculate_ir2(x_perm)
            # Recompute An, Bn, Cn for permuted x
            irxy2_perm, irxyrxry_perm = self.calculate_cross_ir2(Z_perm, T)

            # Compute test statistic for this permutation
            t[j] = self.n * (irxy2_perm + irx2_perm * iry2 - 2 * irxyrxry_perm)

            # print(inverse_r_t"Permutation {j + 1}/{self.n_sim}")
        return t

    def p_cal_func(self, Xs, Ys, condition_set):
        x, y = self.data[:, Xs], self.data[:, Ys]
        irx2, Z = self.calculate_ir2(x)
        iry2, T = self.calculate_ir2(y)
        irxy2, irxyrxry = self.calculate_cross_ir2(Z, T)
        tobserve = self.n * (irxy2 + irx2 * iry2 - 2 * irxyrxry)
        t = self.permutation(x, T, iry2)
        pvalue = np.mean(t > tobserve)
        return pvalue

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)

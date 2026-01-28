import torch
import numpy as np
from torch import nn
import math
from typing import Union, Tuple
from torch.utils.data import Dataset
import torch.nn.functional as F


# ==================== 辅助函数 ====================
def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m


def log_add_exp(a, b):
    maximum = torch.maximum(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个正态分布之间的KL散度（以对数方差形式输入）

    参数:
        mean1: 第一个分布的均值
        logvar1: 第一个分布的对数方差
        mean2: 第二个分布的均值
        logvar2: 第二个分布的对数方差

    返回:
        kl: KL( N(mean1,exp(logvar1)) || N(mean2,exp(logvar2)) )
    """
    # 数值稳定性处理：限制对数方差范围
    logvar1 = torch.clamp(logvar1, min=-30.0, max=20.0)
    logvar2 = torch.clamp(logvar2, min=-30.0, max=20.0)

    # 计算KL散度
    var_ratio = torch.exp(logvar1 - logvar2)
    t1 = 0.5 * (var_ratio - 1.0)
    t2 = 0.5 * (mean1 - mean2).pow(2) * torch.exp(-logvar2)
    t3 = 0.5 * (logvar2 - logvar1)

    kl = t1 + t2 + t3

    # 确保结果非负（数值稳定性）
    return torch.clamp(kl, min=0.0)


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def sum_except_batch(x):
    return x.reshape(x.shape[0], -1).sum(dim=1)


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def sliced_logsumexp(x, slices):
    lse = torch.logcumsumexp(
        torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')),
        dim=-1)

    slices = slices.long()
    slice_starts = slices[:-1]
    slice_ends = slices[1:]
    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(
        slice_lse,
        slice_ends - slice_starts,
        dim=-1
    )
    return slice_lse_repeated


def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))

    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot


def ohe_to_categories(ohe, num_classes):
    offsets = np.cumsum(num_classes)
    x = torch.zeros(ohe.size(0), len(num_classes), dtype=torch.long)
    for i in range(len(num_classes)):
        x[:, i] = ohe[:, offsets[i] - num_classes[i]:offsets[i]].argmax(dim=1)
    return x


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# ==================== 模型核心 ====================
class GaussianMultinomialDiffusion(nn.Module):
    def __init__(self, category_feature_num_list: np.array, numerical_features_num: int, denoise_fn, num_timesteps=1000,
                 gaussian_loss_type='mse', gaussian_parametrization='eps', multinomial_loss_type='vb_stochastic',
                 parametrization='x0', scheduler='cosine', device=torch.device('cpu')):
        super().__init__()
        assert multinomial_loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        self.numerical_features_num = numerical_features_num
        self.category_feature_num_list = category_feature_num_list
        self.category_feature_num = np.sum(category_feature_num_list)
        if np.sum(self.category_feature_num_list) > 0:
            self.num_classes_expanded = torch.from_numpy(np.concatenate(
                [category_feature_num_list[i].repeat(category_feature_num_list[i]) for i in
                 range(len(category_feature_num_list))])).to(device)

            self.slices_for_classes = [np.arange(category_feature_num_list[0])]
            offsets = np.cumsum(self.category_feature_num_list)

            for i in range(1, len(offsets)):
                self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
            self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)
        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        # 计算噪声调度相关参数
        alphas = 1. - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion
        self.posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))).float().to(device)
        self.posterior_mean_coef1 = (betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float().to(device)
        self.posterior_mean_coef2 = (
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas.numpy()) / (1.0 - alphas_cumprod)).float().to(device)

        # 注册缓冲区
        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device))

        # 高斯扩散相关参数
        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))

    def gaussian_q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_1_min_cumprod_alpha, t, x_start.shape)
        return mean, variance, log_variance

    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x.device), (1. - self.alphas)[1:]],
                                   dim=0)
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)

        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError

        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded)
        )
        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded)
        )
        return log_probs

    def predict_start(self, model_out, log_x_t, t, out_dict):
        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def mixed_loss(self, x, out_dict):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')

        x_num = x[:, :self.numerical_features_num]
        x_cat = x[:, self.numerical_features_num:]

        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)  # add noise to original data
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.category_feature_num)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)

        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)
        model_out = self._denoise_fn(x_in, t, **out_dict)  #

        model_out_num = model_out[:, :self.numerical_features_num]
        model_out_cat = model_out[:, self.numerical_features_num:]

        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, out_dict) / len(
                self.category_feature_num)
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)

        return loss_multi.mean(), loss_gauss.mean()

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def gaussian_p_sample(self, model_out, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None, ):
        out = self.gaussian_p_mean_variance(model_out, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                            model_kwargs=model_kwargs, )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"X_num": sample, "pred_xstart": out["pred_xstart"]}

    @torch.no_grad
    def sample(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.numerical_features_num), device=device)

        has_cat = self.category_feature_num != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(y_dist, num_samples=b, replacement=True)  # take effect when denoise_fn.is_y_cond = True
        out_dict = {'y': y.long().to(device)}

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
            model_out_num = model_out[:, :self.numerical_features_num]
            model_out_cat = model_out[:, self.numerical_features_num:]
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['X_num']
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.category_feature_num)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t, out_dict):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t, out_dict=out_dict)
        out = self.log_sample_categorical(model_log_prob)
        return out

    def _vb_terms_bpd(self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], "out_mean": out["mean"], "true_mean": true_mean}

    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        terms = {}
        if self.gaussian_loss_type == 'mse':
            return mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            terms = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            return terms["output"]
        return terms['loss']

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # EV_log_qxt_x0 = self.q_pred(log_x_start, t)

        # print('sum exp', EV_log_qxt_x0.exp().sum(1).mean())
        # assert False

        # log_qxt_x0 = (log_x_t.exp() * EV_log_qxt_x0).sum(dim=1)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32))

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - sliced_logsumexp(unnormed_logprobs, self.offsets)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t, out_dict):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
        else:
            raise ValueError
        return log_model_pred

    def compute_Lt(self, model_out, log_x_start, log_x_t, t, out_dict, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t, out_dict=out_dict)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt, out_dict):
        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_Lt(model_out, log_x_start, log_x_t, t, out_dict)
            kl_prior = self.kl_prior(log_x_start)
            return kl / pt + kl_prior
        elif self.multinomial_loss_type == 'vb_all':
            return -self.nll(log_x_start)
        else:
            raise ValueError()

    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.category_feature_num)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        return index_to_log_onehot(full_sample, self.category_feature_num)

    def q_sample(self, log_x_start, t):
        """对分类数据进行前向扩散过程"""
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

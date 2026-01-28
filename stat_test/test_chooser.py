from stat_test.init import *
from argparse import Namespace

from stat_test.u_cit import ensemble


def stat_test_chooser(data, config, cache_config):
    algo = config.evaluation.algorithm.algo
    test_config = Namespace(**vars(config.test), **vars(cache_config))
    if algo in ['pc', 'deduce_dep_pc','fdr_pc']:
        uit = _stat_test_chooser(data, test_config, config.test.uit)
        cit = _stat_test_chooser(data, test_config, config.test.cit)
    else:
        logging.warning(f'Algorithm does {algo} not support yet !')
        raise ValueError("Unknown algorithm: {}".format(algo))
    return uit, cit


def _stat_test_chooser(data, test_config, test_name):
    keys = ['cache_path', 'use_cache', 'save_cache', 'device', 'save_cache_cycle_seconds', 'alpha']
    try:
        test_kwargs = getattr(test_config, test_name).__dict__
    except AttributeError:
        test_kwargs = test_config.__dict__
    test_kwargs.update({k: getattr(test_config, k) for k in keys})
    if test_name in ['vt', 'pval_ensemble', 'eval_ensemble']:
        test_kwargs.update({
            'cit_l_kwargs': {cit: dict2namespace(
                namespace2dict(getattr(test_config, cit, Namespace())) | {k: getattr(test_config, k) for k in keys}) for
                cit in
                test_kwargs.get('cit_l')},
            'uit_l_kwargs': {uit: dict2namespace(
                namespace2dict(getattr(test_config, uit, Namespace())) | {k: getattr(test_config, k) for k in keys}) for
                uit in
                test_kwargs.get('uit_l')},
        })

    if test_name == fisherz:
        from stat_test.u_cit.correlation import Fisherz
        return Fisherz(data, **test_kwargs)
    elif test_name == spearman:
        from stat_test.u_cit.correlation import Spearmanz
        return Spearmanz(data, **test_kwargs)
    elif test_name == kendall:
        from stat_test.u_cit.correlation import Kendallz
        return Kendallz(data, **test_kwargs)
    elif test_name == robustQn:
        from stat_test.u_cit.correlation import RobustQnz
        return RobustQnz(data, **test_kwargs)
    elif test_name == kci:
        from stat_test.u_cit.kernel import KCI
        return KCI(data, **test_kwargs)
    elif test_name == d_separation:
        # true dag is required!
        from stat_test.ground_truth import D_Separation
        assert hasattr(test_config, 'true_dag'), "D-separation requires true dag !"
        assert isinstance(test_config.true_dag, nx.DiGraph), "True dag must be a networkx.DiGraph"
        return D_Separation(data, **test_kwargs)
    elif test_name == hsic:
        from stat_test.u_cit.hsic import HSIC
        return HSIC(data, **test_kwargs)
    elif test_name == classifier:
        from stat_test.u_cit.classfier import Classifier
        return Classifier(data, **test_kwargs)
    elif test_name == conditional_distance:
        from stat_test.u_cit.cond_dist import ConditionalDistance
        return ConditionalDistance(data, **test_kwargs)
    elif test_name == gcm:
        from stat_test.u_cit.gcm import GCM
        return GCM(data, **test_kwargs)
    elif test_name == knn:
        from stat_test.cit.knn.knn import KNN
        return KNN(data, **test_kwargs)
    elif test_name == gan:
        from stat_test.cit.gan.gan import GAN
        return GAN(data, **test_kwargs)
    elif test_name == lp:
        from stat_test.cit.lp.lp import Lp
        return Lp(data, **test_kwargs)
    elif test_name == diffusion:
        from stat_test.cit.diffusion.diffusion import Diffusion
        return Diffusion(data, **test_kwargs)
    elif test_name == rruit:
        from stat_test.uit.rr import RecurrentRateUIT
        return RecurrentRateUIT(data, **test_kwargs)
    elif test_name == copc:
        from stat_test.u_cit.copc import Copula_Fisherz
        return Copula_Fisherz(data, **test_kwargs)
    elif test_name == wgcm:
        from stat_test.cit.wgcm import WeightedGCM
        return WeightedGCM(data, **test_kwargs)
    elif test_name == dgan:
        from stat_test.cit.dgan.dgan import DoubleGAN
        return DoubleGAN(data, **test_kwargs)
    elif test_name == vt:
        from stat_test.u_cit.ensemble import VotingTest
        return VotingTest(data, **test_kwargs)
    elif test_name == pval_ensemble:
        from stat_test.u_cit.ensemble import PValueEnsemble
        return PValueEnsemble(data, **test_kwargs)
    elif test_name == eval_ensemble:
        from stat_test.u_cit.ensemble import EValueEnsemble
        return EValueEnsemble(data, **test_kwargs)
    else:
        raise ValueError("Unknown test_name: {}".format(test_name))

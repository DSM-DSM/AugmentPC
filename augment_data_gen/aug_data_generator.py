import numpy as np
from copy import deepcopy
from utils.tools import namespace2dict, dict2namespace, WeightDistributionFunction
from torch import device
import logging
from argparse import Namespace
from data_gen.noise_generator import noise_func_chooser
from data_gen.root_generator import cause_func_chooser
from data_gen.grid_data_generator import copula_transform
import os


class AugDataGenerator(object):
    def __init__(self, obj, param, original_data):
        self.obj = obj
        self.param = param
        self.original_data = original_data
        self.ori_n, self.ori_p = original_data.shape
        self.generator_method = param['sample_augment_method']
        self.kv = tuple(param[k] for k in param)
        self.gt_graph = self.obj.original_sample_group[self.kv]['gt_graph']
        self.aug_sample_number = obj.augment_simu_config.n_samples
        self.imbalance_ratio = 0.9
        self.seed = param['seed']

    def generate(self):
        param = deepcopy(self.param)
        if self.generator_method == 'tabdiffusion':
            tabDiffusion = self.obj.original_sample_group[self.kv]['model']
            np.random.seed(self.seed)
            augment_data = tabDiffusion.sample(self.aug_sample_number).detach()
            if augment_data.device != device('cpu'):
                augment_data = augment_data.cpu()
            param['data'], param['graph'] = augment_data.numpy(), self.gt_graph

        elif self.generator_method == 'gt_graph':
            from data_gen.dag_generator import AcyclicGraphGeneratorPlus

            param_dc = deepcopy(param)
            # Specifying DAG Generator Kwargs
            param_dc['n_samples'] = self.aug_sample_number
            param_dc['copula_trans'] = self.obj.simu_config.ori.copula_trans
            param_dc['copula_trans_kwargs'] = self.obj.simu_config.ori.copula_trans_kwargs
            param_dc['dag_type'] = self.obj.simu_config.ori.dag_type
            param_dc['noise_coeff'] = self.obj.simu_config.ori.noise_coeff
            param_dc['designate'] = self.obj.simu_config.ori.designate
            wdf = eval('WeightDistributionFunction')(
                self.obj.weight_distribution_function,
                random_state=self.seed,
                **self.obj.simu_config.weight_kwargs
            )
            param_dc['weight_distribution_function'] = wdf()
            param_dc.update(namespace2dict(self.obj.simu_config.ori.mechanism_kwargs))
            param_dc['noise'], param_dc['root'] = noise_func_chooser(param['noise']), cause_func_chooser(param['root'])

            generator = AcyclicGraphGeneratorPlus(**param_dc)
            generator.designate_graph = self.gt_graph
            # To Confirm Generator Equal to Original One
            # genrator2 = self.obj.original_sample_group[self.kv]['generator']
            # If Mechanism is Polynomial, then
            # flag = genrator2.cfunctions[0].polycause == genrator2.cfunctions[0].polycause

            augment_data, pure_signal, graph = generator.generate(False)
            if param_dc['copula_trans']:
                dfn, dfd = param_dc['copula_trans_kwargs'].dfn, param_dc['copula_trans_kwargs'].dfd
                augment_data = copula_transform(augment_data, dfn=dfn, dfd=dfd)
                pure_signal = copula_transform(pure_signal, dfn=dfn, dfd=dfd)

            param['data'], param['graph'] = augment_data.values, self.gt_graph

        elif self.generator_method == 'SMOTE':
            from imblearn.over_sampling import SMOTE

            os.environ['LOKY_MAX_CPU_COUNT'] = '0'
            np.random.seed(self.param['seed'])
            smote = SMOTE(random_state=self.param['seed'], **namespace2dict(self.obj.model_params.SMOTE))
            y = np.random.binomial(1, self.imbalance_ratio, self.ori_n)
            # How to specify n sample new?
            augment_data, _ = smote.fit_resample(self.original_data, y)
            param['data'], param['graph'] = augment_data[:self.aug_sample_number, :], self.gt_graph

        elif self.generator_method == 'AdaSyn':
            from imblearn.over_sampling import ADASYN

            os.environ['LOKY_MAX_CPU_COUNT'] = '0'
            np.random.seed(self.param['seed'])
            adasyn = ADASYN(random_state=self.param['seed'], **namespace2dict(self.obj.model_params.AdaSyn))
            y = np.random.binomial(1, self.imbalance_ratio, self.ori_n)
            augment_data, _ = adasyn.fit_resample(self.original_data, y)
            param['data'], param['graph'] = augment_data[:self.aug_sample_number, :], self.gt_graph

        else:
            logging.error(f'Augment data generation method {self.generator_method} is not valid !')
            raise NotImplementedError

        return param

import numpy as np
from copy import deepcopy

from cdt.data import AcyclicGraphGenerator

from utils.tools import namespace2dict, dict2namespace, WeightDistributionFunction, TemporaryRandomSeed
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
            param['data'], param['graph'] = self.obj.original_sample_group[self.kv]['gt_aug_sample'], self.gt_graph

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

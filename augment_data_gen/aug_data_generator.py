import numpy as np
from copy import deepcopy

from cdt.data import AcyclicGraphGenerator

from utils.tools import namespace2dict, dict2namespace, WeightDistributionFunction, TemporaryRandomSeed, compute_hash
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
        if self.generator_method == 'tabDiffusion':
            # Initialize Tabular Diffusion Model
            from augment_data_gen.tabDiffusion.pipeline import tabDiffusionPipeline

            np.random.seed(param['seed'])
            y = np.random.rand(self.ori_n, 1)
            data_hash = compute_hash({'original_data': str(self.original_data), 'y': str(y)})
            tabDiffusion_model = tabDiffusionPipeline(
                self.original_data, y, X_cat=None, device=self.obj.configs.test.device, data_hash=data_hash,
                train_hash=self.obj.train_hash,
                model_dir=self.obj.model_dir, **namespace2dict(self.obj.model_params.tabDiffusion)
            )
            # Training
            tabDiffusion_model.train()

            augment_sample_arr = np.zeros((0, self.ori_p))
            while augment_sample_arr.shape[0] < self.aug_sample_number:
                augment_data = tabDiffusion_model.sample(self.ori_n * 10).detach()
                if augment_data.device != device('cpu'):
                    augment_data = augment_data.cpu().numpy()

                from imblearn.under_sampling import EditedNearestNeighbours
                enn = EditedNearestNeighbours(**namespace2dict(self.obj.model_params.ENN))
                y = np.random.binomial(1, self.imbalance_ratio, self.ori_n * 10)
                data, _ = enn.fit_resample(augment_data, y)
                data = data.reshape(-1, self.ori_p)
                augment_data_arr = np.concatenate((augment_sample_arr, data), axis=0)

            param['data'], param['graph'] = augment_data_arr[:self.aug_sample_number, :], self.gt_graph

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

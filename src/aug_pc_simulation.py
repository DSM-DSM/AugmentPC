import time
from copy import deepcopy
from src.simulation import Simulation
import logging, tempfile, joblib, shutil
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.tools import create_folder, compute_hash, dump_yaml, namespace2dict, WeightDistributionFunction
from itertools import product
import pandas as pd
import os, pickle, yaml
import numpy as np
from torch import device


class SimulationAugPC(Simulation):
    def __init__(self, args, configs):
        self.pre_simu_config = None
        self.pre_simu_config_path = None
        self.train_config_path = None
        self.pre_train_config = None
        self.args = args
        self.configs = configs
        self.key2compare = None
        self.file_loaded = False  # 文件是否是从已有文件中读取的
        self.simu_config = configs.simulation
        self.ori_simu_config = self.simu_config.ori
        self.augment_simu_config = self.simu_config.augment
        self.mode = configs.mode
        self.storage = configs.storage

        self.weight_distribution_function = self.simu_config.ori.weight_distribution_function

        self.pre_ori_simu_config = None
        self.model_dir = os.path.join(configs.storage.root_dir, configs.storage.model_dir)
        self.original_sample_group = {}
        self.keys2group = ['noise', 'root', 'causal_mechanism', 'n_samples', 'nodes',
                           'expected_degree', 'g_id', 'd_id', 'sample_augment_method']
        self.augment_sample_group = []
        self.model_params = self.augment_simu_config.model_params
        self.train_hash = compute_hash(namespace2dict(self.model_params))
        self.augment_file_info = {}

    def generate_param_list(self):
        param_combinations = list(
            product(self.ori_simu_config.noise_type, self.ori_simu_config.root,
                    self.ori_simu_config.causal_mechanism, self.ori_simu_config.n_samples,
                    self.ori_simu_config.n_points, self.ori_simu_config.exp_degree,
                    list(range(self.ori_simu_config.graph_num)), list(range(self.ori_simu_config.replicates)),
                    self.augment_simu_config.models))

        param_df = pd.DataFrame(param_combinations, columns=self.keys2group)

        param_df['seed'] = param_df['g_id'] * self.ori_simu_config.graph_num + param_df['d_id'] + self.args.seed
        self.params_combination = [{k: v for k, v in m.items()} for m in param_df.to_dict(orient='records')]

    def generate_grid_true_sample(self, param):
        obj = deepcopy(self)
        obj.simu_config = obj.ori_simu_config
        simu_param = Simulation._get_grid_simulation(obj, param)
        return simu_param['data'], simu_param['graph']

    def generate_original_sample(self):
        def generate_original_sample_group(obj):
            param_df = pd.DataFrame(self.params_combination, columns=self.keys2group + ['seed'])
            true_sample_param_list_dict = [{k: v for k, v in m.items()} for m in param_df.to_dict(orient='records')]
            loop = tqdm(true_sample_param_list_dict, desc='Original Sample Simulation:', leave=True)

            if not obj.mode.parallel:
                original_simu_res_list = []
                for param in loop:
                    true_simu_res = obj.generate_grid_true_sample(param)  # true data and graph
                    original_simu_res_list.append(true_simu_res)
            else:
                temp_folder = tempfile.mkdtemp(prefix='original_simu_', dir=obj.storage.temp_dir)
                try:
                    n_jobs = min(len(loop), obj.mode.n_jobs)
                    with joblib.parallel_backend('loky'):
                        original_simu_res_list = Parallel(n_jobs=n_jobs, temp_folder=temp_folder)(
                            delayed(obj.generate_grid_true_sample)(param) for param in loop)
                finally:
                    shutil.rmtree(temp_folder, ignore_errors=True)

            original_sample_group = {}
            for simu_id, param in enumerate(true_sample_param_list_dict):
                kv = tuple(list(param.values()))
                data, graph = original_simu_res_list[simu_id]
                original_sample_group[kv] = {'ori_data': data, 'gt_graph': graph}
            return original_sample_group

        self.pre_simu_config_path = os.path.join(self.storage.data_dir, 'simulation_config.yml')
        if os.path.exists(self.pre_simu_config_path):
            with open(self.pre_simu_config_path, 'r') as f:
                self.pre_simu_config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.pre_simu_config = None
        try:
            if self.pre_simu_config is not None:
                pkl_file_path = os.path.join(self.storage.data_dir, 'original_sample_group.pkl')
                if self.check_config(self.pre_simu_config.ori, self.ori_simu_config) and os.path.exists(pkl_file_path):
                    with open(pkl_file_path, 'rb') as f:
                        file = pickle.load(f)
                        self.original_sample_group = {k: v for k, v in file['sample_group'].items() if k[-2] in self.augment_simu_config.models}
                        if not self.original_sample_group:
                            self.original_sample_group = generate_original_sample_group(self)
                        else:
                            logging.info(f'Original data has been loaded from {pkl_file_path}')
                else:
                    self.original_sample_group = generate_original_sample_group(self)
            else:
                self.original_sample_group = generate_original_sample_group(self)
        except AttributeError as e:
            logging.error(f'There is an Error: "{e}", when loading original data.')
            self.original_sample_group = generate_original_sample_group(self)

    def check_augment_data_exist_hash(self):
        self.train_config_path = os.path.join(self.storage.data_dir, 'model_params.yml')
        if os.path.exists(self.train_config_path):
            with open(self.train_config_path, 'r') as f:
                self.pre_train_config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.pre_train_config = None
        try:
            # Check original simulation configurations
            if self.pre_simu_config and self.pre_train_config:
                # Original Simulation Configurations
                s1 = self.check_config(self.pre_simu_config.ori, self.ori_simu_config)
                # augment Simulation Configurations
                s2 = self.check_config(self.pre_simu_config.augment, self.augment_simu_config)
                if s1 and s2:
                    # Check training simulation configurations
                    for model in self.augment_simu_config.models:
                        pkl_file_path = os.path.join(self.storage.data_dir, f'{model}_augmented_sample_group.pkl')

                        pre_train_conf = vars(self.pre_train_config)[model]
                        curr_train_conf = vars(self.model_params)[model]
                        self.augment_file_info[model] = {}
                        if os.path.exists(pkl_file_path) and self.check_config(pre_train_conf, curr_train_conf):
                            with open(pkl_file_path, 'rb') as f:
                                file = pickle.load(f)
                                self.augment_file_info[model]['load'] = True
                                self.augment_file_info[model]['sample_group'] = file['sample_group']
                                logging.info(f'Augmented data of {model} have been loaded from {pkl_file_path}')
                        else:
                            # train config mismatch or pkl_file doesn't exist or both
                            self.augment_file_info[model]['load'] = False
                else:
                    # True or augment Simulation Mismatch
                    self.augment_file_info = {model: {'load': False} for model in self.augment_simu_config.models}
            else:
                # can't determine data source
                self.augment_file_info = {model: {'load': False} for model in self.augment_simu_config.models}
        except AttributeError as e:
            logging.error(f'There is an Error: "{e}", when loading augmented data.')
            self.augment_file_info = {model: {'load': False} for model in self.augment_simu_config.models}

    def generate_augment_sample(self):
        self.generate_original_sample()  # self.original_sample_group has been created
        self.check_augment_data_exist_hash()  # augment_file_info

        create_folder(self.storage.model_dir)
        # Training augment Sample Generative Models
        for model, value in self.augment_file_info.items():
            if not value['load']:
                self.train_augment_sample_generator(model)
                logging.info(f'Using {model} to generate augmented sample !')
                augment_sample_group = self.generate_aug_sample(model)
                self.augment_file_info[model]['sample_group'] = augment_sample_group
        self.save_simulation()

    def generate_aug_sample(self, gen_method):
        from augment_data_gen.aug_data_generator import AugDataGenerator

        ori_original_data = np.array([value['ori_data'] for key, value in self.original_sample_group.items()])
        params_combination = [param for param in self.params_combination if
                              param['sample_augment_method'] == gen_method]
        loop = tqdm(params_combination, desc='Augmented Sample Simulation:', leave=True)

        augment_sample_group = []
        for num, param in enumerate(loop):
            generator = AugDataGenerator(self, param, ori_original_data[num])
            augment_sample_group.append(generator.generate())

        return augment_sample_group

    def save_simulation(self):
        dump_yaml(self.pre_simu_config_path, self.simu_config)
        dump_yaml(self.train_config_path, self.model_params)

        # Save augment Sample For All Models
        for model, value in self.augment_file_info.items():
            self.augment_sample_group.extend(value['sample_group'])
            if not value['load']:
                with open(os.path.join(self.storage.data_dir, f'{model}_augmented_sample_group.pkl'), 'wb') as f:
                    pickle.dump({'sample_group': value['sample_group']}, f)

        with open(os.path.join(self.storage.data_dir, 'original_sample_group.pkl'), 'wb') as f:
            pickle.dump({'sample_group': self.original_sample_group}, f)

    def train_augment_sample_generator(self, gen_method):
        def _train_grid_augment_sample_generator(obj, key, value):
            from augment_data_gen.tabDiffusion.pipeline import tabDiffusionPipeline

            gen_method = key[-1]
            if gen_method == 'tabDiffusion':
                np.random.seed(key[-2])
                original_data = value['ori_data']
                y = np.random.rand(original_data.shape[0], 1)
                data_hash = compute_hash({'original_data': str(original_data), 'y': str(y)})
                tabDiffusion_model = tabDiffusionPipeline(
                    original_data, y, X_cat=None, device=obj.configs.test.device, data_hash=data_hash,
                    train_hash=obj.train_hash,
                    model_dir=obj.model_dir, **namespace2dict(obj.model_params.tabDiffusion)
                )
                # Loading Pre-Trained Models of Starting Training
                tabDiffusion_model.train()
                return tabDiffusion_model
            else:
                return None

        sub_original_sample_group = {k: v for k, v in self.original_sample_group.items() if k[-1] == gen_method}
        train_loop = tqdm(sub_original_sample_group.items(), desc=f'Training Data Augment Generator: {gen_method}',
                          leave=True)

        model_obj_list = []
        for key, value in train_loop:
            model_obj = _train_grid_augment_sample_generator(self, key, value)
            model_obj_list.append(model_obj)
            self.original_sample_group[key]['model'] = model_obj

    def __call__(self, *args, **kwargs):
        logging.info('*' * 100)
        create_folder(self.storage.data_dir)
        logging.info('Start to generate Original and Augment samples !')
        self.generate_param_list()
        self.generate_augment_sample()
        logging.info('*' * 100)
        logging.info(f'Generating  Original and Augment samples is over !')
        logging.info('*' * 100)

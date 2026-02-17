import pandas as pd
import numpy as np
from copy import deepcopy
import logging, joblib, shutil, tempfile, hashlib, os
from src.evaluation import Evaluation
from src.aug_pc_simulation import SimulationAugPC
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.tools import find_and_delete_invalid_json_cache, namespace2dict, metric_calculation, create_folder
from stat_test.test_chooser import stat_test_chooser


class EvaluationAugPC(SimulationAugPC, Evaluation):
    def __init__(self, args, configs):
        super().__init__(args, configs)
        self.test_config = configs.test
        self.eval_config = configs.evaluation
        self.cache_config = configs.cache

        # Multiple succession follows MRO order
        # print(EvaluationAugPC.__mro__)
        self.cache_config = self.configs.cache
        self.raw_eval_res = []
        self.ci_sbtester_auxiliary_dir = os.path.join(self.storage.root_dir, self.storage.auxiliary,
                                                      self.configs.test.uit + '_' + self.configs.test.cit)

        self.ori_pc_eval_result = {}
        self.aug_pc_eval_result = {}

    def tester_aggregation(self):
        """Combining Augmented CIT and UIT Testers with Original's"""
        for param, uit, cit in self.raw_eval_res:
            kv = tuple(param[k] for k in param if k in self.keys2group or k == 'seed')
            self.original_sample_group[kv]['aug_uit'] = uit
            self.original_sample_group[kv]['aug_cit'] = cit
            self.original_sample_group[kv]['aug_data'] = param['data']

    def pc(self, key, aug=True):
        param = {self.keys2group[i]: key[i] for i in range(len(self.keys2group))}
        param['graph'] = self.original_sample_group[key]['gt_graph']
        param['seed'] = self.simu_config.seed

        param['data'] = self.original_sample_group[key]['aug_data'] if aug else self.original_sample_group[key][
            'ori_data']

        param, uit, cit = Evaluation._evaluate_grid_parameter(self, param)
        metrics_dict = metric_calculation(param['esti_graph'], param['graph'])
        parma_dc = deepcopy(param)
        del parma_dc['data'], parma_dc['graph']
        parma_dc.update(metrics_dict)

        if aug:
            return parma_dc
        else:
            return parma_dc, uit, cit

    def fdr_pc(self, key, aug=True):
        from fdr_control.multi_test import HybridMultiTest
        from algo.algo_chooser import AlgoChooser

        param = {self.keys2group[i]: key[i] for i in range(len(self.keys2group))}
        gt_graph = self.original_sample_group[key]['gt_graph']
        self.configs.test.true_dag = gt_graph
        hmt = HybridMultiTest(**namespace2dict(self.eval_config.multiple_test_kwargs))
        algo_chooser = AlgoChooser(self.eval_config.algorithm)
        algo = algo_chooser.choose()

        if aug:
            param['esti_graph'] = algo(
                self.original_sample_group[key]['aug_data'],
                self.original_sample_group[key]['aug_uit'],
                self.original_sample_group[key]['aug_cit'],
                self.test_config.alpha,
                hmt=hmt,
                **namespace2dict(self.eval_config.algorithm.fdr_pc),
            )
            metrics_dict = metric_calculation(param['esti_graph'], gt_graph)
            param.update(metrics_dict)
            return param
        else:
            data = self.original_sample_group[key]['ori_data']
            data_hash = hashlib.md5(str(data).encode('utf-8')).hexdigest()
            cache_path = os.path.join(self.storage.cache_dir, data_hash + '.json')
            cache_config = self.cache_config
            cache_config.cache_path = cache_path
            uit, cit = stat_test_chooser(data, self.configs, cache_config)
            param['esti_graph'] = algo(
                data, uit, cit, self.test_config.alpha,
                hmt=hmt, **namespace2dict(self.eval_config.algorithm.fdr_pc),
            )
            metrics_dict = metric_calculation(param['esti_graph'], gt_graph)
            param.update(metrics_dict)
            return param, uit, cit

    def loading_augment_tester(self):
        def loading_tester(obj, param):
            # skipping pc fit for augment sample
            data_hash = hashlib.md5(str(param['data']).encode('utf-8')).hexdigest()
            cache_path = os.path.join(obj.storage.cache_dir, data_hash + '.json')
            cache_config = obj.cache_config
            cache_config.cache_path = cache_path
            obj.configs.test.true_dag = param['graph']

            uit, cit = stat_test_chooser(param['data'], obj.configs, cache_config)
            return param, uit, cit

        json_file_name = [hashlib.md5(str(item['data']).encode('utf-8')).hexdigest() + '.json' for item in
                          self.augment_sample_group]
        find_and_delete_invalid_json_cache(self.storage.cache_dir, json_file_name)
        augment_sample_loop = tqdm(self.augment_sample_group, desc='Loading Augment Tester:', leave=True)

        for param in augment_sample_loop:
            # Sample Concatenation
            kv = tuple(param[k] for k in param if k in self.keys2group or k == 'seed')
            ori_data = self.original_sample_group[kv]['ori_data']
            param['data'] = np.concatenate((ori_data, param['data']), axis=0)
            res = loading_tester(self, param)
            self.raw_eval_res.append(res)

    def eval_original_sample(self):
        json_file_name = list(
            set([hashlib.md5(str(value['ori_data']).encode('utf-8')).hexdigest() + '.json' for key, value in
                 self.original_sample_group.items()]))
        find_and_delete_invalid_json_cache(self.storage.cache_dir, json_file_name)
        original_sample_loop = tqdm(self.original_sample_group, desc='Evaluation of Original Data:', leave=True)

        eval_func = self.pc if self.eval_config.algorithm.algo == 'pc' else self.fdr_pc

        if not self.mode.parallel:
            for key in original_sample_loop:
                pc_fit_res = eval_func(key, aug=False)
                self.ori_pc_eval_result[key] = pc_fit_res
        else:
            temp_folder = tempfile.mkdtemp(prefix='fdr_pc_original_sample_eval_', dir=self.storage.temp_dir)
            # 创建外层进度条
            try:
                logging.info('Evaluation of Original Data is starting !')
                with joblib.parallel_backend('loky'):
                    original_pc_fit_list = Parallel(n_jobs=self.mode.n_jobs, temp_folder=temp_folder)(
                        delayed(eval_func)(key, aug=False) for key in original_sample_loop)
                    self.ori_pc_eval_result = {key: original_pc_fit_list[i] for i, (key, value) in
                                               enumerate(self.original_sample_group.items())}
            finally:
                shutil.rmtree(temp_folder, ignore_errors=True)

        for key in original_sample_loop:
            _, uit, cit = self.ori_pc_eval_result[key]
            self.ori_pc_eval_result[key] = _
            self.original_sample_group[key]['ori_pc_fit'] = self.ori_pc_eval_result[key]['esti_graph']
            self.original_sample_group[key]['ori_uit'] = uit
            self.original_sample_group[key]['ori_cit'] = cit

    def eval_augment_sample(self):
        # delete empty json cache
        json_file_name = []
        for key, value in self.original_sample_group.items():
            ori_data = value['ori_data']
            file_name = hashlib.md5(str(ori_data).encode('utf-8')).hexdigest() + '.json'
            json_file_name.append(file_name)
        find_and_delete_invalid_json_cache(self.storage.cache_dir, json_file_name)
        loop = tqdm(self.original_sample_group, desc='Evaluation of Augment Data:', leave=True)
        eval_func = self.pc if self.eval_config.algorithm.algo == 'pc' else self.fdr_pc

        if not self.mode.parallel:
            for key in loop:
                aug_pc_res = eval_func(key, aug=True)
                self.aug_pc_eval_result[key] = aug_pc_res
        else:
            temp_folder = tempfile.mkdtemp(prefix='fdr_pc_augment_sample_eval_', dir=self.storage.temp_dir)
            # 创建外层进度条
            try:
                with joblib.parallel_backend('loky'):
                    augment_pc_fit_list = Parallel(n_jobs=self.mode.n_jobs, temp_folder=temp_folder)(
                        delayed(eval_func)(key, aug=True) for key in loop)
                    self.aug_pc_eval_result = {key: augment_pc_fit_list[i] for i, (key, value) in
                                               enumerate(self.original_sample_group.items())}
            finally:
                shutil.rmtree(temp_folder, ignore_errors=True)

    def eval_result_aggregation(self, prefix):
        result = getattr(self, f'{prefix}_pc_eval_result')
        agg_file_name = f'{prefix}_pc_eval_res.xlsx'
        raw_file_name = f'{prefix}_pc_raw_eval_res.xlsx'

        res = []
        for key, v in result.items():
            v.update({k: key[i] for i, k in enumerate(self.keys2group)})
            res.append(v)

        df_raw_res = pd.DataFrame(res)
        try:
            df_raw_res.drop(columns=['esti_graph'], inplace=True)
        except KeyError:
            pass
        if prefix == 'ori':
            df_raw_res['sample_augment_method'] = 'original'
            subset = deepcopy(self.keys2group)
            subset.remove('sample_augment_method')
            df_raw_res.drop_duplicates(subset=subset, inplace=True)

        metric_col = ['SHD', 'SHD Anti', 'Normalized SHD', 'Normalized SHD Anti', 'TPR', 'FPR', 'TP', 'FP', 'FN',
                      'TN', 'precision', 'recall', 'F1', 'Accuracy', 'time_spent']
        col2drop = ['g_id', 'd_id']
        col2group = [c for c in df_raw_res.columns if c not in col2drop + metric_col]
        df_raw_res.drop(col2drop, axis=1, inplace=True)
        df_ori_pc_res_grouped = df_raw_res.groupby(col2group)
        data = []
        for n, g in df_ori_pc_res_grouped:
            data.append(list(n) + [str(round(g[metric].mean(), 3)) + '(' + str(
                round(g[metric].std(), 3)) + ')' for metric in metric_col])
        col2group.extend(metric_col)
        df_agg_res = pd.DataFrame(data=data, columns=col2group)

        df_agg_res.loc[:, 'uit'] = self.test_config.uit
        df_agg_res.loc[:, 'cit'] = self.test_config.cit
        df_agg_res.loc[:, 'Augment Size'] = self.simu_config.augment.n_samples
        agg_mask = df_agg_res['sample_augment_method'] == 'original'
        df_agg_res.loc[agg_mask, 'Augment Size'] = pd.NA
        df_agg_res.loc[:, 'algo'] = self.eval_config.algorithm.algo
        df_agg_res.loc[
            :, 'procedure'] = self.eval_config.algorithm.fdr_pc.procedure if self.eval_config.algorithm.algo == 'fdr_pc' else pd.NA
        df_agg_res.loc[
            :, 'fdr_alpha'] = self.eval_config.multiple_test_kwargs.alpha if self.eval_config.algorithm.algo == 'fdr_pc' else pd.NA
        df_agg_res.loc[
            :, 'p_combination'] = self.test_config.pval_ensemble.p_combination if self.test_config.cit == 'pval_ensemble' else pd.NA

        df_raw_res.loc[:, 'uit'] = self.test_config.uit
        df_raw_res.loc[:, 'cit'] = self.test_config.cit
        df_raw_res.loc[:, 'Augment Size'] = self.simu_config.augment.n_samples
        raw_mask = df_raw_res['sample_augment_method'] == 'original'
        df_raw_res.loc[raw_mask, 'Augment Size'] = pd.NA
        df_raw_res.loc[:, 'algo'] = self.eval_config.algorithm.algo
        df_raw_res.loc[
            :, 'procedure'] = self.eval_config.algorithm.fdr_pc.procedure if self.eval_config.algorithm.algo == 'fdr_pc' else pd.NA
        df_raw_res.loc[
            :, 'fdr_alpha'] = self.eval_config.multiple_test_kwargs.alpha if self.eval_config.algorithm.algo == 'fdr_pc' else pd.NA
        df_raw_res.loc[
            :, 'p_combination'] = self.test_config.pval_ensemble.p_combination if self.test_config.cit == 'fdr_pc' else pd.NA

        df_agg_res.to_excel(os.path.join(self.storage.eval_dir, agg_file_name), index=False)
        df_raw_res.to_excel(os.path.join(self.storage.eval_dir, raw_file_name), index=False)
        return df_agg_res

    def save_ori_pc_result(self):
        create_folder(self.storage.eval_dir)

        self.df_ori_pc_agg_res = self.eval_result_aggregation(prefix='ori')

    def save_aug_pc_result(self):
        create_folder(self.storage.eval_dir)

        self.df_aug_pc_agg_res = self.eval_result_aggregation(prefix='aug')

        if self.configs.mode.eval_original_sample:
            df_agg_res = pd.concat([self.df_ori_pc_agg_res, self.df_aug_pc_agg_res], ignore_index=True)
            df_agg_res.to_excel(os.path.join(self.storage.eval_dir, 'results.xlsx'), index=False)

    def __call__(self, *args, **kwargs):
        SimulationAugPC.__call__(self, *args, **kwargs)
        if not self.configs.mode.eval_augment_sample and not self.configs.mode.eval_original_sample:
            logging.info('*' * 100)
            logging.info(f'There is no evaluation task specified.')
        elif self.configs.mode.eval_original_sample:
                logging.info('*' * 100)
                logging.info('Evaluating Original data!')
                self.eval_original_sample()
                logging.info('*' * 100)
                logging.info('Evaluation Results of Original Samples are Saved!')
                self.save_ori_pc_result()
        else:
            logging.info('*' * 100)
            logging.info('Loading Augment testers!')
            if not self.configs.mode.eval_original_sample:
                del_attr = ['model_params', 'ori_simu_config', 'ori_pc_eval_result', 'ori_pc_eval_res',
                            'params_combination']
                for attr_name in del_attr:
                    if hasattr(self, attr_name):
                        delattr(self, attr_name)
            self.loading_augment_tester()
            self.tester_aggregation()
            logging.info('*' * 100)
            logging.info('Testers are loaded!')
            logging.info('*' * 100)
            logging.info('Run Aug-PC !')
            self.eval_augment_sample()
            logging.info('*' * 100)
            logging.info('Evaluation Results with Augment Samples are Saved!')
            self.save_aug_pc_result()
            logging.info('*' * 100)
            logging.info('Evaluation is over! Programs Terminated.')

        logging.info('*' * 100)
        logging.info(f'Program is terminated!')
        logging.info('*' * 100)

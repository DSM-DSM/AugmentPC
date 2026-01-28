import torch
import numpy as np
import os, yaml
from algo.pc_extension import pc, fdr_pc
from utils.tools import dict2namespace, namespace2dict

random_state = 42
np.random.seed(random_state)

alpha = 0.05
n, p = 200, 10
data = np.random.randn(n, p)
cache_path = 'cache_path.json'
augment_sample_size = 100
root_path = r'D:\CausalLearning\code\workflow'
config_path = os.path.join(root_path, 'workflow_config.yml')

with open(config_path, 'r') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    configs = dict2namespace(configs)

# Step1: Data Augmentation
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=random_state, n_neighbors=5)
y = np.random.binomial(1, 0.9, n)
augment_data, _ = adasyn.fit_resample(data, y)
augment_data = augment_data[:augment_sample_size, :]
data = np.concatenate((data, augment_data), axis=0)

# Step2: P Value Ensemble
from stat_test.test_chooser import stat_test_chooser

cache_config = configs.cache
configs.test.device = torch.device('cpu')
cache_config.cache_path = os.path.join(root_path, 'cache_path.json')
uit, cit = stat_test_chooser(data, configs, cache_config)

# Step3: Choose Estimation Algorithm
from algo.algo_chooser import AlgoChooser
from fdr_control.multi_test import HybridMultiTest

algo_chooser = AlgoChooser(configs.evaluation.algorithm)
algo = algo_chooser.choose()
# Using FDR_PC Algorithm
hmt = HybridMultiTest(**namespace2dict(configs.evaluation.multiple_test_kwargs))
estimation = algo(
    data, uit, cit, configs.test.alpha, hmt=hmt,
    **namespace2dict(configs.evaluation.algorithm.fdr_pc)
)

# Using PC Algorithm
# estimation = algo(
#     data, uit, cit, configs.test.alpha,
# )

print(estimation.nx_skel.edges) # Edges of Estimated Skeleton
print(estimation.nx_graph.edges) # Edges of Estimated CPDAG

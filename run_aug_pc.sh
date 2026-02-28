#!/bin/bash

#SBATCH --account=u2024104095
#SBATCH --comment=causal_learning
#SBATCH --job-name=10_3_pc_kci_gt_graph_nn_100
#SBATCH --nodes=1

#SBATCH --partition=cpu64c # cpu64c6530, cpu64c1t, cpu64c, cpu40c, cpu24c, gpu-titan, gpu-5090
###SBATCH --gres=gpu:1
#SBATCH --ntasks=64
#SBATCH --output=logs/hpc/10_3_pc_kci_gt_graph_nn_100.out

### hostname
# 使用 bash 显式解释器
export SHELL=/bin/bash
export PATH=/home/u2024104095/.conda/envs/causal_learning/bin:$PATH
eval "$(/opt/app/anaconda3/bin/conda shell.bash hook)"
### export CUDA_VISIBLE_DEVICES=0
source activate causal_learning

python run_aug_pc.py --config augment_pc_config_10.yml --seed 8888

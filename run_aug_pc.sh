#!/bin/bash

#SBATCH --account=u2024104095
#SBATCH --comment=causal_learning
#SBATCH --job-name=50_2_pc_spearman_gt_graph_AdaSyn_50
#SBATCH --nodes=1

#SBATCH --partition=cpu40c # cpu64c6530, cpu64c1t, cpu64c, cpu40c, cpu24c, gpu-titan, gpu-5090
###SBATCH --gres=gpu:1
#SBATCH --ntasks=40
#SBATCH --output=logs/hpc/50_2_pc_spearman_gt_graph_AdaSyn_50.out

### hostname
# 使用 bash 显式解释器
export SHELL=/bin/bash
export PATH=/home/u2024104095/.conda/envs/causal_learning/bin:$PATH
eval "$(/opt/app/anaconda3/bin/conda shell.bash hook)"
### export CUDA_VISIBLE_DEVICES=0
source activate causal_learning

python run_aug_pc.py --config augment_pc_config_50.yml --seed 8888

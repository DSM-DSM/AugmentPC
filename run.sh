#!/bin/bash

#SBATCH --comment=causal_learning
#SBATCH --job-name=pc_kci
#SBATCH --nodes=1


#SBATCH --partition=cpu64c  # cpu64c1t, cpu64c, cpu40c, cpu24c, gpu-titan
###SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --output=logs/hpc/pc_kci.out

### hostname
# 使用 bash 显式解释器
export SHELL=/bin/bash
export PATH=/home/u2024104095/.conda/envs/causal_learning/bin:$PATH
eval "$(/opt/app/anaconda3/bin/conda shell.bash hook)"
### export CUDA_VISIBLE_DEVICES=0
source activate causal_learning

python run.py --config config_10.yml --seed 8888

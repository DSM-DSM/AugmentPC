salloc --nodes=1 --gres=gpu:1 --partition=gpu-titan --ntasks=1 --time=00:40:00 --comment=causal_learning
# 以下为屏幕反馈信息
# salloc: Granted job allocation 49933
# salloc: Waiting for resource configuration
# salloc: Nodes titan-1 are ready for job
# 登录到当前计算节点
ssh cpu40c-1
conda activate causal_learning

python run.py --config config.yml --seed 1234

# 退出当前计算节点
exit
# 以下为屏幕反馈信息
# logout
# Connection to titan-1 closed.

# 释放当前 salloc 计算资源
exit
# 以下为屏幕反馈信息
# salloc: Relinquishing job allocation 49933
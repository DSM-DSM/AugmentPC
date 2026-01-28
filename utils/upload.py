import paramiko
import warnings
from cryptography.utils import CryptographyDeprecationWarning

# 忽略所有 CryptographyDeprecationWarning 警告
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.224.255.112', username='s2024104095', password='xi3tgvlfg2')
print('Connected!')

# 提交文件
sftp = ssh.open_sftp()

file1 = r'D:\CausalLearning\src\evaluation.py'
file2 = r'D:\CausalLearning\src\experiment.py'
file3 = r'D:\CausalLearning\src\simulation.py'
file4 = r'D:\CausalLearning\src\utils.py'
file5 = r'D:\CausalLearning\src\noise_and_cause.py'
file6 = r'D:\Anaconda3\Lib\site-packages\causallearn\utils\cit.py'
file7 = r'D:\CausalLearning\src\ACGeneratorPlus.py'
file8 = r'D:\CausalLearning\src\grid_data_generator.py'
file9 = r'D:\Anaconda3\Lib\site-packages\causallearn\search\ConstraintBased\PC.py'
file10 = r'D:\CausalLearning\src\algo.py'
file11 = r'D:\CausalLearning\src\citPlus.py'
file12 = r'D:\CausalLearning\src\CIT_R2python.py'
file13 = r'D:\CausalLearning\src\SkeletonDiscoveryPlus.py'
file15 = r'D:\CausalLearning\src\settings.py'

# sftp.put('poker-hand.data', '/home/s2024104095/poker-hand.data')

sftp.put(file1, '/home/s2024104095/CausalLearning/src/evaluation.py')
sftp.put(file2, '/home/s2024104095/CausalLearning/src/run.py')
sftp.put(file3, '/home/s2024104095/CausalLearning/src/simulation.py')
sftp.put(file4, '/home/s2024104095/CausalLearning/src/tools.py')
sftp.put(file5, '/home/s2024104095/CausalLearning/src/noise_generator.py')
sftp.put(file6, '/home/s2024104095/anaconda3/lib/python3.12/site-packages/causallearn/utils/cit.py')
sftp.put(file7, '/home/s2024104095/CausalLearning/src/dag_generator.py')
sftp.put(file8, '/home/s2024104095/CausalLearning/src/grid_data_generator.py')
sftp.put(file9, '/home/s2024104095/anaconda3/lib/python3.12/site-packages/causallearn/search/ConstraintBased/algo.py')
sftp.put(file10, '/home/s2024104095/CausalLearning/src/algo.py')
sftp.put(file11, '/home/s2024104095/CausalLearning/src/citPlus.py')
sftp.put(file12, '/home/s2024104095/CausalLearning/src/CIT_R2python.py')
sftp.put(file13, '/home/s2024104095/CausalLearning/src/SkeletonDiscoveryPlus.py')
sftp.put(file15, '/home/s2024104095/CausalLearning/src/settings.py')
# "/home/s2024104095/anaconda3/lib/python3.12/site-packages/causallearn/utils/PCUtils/skeleton_discovery.py"
sftp.close()
ssh.close()
print('Uploaded!')

# 提交任务
# 2. 提交任务
"""
你可以通过 exec_command 方法执行任意命令行指令来提交任务。例如，如果你想提交一个长时间运行的任务（如备份数据库），可以这样做：
stdin, stdout, stderr = ssh.exec_command('nohup /path/to/your/script.sh > /dev/null 2>&1 & echo $!')
pid = stdout.read().decode().strip()  # 获取进程 ID
print(f"Task submitted with PID: {pid}")


conda activate R4.4.1
rm -rf Gaussion F_dist Cauchy_Gaussion_Mixed
mkdir Gaussion F_dist Cauchy_Gaussion_Mixed

检查修改source、网格参数函数再提交任务
nohup Rscript parallel_computation.R > paral_c.log 2>&1 &
pkill -f s2024104095
ps aux | grep '[s]2024104095' | awk '{print $2}' | xargs kill -9

conda deactivate

ssh s2024104095@10.224.255.112
conda activate mypython
cd CausalLearning/src
nohup python3 run.py > res.log 2>&1 &
tail -f res.log

killall -u s2024104095 python3
"""
# 下载文件
"""
# download
scp -r s2024104095@10.224.255.112:Gaussion C:/Users/俞锦越/Desktop
scp -r s2024104095@10.224.255.112:F_dist C:/Users/俞锦越/Desktop
scp -r s2024104095@10.224.255.112:Cauchy_Gaussion_Mixed C:/Users/俞锦越/Desktop
scp -r s2024104095@10.224.255.112:causal_learning/Shubhadeep[2022]/Normal C:/Users/俞锦越/Desktop

scp -r s2024104095@10.224.255.112:CausalLearning/src/Eval_Results C:/Users/俞锦越/Desktop
scp -r s2024104095@10.224.255.112:CausalLearning/src/Eval_Results/2025_0508_2243 D:\\CausalLearning\\src\\Eval_Results
scp -r s2024104095@10.224.255.112:CausalLearning/src/cit_cache_1.26.4 D:\\CausalLearning\\src
scp -r s2024104095@10.224.255.112:CausalLearning/src/copc_sample D:\\CausalLearning\\src\\copc_sample

scp -r -P 22713 u2024104095@10.10.252.11:~/sEFPC_code/simulations/results D:\\Exp_family_PCA\\sEFPC\\sEFPC_code\\simulations 
scp -r -P 22713 u2024104095@10.10.252.11:~/sEFPC_code/simulations/resultsPoisson D:\\Exp_family_PCA\\sEFPC\\sEFPC_code\\simulations 

# upload
scp -r D:\\CausalLearning\\src\\cit_cache_1.26.4 s2024104095@10.224.255.112:CausalLearning/src/cit_cache_1.26.4
scp -r /home/s2024104095/CausalLearning/src/cit_cache_1.26.4 /tmp/pycharm_project_68/src/cit_cache_1.26.4
scp -r s2024104095@10.224.255.112:/tmp/pycharm_project_68/src/cit_cache_1.26.4 D:\\CausalLearning\\src\\cit_cache_1.26.4

scp -r -P 22713 D:\\Exp_family_PCA\\sEFPC u2024104095@10.10.252.11:~/sEFPC_code
"""

"""
# 打开配置文件
nano ~/.bashrc  # 或 ~/.bash_profile

# 在文件末尾添加以下内容（示例：添加 Java 环境变量）
export R_HOME="/home/s2024104095/anaconda3/envs/R4.4.1/lib/R"

# 保存后重新加载配置
source ~/.bashrc
"""

"""
> .libPaths()
[1] "/home/s2024104095/anaconda3/envs/R4.4.1/lib/R/library"

> .libPaths()
[1] "/home/s2024104095/R/x86_64-pc-linux-gnu-library/3.6"
[2] "/usr/local/lib/R/site-library"
[3] "/usr/lib/R/site-library"
[4] "/usr/lib/R/library"

"""
"""
rm -rf ~/R-4.3.1
tar -xvzf R-4.3.1.tar.gz
cd R-4.3.1

./configure --with-x=no
ls -l Makefile
make
make install

"""

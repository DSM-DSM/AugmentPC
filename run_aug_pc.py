import numpy as np
from utils.tools import dict2namespace, create_folder
import os, yaml, time, torch, logging, argparse, traceback
import sys
import os

# 获取项目根目录
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Put this before importing tensorflow
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--configs', type=str, default='augment_pc_config_10.yml', help='configuration file path')
    parser.add_argument('--seed', type=int, default=8888, help='Random seed')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    args = parser.parse_args()

    # load configs
    with open(os.path.join(root_path, 'configs', args.configs), 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = dict2namespace(configs)
    configs.simulation.weight_kwargs = configs.simulation.ori.weight_kwargs.__dict__

    # create log
    run_id = str(os.getpid())
    run_time = time.strftime('%Y_%b_%d_%H_%M_%S')

    if configs.evaluation.algorithm.algo == 'pc':
        doc = [run_time, configs.test.uit, configs.test.cit, configs.evaluation.algorithm.algo]
    elif configs.evaluation.algorithm.algo == 'fdr_pc':
        doc = [run_time, configs.test.uit, configs.test.cit, configs.evaluation.algorithm.algo,
               configs.evaluation.algorithm.fdr_pc.procedure]
    else:
        raise RuntimeError('Unknown algorithm: {}'.format(configs.evaluation.algorithm.algo))
    doc.extend(configs.simulation.augment.models + [str(configs.simulation.augment.n_samples), run_id])
    # Initialize Storage Path and Dictory
    configs.doc = '_'.join(doc)
    configs.storage.root_dir = root_path
    configs.storage.temp_dir = os.path.join(root_path, configs.storage.temp_dir)
    configs.storage.data_dir = os.path.join(root_path, configs.storage.data_dir)
    configs.storage.cache_dir = os.path.join(root_path, configs.storage.cache_dir)
    configs.storage.model_dir = os.path.join(root_path, configs.storage.model_dir)

    configs.storage.log_dir = os.path.join(root_path, configs.storage.log_dir, configs.doc)
    configs.storage.eval_dir = os.path.join(root_path, configs.storage.eval_dir, configs.doc)

    create_folder(configs.storage.log_dir)
    create_folder(configs.storage.temp_dir)
    create_folder(configs.storage.cache_dir)
    configs.simulation.seed = args.seed

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()  # 将日志输出到控制台
    handler2 = logging.FileHandler(os.path.join(configs.storage.log_dir, 'stdout.txt'))  # 将日志写入文件
    # 定义日志的格式，包括日志级别、文件名、时间戳和消息
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()  # 全局日志记录器
    logger.addHandler(handler1)  # 为日志记录器添加处理器
    logger.addHandler(handler2)
    logger.setLevel(level)  # 设置日志级别

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    configs.test.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    with open(os.path.join(configs.storage.log_dir, 'configs.yml'), 'w') as f:
        yaml.dump(configs, f, default_flow_style=False)
    return args, configs


def main():
    args, configs = parse_args_and_config()
    logging.info("Writing evaluation result to {}".format(configs.doc))
    logging.info("-" * 100)
    logging.info("Experiment instance id = {}".format(os.getpid()))
    logging.info("-" * 100)
    logging.info("Experiment description = {}".format(configs.other.description))
    logging.info("-" * 100)
    logging.info(f"augment sample generating models: {configs.simulation.augment.models}")
    logging.info(f"augment sample size: {configs.simulation.augment.n_samples}")
    logging.info("-" * 100)
    logging.info(f"Using UIT is {configs.test.uit}, CIT is {configs.test.cit}")
    if configs.test.uit in ['vt', 'pval_ensemble', 'eval_ensemble']:
        logging.info(f"Using UIT List: {getattr(configs.test, configs.test.uit).uit_l}")
    if configs.test.cit in ['vt', 'pval_ensemble', 'eval_ensemble']:
        logging.info(f"Using CIT List: {getattr(configs.test, configs.test.cit).cit_l}")
    if configs.test.uit == 'pval_ensemble' or configs.test.cit == 'pval_ensemble':
        logging.info(f"P Value Ensemble Method is {configs.test.pval_ensemble.p_combination}")
    if configs.test.uit == 'eval_ensemble' or configs.test.cit == 'eval_ensemble':
        logging.info(f"E Value Ensemble Function is {configs.test.eval_ensemble.func}")
    logging.info("-" * 100)
    logging.info(f"Algorithm is {configs.evaluation.algorithm.algo}")
    if configs.evaluation.algorithm.algo == 'fdr_pc':
        logging.info(f"FDR Method is {configs.evaluation.algorithm.fdr_pc.procedure}")
        logging.info(f"FDR Significant Level is {configs.evaluation.multiple_test_kwargs.alpha}")
    logging.info("-" * 100)
    for model in configs.simulation.augment.models:
        logging.info(f"{model} training kwargs is {vars(configs.simulation.augment.model_params)[model]}")
        logging.info("-" * 100)
    try:
        # from src.ebh_pc_evaluation import SimulationAugPC
        # ebh_pc_simulation = SimulationAugPC(args, configs)
        # ebh_pc_simulation()
        from src.aug_pc_evaluation import EvaluationAugPC
        ebh_pc_evaluation = EvaluationAugPC(args, configs)
        ebh_pc_evaluation()
        return 0
    except:
        logging.error(traceback.format_exc())
        return -1


if __name__ == '__main__':
    main()

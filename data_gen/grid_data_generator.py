import networkx as nx
from data_gen.noise_generator import noise_func_chooser
from data_gen.dag_generator import AcyclicGraphGeneratorPlus
from data_gen.root_generator import cause_func_chooser
from utils.tools import TemporaryRandomSeed
from scipy.stats import f, cauchy
import numpy as np


def copula_transform(data, dfn=1, dfd=1):
    cauchy_dist = cauchy(loc=0, scale=1)
    uniform_data = cauchy_dist.cdf(data)
    f_distribution_data = f.ppf(uniform_data, dfn=dfn, dfd=dfd)
    assert not np.isinf(f_distribution_data).any(), 'Copula Transformation failed due to Inf value!'
    return f_distribution_data


def data_graph_generator(configs):
    if configs.nodes < 2:
        raise ValueError("p must be greater than or equal to 2")
    if configs.nodes > configs.n_samples:
        raise ValueError("p should not greater than or equal to n")
    if configs.expected_degree < 0:
        raise ValueError("exp_degree must be greater than or equal to 0")
    if configs.expected_degree > configs.nodes - 1:
        raise ValueError("exp_degree must be less than or equal to p - 1")
    return _data_graph_generator(configs)


def _data_graph_generator(configs):
    """
    Generate a random Directed Acyclic Graph (DAG)
    :return:
        dict : {
        'data': list of numpy.ndarray,
        'graph': networkx.DiGraph
        }
    Reference: https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/data.html
    """
    configs.noise, configs.root = noise_func_chooser(configs.noise), cause_func_chooser(
        configs.root)
    dag_kwarg = configs.__dict__
    # When dag_type is default, specify expected_degree be an idle parameter，See AcyclicGraphGeneratorPlus.generate_dag
    if configs.designate:
        with TemporaryRandomSeed(configs.g_id):
            generator = AcyclicGraphGeneratorPlus(**dag_kwarg)
            # When designate is True, data will be generated from  FIXED DAG graph.
            generator.designate_graph = nx.from_numpy_array(generator.generate_dag(), create_using=nx.DiGraph)
    else:
        generator = AcyclicGraphGeneratorPlus(**dag_kwarg)

    with TemporaryRandomSeed(configs.seed):
        data, pure_signal, graph = generator.generate(configs.rescale)
    nx.relabel_nodes(graph, {node: int(node.strip('V')) for node in graph.nodes()}, copy=False)  # 修改graph的图节点名
    # print(graph.edges)
    data = data.values
    if configs.copula_trans:
        dfn, dfd = configs.copula_trans_kwargs.dfn, configs.copula_trans_kwargs.dfd
        data = copula_transform(data, dfn=dfn, dfd=dfd)
        pure_signal = copula_transform(pure_signal, dfn=dfn, dfd=dfd)
    return data, graph

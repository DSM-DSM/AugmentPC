import numpy as np
import pandas as pd
import networkx as nx
from cdt.data import AcyclicGraphGenerator
from sklearn.preprocessing import scale
from cdt.data.causal_mechanisms import *
from utils.tools import normalize_to_neg1_1


class AcyclicGraphGeneratorPlus(AcyclicGraphGenerator):
    def __init__(self, nodes, n_samples, noise, noise_coeff, expected_degree, dag_type, root,
                 parents_max=5, **kwargs):
        """

        :param designate: 是否数据生成方式是否只根据一个graph生成。
            若为假，则每调用一次AcyclicGraphGeneratorPlus.generate(),则新生成一个DAG和对应的一组数据；
            若为真，则每调用一次AcyclicGraphGeneratorPlus.generate(),则会记住上一次的self.g(若没有则会按照随机数种子生成一个)并生成对应的一组数据
        :param weight_distribution_function: 有向无环图中线性相加的weight_ij分布类
        :param kwargs:
        """
        self.data = pd.DataFrame(None, columns=["V{}".format(i) for i in range(nodes)])
        self.pure_signal = pd.DataFrame(None, columns=["V{}".format(i) for i in range(nodes)])
        self.nodes = nodes
        self.n_samples = n_samples
        self.noise = noise
        self.noise_coeff = noise_coeff
        self.adjacency_matrix = np.zeros((nodes, nodes))
        self.parents_max = parents_max
        self.expected_degree = expected_degree
        self.dag_type = dag_type
        self.initial_generator = root
        self.cfunctions = None
        self.g = None
        self.designate = kwargs.get('designate', False)
        self.designate_graph = kwargs.get('designate_graph', None)
        causal_mechanism = kwargs.get('causal_mechanism', 'linear')
        self.mechanism = {'linear': LinearMechanismPlus,
                          'polynomial': Polynomial_MechanismPlus,
                          'sigmoid_add': SigmoidAM_MechanismPlus,
                          'sigmoid_mix': SigmoidMix_MechanismPlus,
                          'gp_add': GaussianProcessAdd_MechanismPlus,
                          'gp_mix': GaussianProcessMix_MechanismPlus,
                          'nn': NN_MechanismPlus}[causal_mechanism]
        self.mechanism_kwargs = kwargs

    def generate_dag(self):
        adjacency_matrix = np.zeros((self.nodes, self.nodes))
        # self.dag_type为default时，生成的DAG只受随机数种子影响
        if self.dag_type == 'default':
            for j in range(1, self.nodes):
                nb_parents = np.random.randint(0, min([self.parents_max, j]) + 1)
                for i in np.random.choice(range(0, j), nb_parents, replace=False):
                    adjacency_matrix[i, j] = 1

        elif self.dag_type == 'erdos':
            nb_edges = self.expected_degree * self.nodes
            prob_connection = 2 * nb_edges / (self.nodes ** 2 - self.nodes)
            causal_order = np.random.permutation(np.arange(self.nodes))

            for i in range(self.nodes - 1):
                node = causal_order[i]
                possible_parents = causal_order[(i + 1):]
                num_parents = np.random.binomial(n=self.nodes - i - 1,
                                                 p=prob_connection)
                parents = np.random.choice(possible_parents, size=num_parents,
                                           replace=False)
                adjacency_matrix[parents, node] = 1
        return adjacency_matrix

    def init_dag(self, verbose):
        if self.designate_graph is not None:
            self.adjacency_matrix = nx.to_numpy_array(self.designate_graph)
            self.g = self.designate_graph

        elif self.designate:
            designate_graph_adj = self.generate_dag()
            self.adjacency_matrix = designate_graph_adj
            self.designate_graph = nx.DiGraph(designate_graph_adj)
            self.g = self.designate_graph

        else:
            self.adjacency_matrix = self.generate_dag()
            try:
                self.g = nx.DiGraph(self.adjacency_matrix)
                assert not list(nx.simple_cycles(self.g))

            except AssertionError:
                if verbose:
                    print("Regenerating, graph non valid...")
                self.init_dag(verbose=verbose)

    def init_variables(self, verbose=False):
        self.init_dag(verbose)

        self.cfunctions = [
            self.mechanism(int(sum(self.adjacency_matrix[:, i])), self.n_samples, self.noise,
                           noise_coeff=self.noise_coeff,
                           **self.mechanism_kwargs)
            if sum(self.adjacency_matrix[:, i])
            else self.initial_generator for i in range(self.nodes)
        ]

    def generate(self, rescale=False):
        if self.cfunctions is None:
            self.init_variables()

        for i in nx.topological_sort(self.g):
            # Root cause
            if not sum(self.adjacency_matrix[:, i]):
                self.data['V{}'.format(i)] = self.cfunctions[i](self.n_samples)
                self.pure_signal['V{}'.format(i)] = self.cfunctions[i](self.n_samples)
            # Generating causes
            else:
                self.data['V{}'.format(i)], self.pure_signal['V{}'.format(i)] = self.cfunctions[i](
                    self.data.iloc[:, self.adjacency_matrix[:, i].nonzero()[0]].values)
            if rescale:
                # self.data['V{}'.format(i)] = scale(self.data['V{}'.format(i)].values)
                x = self.data['V{}'.format(i)].values
                self.data['V{}'.format(i)] = (x - np.mean(x)) / np.std(x)

                pure_signal_x = self.pure_signal['V{}'.format(i)].values
                self.pure_signal['V{}'.format(i)] = (pure_signal_x - np.mean(pure_signal_x)) / np.std(pure_signal_x)

        return self.data, self.pure_signal, nx.relabel_nodes(self.g, {i: 'V' + str(i) for i in self.g.nodes}, copy=True)


class LinearMechanismPlus(LinearMechanism):
    def __init__(self, ncauses, points, noise_function, noise_coeff=0.4, **kwargs):
        self.n_causes = ncauses
        self.points = points
        self.coefflist = []
        weight_distribution_function = kwargs.get('weight_distribution_function')
        for i in range(ncauses):
            self.coefflist.append(weight_distribution_function())

        self.noise = noise_coeff * noise_function(points)

    def __call__(self, causes):
        """Run the mechanism."""
        # Additive only, for now
        effect = np.zeros((self.points, 1))
        pure_signal = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            pure_signal[:, 0] = pure_signal[:, 0] + self.coefflist[par] * causes[:, par]
        effect[:, 0] = pure_signal[:, 0] + self.noise[:, 0]

        return effect, pure_signal


class Polynomial_MechanismPlus(Polynomial_Mechanism):
    def __init__(self, ncauses, points, noise_function, d=2, noise_coeff=.4, **kwargs):
        self.n_causes = ncauses
        self.points = points
        self.d = d
        self.polycause = []

        weight_distribution_function = kwargs.get('weight_distribution_function')
        if 'noise_mechanism' in kwargs.keys():
            noise_mechanism = kwargs.get('noise_mechanism', 'add')
            assert noise_mechanism in ['additive', 'multiplicative']
            if noise_mechanism == 'additive':
                self.ber = 0
            else:
                self.ber = 1
        else:
            self.ber = np.random.binomial(1, 0.5)  # 用于控制是使用加性噪声还是乘性噪声

        for c in range(ncauses):
            self.coefflist = []
            for j in range(self.d + 1):
                self.coefflist.append(weight_distribution_function())
            self.polycause.append(self.coefflist)

        self.noise = noise_coeff * noise_function(points)

    def mechanism(self, x, par):
        """Mechanism function."""
        list_coeff = self.polycause[par]
        result = np.zeros((self.points, 1))
        for i in range(self.points):
            for j in range(self.d + 1):
                result[i, 0] += list_coeff[j] * np.power(x[i], j)
        return result

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        pure_signal = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            pure_signal[:, 0] = pure_signal[:, 0] + self.mechanism(causes[:, par], par)[:, 0]

        if (self.ber > 0 and causes.shape[1] > 0):
            effect[:, 0] = pure_signal[:, 0] * self.noise[:, 0]
        else:
            effect[:, 0] = pure_signal[:, 0] + self.noise[:, 0]
        # secure the numerical stability
        effect = normalize_to_neg1_1(effect)
        pure_signal = normalize_to_neg1_1(pure_signal)
        return effect, pure_signal


class SigmoidAM_MechanismPlus(SigmoidAM_Mechanism):
    def __init__(self, ncauses, points, noise_function, noise_coeff=.4, **kwargs):
        self.n_causes = ncauses
        self.points = points

        self.a = np.random.exponential(1 / 4) + 1
        ber = np.random.binomial(1, 0.5)
        self.b = ber * np.random.uniform(-2, -0.5) + (1 - ber) * np.random.uniform(0.5, 2)
        self.c = np.random.uniform(-2, 2)
        self.noise = noise_coeff * noise_function(points)

    def __call__(self, causes):
        """Run the mechanism."""
        # Additive only
        effect = np.zeros((self.points, 1))
        pure_signal = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            pure_signal[:, 0] = pure_signal[:, 0] + self.mechanism(causes[:, par])[:, 0]

        effect[:, 0] = pure_signal[:, 0] + self.noise[:, 0]

        return effect, pure_signal


class SigmoidMix_MechanismPlus(SigmoidMix_Mechanism):
    def __init__(self, ncauses, points, noise_function, noise_coeff=.4, **kwargs):
        """Init the mechanism."""
        self.n_causes = ncauses
        self.points = points

        self.a = np.random.exponential(1 / 4) + 1
        ber = np.random.binomial(1, 0.5)
        self.b = ber * np.random.uniform(-2, -0.5) + (1 - ber) * np.random.uniform(0.5, 2)
        self.c = np.random.uniform(-2, 2)

        self.noise = noise_coeff * noise_function(points)

    def mechanism(self, causes, noise=True):
        """Mechanism function."""
        result = np.zeros((self.points, 1))
        for i in range(self.points):
            pre_add_effect = 0
            for c in range(causes.shape[1]):
                pre_add_effect += causes[i, c]

            if noise:
                pre_add_effect += self.noise[i]

            result[i, 0] = self.a * self.b * \
                           (pre_add_effect + self.c) / (1 + abs(self.b * (pre_add_effect + self.c)))

        return result

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        pure_signal = np.zeros((self.points, 1))
        # Compute each cause's contribution

        pure_signal[:, 0] = self.mechanism(causes, noise=False)[:, 0]
        effect[:, 0] = self.mechanism(causes, noise=True)[:, 0]
        return effect, pure_signal


class GaussianProcessAdd_MechanismPlus(GaussianProcessAdd_Mechanism):
    def __init__(self, ncauses, points, noise_function, noise_coeff=.4, **kwargs):
        """Init the mechanism."""
        self.n_causes = ncauses
        self.points = points

        self.noise = noise_coeff * noise_function(points)
        self.nb_step = 0

    def __call__(self, causes):
        """Run the mechanism."""
        # Additive only
        effect = np.zeros((self.points, 1))
        pure_signal = np.zeros((self.points, 1))

        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            pure_signal[:, 0] = pure_signal[:, 0] + self.mechanism(causes[:, par])

        effect[:, 0] = pure_signal[:, 0] + self.noise[:, 0]

        return effect, pure_signal


class GaussianProcessMix_MechanismPlus(GaussianProcessMix_Mechanism):
    def __init__(self, ncauses, points, noise_function, noise_coeff=.4, **kwargs):
        """Init the mechanism."""
        self.n_causes = ncauses
        self.points = points
        self.noise = noise_coeff * noise_function(points)
        self.nb_step = 0

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        pure_signal = np.zeros((self.points, 1))
        # Compute each cause's contribution
        if (causes.shape[1] > 0):
            mix = np.hstack((causes, self.noise))
            effect[:, 0] = self.mechanism(mix)
            pure_signal[:, 0] = self.mechanism(causes)
        else:
            effect[:, 0] = self.mechanism(self.noise)

        return effect, pure_signal


class NN_MechanismPlus(NN_Mechanism):
    def __init__(self, ncauses, points, noise_function, nh=20, noise_coeff=.4, hidden_layer_num=2,
                 activation_function='Tanh', **kwargs):
        self.n_causes = ncauses
        self.points = points
        self.noise = noise_coeff * noise_function(points)
        self.nb_step = 0
        self.nh = nh
        self.hidden_layer_num = hidden_layer_num
        self.activation_function = getattr(th.nn, activation_function)()
        wdf = kwargs.get('weight_distribution_function')

        def weight_distribution_function(shape):
            total_elements = np.prod(shape)
            flat_array = np.array([wdf() for _ in range(total_elements)], dtype=np.float32)
            return flat_array.reshape(shape)

        self.weight_distribution_function = weight_distribution_function

    def _initialize_weights(self, module):
        """Custom weight initialization for linear layers."""
        if isinstance(module, th.nn.Linear):
            w_shape = module.weight.data.shape
            module.weight.data = th.from_numpy(self.weight_distribution_function(w_shape))
            if module.bias is not None:
                b_shape = module.bias.data.shape
                module.bias.data = th.from_numpy(self.weight_distribution_function(b_shape))

    def mechanism(self, x):
        """Mechanism function."""
        with th.no_grad():
            layers = []

            # 创建线性层
            input_layer = th.nn.Linear(self.n_causes, self.nh)
            output_layer = th.nn.Linear(self.nh, 1)

            layers.append(input_layer)
            layers.append(self.activation_function)

            for i in range(self.hidden_layer_num):
                hidden_layer = th.nn.Linear(self.nh, self.nh)
                layers.append(hidden_layer)
                layers.append(self.activation_function)

            layers.append(output_layer)

            self.layers = th.nn.Sequential(*layers)
            self.layers.apply(self._initialize_weights)
            data = x.astype('float32')
            data_th = th.from_numpy(data)

            effect_data = self.layers(data_th)
            effect_data = effect_data.detach().numpy()
        return effect_data

    def __call__(self, causes):
        """Run the mechanism."""
        # Compute each cause's contribution
        pure_signal = self.mechanism(causes)

        effect = pure_signal + self.noise
        return effect, pure_signal

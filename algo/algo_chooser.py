import logging
from stat_test.test_chooser import stat_test_chooser


class AlgoChooser(object):
    def __init__(self, algo_config):
        if not hasattr(algo_config, 'algo'):
            logging.error("Invalid eval_config: missing 'algo' attribute")
        self.algo_config = algo_config

    def choose(self):
        if self.algo_config.algo == 'pc':
            from algo.pc_extension import pc
            return pc

        elif self.algo_config.algo == 'fdr_pc':
            from algo.pc_extension import fdr_pc
            return fdr_pc

        elif self.algo_config.algo == 'deduce_dep_pc':
            from algo.pc_extension import deduce_dep_pc
            return deduce_dep_pc

        else:
            logging.error(f'Algorithm {self.algo_config.algo} does not supported !')
            raise ValueError

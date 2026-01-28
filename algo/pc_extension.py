from causallearn.search.ConstraintBased.PC import *
from algo.skeleton_discovery import skeleton_discovery_deduce_dep, skeleton_discovery, fdr_skeleton_discovery
from numpy import ndarray
from typing import List
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from algo.graphs import CausalGraphPlus
from algo.skeleton_orientation import orientation
from fdr_control.multi_test import HybridMultiTest
import logging


def fdr_pc(data, uit, cit, alpha, hmt: object = None, stable: bool = True,
           uc_rule: int = 0, uc_priority: int = 2, correction_name: str = 'MV_Crtn_Fisher_Z',
           background_knowledge: BackgroundKnowledge | None = None, verbose: bool = False, show_progress: bool = False,
           node_names: List[str] | None = None, **kwargs):
    if data.shape[0] < data.shape[1]:
        logging.warnings("The number of features is much larger than the X_num size!")

    if not isinstance(hmt, HybridMultiTest):
        logging.error(f"Instance hmt should be a HybridMultiTest instance !")

    return fdr_pc_alg(data=data, uit=uit, cit=cit, alpha=alpha, hmt=hmt, node_names=node_names,
                      no_of_var=data.shape[1], stable=stable, uc_rule=uc_rule, uc_priority=uc_priority,
                      background_knowledge=background_knowledge, verbose=verbose, show_progress=show_progress, **kwargs)


def fdr_pc_alg(data, uit, cit, hmt, node_names: List[str] | None, alpha: float, no_of_var: int, stable: bool,
               uc_rule: int, uc_priority: int, background_knowledge: BackgroundKnowledge | None = None,
               verbose: bool = False, show_progress: bool = False, **kwargs) -> CausalGraphPlus:
    start = time.time()

    cg_1 = fdr_skeleton_discovery(data, uit, cit, hmt, alpha, stable, background_knowledge=background_knowledge,
                                  verbose=verbose, show_progress=show_progress, node_names=node_names, **kwargs)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    cg = orientation(cg_1, alpha, uc_rule, uc_priority, background_knowledge)
    end = time.time()

    cg.PC_elapsed = end - start
    cg.to_nx_skeleton()
    cg.to_nx_graph()
    return cg


def pc(data: ndarray, uit, cit, alpha=0.05, stable: bool = True,
       uc_rule: int = 0, uc_priority: int = 2, correction_name: str = 'MV_Crtn_Fisher_Z',
       background_knowledge: BackgroundKnowledge | None = None, verbose: bool = False, show_progress: bool = False,
       node_names: List[str] | None = None, **kwargs):
    if data.shape[0] < data.shape[1]:
        logging.warnings("The number of features is much larger than the X_num size!")

    return pc_alg(data=data, uit=uit, cit=cit, node_names=node_names, alpha=alpha, stable=stable,
                  uc_rule=uc_rule, uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose,
                  show_progress=show_progress, **kwargs)


def pc_alg(data: ndarray, uit, cit, node_names: List[str] | None, alpha: float, stable: bool,
           uc_rule: int, uc_priority: int, background_knowledge: BackgroundKnowledge | None = None,
           verbose: bool = False, show_progress: bool = False, **kwargs) -> CausalGraphPlus:
    start = time.time()
    cg_1 = skeleton_discovery(data, uit, cit, alpha, stable, background_knowledge=background_knowledge, verbose=verbose,
                              show_progress=show_progress, node_names=node_names, **kwargs)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    cg = orientation(cg_1, alpha, uc_rule, uc_priority, background_knowledge)
    end = time.time()

    cg.PC_elapsed = end - start
    cg.to_nx_skeleton()
    cg.to_nx_graph()
    return cg


def deduce_dep_pc(data: ndarray, uit, cit, alpha=0.05, stable: bool = True,
                  uc_rule: int = 0, uc_priority: int = 2, mvpc: bool = False, correction_name: str = 'MV_Crtn_Fisher_Z',
                  background_knowledge: BackgroundKnowledge | None = None, verbose: bool = False,
                  show_progress: bool = False, node_names: List[str] | None = None, deduce_dep_k: int = 1, **kwargs):
    start = time.time()
    cg_1 = skeleton_discovery_deduce_dep(data, uit, cit, alpha, stable,
                                         background_knowledge=background_knowledge, verbose=verbose,
                                         show_progress=show_progress, node_names=node_names, deduce_dep_k=deduce_dep_k)
    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    cg = orientation(cg_1, alpha, uc_rule, uc_priority, background_knowledge)

    end = time.time()

    cg.PC_elapsed = end - start

    return cg

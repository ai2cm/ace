"""
Supported Modulus graph configs
"""

import torch

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class GraphConf:
    func_arch: bool = MISSING
    func_arch_allow_partial_hessian: bool = MISSING


@dataclass
class DefaultGraphConf(GraphConf):
    func_arch: bool = False
    func_arch_allow_partial_hessian: bool = True


def register_graph_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="graph",
        name="default",
        node=DefaultGraphConf,
    )

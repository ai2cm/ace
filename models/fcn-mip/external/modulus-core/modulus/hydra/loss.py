"""
Supported Modulus loss aggregator configs
"""

import torch

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import Any


@dataclass
class LossConf:
    _target_: str = MISSING
    weights: Any = None


@dataclass
class AggregatorSumConf(LossConf):
    _target_: str = "modulus.loss.aggregator.Sum"


@dataclass
class AggregatorGradNormConf(LossConf):
    _target_: str = "modulus.loss.aggregator.GradNorm"
    alpha: float = 1.0


@dataclass
class AggregatorResNormConf(LossConf):
    _target_: str = "modulus.loss.aggregator.ResNorm"
    alpha: float = 1.0


@dataclass
class AggregatorHomoscedasticConf(LossConf):
    _target_: str = "modulus.loss.aggregator.HomoscedasticUncertainty"


@dataclass
class AggregatorLRAnnealingConf(LossConf):
    _target_: str = "modulus.loss.aggregator.LRAnnealing"
    update_freq: int = 1
    alpha: float = 0.01
    ref_key: Any = None  # Change to Union[None, str] when supported by hydra
    eps: float = 1e-8


@dataclass
class AggregatorSoftAdaptConf(LossConf):
    _target_: str = "modulus.loss.aggregator.SoftAdapt"
    eps: float = 1e-8


@dataclass
class AggregatorRelobraloConf(LossConf):
    _target_: str = "modulus.loss.aggregator.Relobralo"
    alpha: float = 0.95
    beta: float = 0.99
    tau: float = 1.0
    eps: float = 1e-8


@dataclass
class NTKConf:
    use_ntk: bool = False
    save_name: Any = None  # Union[str, None]
    run_freq: int = 1000


def register_loss_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="loss",
        name="sum",
        node=AggregatorSumConf,
    )
    cs.store(
        group="loss",
        name="grad_norm",
        node=AggregatorGradNormConf,
    )
    cs.store(
        group="loss",
        name="res_norm",
        node=AggregatorResNormConf,
    )
    cs.store(
        group="loss",
        name="homoscedastic",
        node=AggregatorHomoscedasticConf,
    )
    cs.store(
        group="loss",
        name="lr_annealing",
        node=AggregatorLRAnnealingConf,
    )
    cs.store(
        group="loss",
        name="soft_adapt",
        node=AggregatorSoftAdaptConf,
    )
    cs.store(
        group="loss",
        name="relobralo",
        node=AggregatorRelobraloConf,
    )

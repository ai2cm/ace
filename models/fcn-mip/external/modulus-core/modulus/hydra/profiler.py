"""
Profiler config
"""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI, II
from typing import Any, Union, List, Dict


@dataclass
class ProfilerConf:
    profile: bool = MISSING
    start_step: int = MISSING
    end_step: int = MISSING


@dataclass
class NvtxProfiler(ProfilerConf):
    name: str = "nvtx"
    profile: bool = False
    start_step: int = 0
    end_step: int = 100


@dataclass
class TensorBoardProfiler(ProfilerConf):
    name: str = "tensorboard"
    profile: bool = False
    start_step: int = 0
    end_step: int = 100
    warmup: int = 5
    repeat: int = 1
    filename: str = "${hydra.job.override_dirname}-${hydra.job.name}.profile"


def register_profiler_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="profiler",
        name="nvtx",
        node=NvtxProfiler,
    )
    cs.store(
        group="profiler",
        name="tensorboard",
        node=TensorBoardProfiler,
    )

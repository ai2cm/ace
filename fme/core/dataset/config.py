import abc
from collections.abc import Sequence

import torch

from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule


class DatasetConfigABC(abc.ABC):
    @abc.abstractmethod
    def build(
        self,
        names: Sequence[str],
        n_timesteps: IntSchedule,
        allow_variable_masking: bool = False,
    ) -> tuple[torch.utils.data.Dataset, DatasetProperties]:
        pass

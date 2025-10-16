import abc
from collections.abc import Sequence

import torch

from fme.core.dataset.properties import DatasetProperties


class DatasetConfigABC(abc.ABC):
    @abc.abstractmethod
    def build(
        self,
        names: Sequence[str],
        n_timesteps: int,
    ) -> tuple[torch.utils.data.Dataset, DatasetProperties]:
        pass

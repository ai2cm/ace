from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeAlias

import torch

from fme.core.typing_ import TensorMapping

SingleInputFunc: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
ComparisonInputFunc: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class _ComparisonAggregator(Protocol):
    def record_batch(
        self, target: TensorMapping, prediction: TensorMapping
    ) -> None: ...

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]: ...


class _CoarseComparisonAggregator(Protocol):
    def record_batch(
        self, target: TensorMapping, prediction: TensorMapping, coarse: TensorMapping
    ) -> None: ...

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]: ...


class _DynamicMetricComparisonAggregator(Protocol):
    def record_batch(
        self,
        target: TensorMapping,
        prediction: TensorMapping,
        dynamic_metric: ComparisonInputFunc | None = None,
    ) -> None: ...

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]: ...

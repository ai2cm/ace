"""
This file contains code for computing metrics of single variables on batches of data,
and aggregating them into a single metric value. The functions here mainly exist
to turn metric functions that may have different APIs into a common API,
so that they can be iterated over and called in the same way in a loop.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

import torch

from fme.core.typing_ import TensorDict, TensorMapping


class AreaWeightedFunction(Protocol):
    """
    A function that computes a metric on the true and predicted values,
    weighted by area.
    """

    def __call__(
        self,
        truth: TensorMapping,
        predicted: TensorMapping,
    ) -> TensorDict: ...


class ReducedMetric(Protocol):
    """Used to record a metric value on batches of data (potentially out-of-memory)
    and then get the total metric at the end.
    """

    def record(self, target: TensorMapping, gen: TensorMapping):
        """
        Update metric for a batch of data.
        """
        ...

    def get(self) -> TensorDict:
        """
        Get the total metric value, not divided by number of recorded batches.
        """
        ...

    def get_channel_mean(self) -> torch.Tensor:
        """
        Get the channel-mean metric value, not divided by number of record batches.
        """
        ...


class AreaWeightedReducedMetric:
    """
    A wrapper around an area-weighted metric function.
    """

    def __init__(
        self,
        device: torch.device,
        compute_metric: AreaWeightedFunction,
        channel_mean_names: Sequence[str] | None = None,
    ):
        self._compute_metric = compute_metric
        self._total: TensorDict | None = None
        self._channel_mean: torch.Tensor | None = None
        self._device = device
        self._channel_mean_names = channel_mean_names

    def _get_channel_mean_names(self, tensors: TensorDict) -> Sequence[str]:
        if self._channel_mean_names is None:
            return list(tensors.keys())
        return self._channel_mean_names

    def _update_total(self, tensors: TensorDict):
        if self._total is None:
            self._total = {
                name: torch.zeros_like(tensor, device=self._device)
                for name, tensor in tensors.items()
            }
            self._channel_mean = torch.tensor(0.0, device=self._device)
        missing_names = set(self._total) - set(tensors)
        if len(missing_names) > 0:
            raise ValueError(
                f"Missing metrics for {missing_names} which were "
                "present the first time metrics were recorded."
            )
        channel_mean_names = self._get_channel_mean_names(tensors)
        for name, tensor in tensors.items():
            try:
                self._total[name] += tensor
                if name in channel_mean_names:
                    self._channel_mean += tensor / len(channel_mean_names)
            except KeyError:
                raise ValueError(
                    "Attempted to record the area weighted reduced metric for "
                    f"'{name}' but it was not present the first time metrics "
                    "were recorded."
                )

    def record(self, target: TensorMapping, gen: TensorMapping):
        """Add a batch of data to the metric.

        Args:
            target: Target data. Should have shape [batch, time, height, width].
            gen: Generated data. Should have shape [batch, time, height, width].
        """
        # Update totals for each variable
        batch_avgs = {
            name: tensor.mean(dim=0)
            for name, tensor in self._compute_metric(target, gen).items()
        }
        self._update_total(batch_avgs)

    def get(self) -> TensorDict:
        """Returns the metric."""
        if self._total is None:
            return defaultdict(lambda: torch.tensor(torch.nan))
        return self._total

    def get_channel_mean(self) -> torch.Tensor:
        """Returns the channel-mean metric."""
        if self._channel_mean is None:
            return torch.tensor(torch.nan)
        return self._channel_mean

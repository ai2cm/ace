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

from fme.core.tensor_dict_accumulator import TensorDictAccumulator
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
        self._accumulator = TensorDictAccumulator(device=device)
        self._channel_mean: torch.Tensor | None = None
        self._device = device
        self._channel_mean_names = channel_mean_names
        # Variables whose target is entirely NaN (e.g. filled by
        # allow_missing_variables); detected once on the first batch since
        # missingness is constant, then excluded from the channel mean.
        self._all_nan_target_names: set[str] | None = None

    def _get_channel_mean_names(self, tensors: TensorDict) -> Sequence[str]:
        if self._channel_mean_names is None:
            return list(tensors.keys())
        return self._channel_mean_names

    def record(self, target: TensorMapping, gen: TensorMapping):
        """Add a batch of data to the metric.

        Args:
            target: Target data. Should have shape [batch, time, height, width].
            gen: Generated data. Should have shape [batch, time, height, width].
        """
        batch_avgs = {
            name: tensor.mean(dim=0)
            for name, tensor in self._compute_metric(target, gen).items()
        }
        if self._channel_mean is None:
            self._channel_mean = torch.tensor(0.0, device=self._device)
        channel_mean_names = self._get_channel_mean_names(batch_avgs)
        missing = [n for n in channel_mean_names if n not in batch_avgs]
        if missing:
            raise KeyError(
                f"channel_mean_names contains entries not present in the "
                f"recorded data: {missing}. Available: {sorted(batch_avgs)}."
            )
        if self._all_nan_target_names is None:
            self._all_nan_target_names = {
                name for name in channel_mean_names if torch.isnan(target[name]).all()
            }
        # Targets may be entirely NaN if filled by allow_missing_variables
        non_nan_targets = [
            name
            for name in channel_mean_names
            if name not in self._all_nan_target_names
        ]
        if not non_nan_targets:
            raise ValueError(
                "All target variables are NaN; cannot compute channel mean."
            )
        for name in non_nan_targets:
            self._channel_mean += batch_avgs[name] / len(non_nan_targets)

        self._accumulator.add(batch_avgs)

    def get(self) -> TensorDict:
        """Returns the metric."""
        total = self._accumulator.get_sum()
        if total is None:
            return defaultdict(lambda: torch.tensor(torch.nan))
        return total

    def get_channel_mean(self) -> torch.Tensor:
        """Returns the channel-mean metric."""
        if self._channel_mean is None:
            return torch.tensor(torch.nan)
        return self._channel_mean

"""
This file contains code for computing metrics of single variables on batches of data,
and aggregating them into a single metric value. The functions here mainly exist
to turn metric functions that may have different APIs into a common API,
so that they can be iterated over and called in the same way in a loop.
"""

from typing import Protocol

import torch


class AreaWeightedFunction(Protocol):
    """
    A function that computes a metric on the true and predicted values,
    weighted by area.
    """

    def __call__(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
    ) -> torch.Tensor: ...


class ReducedMetric(Protocol):
    """Used to record a metric value on batches of data (potentially out-of-memory)
    and then get the total metric at the end.
    """

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        """
        Update metric for a batch of data.
        """
        ...

    def get(self) -> torch.Tensor:
        """
        Get the total metric value, not divided by number of recorded batches.
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
    ):
        self._compute_metric = compute_metric
        self._total = None
        self._device = device

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        """Add a batch of data to the metric.

        Args:
            target: Target data. Should have shape [batch, time, height, width].
            gen: Generated data. Should have shape [batch, time, height, width].
        """
        new_value = self._compute_metric(target, gen).mean(dim=0)
        if self._total is None:
            self._total = torch.zeros_like(new_value, device=self._device)
        self._total += new_value

    def get(self) -> torch.Tensor:
        """Returns the metric."""
        return self._total

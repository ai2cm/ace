"""Inference SubAggregator for residual (tendency) power spectra."""

import dataclasses
from typing import Any, Literal

import xarray as xr

from fme.core.fill import SmoothFloodFill

from ..residual_spectrum import temporal_diffs
from .build_context import MetricBuildContext, MetricNotSupportedError, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator
from .spectrum import PairedSphericalPowerSpectrumAggregator


class ResidualSpectrumAggregator:
    """Computes power spectra of temporal tendency fields.

    For each batch, computes consecutive temporal differences within the
    prediction and target windows, then feeds those tendency fields to
    a ``PairedSphericalPowerSpectrumAggregator``.
    """

    def __init__(self, inner: PairedSphericalPowerSpectrumAggregator):
        self._inner = inner

    def record_batch(self, data: InferenceBatchData) -> None:
        if not data.has_target:
            return
        gen_diffs = temporal_diffs(data.prediction)
        tgt_diffs = temporal_diffs(data.target)
        if gen_diffs:
            self._inner.record_paired_data(prediction=gen_diffs, target=tgt_diffs)

    def get_logs(self, label: str) -> dict[str, Any]:
        return self._inner.get_logs(label)

    def get_dataset(self) -> xr.Dataset:
        return self._inner.get_dataset()


@dataclasses.dataclass
class ResidualSpectrumMetricConfig:
    """Metric config for residual (tendency) power spectra.

    Parameters:
        variables: Optional variable filter.  If ``None``, all variables
            present in both prediction and target are included.
        name: Name used as the aggregator key in logs.
    """

    type: Literal["residual_spectrum"] = "residual_spectrum"
    variables: list[str] | None = None
    name: str = "residual_spectrum"

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        try:
            inner = PairedSphericalPowerSpectrumAggregator(
                gridded_operations=ctx.ops,
                nan_fill_fn=SmoothFloodFill(num_steps=4),
                report_plot=True,
                variable_metadata=ctx.variable_metadata,
            )
        except NotImplementedError as e:
            raise MetricNotSupportedError(str(e)) from e
        agg: SubAggregator = ResidualSpectrumAggregator(inner)
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))

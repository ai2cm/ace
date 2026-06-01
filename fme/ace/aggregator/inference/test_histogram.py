"""Tests for the inference-side histogram aggregator wrapper.

The core histogram machinery lives in ``fme.core.histogram`` and is
covered by ``fme/core/test_histogram.py``. The wrapper here adds the
``percentile_variables`` allowlist that gates which variables' tail
percentile metrics get emitted into the W&B run — useful when the
histogram plot is wanted cohort-wide but the noisy 99.9999th-
percentile keys are only worth tracking for a small variable list
(precipitation, etc.).
"""

import torch

import fme
from fme.ace.aggregator.inference.data import InferenceBatchData, make_dummy_time
from fme.ace.aggregator.inference.histogram import HistogramAggregator


def _batch_two_vars(nlat: int = 8, nlon: int = 16) -> InferenceBatchData:
    rng = torch.Generator(device=fme.get_device()).manual_seed(0)
    data = {
        "PRATEsfc": torch.rand(
            (2, 3, nlat, nlon), generator=rng, device=fme.get_device()
        ),
        "TMP2m": torch.rand((2, 3, nlat, nlon), generator=rng, device=fme.get_device())
        + 280,
    }
    return InferenceBatchData(
        prediction=data,
        prediction_norm={},
        target=data,
        target_norm={},
        time=make_dummy_time(2, 3),
        i_time_start=0,
    )


def test_histogram_aggregator_percentile_variables_none_is_default():
    """Default ``percentile_variables=None`` emits percentile keys for
    every variable the underlying ``ComparedDynamicHistograms`` reports
    on — backwards-compatible behaviour."""
    agg = HistogramAggregator()
    agg.record_batch(_batch_two_vars())
    logs = agg.get_logs("hist")
    pct_keys = [k for k in logs if "th-percentile/" in k]
    fields = {k.rsplit("/", 1)[-1] for k in pct_keys}
    assert "PRATEsfc" in fields
    assert "TMP2m" in fields


def test_histogram_aggregator_percentile_variables_filters():
    """An allowlist drops percentile keys for variables not in it. The
    histogram-plot keys (non-``th-percentile`` entries) are left in
    place for every variable — distribution shape stays observable
    cohort-wide while tail-percentile noise is restricted."""
    agg = HistogramAggregator(percentile_variables=["PRATEsfc"])
    agg.record_batch(_batch_two_vars())
    logs = agg.get_logs("hist")

    pct_keys = [k for k in logs if "th-percentile/" in k]
    fields = {k.rsplit("/", 1)[-1] for k in pct_keys}
    assert fields == {"PRATEsfc"}, fields  # only PRATEsfc

    # Histogram-plot keys for both variables still present.
    plot_keys = [k for k in logs if "th-percentile/" not in k]
    plot_fields = {k.rsplit("/", 1)[-1] for k in plot_keys}
    assert "PRATEsfc" in plot_fields
    assert "TMP2m" in plot_fields


def test_histogram_aggregator_percentile_variables_empty_drops_all():
    """An empty allowlist drops every percentile key but leaves the
    histogram plots in place. Useful when histograms are wanted as
    visual sanity checks but no scalar tail metric is desired."""
    agg = HistogramAggregator(percentile_variables=[])
    agg.record_batch(_batch_two_vars())
    logs = agg.get_logs("hist")
    pct_keys = [k for k in logs if "th-percentile/" in k]
    assert pct_keys == []
    # Plots survive.
    assert any("hist/PRATEsfc" == k or k.endswith("/PRATEsfc") for k in logs)

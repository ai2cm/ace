from fme.core.aggregator.inference.reduced import AreaWeightedReducedMetric
import torch
import numpy as np
from fme.core.device import get_device


def test_area_weighted_reduced_metric_includes_later_window_starts():
    """
    The area weighted reduced metric should assume that the start
    of a window is always recorded, as we clip it before calling.
    """

    def compute_metric(truth, predicted, weights=None, dim=()):
        return truth.mean(dim=(2, 3))

    metric = AreaWeightedReducedMetric(
        area_weights=torch.ones([4]),
        device=get_device(),
        compute_metric=compute_metric,
        n_timesteps=7,
    )

    data = torch.ones([2, 3, 4, 4], device=get_device())
    metric.record(data, data, 0)
    data[:, 0, :, :] = np.nan
    metric.record(data, data, 2)
    metric.record(data, data, 4)
    result = metric.get()
    result = result.cpu().numpy()
    # assert tensor is all ones
    assert np.sum(np.isnan(result)) == 2
    assert np.isnan(result[2])
    assert np.isnan(result[4])

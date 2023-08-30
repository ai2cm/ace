from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from fme.core.aggregator.inference.video import VideoAggregator
from fme.core.device import get_device


def _make_data(
    names: List[str], n_samples: int, nx: int, ny: int, offsets: np.ndarray
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Make a 2D array with a pattern of offsets.

    Each variable will have a random horizontal pattern consistent between
    the generated and target data, but with a constant time-dependent offset
    between the two.

    Args:
        names: Names of the variables.
        n_samples: Number of samples in the batch.
        nx: Number of x points.
        ny: Number of y points.
        offsets: 1-D (time-axis) array of offsets, which also determines the
            length of the time axis.
    """
    nt = len(offsets)
    data = np.random.normal(size=(n_samples, nt, len(names), ny, nx))
    gen = {}
    target = {}
    for i_name, name in enumerate(names):
        target[name] = torch.tensor(data[:, :, i_name, :, :], device=get_device())
        gen[name] = torch.tensor(data[:, :, i_name, :, :], device=get_device())
        # add time-variable offset to generated data
        for i_offset, offset in enumerate(offsets):
            gen[name][:, i_offset, :, :] += offset
    return target, gen


def time_select(
    i_start: int,
    i_end: int,
    gen: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Select a time range from the generated and target data.

    Args:
        i_start: Start index.
        i_end: End index.
        gen: Generated data.
        target: Target data.
    """
    gen_out = {}
    target_out = {}
    for name in gen.keys():
        gen_out[name] = gen[name][:, i_start:i_end]
        target_out[name] = target[name][:, i_start:i_end]
    return target_out, gen_out


@pytest.mark.parametrize(
    "offsets",
    [
        np.array([0, 1]),
        np.array([0, 1, 2, 3, 4, 5.5]),
    ],
)
def test_video_data(offsets: np.ndarray):
    aggregator = VideoAggregator(n_timesteps=len(offsets), enable_extended_videos=True)
    names = ["a"]
    n_samples = 5
    nx = 8
    ny = 10
    target, gen = _make_data(names, n_samples, nx, ny, offsets)
    n_window_in_memory = 2
    for i_start in range(0, len(offsets) - n_window_in_memory + 1):
        i_end = i_start + n_window_in_memory
        target_window, gen_window = time_select(i_start, i_end, gen, target)
        aggregator.record_batch(
            loss=0, target_data=target_window, gen_data=gen_window, i_time_start=i_start
        )
    data = aggregator._get_data("video")
    assert data["video/bias/a"].target is None
    assert data["video/min_err/a"].target is None
    assert data["video/max_err/a"].target is None
    assert data["video/a"].gen.shape[0] == len(offsets)
    assert data["video/a"].target.shape[0] == len(offsets)
    assert data["video/bias/a"].gen.shape[0] == len(offsets)
    assert data["video/min_err/a"].gen.shape[0] == len(offsets)
    assert data["video/max_err/a"].gen.shape[0] == len(offsets)
    for i, offset_i in enumerate(offsets):
        np.testing.assert_allclose(
            data["video/a"].gen[i, :], data["video/a"].target[i, :] + offset_i
        )
        np.testing.assert_allclose(data["video/bias/a"].gen[i, :], offset_i)
        np.testing.assert_allclose(data["video/min_err/a"].gen[i, :], offset_i)
        np.testing.assert_allclose(data["video/max_err/a"].gen[i, :], offset_i)
        np.testing.assert_allclose(data["video/gen_var/a"].gen[i, :], 1.0)


@pytest.mark.parametrize(
    "offsets",
    [
        np.array([0, 1]),
        np.array([0, 1, 2, 3, 4, 5.5]),
    ],
)
def test_video_data_without_extended_videos(offsets: np.ndarray):
    aggregator = VideoAggregator(n_timesteps=len(offsets), enable_extended_videos=False)
    names = ["a"]
    n_samples = 5
    nx = 8
    ny = 10
    target, gen = _make_data(names, n_samples, nx, ny, offsets)
    n_window_in_memory = 2
    for i_start in range(0, len(offsets) - n_window_in_memory + 1):
        i_end = i_start + n_window_in_memory
        target_window, gen_window = time_select(i_start, i_end, gen, target)
        aggregator.record_batch(
            loss=0, target_data=target_window, gen_data=gen_window, i_time_start=i_start
        )
    data = aggregator._get_data("video")
    assert len(data) == 1
    assert data["video/a"].gen.shape[0] == len(offsets)
    assert data["video/a"].target.shape[0] == len(offsets)
    for i, offset_i in enumerate(offsets):
        np.testing.assert_allclose(
            data["video/a"].gen[i, :], data["video/a"].target[i, :] + offset_i
        )


def slice_samples(data: Dict[str, torch.Tensor], i_start: int, i_end: int):
    data_out = {}
    for name in data.keys():
        data_out[name] = data[name][i_start:i_end]
    return data_out


@pytest.mark.parametrize("n_batches", [1, 5])
def test_video_data_values_on_random_inputs(n_batches: int):
    torch.manual_seed(0)
    names = ["a"]
    n_samples = 10
    samples_per_batch = n_samples // n_batches
    nx = 8
    ny = 10
    nt = 7
    aggregator = VideoAggregator(n_timesteps=nt, enable_extended_videos=True)
    offsets = np.zeros([nt])
    target, _ = _make_data(names, n_samples, nx, ny, offsets)
    gen, _ = _make_data(names, n_samples, nx, ny, offsets)
    n_window_in_memory = 2
    for i_start in range(0, len(offsets) - n_window_in_memory + 1):
        i_end = i_start + n_window_in_memory
        target_window, gen_window = time_select(i_start, i_end, gen, target)
        for nb in range(n_batches):  # shouldn't affect results to duplicate batches
            aggregator.record_batch(
                loss=0,
                target_data=slice_samples(
                    target_window,
                    i_start=nb * samples_per_batch,
                    i_end=(nb + 1) * samples_per_batch,
                ),
                gen_data=slice_samples(
                    gen_window,
                    i_start=nb * samples_per_batch,
                    i_end=(nb + 1) * samples_per_batch,
                ),
                i_time_start=i_start,
            )
    data = aggregator._get_data("video")
    assert data["video/bias/a"].target is None
    assert data["video/min_err/a"].target is None
    assert data["video/max_err/a"].target is None
    assert data["video/a"].gen.shape[0] == len(offsets)
    assert data["video/a"].target.shape[0] == len(offsets)
    assert data["video/bias/a"].gen.shape[0] == len(offsets)
    assert data["video/min_err/a"].gen.shape[0] == len(offsets)
    assert data["video/max_err/a"].gen.shape[0] == len(offsets)
    np.testing.assert_allclose(
        data["video/a"].gen.cpu().numpy(), gen["a"].mean(dim=0).cpu().numpy()
    )
    np.testing.assert_allclose(
        data["video/a"].target.cpu().numpy(), target["a"].mean(dim=0).cpu().numpy()
    )
    np.testing.assert_allclose(
        data["video/bias/a"].gen.cpu().numpy(),
        (gen["a"] - target["a"]).mean(dim=0).cpu().numpy(),
    )
    np.testing.assert_allclose(
        data["video/min_err/a"].gen.cpu().numpy(),
        (gen["a"] - target["a"]).min(dim=0)[0].cpu().numpy(),
    )
    np.testing.assert_allclose(
        data["video/max_err/a"].gen.cpu().numpy(),
        (gen["a"] - target["a"]).max(dim=0)[0].cpu().numpy(),
    )
    np.testing.assert_allclose(
        data["video/gen_var/a"].gen.cpu().numpy(),
        (gen["a"].var(dim=0) / target["a"].var(dim=0)).cpu().numpy(),
    )

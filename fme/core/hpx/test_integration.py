import pathlib
import tempfile

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.testing.compare
import pytest
import torch

from fme.ace.aggregator.plotting import plot_imshow
from fme.core.device import get_device
from fme.core.gridded_ops import HEALPixOperations
from fme.core.hpx.reorder import get_reordering_ring_to_xy

DIR = pathlib.Path(__file__).parent


def regression_test_image(filename: pathlib.Path, fig: matplotlib.figure.Figure):
    if not filename.exists():
        fig.savefig(filename)
        assert False, f"Regression test image not found, created {filename}"
    else:
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            fig.savefig(temp_file.name)
            diff = matplotlib.testing.compare.compare_images(
                temp_file.name, filename, tol=1e-6
            )
            assert (
                diff is None
            ), f"Regression test image differs from {filename}:\n{diff}"


@pytest.mark.parametrize("i_l, i_m", [(1, 0), (1, 1), (1, 2), (2, 1)])
def test_plot_isht_result(i_l: int, i_m: int):
    nside = 8
    device = get_device()
    ops = HEALPixOperations(nside=nside)
    isht = ops.get_real_isht()
    x_hat = torch.zeros([isht.lmax, isht.mmax], dtype=torch.complex64, device=device)
    x_hat[i_l, i_m] = 1
    isht_result = isht(x_hat)
    reordering = get_reordering_ring_to_xy(nside=nside, device=device)
    isht_result_xy = isht_result[..., reordering].reshape(12, nside, nside)
    fig, ax = plt.subplots()
    plot_imshow(ax=ax, data=isht_result_xy.cpu().numpy(), cmap="viridis")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    regression_test_image(
        DIR / "testdata" / f"test_plot_isht_result_{i_l}_{i_m}.png", fig
    )

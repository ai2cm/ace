import pytest
import torch
import torch_harmonics.distributed as thd

from fme.core import get_device
from fme.core.distributed import Distributed
from fme.core.sht import DistributedInverseRealSHT, DistributedRealSHT

IMG_SHAPE = (8, 16)


def _get_local_block(x: torch.Tensor) -> torch.Tensor:
    """This rank's block of ``x``, with the second-to-last dimension split
    across the polar group and the last dimension split across the azimuth
    group."""
    polar_shapes = thd.compute_split_shapes(x.shape[-2], thd.polar_group_size())
    azimuth_shapes = thd.compute_split_shapes(x.shape[-1], thd.azimuth_group_size())
    polar_start = sum(polar_shapes[: thd.polar_group_rank()])
    azimuth_start = sum(azimuth_shapes[: thd.azimuth_group_rank()])
    return x[
        ...,
        polar_start : polar_start + polar_shapes[thd.polar_group_rank()],
        azimuth_start : azimuth_start + azimuth_shapes[thd.azimuth_group_rank()],
    ]


def _assert_compiled_matches_eager(
    module: torch.nn.Module, input_shape: tuple[int, ...], dtype: torch.dtype
):
    """Compile ``module``, call it twice on fresh local inputs, and check
    each result against eager."""
    torch._dynamo.reset()
    compiled = torch.compile(module, backend="aot_eager")
    try:
        for _ in range(2):
            x = _get_local_block(
                torch.randn(*input_shape, dtype=dtype, device=get_device())
            )
            torch.testing.assert_close(compiled(x), module(x))
    finally:
        torch._dynamo.reset()


def _assert_matches_upstream(
    module: torch.nn.Module,
    upstream: torch.nn.Module,
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
):
    """Call ``module`` and ``upstream`` eagerly on the same fresh local input
    and check that the results are equal."""
    x = _get_local_block(torch.randn(*input_shape, dtype=dtype, device=get_device()))
    torch.testing.assert_close(module(x), upstream(x), rtol=0, atol=0)


@pytest.mark.parallel
def test_compiled_sht_matches_eager():
    """The distributed SHT matches eager across repeated compiled calls."""
    torch.manual_seed(0)
    nlat, nlon = IMG_SHAPE
    sht = Distributed.get_instance().get_sht(nlat, nlon).to(get_device())
    _assert_compiled_matches_eager(sht, (2, 3, nlat, nlon), torch.float32)


@pytest.mark.parallel
def test_compiled_isht_matches_eager():
    """The distributed inverse SHT matches eager across repeated compiled calls."""
    torch.manual_seed(0)
    nlat, nlon = IMG_SHAPE
    isht = Distributed.get_instance().get_isht(nlat, nlon).to(get_device())
    _assert_compiled_matches_eager(isht, (2, 3, isht.lmax, isht.mmax), torch.complex64)


@pytest.mark.parallel
def test_sht_matches_upstream():
    """The distributed SHT matches the upstream torch-harmonics class."""
    torch.manual_seed(0)
    nlat, nlon = IMG_SHAPE
    sht = DistributedRealSHT(nlat, nlon).to(get_device())
    upstream = thd.DistributedRealSHT(nlat, nlon).to(get_device())
    _assert_matches_upstream(sht, upstream, (2, 3, nlat, nlon), torch.float32)


@pytest.mark.parallel
def test_isht_matches_upstream():
    """The distributed inverse SHT matches the upstream torch-harmonics class."""
    torch.manual_seed(0)
    nlat, nlon = IMG_SHAPE
    isht = DistributedInverseRealSHT(nlat, nlon).to(get_device())
    upstream = thd.DistributedInverseRealSHT(nlat, nlon).to(get_device())
    _assert_matches_upstream(
        isht, upstream, (2, 3, isht.lmax, isht.mmax), torch.complex64
    )

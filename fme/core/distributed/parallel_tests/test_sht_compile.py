import pytest
import torch
import torch_harmonics.distributed as thd

from fme.core import get_device
from fme.core.distributed import Distributed

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


@pytest.mark.parallel
def test_compiled_sht_matches_eager():
    """The distributed SHT runs repeatedly under torch.compile and matches
    eager.

    With both latitude and longitude split, torch-harmonics hands a complex
    view of a real tensor to its polar transpose. Reusing the compiled code on
    the second call fails in AOT autograd ("setStorage ... out of bounds")
    unless the view is cloned first.
    """
    torch.manual_seed(0)
    nlat, nlon = IMG_SHAPE
    sht = Distributed.get_instance().get_sht(nlat, nlon).to(get_device())
    _assert_compiled_matches_eager(sht, (2, 3, nlat, nlon), torch.float32)


@pytest.mark.parallel
def test_compiled_isht_matches_eager():
    """The distributed inverse SHT runs repeatedly under torch.compile and
    matches eager.

    With longitude split and latitude not, torch-harmonics hands a complex view
    of a real tensor to its azimuth transpose. The first call compiles the
    transpose; reusing that compiled code on the next call fails in AOT
    autograd ("setStorage ... out of bounds") unless the view is cloned first.
    """
    torch.manual_seed(0)
    nlat, nlon = IMG_SHAPE
    isht = Distributed.get_instance().get_isht(nlat, nlon).to(get_device())
    _assert_compiled_matches_eager(isht, (2, 3, isht.lmax, isht.mmax), torch.complex64)

import pytest
import torch
import torch_harmonics.distributed as thd

from fme.core import get_device
from fme.core.distributed import Distributed

IMG_SHAPE = (8, 16)


def _get_local_coefficients(
    n_samples: int, n_channels: int, lmax: int, mmax: int
) -> torch.Tensor:
    """Spectral coefficients laid out as the distributed inverse SHT expects:
    degree ``l`` split across the polar group and order ``m`` split across the
    azimuth group, with channels local."""
    coefficients = torch.randn(
        n_samples, n_channels, lmax, mmax, dtype=torch.complex64, device=get_device()
    )
    l_shapes = thd.compute_split_shapes(lmax, thd.polar_group_size())
    m_shapes = thd.compute_split_shapes(mmax, thd.azimuth_group_size())
    l_start = sum(l_shapes[: thd.polar_group_rank()])
    m_start = sum(m_shapes[: thd.azimuth_group_rank()])
    return coefficients[
        ...,
        l_start : l_start + l_shapes[thd.polar_group_rank()],
        m_start : m_start + m_shapes[thd.azimuth_group_rank()],
    ]


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
    torch._dynamo.reset()
    compiled_isht = torch.compile(isht, backend="aot_eager")
    try:
        for _ in range(2):
            x = _get_local_coefficients(2, 3, isht.lmax, isht.mmax)
            torch.testing.assert_close(compiled_isht(x), isht(x))
    finally:
        torch._dynamo.reset()

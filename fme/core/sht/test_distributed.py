from collections.abc import Callable

import pytest
import torch
import torch_harmonics.distributed as thd

from fme.core import get_device
from fme.core.distributed import Distributed
from fme.core.sht import DistributedInverseRealSHT, DistributedRealSHT

NLAT, NLON = 8, 16
BATCH_CHANNEL_SHAPE = (2, 3)

# (lmax, mmax): the transform defaults, and modes truncated as SFNO does with
# hard_thresholding_fraction=0.5 (lmax=int(nlat*0.5), mmax=int((nlon//2+1)*0.5)).
MODES = pytest.mark.parametrize(
    "lmax, mmax",
    [
        pytest.param(None, None, id="default_modes"),
        pytest.param(NLAT // 2, (NLON // 2 + 1) // 2, id="truncated_modes"),
    ],
)


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


def _random_local_input(
    spatial_shape: tuple[int, int], dtype: torch.dtype
) -> Callable[[], torch.Tensor]:
    """A factory of random inputs with the given global spatial shape,
    restricted to this rank's block. Every rank draws the same global tensor."""

    def make_input() -> torch.Tensor:
        x = torch.randn(
            *BATCH_CHANNEL_SHAPE, *spatial_shape, dtype=dtype, device=get_device()
        )
        return _get_local_block(x)

    return make_input


def _sht_input(sht: torch.nn.Module) -> Callable[[], torch.Tensor]:
    return _random_local_input((sht.nlat, sht.nlon), torch.float32)


def _isht_input(isht: torch.nn.Module) -> Callable[[], torch.Tensor]:
    return _random_local_input((isht.lmax, isht.mmax), torch.complex64)


def _assert_compiled_matches_eager(
    module: torch.nn.Module, make_input: Callable[[], torch.Tensor]
):
    """Compile ``module``, call it twice on fresh local inputs, and check
    each result against eager.

    Without the ``.clone()`` in ``fme.core.sht._distributed``, the second
    compiled call raises "setStorage ... out of bounds" on torch 2.7-2.9 when
    the polar dimension is split (H > 1); torch 2.10 is unaffected.
    """
    torch.manual_seed(0)
    torch._dynamo.reset()
    compiled = torch.compile(module, backend="aot_eager")
    try:
        for _ in range(2):
            x = make_input()
            torch.testing.assert_close(compiled(x), module(x))
    finally:
        torch._dynamo.reset()


def _assert_matches_upstream(
    module: torch.nn.Module,
    upstream: torch.nn.Module,
    make_input: Callable[[], torch.Tensor],
):
    """Call ``module`` and ``upstream`` eagerly on the same fresh local input
    and check that the results are equal."""
    torch.manual_seed(0)
    x = make_input()
    torch.testing.assert_close(module(x), upstream(x), rtol=0, atol=0)


@pytest.mark.parallel
@MODES
def test_compiled_sht_matches_eager(lmax, mmax):
    """The SHT from ``Distributed.get_sht`` matches eager across repeated
    compiled calls. Only spatial parallelism (``model`` backend) returns the
    patched ``DistributedRealSHT``; otherwise this checks a non-distributed SHT."""
    dist = Distributed.get_instance()
    sht = dist.get_sht(NLAT, NLON, lmax=lmax, mmax=mmax)
    if dist.has_spatial_parallelism:
        assert isinstance(sht, DistributedRealSHT)
    _assert_compiled_matches_eager(sht.to(get_device()), _sht_input(sht))


@pytest.mark.parallel
@MODES
def test_compiled_isht_matches_eager(lmax, mmax):
    """The inverse SHT from ``Distributed.get_isht`` matches eager across
    repeated compiled calls. Only spatial parallelism (``model`` backend)
    returns the patched ``DistributedInverseRealSHT``; otherwise this checks a
    non-distributed inverse SHT."""
    dist = Distributed.get_instance()
    isht = dist.get_isht(NLAT, NLON, lmax=lmax, mmax=mmax)
    if dist.has_spatial_parallelism:
        assert isinstance(isht, DistributedInverseRealSHT)
    _assert_compiled_matches_eager(isht.to(get_device()), _isht_input(isht))


@pytest.mark.parallel
@MODES
def test_sht_matches_upstream(lmax, mmax):
    """The distributed SHT matches the upstream torch-harmonics class."""
    sht = DistributedRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax).to(get_device())
    upstream = thd.DistributedRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax).to(get_device())
    _assert_matches_upstream(sht, upstream, _sht_input(sht))


@pytest.mark.parallel
@MODES
def test_isht_matches_upstream(lmax, mmax):
    """The distributed inverse SHT matches the upstream torch-harmonics class."""
    isht = DistributedInverseRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax).to(get_device())
    upstream = thd.DistributedInverseRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax).to(
        get_device()
    )
    _assert_matches_upstream(isht, upstream, _isht_input(isht))


@MODES
def test_isht_compiles_without_graph_break(lmax, mmax):
    """The inverse SHT zeroes the imaginary DC and Nyquist parts in a form
    dynamo can trace, so on a single rank it compiles to a single graph.
    Upstream's ``.imag`` setattr breaks the graph there. Not run under spatial
    parallelism, where the distributed transposes break the graph anyway."""
    isht = DistributedInverseRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax).to(get_device())
    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(isht)(_isht_input(isht)())
    finally:
        torch._dynamo.reset()
    assert explanation.graph_break_count == 0, explanation.break_reasons

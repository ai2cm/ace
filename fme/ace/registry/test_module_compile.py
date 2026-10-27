"""Inductor coverage for every registered module builder.

The test suite compiles with the fast ``aot_eager`` backend (see
:data:`fme.core.testing.TEST_COMPILE_BACKEND`), which traces every network a
step builds but never reaches inductor's code generation, where bugs of their
own live (e.g. stride mismatches in generated kernels). Production compiles
with inductor, so this module compiles one small network per registered
builder with inductor and checks that its outputs and gradients match eager.

Every production builder in the registry has one entry in
:data:`BUILDER_CASES`, so registering a builder without deciding whether its
networks compile fails :func:`test_builder_cases_cover_every_registered_builder`.

These tests run on CUDA, and on CPU where inductor's C++ toolchain is
available, skipping otherwise.
"""

import contextlib
import dataclasses
import functools
import logging
import sys
from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

import pytest
import torch
import torch._inductor.config
from torch import nn

from fme.ace.models.graphcast import GRAPHCAST_AVAIL
from fme.ace.models.healpix.healpix_blocks import (
    AvgPoolDownsamplingBlockConfig,
    BasicConvBlockConfig,
    ConvNeXtBlockConfig,
    TransposedConvUpsampleBlockConfig,
)
from fme.ace.registry.hpx import UNetDecoderConfig, UNetEncoderConfig
from fme.core.compile import compile_backend, compile_enabled
from fme.core.coordinates import HEALPixCoordinates, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.labels import BatchLabels
from fme.core.rand import use_generator
from fme.core.registry.module import Module, ModuleSelector
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.testing import get_dataset_info, skip_without_inductor

LATLON_SHAPE = (8, 16)
"""Image shape for the builders that take a regular lat/lon grid."""

WINDOWED_SHAPE = (16, 32)
"""Image shape for the Swin builders, whose window/patch sizes need more room."""

HEALPIX_NSIDE = 8
"""Face height and width of the HEALPix test grid."""

LABELS = {"label_a", "label_b"}
"""Batch labels used by the conditional builders."""

N_SAMPLES = 2

OUTPUT_TOLERANCE = 1e-3
"""Maximum relative L2 error of the compiled network's output against eager.

Inductor may reassociate and fuse floating point operations (and pick other
kernels for TF32-eligible convolutions on CUDA), so bitwise equality is not
guaranteed; this is loose enough for that and far tighter than any difference
that would change a prediction.
"""

GRAD_TOLERANCE = 5e-3
"""Maximum relative L2 error of each parameter's gradient against eager.

Looser than :data:`OUTPUT_TOLERANCE` since the backward accumulates the
forward's reassociation error through a second pass. Each error is relative
to the larger of that parameter's gradient norm and
:data:`GRAD_NORM_FLOOR` times the norm of the whole gradient, since some
parameters (e.g. normalization biases at initialization) get gradients that
are zero up to rounding, whose relative error is meaningless.
"""

GRAD_NORM_FLOOR = 1e-3
"""Fraction of the whole gradient's norm below which a parameter's gradient
counts as zero for :data:`GRAD_TOLERANCE`."""

INDUCTOR_TIMEOUT_SECONDS = 300
"""Per-test timeout for compiling a network's forward and backward with
inductor, which for the Swin builders takes about two minutes on the CI GPU."""

SWIN_BUILDERS = frozenset({"SwinTransformer", "NoiseConditionedSwinTransformer"})

SEGFAULTS_ON_MACOS_CPU = frozenset({"Samudra"})
"""Builders whose inductor backward crashes the process on macOS CPU.

Seen with torch 2.7.1 on Apple silicon: the generated backward calls
``aten.convolution_backward`` on a ``slice_scatter`` output and segfaults
inside aten. Production compiles on CUDA, where the case runs; Linux CPU runs
it too.
"""


def _is_production_builder(builder_type: str) -> bool:
    # test modules register mock builders into the same process-global
    # registry when they share a pytest worker with this one; those live in
    # test_*.py modules and have no production networks
    module_name = ModuleSelector.registry._types[builder_type].__module__
    return not module_name.rsplit(".", 1)[-1].startswith("test_")


def _latlon_dataset_info(
    img_shape: tuple[int, int], all_labels: set[str] | None = None
) -> DatasetInfo:
    return get_dataset_info(
        img_shape=img_shape, all_labels=all_labels, device=get_device()
    )


def _healpix_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=HEALPixCoordinates(
            face=torch.arange(12, device=get_device()),
            height=torch.arange(HEALPIX_NSIDE, device=get_device()),
            width=torch.arange(HEALPIX_NSIDE, device=get_device()),
        )
    )


def _floenet_dataset_info(img_shape: tuple[int, int]) -> DatasetInfo:
    """DatasetInfo with the mask and meshgrid coordinates FloeNet needs.

    FloeNet places its mesh nodes from the horizontal coordinates and selects
    the wet points with a 2D mask, and builds nothing without them.
    """
    height, width = img_shape
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.linspace(-90, 90, height),
            lon=torch.linspace(0, 360, width + 1)[:-1],
        ),
        spatial_mask_provider=SpatialMaskProvider(
            masks={"mask_2d": torch.ones(height, width, dtype=torch.bool)}
        ),
    )


def _healpix_unet_config() -> dict[str, Any]:
    """A two-level HEALPix UNet, the smallest shape the encoder/decoder allow.

    ``karlbauer`` padding is pure torch, so this case does not need the
    optional ``earth2grid`` dependency.
    """
    encoder = UNetEncoderConfig(
        conv_block=ConvNeXtBlockConfig(),
        down_sampling_block=AvgPoolDownsamplingBlockConfig(pooling=2),
        n_channels=[8, 16],
        n_layers=[1, 1],
    )
    decoder = UNetDecoderConfig(
        conv_block=ConvNeXtBlockConfig(),
        up_sampling_block=TransposedConvUpsampleBlockConfig(stride=2),
        output_layer=BasicConvBlockConfig(n_layers=1, kernel_size=1),
        n_channels=[16, 8],
        n_layers=[1, 1],
    )
    return {
        "encoder": dataclasses.asdict(encoder),
        "decoder": dataclasses.asdict(decoder),
        "hpx_padding_mode": "karlbauer",
    }


def _prebuilt_selector() -> ModuleSelector:
    """A selector for the prebuilt builder, which returns the network it is given.

    A fresh network is built per call so that cases do not share parameters.
    """
    return ModuleSelector(
        type="prebuilt",
        config={"module": nn.Sequential(nn.Conv2d(3, 2, kernel_size=1), nn.GELU())},
    )


@dataclasses.dataclass(frozen=True)
class BuilderCase:
    """One registered module builder, configured small enough to compile in a test.

    Parameters:
        name: The registered builder type, e.g. ``"MLP"``.
        build_selector: Builds the selector; a callable so that no network is
            constructed at collection time and cases never share parameters.
        build_dataset_info: Builds the DatasetInfo the builder needs.
        input_shape: Shape of one sample of the network input, without the
            leading batch dimension. HEALPix networks take ``(12, channels,
            face, face)``; the rest take ``(channels, lat, lon)``.
        n_in: Number of input channels passed to the builder.
        n_out: Number of output channels passed to the builder.
        conditional: Whether the selector conditions on batch labels, in which
            case the tests pass a :class:`BatchLabels` alongside the input.
        expect_unsupported: Whether the builder declares ``torch.compile``
            unsupported, so ``Module.compile()`` must leave it uncompiled
            instead of tracing.
        marks: Marks applied to the parametrized cases, e.g. dependency skips.
    """

    name: str
    build_selector: Callable[[], ModuleSelector]
    build_dataset_info: Callable[[], DatasetInfo]
    input_shape: tuple[int, ...]
    n_in: int
    n_out: int
    conditional: bool = False
    expect_unsupported: bool = False
    marks: tuple[pytest.MarkDecorator, ...] = ()


BUILDER_CASES: list[BuilderCase] = [
    BuilderCase(
        name="MLP",
        build_selector=functools.partial(
            ModuleSelector, type="MLP", config={"hidden_dim": 8, "depth": 2}
        ),
        build_dataset_info=functools.partial(_latlon_dataset_info, LATLON_SHAPE),
        input_shape=(3, *LATLON_SHAPE),
        n_in=3,
        n_out=2,
    ),
    BuilderCase(
        name="prebuilt",
        build_selector=_prebuilt_selector,
        build_dataset_info=functools.partial(_latlon_dataset_info, LATLON_SHAPE),
        input_shape=(3, *LATLON_SHAPE),
        n_in=3,
        n_out=2,
    ),
    BuilderCase(
        name="LandNet",
        build_selector=functools.partial(
            ModuleSelector, type="LandNet", config={"hidden_dims": [8, 8]}
        ),
        build_dataset_info=functools.partial(_latlon_dataset_info, LATLON_SHAPE),
        input_shape=(3, *LATLON_SHAPE),
        n_in=3,
        n_out=2,
    ),
    BuilderCase(
        name="SphericalFourierNeuralOperatorNet",
        build_selector=functools.partial(
            ModuleSelector,
            type="SphericalFourierNeuralOperatorNet",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
        ),
        build_dataset_info=functools.partial(_latlon_dataset_info, LATLON_SHAPE),
        input_shape=(3, *LATLON_SHAPE),
        n_in=3,
        n_out=2,
    ),
    BuilderCase(
        name="SFNO-v0.1.0",
        build_selector=functools.partial(
            ModuleSelector,
            type="SFNO-v0.1.0",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
        ),
        build_dataset_info=functools.partial(_latlon_dataset_info, LATLON_SHAPE),
        input_shape=(3, *LATLON_SHAPE),
        n_in=3,
        n_out=2,
    ),
    BuilderCase(
        name="Samudra",
        build_selector=functools.partial(
            ModuleSelector,
            type="Samudra",
            config={"ch_width": [8], "dilation": [1], "n_layers": [1]},
        ),
        build_dataset_info=functools.partial(_latlon_dataset_info, LATLON_SHAPE),
        input_shape=(3, *LATLON_SHAPE),
        n_in=3,
        n_out=2,
    ),
    BuilderCase(
        name="AnkurLocalNet",
        build_selector=functools.partial(
            ModuleSelector, type="AnkurLocalNet", config={"embed_dim": 8}
        ),
        build_dataset_info=functools.partial(_latlon_dataset_info, LATLON_SHAPE),
        input_shape=(3, *LATLON_SHAPE),
        n_in=3,
        n_out=2,
    ),
    BuilderCase(
        name="LocalNet",
        build_selector=functools.partial(
            ModuleSelector,
            type="LocalNet",
            conditional=True,
            config={
                "embed_dim": 8,
                "noise_embed_dim": 4,
                "noise_type": "isotropic",
                "block_types": ["disco", "conv1x1"],
                "label_embed_dim": 2,
            },
        ),
        build_dataset_info=functools.partial(
            _latlon_dataset_info, LATLON_SHAPE, LABELS
        ),
        input_shape=(3, *LATLON_SHAPE),
        n_in=3,
        n_out=2,
        conditional=True,
    ),
    BuilderCase(
        name="NoiseConditionedSFNO",
        build_selector=functools.partial(
            ModuleSelector,
            type="NoiseConditionedSFNO",
            conditional=True,
            config={
                "embed_dim": 4,
                "noise_embed_dim": 4,
                "noise_type": "isotropic",
                "filter_type": "linear",
                "filter_num_groups": 2,
                "context_pos_embed_dim": 2,
                "pos_embed": False,
                "num_layers": 2,
                "local_blocks": [0],
                "affine_norms": True,
                "label_embed_dim": 2,
            },
        ),
        build_dataset_info=functools.partial(
            _latlon_dataset_info, LATLON_SHAPE, LABELS
        ),
        input_shape=(3, *LATLON_SHAPE),
        n_in=3,
        n_out=2,
        conditional=True,
    ),
    BuilderCase(
        name="SwinTransformer",
        build_selector=functools.partial(
            ModuleSelector,
            type="SwinTransformer",
            conditional=True,
            config={
                "embed_dim": 16,
                "num_heads": [2, 2, 2, 2],
                "window_size": [4, 4],
                "mlp_ratio": 2.0,
                "drop_path_rate": 0.0,
            },
        ),
        build_dataset_info=functools.partial(
            _latlon_dataset_info, WINDOWED_SHAPE, LABELS
        ),
        input_shape=(3, *WINDOWED_SHAPE),
        n_in=3,
        n_out=2,
        conditional=True,
    ),
    BuilderCase(
        name="NoiseConditionedSwinTransformer",
        build_selector=functools.partial(
            ModuleSelector,
            type="NoiseConditionedSwinTransformer",
            conditional=True,
            config={
                "embed_dim": 16,
                "num_heads": [2, 2, 2, 2],
                "window_size": [4, 4],
                "mlp_ratio": 2.0,
                "drop_path_rate": 0.0,
                "noise_embed_dim": 8,
                "label_embed_dim": 2,
            },
        ),
        build_dataset_info=functools.partial(
            _latlon_dataset_info, WINDOWED_SHAPE, LABELS
        ),
        input_shape=(3, *WINDOWED_SHAPE),
        n_in=3,
        n_out=2,
        conditional=True,
    ),
    BuilderCase(
        name="HEALPixUNet",
        build_selector=functools.partial(
            ModuleSelector, type="HEALPixUNet", config=_healpix_unet_config()
        ),
        build_dataset_info=_healpix_dataset_info,
        input_shape=(12, 3, HEALPIX_NSIDE, HEALPIX_NSIDE),
        n_in=3,
        n_out=2,
    ),
    BuilderCase(
        name="FloeNet",
        build_selector=functools.partial(
            ModuleSelector,
            type="FloeNet",
            config={
                "latent_dimension": 8,
                "meshes": 6,
                "M0": 0,
                "processor_steps": 1,
            },
        ),
        build_dataset_info=functools.partial(_floenet_dataset_info, (9, 18)),
        input_shape=(2, 9, 18),
        n_in=2,
        n_out=2,
        expect_unsupported=True,
        marks=(
            pytest.mark.skipif(
                not GRAPHCAST_AVAIL, reason="trimesh/rtree are not available"
            ),
        ),
    ),
]


def _params(cases: list[BuilderCase]) -> list[Any]:
    return [pytest.param(case, id=case.name, marks=list(case.marks)) for case in cases]


COMPILABLE_CASES = _params(
    [case for case in BUILDER_CASES if not case.expect_unsupported]
)
UNSUPPORTED_CASES = _params([case for case in BUILDER_CASES if case.expect_unsupported])


def _build_module(case: BuilderCase) -> Module:
    return (
        case.build_selector()
        .build(
            n_in_channels=case.n_in,
            n_out_channels=case.n_out,
            dataset_info=case.build_dataset_info(),
        )
        .to(get_device())
    )


def _make_input(case: BuilderCase) -> torch.Tensor:
    return torch.randn(N_SAMPLES, *case.input_shape, device=get_device())


def _make_labels(case: BuilderCase) -> BatchLabels | None:
    if not case.conditional:
        return None
    return BatchLabels.new_from_set(LABELS, n_samples=N_SAMPLES, device=get_device())


def _seeded_forward(
    module: Module, x: torch.Tensor, labels: BatchLabels | None
) -> torch.Tensor:
    """Call ``module`` with its random draws taken from a fixed generator.

    The noise-conditioned builders draw their conditioning noise through
    ``fme.core.rand``; with a generator active the draws happen on the CPU
    outside inductor's own random stream, so compiled and eager calls see the
    same noise, as seeded inference does in production.
    """
    with use_generator(torch.Generator().manual_seed(0)):
        return module(x, labels)


def _relative_error(
    actual: torch.Tensor, expected: torch.Tensor, floor: float = 0.0
) -> float:
    scale = max(expected.norm().item(), floor, 1e-12)
    return (actual - expected).norm().item() / scale


def _total_norm(tensors: dict[str, torch.Tensor]) -> float:
    return torch.stack([tensor.norm() for tensor in tensors.values()]).norm().item()


def _grads(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


@contextlib.contextmanager
def _inductor_workarounds(case: BuilderCase) -> Iterator[None]:
    """Work around a torch inductor bug for the Swin builders.

    torch 2.10's inductor fuses pairs of reductions whose loop orders are
    swapped (``triton.mix_order_reduction``, on by default on CUDA). Scoring
    that fusion for the Swin backward trips an assertion inside the scheduler
    (``has_mix_reduction_orders``), so compilation fails before any kernel
    runs. Users who hit it can set ``TORCHINDUCTOR_MIX_ORDER_REDUCTION=0``.
    The option is read when the backward graph compiles on the first
    ``backward()`` call, so it is patched around forward and backward.
    """
    if case.name not in SWIN_BUILDERS or not hasattr(
        torch._inductor.config.triton, "mix_order_reduction"
    ):
        yield
        return
    with torch._inductor.config.patch({"triton.mix_order_reduction": False}):
        yield


def test_builder_cases_cover_every_registered_builder():
    """Every registered builder must declare whether its networks compile.

    A new builder shows up here as a missing case, forcing whoever adds it to
    either give it an inductor test or declare compilation unsupported.
    """
    covered = {case.name for case in BUILDER_CASES}
    registered = {
        name
        for name in ModuleSelector.get_available_types()
        if _is_production_builder(name)
    }
    assert covered == registered


@pytest.mark.parametrize("case", _params(BUILDER_CASES))
def test_builder_case_matches_registration(case: BuilderCase):
    """The case name is the registry key the coverage test relies on, and the
    case expects compilation unsupported exactly when the builder declared so
    at registration."""
    assert case.build_selector().type == case.name
    reason = ModuleSelector.registry.get_compile_unsupported_reason(case.name)
    assert case.expect_unsupported == (reason is not None)


@pytest.mark.slow
@pytest.mark.timeout(INDUCTOR_TIMEOUT_SECONDS)
@pytest.mark.parametrize("case", COMPILABLE_CASES)
def test_inductor_compiled_builder_matches_eager(case: BuilderCase):
    """A network compiled with inductor, as production compiles it, trains
    like the eager network: the same outputs and gradients within floating
    point tolerance, with gradients reaching the same parameters.

    Some builders own parameters that no forward pass reaches (e.g. embeddings
    for a conditioning path this configuration does not use), so the contract
    is that compilation does not change which parameters get gradients.
    """
    skip_without_inductor()
    if (
        case.name in SEGFAULTS_ON_MACOS_CPU
        and sys.platform == "darwin"
        and get_device().type == "cpu"
    ):
        pytest.skip(f"{case.name} inductor backward segfaults on macOS CPU")
    torch.manual_seed(0)
    module = _build_module(case)
    module.torch_module.train()
    with compile_enabled(True), compile_backend("inductor"):
        compiled = module.compile()
    assert compiled.is_compiled
    x = _make_input(case)
    labels = _make_labels(case)

    eager_output = _seeded_forward(module, x, labels)
    eager_output.square().sum().backward()
    eager_grads = _grads(module.torch_module)
    module.torch_module.zero_grad(set_to_none=True)

    with _inductor_workarounds(case):
        compiled_output = _seeded_forward(compiled, x, labels)
        compiled_output.square().sum().backward()
    compiled_grads = _grads(module.torch_module)

    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
    assert compiled_output.shape == eager_output.shape
    assert torch.isfinite(compiled_output).all()
    assert _relative_error(compiled_output, eager_output) < OUTPUT_TOLERANCE
    assert compiled_grads.keys() == eager_grads.keys()
    floor = GRAD_NORM_FLOOR * _total_norm(eager_grads)
    for name, grad in compiled_grads.items():
        assert torch.isfinite(grad).all(), name
        error = _relative_error(grad, eager_grads[name], floor)
        assert error < GRAD_TOLERANCE, (name, error)


@pytest.mark.parametrize("case", UNSUPPORTED_CASES)
def test_compile_unsupported_builder_runs_eagerly_without_tracing(
    case: BuilderCase, caplog: pytest.LogCaptureFixture
):
    """A builder that declares compilation unsupported is left uncompiled.

    Letting dynamo attempt the trace instead would burn minutes before falling
    back to eager, so the reason is checked before ``torch.compile`` is called
    at all, and logged so the eager run is not a surprise.
    """
    module = _build_module(case)
    with (
        compile_enabled(True),
        caplog.at_level(logging.INFO),
        mock.patch("torch.compile") as mock_compile,
    ):
        compiled = module.compile()
    mock_compile.assert_not_called()
    assert not compiled.is_compiled
    assert "torch.compile skipped" in caplog.text

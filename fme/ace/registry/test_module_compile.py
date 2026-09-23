"""Cross-builder ``torch.compile`` coverage for every registered module builder.

Every production builder in the registry gets one entry in
:data:`BUILDER_CASES`, so adding a builder without deciding whether its
networks can be compiled fails
:func:`test_builder_cases_cover_every_registered_builder`.

The equivalence and training tests run through the ``aot_eager`` backend:
it exercises the full dynamo tracing and AOTAutograd path (so tracing errors
and graph breaks still surface) without paying inductor's per-network codegen
cost, which for the transformer builders is tens of seconds. The two tests
that must cover the backend production actually uses (inductor) are kept to a
single tiny network (:func:`test_inductor_compiled_mlp_matches_eager`) and a
single rollout (:func:`test_compiled_rollout_matches_eager`).

Per-builder graph-break counts are deliberately not asserted here: for whole
networks they are a property of the torch version as much as of our code, so
they are tabulated in the pull request instead. The specific breaks removed
for traceability (``irfft``, ``NullTimer.child``, ``CappedGELU``, Samudra) are
pinned down by zero-break tests next to each fix, since each of those targets
one construct we control.
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Any
from unittest import mock

import pytest
import torch
from torch import nn

from fme.ace.models.graphcast import GRAPHCAST_AVAIL
from fme.ace.models.healpix.healpix_blocks import (
    AvgPoolDownsamplingBlockConfig,
    BasicConvBlockConfig,
    ConvNeXtBlockConfig,
    TransposedConvUpsampleBlockConfig,
)
from fme.ace.registry.hpx import UNetDecoderConfig, UNetEncoderConfig
from fme.core.coordinates import HEALPixCoordinates, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.labels import BatchLabels
from fme.core.registry.module import Module, ModuleSelector
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.step.args import StepArgs
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepABC, StepSelector
from fme.core.testing import (
    dynamo_hygiene,  # noqa: F401  autouse in this module
    get_dataset_info,
    trivial_network_and_loss_normalization,
)

LATLON_SHAPE = (8, 16)
"""Image shape for the builders that take a regular lat/lon grid."""

WINDOWED_SHAPE = (16, 32)
"""Image shape for the Swin builders, whose window/patch sizes need more room."""

HEALPIX_NSIDE = 8
"""Face height and width of the HEALPix test grid."""

LABELS = {"label_a", "label_b"}
"""Batch labels used by the conditional builders."""

N_SAMPLES = 2

TOLERANCE = dict(atol=1e-4, rtol=1e-4)
"""Single-forward tolerance between the compiled and eager networks.

Compilation may reassociate and fuse floating point operations, so bitwise
equality is not guaranteed; this is loose enough for that and far tighter than
any difference that would change a prediction.
"""

TEST_ONLY_BUILDERS = frozenset(
    {"mock", "mock_with_default", "mock_with_deprecation", "mock_compile_unsupported"}
)
"""Builders registered by test modules, which have no production networks."""


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
            unsupported, so ``Module.compile()`` must raise instead of tracing.
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


def _make_input(case: BuilderCase, n_samples: int = N_SAMPLES) -> torch.Tensor:
    return torch.randn(n_samples, *case.input_shape, device=get_device())


def _make_labels(case: BuilderCase, n_samples: int = N_SAMPLES) -> BatchLabels | None:
    if not case.conditional:
        return None
    return BatchLabels.new_from_set(LABELS, n_samples=n_samples, device=get_device())


def _seeded_forward(
    module: Module, x: torch.Tensor, labels: BatchLabels | None
) -> torch.Tensor:
    """Call ``module`` from a fixed global RNG state.

    The noise-conditioned builders draw their conditioning noise from the
    global torch RNG (see ``fme.core.rand``), so the compiled and eager calls
    only agree if each starts from the same seed.
    """
    torch.manual_seed(0)
    return module(x, labels)


def test_builder_cases_cover_every_registered_builder():
    """Every registered builder must declare whether its networks compile.

    A new builder shows up here as a missing case, forcing whoever adds it to
    either give it a compile test or declare compilation unsupported.
    """
    covered = {case.name for case in BUILDER_CASES}
    registered = set(ModuleSelector.get_available_types()) - TEST_ONLY_BUILDERS
    assert covered == registered


@pytest.mark.parametrize("case", _params(BUILDER_CASES))
def test_builder_case_name_matches_selector_type(case: BuilderCase):
    """The case name is the registry key, which the coverage test relies on."""
    assert case.build_selector().type == case.name


@pytest.mark.slow
@pytest.mark.parametrize("case", COMPILABLE_CASES)
def test_compiled_module_matches_eager(case: BuilderCase):
    """Compiling a built network changes neither its outputs nor its state.

    Run in eval mode under ``no_grad``, which is how inference calls the
    network.
    """
    module = _build_module(case)
    module.torch_module.eval()
    compiled = module.compile(backend="aot_eager")
    x = _make_input(case)
    labels = _make_labels(case)

    with torch.no_grad():
        eager_output = _seeded_forward(module, x, labels)
        compiled_output = _seeded_forward(compiled, x, labels)

    torch.testing.assert_close(compiled_output, eager_output, **TOLERANCE)
    assert compiled.get_state().keys() == module.get_state().keys()
    assert compiled.torch_module is module.torch_module


def _sum_of_squares(output: torch.Tensor) -> torch.Tensor:
    return output.square().sum()


def _grads_missing(module: nn.Module) -> set[str]:
    return {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    }


@pytest.mark.slow
@pytest.mark.parametrize("case", COMPILABLE_CASES)
def test_compiled_module_backward_matches_eager(case: BuilderCase):
    """Training through the compiled network produces usable gradients.

    Some builders own parameters that no forward pass reaches (e.g. embeddings
    for a conditioning path this configuration does not use), so "every
    parameter has a gradient" is not the contract. The contract is that
    compilation does not *change* which parameters get gradients, and that the
    gradients it does produce are finite.
    """
    module = _build_module(case)
    module.torch_module.train()
    compiled = module.compile(backend="aot_eager")
    x = _make_input(case)
    labels = _make_labels(case)

    module.torch_module.zero_grad(set_to_none=True)
    _sum_of_squares(_seeded_forward(compiled, x, labels)).backward()
    compiled_missing = _grads_missing(module.torch_module)
    for name, parameter in module.torch_module.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name

    module.torch_module.zero_grad(set_to_none=True)
    _sum_of_squares(_seeded_forward(module, x, labels)).backward()
    assert compiled_missing == _grads_missing(module.torch_module)


@pytest.mark.slow
def test_inductor_compiled_mlp_matches_eager():
    """The default (inductor) backend compiles and runs.

    Everything else here uses ``aot_eager`` for speed, which never reaches
    inductor's codegen; this keeps one case honest about the backend
    production actually uses.
    """
    case = next(case for case in BUILDER_CASES if case.name == "MLP")
    module = _build_module(case)
    module.torch_module.eval()
    compiled = module.compile()
    x = _make_input(case)

    with torch.no_grad():
        eager_output = module(x)
        compiled_output = compiled(x)

    torch.testing.assert_close(compiled_output, eager_output, **TOLERANCE)


ROLLOUT_PROGNOSTIC_NAMES = ["prog_a", "prog_b"]
ROLLOUT_FORCING_NAME = "forcing"
ROLLOUT_STEPS = 10
ROLLOUT_GROWTH_BOUND = 2.0


@dataclasses.dataclass(frozen=True)
class RolloutCase:
    """A builder configuration to roll out autoregressively.

    Parameters:
        name: Identifier for the parametrized test.
        builder_type: Registered builder type.
        builder_config: Tiny configuration for that builder.
        img_shape: Image shape to roll out on.
        init_scale: Factor applied to every initial weight. Chosen per
            architecture so the residual update is large enough that the
            trajectory genuinely evolves, while staying inside the
            non-expansion precondition the tolerance argument rests on; the
            test asserts the precondition rather than trusting the number.
    """

    name: str
    builder_type: str
    builder_config: Mapping[str, Any]
    img_shape: tuple[int, int]
    init_scale: float


ROLLOUT_CASES = [
    pytest.param(
        RolloutCase(
            name="sfno",
            builder_type="SphericalFourierNeuralOperatorNet",
            builder_config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
            img_shape=LATLON_SHAPE,
            init_scale=4.0,
        ),
        id="sfno",
    ),
    pytest.param(
        RolloutCase(
            name="swin",
            builder_type="SwinTransformer",
            builder_config={
                "embed_dim": 16,
                "num_heads": [2, 2, 2, 2],
                "window_size": [4, 4],
                "mlp_ratio": 2.0,
                "drop_path_rate": 0.0,
            },
            img_shape=WINDOWED_SHAPE,
            init_scale=0.5,
        ),
        id="swin",
        marks=pytest.mark.skipif(
            get_device().type != "cuda",
            # inductor needs ~30 s to compile the Swin network on CPU, which
            # is too close to the suite's per-test timeout under xdist.
            reason="inductor compilation of Swin is too slow on CPU",
        ),
    ),
]


def _rollout_step_selector(case: RolloutCase, compile: bool) -> StepSelector:
    """A residual single-module step with two prognostics and one forcing."""
    all_names = [*ROLLOUT_PROGNOSTIC_NAMES, ROLLOUT_FORCING_NAME]
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type=case.builder_type, config=case.builder_config
                ),
                in_names=all_names,
                out_names=list(ROLLOUT_PROGNOSTIC_NAMES),
                normalization=trivial_network_and_loss_normalization(all_names),
                residual_prediction=True,
                compile=compile,
            )
        ),
    )


def _build_rollout_step(case: RolloutCase, compile: bool) -> StepABC:
    def scale_weights(modules: list[nn.Module]) -> None:
        with torch.no_grad():
            for module in modules:
                for parameter in module.parameters():
                    parameter.mul_(case.init_scale)

    torch.manual_seed(0)
    return _rollout_step_selector(case, compile).get_step(
        get_dataset_info(img_shape=case.img_shape, device=get_device()),
        scale_weights,
    )


def _rollout(
    step: StepABC,
    initial: dict[str, torch.Tensor],
    forcing: dict[str, torch.Tensor],
    n_steps: int,
) -> list[dict[str, torch.Tensor]]:
    """Feed each step's prognostic outputs back in as the next step's input."""
    state = dict(initial)
    trajectory = []
    with torch.no_grad():
        for _ in range(n_steps):
            output = step.step(
                StepArgs(input={**state, **forcing}, next_step_input_data=forcing)
            ).output
            state = {name: output[name] for name in ROLLOUT_PROGNOSTIC_NAMES}
            trajectory.append(state)
    return trajectory


def _max_abs(state: Mapping[str, torch.Tensor]) -> float:
    return max(tensor.abs().max().item() for tensor in state.values())


@pytest.mark.slow
@pytest.mark.parametrize("case", ROLLOUT_CASES)
def test_compiled_rollout_matches_eager(case: RolloutCase):
    """Compiled and eager rollouts stay together over a multi-step rollout.

    A single forward differs from eager only by floating point reassociation,
    but inference feeds each output back in, so the question this answers is
    whether that difference amplifies. The tolerance is the single-forward
    tolerance times the number of steps: the test first asserts that the
    eager trajectory is finite and non-expanding
    (``max|x_t| <= 2 max|x_0|``), under which the per-step perturbation can
    accumulate at most linearly. Measured drift on CPU is ~1e-6, so the bound
    is not tight here; it exists for GPU, where reduced-precision matmuls make
    the per-step difference far larger.

    Uses the default inductor backend, since ``aot_eager`` is bit-identical to
    eager for these builders and would make the test vacuous.
    """
    eager_step = _build_rollout_step(case, compile=False)
    compiled_step = _build_rollout_step(case, compile=True)
    # the two steps are built from the same seed, but load the state anyway so
    # the comparison cannot be confounded by a difference in initialization
    compiled_step.load_state(eager_step.get_state())
    eager_step.eval()
    compiled_step.eval()

    torch.manual_seed(1)
    initial = {
        name: torch.randn(N_SAMPLES, *case.img_shape, device=get_device())
        for name in ROLLOUT_PROGNOSTIC_NAMES
    }
    forcing = {
        ROLLOUT_FORCING_NAME: torch.randn(
            N_SAMPLES, *case.img_shape, device=get_device()
        )
    }

    eager_trajectory = _rollout(eager_step, initial, forcing, ROLLOUT_STEPS)
    initial_scale = _max_abs(initial)
    for index, state in enumerate(eager_trajectory):
        for name, tensor in state.items():
            assert torch.isfinite(tensor).all(), f"step {index}, {name}"
        assert _max_abs(state) <= ROLLOUT_GROWTH_BOUND * initial_scale, (
            f"eager rollout is expanding at step {index}: the drift tolerance "
            "assumes a non-expanding map"
        )

    compiled_trajectory = _rollout(compiled_step, initial, forcing, ROLLOUT_STEPS)
    for name in ROLLOUT_PROGNOSTIC_NAMES:
        torch.testing.assert_close(
            compiled_trajectory[-1][name],
            eager_trajectory[-1][name],
            atol=ROLLOUT_STEPS * TOLERANCE["atol"],
            rtol=ROLLOUT_STEPS * TOLERANCE["rtol"],
        )


@pytest.mark.parametrize("case", UNSUPPORTED_CASES)
def test_compile_unsupported_builder_raises_without_tracing(case: BuilderCase):
    """A builder that declares compilation unsupported must fail immediately.

    Letting dynamo attempt the trace instead would burn minutes before falling
    back to eager, so the reason is checked before ``torch.compile`` is called
    at all.
    """
    module = _build_module(case)
    with mock.patch("torch.compile") as mock_compile:
        with pytest.raises(NotImplementedError) as err:
            module.compile()
    mock_compile.assert_not_called()
    assert case.name in str(err.value)

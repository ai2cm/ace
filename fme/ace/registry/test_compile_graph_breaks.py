"""Every registered network traces into a single torch.compile graph.

Steps compile the networks they build from a ``ModuleSelector`` (see
``Module.compile``). A graph break does not change results, so numerics tests
keep passing, but it splits the compiled forward pass into pieces with Python
running in between. These tests trace each registered network the way a step
calls it and fail on any break.
"""

import dataclasses

import pytest
import torch

import fme
from fme.ace.models.healpix.healpix_blocks import (
    AvgPoolDownsamplingBlockConfig,
    BasicConvBlockConfig,
    ConvNeXtBlockConfig,
    TransposedConvUpsampleBlockConfig,
)
from fme.ace.registry.hpx import UNetDecoderConfig, UNetEncoderConfig
from fme.core.coordinates import (
    HEALPixCoordinates,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.dataset_info import DatasetInfo
from fme.core.registry import ModuleSelector

_LAT_LON_SHAPE = (16, 32)
_HEALPIX_NSIDE = 8
_N_IN, _N_OUT = 3, 2
_BATCH_SIZE = 2
_LABELS = {"a", "b"}


@dataclasses.dataclass
class _NetworkCase:
    selector: ModuleSelector
    healpix: bool = False

    def dataset_info(self) -> DatasetInfo:
        device = fme.get_device()
        if self.healpix:
            horizontal_coordinates: LatLonCoordinates | HEALPixCoordinates = (
                HEALPixCoordinates(
                    torch.arange(12),
                    torch.arange(_HEALPIX_NSIDE),
                    torch.arange(_HEALPIX_NSIDE),
                )
            )
        else:
            horizontal_coordinates = LatLonCoordinates(
                lat=torch.linspace(-80, 80, _LAT_LON_SHAPE[0], device=device),
                lon=torch.linspace(0, 360, _LAT_LON_SHAPE[1] + 1, device=device)[:-1],
            )
        return DatasetInfo(
            horizontal_coordinates=horizontal_coordinates,
            vertical_coordinate=HybridSigmaPressureCoordinate(
                ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
            ),
            all_labels=_LABELS if self.selector.conditional else None,
        )

    def input(self) -> torch.Tensor:
        if self.healpix:
            shape: tuple[int, ...] = (12, _N_IN, _HEALPIX_NSIDE, _HEALPIX_NSIDE)
        else:
            shape = (_N_IN, *_LAT_LON_SHAPE)
        return torch.randn(_BATCH_SIZE, *shape, device=fme.get_device())


def _healpix_unet_config() -> dict:
    return {
        "encoder": dataclasses.asdict(
            UNetEncoderConfig(
                conv_block=ConvNeXtBlockConfig(),
                down_sampling_block=AvgPoolDownsamplingBlockConfig(pooling=2),
                n_channels=[4, 8],
                n_layers=[1, 1],
            )
        ),
        "decoder": dataclasses.asdict(
            UNetDecoderConfig(
                conv_block=ConvNeXtBlockConfig(),
                up_sampling_block=TransposedConvUpsampleBlockConfig(stride=2),
                output_layer=BasicConvBlockConfig(n_layers=1, kernel_size=1),
                n_channels=[8, 4],
                n_layers=[1, 1],
            )
        ),
    }


_SMALL_SFNO = {"embed_dim": 4, "num_layers": 2, "scale_factor": 1}
_SMALL_NOISE_CONDITIONED_SFNO = {
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
}
_SMALL_LOCAL_NET = {"embed_dim": 4, "block_types": ["disco", "conv1x1"]}
_SMALL_SWIN = {
    "embed_dim": 8,
    "num_heads": [2, 2, 2, 2],
    "window_size": [4, 4],
    "mlp_ratio": 2.0,
    "drop_path_rate": 0.0,
}
_SMALL_SAMUDRA = {"ch_width": [4, 4], "dilation": [1, 2], "n_layers": [1, 1]}


def _case(type: str, config: dict, conditional: bool = False, **kwargs) -> _NetworkCase:
    return _NetworkCase(
        ModuleSelector(type=type, config=config, conditional=conditional), **kwargs
    )


_NETWORK_CASES: dict[str, _NetworkCase] = {
    "MLP": _case("MLP", {"hidden_dim": 8}),
    "LandNet": _case("LandNet", {"hidden_dims": [8, 8]}),
    "SphericalFourierNeuralOperatorNet": _case(
        "SphericalFourierNeuralOperatorNet", _SMALL_SFNO
    ),
    "SFNO-v0.1.0": _case("SFNO-v0.1.0", _SMALL_SFNO),
    "NoiseConditionedSFNO": _case(
        "NoiseConditionedSFNO", _SMALL_NOISE_CONDITIONED_SFNO
    ),
    "NoiseConditionedSFNO-conditional": _case(
        "NoiseConditionedSFNO", _SMALL_NOISE_CONDITIONED_SFNO, conditional=True
    ),
    "AnkurLocalNet": _case("AnkurLocalNet", {"embed_dim": 4}),
    "LocalNet": _case("LocalNet", _SMALL_LOCAL_NET),
    "LocalNet-conditional": _case("LocalNet", _SMALL_LOCAL_NET, conditional=True),
    "SwinTransformer": _case("SwinTransformer", _SMALL_SWIN),
    "SwinTransformer-conditional": _case(
        "SwinTransformer", _SMALL_SWIN, conditional=True
    ),
    "NoiseConditionedSwinTransformer": _case(
        "NoiseConditionedSwinTransformer", {**_SMALL_SWIN, "noise_embed_dim": 4}
    ),
    "NoiseConditionedSwinTransformer-conditional": _case(
        "NoiseConditionedSwinTransformer",
        {**_SMALL_SWIN, "noise_embed_dim": 4},
        conditional=True,
    ),
    "Samudra": _case("Samudra", _SMALL_SAMUDRA),
    "Samudra-noise-conditioned": _case(
        "Samudra",
        {**_SMALL_SAMUDRA, "noise_embed_dim": 2, "conditioned_blocks": "bottleneck"},
    ),
    "HEALPixUNet": _case("HEALPixUNet", _healpix_unet_config(), healpix=True),
}

_UNCOVERED_NETWORK_TYPES = {
    "prebuilt": "returns a module supplied in code rather than a network of its own",
}
"""Registered builders compile supports but these tests do not trace, and why."""


def _is_production_network_type(type_name: str) -> bool:
    # test modules register mock builders into the same process-global
    # registry when they share a pytest worker with this one
    module_name = ModuleSelector.registry._types[type_name].__module__
    return not module_name.rsplit(".", 1)[-1].startswith("test_")


def test_network_cases_cover_every_registered_network_type():
    """Every registered builder is traced below, declares compilation
    unsupported at registration (so it runs eagerly), or is listed as
    uncovered, so a new builder cannot silently skip graph-break coverage."""
    covered = {case.selector.type for case in _NETWORK_CASES.values()}
    registered = {
        type_name
        for type_name in ModuleSelector.get_available_types()
        if _is_production_network_type(type_name)
    }
    compile_unsupported = {
        type_name
        for type_name in registered
        if ModuleSelector.registry.get_compile_unsupported_reason(type_name) is not None
    }
    assert covered.isdisjoint(compile_unsupported)
    assert covered | compile_unsupported | _UNCOVERED_NETWORK_TYPES.keys() == (
        registered
    )


def _explain_case(
    case: _NetworkCase, training: bool
) -> torch._dynamo.backends.debugging.ExplainOutput:
    torch.manual_seed(0)
    module = case.selector.build(
        n_in_channels=_N_IN, n_out_channels=_N_OUT, dataset_info=case.dataset_info()
    )
    # Module.compile hands torch_module to torch.compile, and Module.__call__
    # passes encoded labels as a keyword argument to conditional networks
    torch_module = module.torch_module.to(fme.get_device())
    torch_module.train(training)
    kwargs = {}
    if case.selector.conditional:
        kwargs["labels"] = torch.eye(len(_LABELS), device=fme.get_device())[
            torch.arange(_BATCH_SIZE) % len(_LABELS)
        ]
    with torch.set_grad_enabled(training):
        return torch._dynamo.explain(torch_module)(case.input(), **kwargs)


@pytest.mark.medium_duration
@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("case_name", _NETWORK_CASES.keys())
def test_network_has_no_graph_breaks(case_name: str, training: bool):
    """The network's forward pass traces into one graph, in training mode with
    gradients and in evaluation mode without (batch norm and dropout take
    different paths in each)."""
    explanation = _explain_case(_NETWORK_CASES[case_name], training)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    assert explanation.graph_count == 1

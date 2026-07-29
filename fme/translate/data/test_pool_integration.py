"""End-to-end test of the stream -> domain -> component pairing key.

The loader publishes ``dataset_info`` keyed by domain name; that mapping is
exactly what :meth:`ComponentPoolConfig.build` consumes to bind each component
to a grid. This exercises the whole chain on tiny synthetic multi-resolution
data: load, build the pool against the loader's dataset_info, and push one
loaded batch through a transform built that way.
"""

import dataclasses

import pytest
import torch

from fme.ace.stepper.single_module import StepperConfig
from fme.core.registry.module import ModuleSelector
from fme.core.step import SingleModuleStepConfig, StepSelector
from fme.core.testing import trivial_network_and_loss_normalization
from fme.translate.components import (
    BackboneConfig,
    ComponentPoolConfig,
    TransformConfig,
)
from fme.translate.data.getters import get_gridded_data
from fme.translate.data.test_data_loader import (
    NAMES,
    SHAPES,
    _multi_resolution_config,
    _requirements,
)
from fme.translate.domains import DomainConfig, LatentChannels
from fme.translate.modules import TransformSelector

N_LATENT = 4
LATENT_NAMES = [f"z_{i}" for i in range(N_LATENT)]


def _interpolate() -> TransformSelector:
    return TransformSelector(type="interpolate", config={})


def _latent_stepper() -> StepperConfig:
    return StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="SphericalFourierNeuralOperatorNet",
                        config={"scale_factor": 1, "embed_dim": 2, "num_layers": 1},
                    ),
                    in_names=LATENT_NAMES,
                    out_names=LATENT_NAMES,
                    normalization=trivial_network_and_loss_normalization(LATENT_NAMES),
                )
            ),
        ),
    )


def _pool_config() -> ComponentPoolConfig:
    """A 1°/2°/4° pool: per-resolution encoders into a 4°-grid latent."""
    return ComponentPoolConfig(
        domains={
            "atmos_1deg": DomainConfig(channels=list(NAMES)),
            "atmos_2deg": DomainConfig(channels=list(NAMES)),
            "atmos_4deg": DomainConfig(channels=list(NAMES)),
            "latent": DomainConfig(
                channels=[LatentChannels(name="z", channels=N_LATENT)],
                grid_like="atmos_4deg",
            ),
        },
        transforms={
            f"encoder_{resolution}": TransformConfig(
                module=_interpolate(),
                in_domain=f"atmos_{resolution}",
                out_domain="latent",
            )
            for resolution in ["1deg", "2deg", "4deg"]
        }
        | {
            "decoder_4deg": TransformConfig(
                module=_interpolate(), in_domain="latent", out_domain="atmos_4deg"
            )
        },
        backbones={
            "stepper": BackboneConfig(domain="latent", stepper=_latent_stepper())
        },
    )


@pytest.mark.medium_duration
def test_loader_dataset_info_builds_a_multi_resolution_pool(tmp_path):
    data = get_gridded_data(
        _multi_resolution_config(tmp_path),
        _requirements({name: NAMES for name in SHAPES}),
        train=True,
    )
    pool = _pool_config().build(data.dataset_info)

    # Every domain, including the latent one that inherits its grid, resolved to
    # the grid of the stream bound to it.
    for name, shape in SHAPES.items():
        domain = name.replace("era5_", "atmos_")
        assert pool.dataset_info[domain].img_shape == shape
    assert pool.dataset_info["latent"].img_shape == SHAPES["era5_4deg"]

    # A component built against the 1° domain's grid consumes a 1° batch and
    # lands on the latent (4°) grid.
    batch = next(iter(data.loader))["era5_1deg"]
    fields = torch.cat([batch.data[name][:, 0].unsqueeze(1) for name in NAMES], dim=1)
    assert fields.shape == (data.batch_size, len(NAMES), *SHAPES["era5_1deg"])
    latent = pool.transforms["encoder_1deg"](fields)
    assert latent.shape == (data.batch_size, N_LATENT, *SHAPES["era5_4deg"])


@pytest.mark.medium_duration
def test_pool_build_rejects_a_domain_no_stream_serves(tmp_path):
    """A domain the loader publishes nothing for is caught by the pool."""
    data = get_gridded_data(
        _multi_resolution_config(tmp_path),
        _requirements({"era5_1deg": NAMES, "era5_2deg": NAMES, "era5_4deg": NAMES}),
        train=True,
    )
    config = _pool_config()
    config.domains["atmos_8deg"] = DomainConfig(channels=list(NAMES))
    with pytest.raises(ValueError, match="Missing dataset_info"):
        config.build(data.dataset_info)

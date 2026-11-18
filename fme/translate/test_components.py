import dataclasses
from collections.abc import Mapping

import pytest
import torch

from fme.ace.stepper.parameter_init import (
    FrozenParameterConfig,
    ParameterClassification,
    ParameterInitializationConfig,
)
from fme.ace.stepper.single_module import (
    CheckpointStepperConfig,
    Stepper,
    StepperConfig,
)
from fme.core.distributed.non_distributed import DummyWrapper
from fme.core.registry.module import ModuleSelector
from fme.core.step import SingleModuleStepConfig, StepSelector
from fme.core.testing import get_dataset_info, trivial_network_and_loss_normalization
from fme.core.weight_ops import strip_leading_module
from fme.translate.components import (
    BackboneConfig,
    ComponentPool,
    ComponentPoolConfig,
    TransformConfig,
)
from fme.translate.domains import DomainConfig, LatentChannels
from fme.translate.modules import TransformSelector

IMG_SHAPE = (5, 5)


def _sfno_selector(embed_dim: int = 2) -> TransformSelector:
    return TransformSelector(
        type="same_grid",
        config={
            "module": {
                "type": "SphericalFourierNeuralOperatorNet",
                "config": {
                    "scale_factor": 1,
                    "embed_dim": embed_dim,
                    "num_layers": 1,
                },
            }
        },
    )


def _interpolate_selector() -> TransformSelector:
    return TransformSelector(type="interpolate", config={})


def _stepper_config(in_names: list[str], out_names: list[str]) -> StepperConfig:
    return StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="SphericalFourierNeuralOperatorNet",
                        config={
                            "scale_factor": 1,
                            "embed_dim": 2,
                            "num_layers": 1,
                        },
                    ),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=trivial_network_and_loss_normalization(
                        list(set(in_names) | set(out_names))
                    ),
                )
            ),
        ),
    )


def _all_frozen_init() -> ParameterInitializationConfig:
    return ParameterInitializationConfig(
        parameters=[
            ParameterClassification(frozen=FrozenParameterConfig(include=["*"]))
        ]
    )


def _save_stepper_checkpoint(stepper: Stepper, path) -> None:
    torch.save({"stepper": stepper.get_state()}, path)


def _encoder_decoder_config(
    parameter_init: ParameterInitializationConfig | None = None,
    with_backbone: bool = True,
) -> ComponentPoolConfig:
    """A pool with a physical domain, a same-grid latent domain (grid_like),
    an encoder/decoder pair between them, and optionally a latent backbone.
    """
    kwargs = {} if parameter_init is None else {"parameter_init": parameter_init}
    backbones = {}
    if with_backbone:
        backbones["backbone"] = BackboneConfig(
            domain="latent",
            stepper=_stepper_config(["z_0", "z_1"], ["z_0", "z_1"]),
        )
    return ComponentPoolConfig(
        domains={
            "physical": DomainConfig(channels=["a", "b"]),
            "latent": DomainConfig(
                channels=[LatentChannels(name="z", channels=2)],
                grid_like="physical",
            ),
        },
        transforms={
            "encoder": TransformConfig(
                _sfno_selector(), "physical", "latent", **kwargs
            ),
            "decoder": TransformConfig(_sfno_selector(), "latent", "physical"),
        },
        backbones=backbones,
    )


# ---------------------------------------------------------------------------
# Build + modules exposure
# ---------------------------------------------------------------------------


def test_pool_builds_and_exposes_flattened_modules():
    config = _encoder_decoder_config()
    pool = config.build({"physical": get_dataset_info(img_shape=IMG_SHAPE)})

    assert set(pool.transforms) == {"encoder", "decoder"}
    assert set(pool.backbones) == {"backbone"}
    assert set(pool.dataset_info) == {"physical", "latent"}

    modules = pool.modules
    assert isinstance(modules, torch.nn.ModuleList)
    # two transforms + one backbone module
    assert len(modules) == 3
    # every parameter of every component is reachable through .modules
    pool_param_count = sum(p.numel() for p in modules.parameters())
    component_param_count = sum(
        p.numel() for t in pool.transforms.values() for p in t.torch_module.parameters()
    ) + sum(
        p.numel()
        for b in pool.backbones.values()
        for m in b.modules
        for p in m.parameters()
    )
    assert pool_param_count == component_param_count
    assert pool_param_count > 0


def test_transform_channel_counts_come_from_domains():
    config = ComponentPoolConfig(
        domains={
            "physical": DomainConfig(channels=["a", "b", "c"]),
            "latent": DomainConfig(
                channels=[LatentChannels(name="z", channels=7)],
                grid_like="physical",
            ),
        },
        transforms={"encoder": TransformConfig(_sfno_selector(), "physical", "latent")},
    )
    pool = config.build({"physical": get_dataset_info(img_shape=IMG_SHAPE)})
    out = pool.transforms["encoder"](torch.zeros(1, 3, *IMG_SHAPE))
    assert out.shape == (1, 7, *IMG_SHAPE)


def test_multi_resolution_chain_builds_and_runs():
    """Jeremy's multi-resolution scenario at toy scale: three physical domains
    at descending resolution, resolution-changing transforms between adjacent
    resolutions, and a backbone stepping in the coarsest domain.
    """
    shapes = {
        "one_deg": (8, 16),
        "two_deg": (4, 8),
        "four_deg": (2, 4),
    }
    config = ComponentPoolConfig(
        domains={name: DomainConfig(channels=["a", "b"]) for name in shapes},
        transforms={
            "enc_1to2": TransformConfig(_interpolate_selector(), "one_deg", "two_deg"),
            "enc_2to4": TransformConfig(_interpolate_selector(), "two_deg", "four_deg"),
            "dec_4to2": TransformConfig(_interpolate_selector(), "four_deg", "two_deg"),
            "dec_2to1": TransformConfig(_interpolate_selector(), "two_deg", "one_deg"),
        },
        backbones={
            "backbone": BackboneConfig(
                domain="four_deg", stepper=_stepper_config(["a", "b"], ["a", "b"])
            )
        },
    )
    pool = config.build(
        {name: get_dataset_info(img_shape=shape) for name, shape in shapes.items()}
    )

    x = torch.randn(1, 2, *shapes["one_deg"])
    coarse = pool.transforms["enc_2to4"](pool.transforms["enc_1to2"](x))
    assert coarse.shape == (1, 2, *shapes["four_deg"])
    fine = pool.transforms["dec_2to1"](pool.transforms["dec_4to2"](coarse))
    assert fine.shape == x.shape


# ---------------------------------------------------------------------------
# Domain wiring validation
# ---------------------------------------------------------------------------


def test_unknown_transform_domain_rejected():
    with pytest.raises(ValueError, match="out_domain='nowhere'"):
        ComponentPoolConfig(
            domains={"physical": DomainConfig(channels=["a"])},
            transforms={
                "encoder": TransformConfig(_sfno_selector(), "physical", "nowhere")
            },
        )


def test_unknown_backbone_domain_rejected():
    with pytest.raises(ValueError, match="domain='nowhere'"):
        ComponentPoolConfig(
            domains={"physical": DomainConfig(channels=["a"])},
            backbones={
                "backbone": BackboneConfig(
                    domain="nowhere", stepper=_stepper_config(["a"], ["a"])
                )
            },
        )


def test_unknown_grid_like_rejected():
    with pytest.raises(ValueError, match="grid_like='nowhere'"):
        ComponentPoolConfig(
            domains={
                "latent": DomainConfig(
                    channels=[LatentChannels(name="z", channels=2)],
                    grid_like="nowhere",
                )
            },
        )


def test_chained_grid_like_rejected():
    with pytest.raises(ValueError, match="itself has grid_like"):
        ComponentPoolConfig(
            domains={
                "physical": DomainConfig(channels=["a"]),
                "latent": DomainConfig(
                    channels=[LatentChannels(name="z", channels=2)],
                    grid_like="physical",
                ),
                "latent2": DomainConfig(
                    channels=[LatentChannels(name="w", channels=2)],
                    grid_like="latent",
                ),
            },
        )


def test_missing_and_extra_dataset_info_rejected():
    config = _encoder_decoder_config()
    with pytest.raises(ValueError, match="Missing dataset_info.*physical"):
        config.build({})
    info = get_dataset_info(img_shape=IMG_SHAPE)
    with pytest.raises(ValueError, match="not data-backed domains.*latent"):
        config.build({"physical": info, "latent": info})


def test_backbone_variables_must_be_domain_channels():
    config = ComponentPoolConfig(
        domains={"physical": DomainConfig(channels=["a"])},
        backbones={
            "backbone": BackboneConfig(
                domain="physical", stepper=_stepper_config(["a", "b"], ["a"])
            )
        },
    )
    with pytest.raises(ValueError, match="not channels of that domain.*'b'"):
        config.build({"physical": get_dataset_info(img_shape=IMG_SHAPE)})


# ---------------------------------------------------------------------------
# Freeze-before-wrap correctness
# ---------------------------------------------------------------------------


def _assert_fully_frozen_dummy_wrapped(modules):
    assert len(modules) > 0
    for module in modules:
        assert isinstance(module, DummyWrapper)
        params = list(module.parameters())
        assert len(params) > 0
        assert all(not p.requires_grad for p in params)


def test_frozen_backbone_from_checkpoint_is_frozen_and_not_ddp(tmp_path):
    """A configured-frozen donor backbone sourced from a checkpoint has all
    parameters frozen and is wrapped as a DummyWrapper (no live-gradient DDP
    expectations), because freezing precedes wrapping.
    """
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    donor = _stepper_config(["a"], ["a"]).get_stepper(dataset_info)
    ckpt = tmp_path / "donor.pt"
    _save_stepper_checkpoint(donor, ckpt)

    backbone = BackboneConfig(
        domain="physical",
        checkpoint=CheckpointStepperConfig(checkpoint_path=str(ckpt)),
        init_from_checkpoint=True,
        parameter_init=_all_frozen_init(),
    ).build(dataset_info, DomainConfig(channels=["a"]))

    _assert_fully_frozen_dummy_wrapped(backbone.modules)

    # weights were actually sourced from the donor (name-matched)
    donor_state = strip_leading_module(donor.modules[0].state_dict())
    loaded_state = strip_leading_module(backbone.modules[0].state_dict())
    assert set(donor_state) == set(loaded_state)
    for name, value in donor_state.items():
        torch.testing.assert_close(loaded_state[name], value)


def test_fresh_backbone_freeze_before_wrap():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    backbone = BackboneConfig(
        domain="physical",
        stepper=_stepper_config(["a"], ["a"]),
        parameter_init=_all_frozen_init(),
    ).build(dataset_info, DomainConfig(channels=["a"]))
    _assert_fully_frozen_dummy_wrapped(backbone.modules)


def test_unfrozen_fresh_backbone_is_trainable():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    backbone = BackboneConfig(
        domain="physical", stepper=_stepper_config(["a"], ["a"])
    ).build(dataset_info, DomainConfig(channels=["a"]))
    assert any(p.requires_grad for p in backbone.modules.parameters())


def test_transform_freeze_before_wrap():
    config = _encoder_decoder_config(parameter_init=_all_frozen_init())
    pool = config.build({"physical": get_dataset_info(img_shape=IMG_SHAPE)})
    encoder = pool.transforms["encoder"]
    assert isinstance(encoder.torch_module, DummyWrapper)
    params = list(encoder.torch_module.parameters())
    assert len(params) > 0
    assert all(not p.requires_grad for p in params)
    # the decoder was not configured frozen
    assert any(
        p.requires_grad for p in pool.transforms["decoder"].torch_module.parameters()
    )


# ---------------------------------------------------------------------------
# Name-matched partial checkpoint init
# ---------------------------------------------------------------------------


def _transform_state(pool: ComponentPool, name: str) -> Mapping[str, torch.Tensor]:
    return strip_leading_module(pool.transforms[name].torch_module.state_dict())


def test_transform_name_matched_init(tmp_path):
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)

    # A "donor" pool whose encoder's raw (pre-wrap) state_dict we save.
    torch.manual_seed(0)
    donor_pool = _encoder_decoder_config(with_backbone=False).build(
        {"physical": dataset_info}
    )
    donor_state = _transform_state(donor_pool, "encoder")
    ckpt = tmp_path / "transform.pt"
    torch.save(donor_state, ckpt)

    # A fresh encoder of matching shape, initialized from the donor by name.
    torch.manual_seed(1)
    pool = _encoder_decoder_config(
        parameter_init=ParameterInitializationConfig(weights_path=str(ckpt)),
        with_backbone=False,
    ).build({"physical": dataset_info})

    loaded_state = _transform_state(pool, "encoder")
    assert set(loaded_state) == set(donor_state)
    for name, value in donor_state.items():
        torch.testing.assert_close(loaded_state[name], value)


def test_transform_partial_init_excludes_named_params(tmp_path):
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    torch.manual_seed(0)
    donor_pool = _encoder_decoder_config(with_backbone=False).build(
        {"physical": dataset_info}
    )
    donor_state = _transform_state(donor_pool, "encoder")
    ckpt = tmp_path / "transform.pt"
    torch.save(donor_state, ckpt)

    # exclude a parameter that differs between the donor and a fresh init, so
    # the assertions distinguish "kept fresh" from "overwritten from donor".
    torch.manual_seed(1)
    fresh_reference = _transform_state(
        _encoder_decoder_config(with_backbone=False).build({"physical": dataset_info}),
        "encoder",
    )
    excluded = next(
        name
        for name in sorted(donor_state)
        if not torch.allclose(fresh_reference[name], donor_state[name])
    )
    other = next(
        name
        for name in sorted(donor_state)
        if name != excluded
        and not torch.allclose(fresh_reference[name], donor_state[name])
    )

    torch.manual_seed(1)
    pool = _encoder_decoder_config(
        parameter_init=ParameterInitializationConfig(
            weights_path=str(ckpt),
            parameters=[ParameterClassification(exclude=[excluded])],
        ),
        with_backbone=False,
    ).build({"physical": dataset_info})

    loaded_state = _transform_state(pool, "encoder")
    # the excluded parameter kept its fresh initialization
    torch.testing.assert_close(loaded_state[excluded], fresh_reference[excluded])
    # a non-excluded parameter was overwritten from the donor
    torch.testing.assert_close(loaded_state[other], donor_state[other])


# ---------------------------------------------------------------------------
# State round-trip
# ---------------------------------------------------------------------------


def test_component_pool_state_round_trip():
    pool = _encoder_decoder_config().build(
        {"physical": get_dataset_info(img_shape=IMG_SHAPE)}
    )
    state = pool.get_state()

    reloaded = ComponentPool.from_state(state)
    assert set(reloaded.transforms) == set(pool.transforms)
    assert set(reloaded.backbones) == set(pool.backbones)
    assert set(reloaded.dataset_info) == set(pool.dataset_info)

    orig = strip_leading_module(pool.transforms["encoder"].torch_module.state_dict())
    new = strip_leading_module(reloaded.transforms["encoder"].torch_module.state_dict())
    assert set(orig) == set(new)
    for name, value in orig.items():
        torch.testing.assert_close(new[name], value)

    orig_b = strip_leading_module(pool.backbones["backbone"].modules[0].state_dict())
    new_b = strip_leading_module(reloaded.backbones["backbone"].modules[0].state_dict())
    for name, value in orig_b.items():
        torch.testing.assert_close(new_b[name], value)


def test_component_pool_load_state_into_built_pool():
    config = _encoder_decoder_config(with_backbone=False)
    dataset_info = {"physical": get_dataset_info(img_shape=IMG_SHAPE)}
    pool = config.build(dataset_info)
    state = pool.get_state()

    other = config.build(dataset_info)
    # perturb every parameter so the load is observable
    with torch.no_grad():
        for param in other.modules.parameters():
            param.add_(1.0)
    before = {
        name: value.clone()
        for name, value in strip_leading_module(
            other.transforms["encoder"].torch_module.state_dict()
        ).items()
    }
    other.load_state(state)
    after = strip_leading_module(other.transforms["encoder"].torch_module.state_dict())
    saved = strip_leading_module(pool.transforms["encoder"].torch_module.state_dict())
    for name, value in saved.items():
        torch.testing.assert_close(after[name], value)
    # sanity: at least one parameter actually changed on load
    assert any(not torch.allclose(before[name], after[name]) for name in before)


def test_config_dacite_round_trip():
    config = _encoder_decoder_config()
    reloaded = ComponentPoolConfig.from_state(config.get_state())
    assert set(reloaded.domains) == {"physical", "latent"}
    assert reloaded.domains["physical"].names == ["a", "b"]
    # the latent block deserializes back to a LatentChannels entry
    assert reloaded.domains["latent"].channels == [LatentChannels(name="z", channels=2)]
    assert reloaded.domains["latent"].grid_like == "physical"
    assert set(reloaded.transforms) == {"encoder", "decoder"}
    assert reloaded.transforms["encoder"].in_domain == "physical"
    assert reloaded.transforms["encoder"].out_domain == "latent"
    assert set(reloaded.backbones) == {"backbone"}
    assert reloaded.backbones["backbone"].domain == "latent"
    assert reloaded.backbones["backbone"].stepper is not None
    # the rebuilt config still builds
    reloaded.build({"physical": get_dataset_info(img_shape=IMG_SHAPE)})


def test_checkpoint_backbone_config_dacite_round_trip():
    """The frozen-donor arm's config (a checkpoint-sourced backbone) round-trips
    through get_state/from_state, preserving the checkpoint source and freeze.
    """
    config = ComponentPoolConfig(
        domains={"physical": DomainConfig(channels=["a"])},
        backbones={
            "backbone": BackboneConfig(
                domain="physical",
                checkpoint=CheckpointStepperConfig(checkpoint_path="donor.pt"),
                init_from_checkpoint=True,
                parameter_init=_all_frozen_init(),
            )
        },
    )
    reloaded = ComponentPoolConfig.from_state(config.get_state())
    backbone = reloaded.backbones["backbone"]
    assert backbone.stepper is None
    assert backbone.checkpoint is not None
    assert backbone.checkpoint.checkpoint_path == "donor.pt"
    assert backbone.init_from_checkpoint is True
    assert backbone.parameter_init.parameters[0].frozen.include == ["*"]


# ---------------------------------------------------------------------------
# train / eval / epoch fan-out
# ---------------------------------------------------------------------------


def test_set_train_eval_fans_out():
    pool = _encoder_decoder_config().build(
        {"physical": get_dataset_info(img_shape=IMG_SHAPE)}
    )

    pool.set_eval()
    assert not pool.transforms["encoder"].torch_module.training
    assert all(not m.training for m in pool.backbones["backbone"].modules)

    pool.set_train()
    assert pool.transforms["encoder"].torch_module.training
    assert all(m.training for m in pool.backbones["backbone"].modules)

    # set_epoch must not raise even with only transforms/backbones present
    pool.set_epoch(3)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_backbone_requires_exactly_one_source():
    with pytest.raises(ValueError, match="Exactly one"):
        BackboneConfig(domain="physical")
    with pytest.raises(ValueError, match="Exactly one"):
        BackboneConfig(
            domain="physical",
            stepper=_stepper_config(["a"], ["a"]),
            checkpoint=CheckpointStepperConfig(checkpoint_path="x"),
        )


def test_backbone_checkpoint_weights_path_conflict():
    with pytest.raises(ValueError, match="weights_path must be None"):
        BackboneConfig(
            domain="physical",
            checkpoint=CheckpointStepperConfig(checkpoint_path="x"),
            init_from_checkpoint=True,
            parameter_init=ParameterInitializationConfig(weights_path="y"),
        )


def test_duplicate_component_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        ComponentPoolConfig(
            domains={"physical": DomainConfig(channels=["a"])},
            transforms={
                "shared": TransformConfig(_sfno_selector(), "physical", "physical")
            },
            backbones={
                "shared": BackboneConfig(
                    domain="physical", stepper=_stepper_config(["a"], ["a"])
                )
            },
        )

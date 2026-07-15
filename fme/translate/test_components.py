import dataclasses

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

IMG_SHAPE = (5, 5)


def _sfno_selector(embed_dim: int = 2) -> ModuleSelector:
    return ModuleSelector(
        type="SphericalFourierNeuralOperatorNet",
        config={"scale_factor": 1, "embed_dim": embed_dim, "num_layers": 1},
    )


def _stepper_config(in_names: list[str], out_names: list[str]) -> StepperConfig:
    return StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=_sfno_selector(),
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


# ---------------------------------------------------------------------------
# Build + modules exposure
# ---------------------------------------------------------------------------


def test_pool_builds_and_exposes_flattened_modules():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    config = ComponentPoolConfig(
        transforms={
            "encoder": TransformConfig(_sfno_selector(), ["a", "b"], ["z"]),
            "decoder": TransformConfig(_sfno_selector(), ["z"], ["a", "b"]),
        },
        backbones={"backbone": BackboneConfig(stepper=_stepper_config(["z"], ["z"]))},
    )
    pool = config.build(dataset_info)

    assert set(pool.transforms) == {"encoder", "decoder"}
    assert set(pool.backbones) == {"backbone"}

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


def test_transform_channel_counts_from_names():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    transform = TransformConfig(_sfno_selector(), ["a", "b", "c"], ["x"]).build(
        dataset_info
    )
    grid = torch.zeros(1, 3, *IMG_SHAPE)
    out = transform(grid)
    assert out.shape == (1, 1, *IMG_SHAPE)


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
        checkpoint=CheckpointStepperConfig(checkpoint_path=str(ckpt)),
        init_from_checkpoint=True,
        parameter_init=_all_frozen_init(),
    ).build(dataset_info)

    _assert_fully_frozen_dummy_wrapped(backbone.modules)

    # weights were actually sourced from the donor (name-matched)
    donor_state = strip_leading_module(donor.modules[0].state_dict())
    loaded_state = strip_leading_module(backbone.modules[0].state_dict())
    assert set(donor_state) == set(loaded_state)
    for name, value in donor_state.items():
        torch.testing.assert_close(loaded_state[name], value)


def test_fresh_backbone_freeze_before_wrap(tmp_path):
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    backbone = BackboneConfig(
        stepper=_stepper_config(["a"], ["a"]),
        parameter_init=_all_frozen_init(),
    ).build(dataset_info)
    _assert_fully_frozen_dummy_wrapped(backbone.modules)


def test_unfrozen_fresh_backbone_is_trainable():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    backbone = BackboneConfig(stepper=_stepper_config(["a"], ["a"])).build(dataset_info)
    assert any(p.requires_grad for p in backbone.modules.parameters())


def test_transform_freeze_before_wrap():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    transform = TransformConfig(
        _sfno_selector(),
        ["a"],
        ["a"],
        parameter_init=_all_frozen_init(),
    ).build(dataset_info)
    assert isinstance(transform.torch_module, DummyWrapper)
    params = list(transform.torch_module.parameters())
    assert len(params) > 0
    assert all(not p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# Name-matched partial checkpoint init
# ---------------------------------------------------------------------------


def test_transform_name_matched_init(tmp_path):
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    selector = _sfno_selector()

    # A "donor" transform whose raw (pre-wrap) state_dict we save.
    torch.manual_seed(0)
    donor_raw = selector.build(
        n_in_channels=1, n_out_channels=1, dataset_info=dataset_info
    )
    donor_state = donor_raw.torch_module.state_dict()
    ckpt = tmp_path / "transform.pt"
    torch.save(donor_state, ckpt)

    # A fresh transform of matching shape, initialized from the donor by name.
    torch.manual_seed(1)
    transform = TransformConfig(
        selector,
        ["a"],
        ["a"],
        parameter_init=ParameterInitializationConfig(weights_path=str(ckpt)),
    ).build(dataset_info)

    loaded_state = strip_leading_module(transform.torch_module.state_dict())
    assert set(loaded_state) == set(donor_state)
    for name, value in donor_state.items():
        torch.testing.assert_close(loaded_state[name], value)


def test_transform_partial_init_excludes_named_params(tmp_path):
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    selector = _sfno_selector()
    torch.manual_seed(0)
    donor_raw = selector.build(
        n_in_channels=1, n_out_channels=1, dataset_info=dataset_info
    )
    donor_state = donor_raw.torch_module.state_dict()
    ckpt = tmp_path / "transform.pt"
    torch.save(donor_state, ckpt)

    # exclude a parameter that differs between the donor and a fresh init, so
    # the assertions distinguish "kept fresh" from "overwritten from donor".
    torch.manual_seed(1)
    fresh_reference = strip_leading_module(
        TransformConfig(selector, ["a"], ["a"])
        .build(dataset_info)
        .torch_module.state_dict()
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
    transform = TransformConfig(
        selector,
        ["a"],
        ["a"],
        parameter_init=ParameterInitializationConfig(
            weights_path=str(ckpt),
            parameters=[ParameterClassification(exclude=[excluded])],
        ),
    ).build(dataset_info)

    loaded_state = strip_leading_module(transform.torch_module.state_dict())
    # the excluded parameter kept its fresh initialization
    torch.testing.assert_close(loaded_state[excluded], fresh_reference[excluded])
    # a non-excluded parameter was overwritten from the donor
    torch.testing.assert_close(loaded_state[other], donor_state[other])


# ---------------------------------------------------------------------------
# State round-trip
# ---------------------------------------------------------------------------


def test_component_pool_state_round_trip():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    config = ComponentPoolConfig(
        transforms={"encoder": TransformConfig(_sfno_selector(), ["a"], ["z"])},
        backbones={"backbone": BackboneConfig(stepper=_stepper_config(["z"], ["z"]))},
    )
    pool = config.build(dataset_info)
    state = pool.get_state()

    reloaded = ComponentPool.from_state(state)
    assert set(reloaded.transforms) == set(pool.transforms)
    assert set(reloaded.backbones) == set(pool.backbones)

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
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    config = ComponentPoolConfig(
        transforms={"encoder": TransformConfig(_sfno_selector(), ["a"], ["z"])},
    )
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
    config = ComponentPoolConfig(
        transforms={"encoder": TransformConfig(_sfno_selector(), ["a", "b"], ["z"])},
        backbones={"backbone": BackboneConfig(stepper=_stepper_config(["z"], ["z"]))},
    )
    reloaded = ComponentPoolConfig.from_state(config.get_state())
    assert set(reloaded.transforms) == {"encoder"}
    assert reloaded.transforms["encoder"].in_names == ["a", "b"]
    assert reloaded.transforms["encoder"].out_names == ["z"]
    assert set(reloaded.backbones) == {"backbone"}
    assert reloaded.backbones["backbone"].stepper is not None
    # the rebuilt config still builds
    reloaded.build(get_dataset_info(img_shape=IMG_SHAPE))


# ---------------------------------------------------------------------------
# train / eval / epoch fan-out
# ---------------------------------------------------------------------------


def test_set_train_eval_fans_out():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE)
    config = ComponentPoolConfig(
        transforms={"encoder": TransformConfig(_sfno_selector(), ["a"], ["z"])},
        backbones={"backbone": BackboneConfig(stepper=_stepper_config(["z"], ["z"]))},
    )
    pool = config.build(dataset_info)

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
        BackboneConfig()
    with pytest.raises(ValueError, match="Exactly one"):
        BackboneConfig(
            stepper=_stepper_config(["a"], ["a"]),
            checkpoint=CheckpointStepperConfig(checkpoint_path="x"),
        )


def test_backbone_checkpoint_weights_path_conflict():
    with pytest.raises(ValueError, match="weights_path must be None"):
        BackboneConfig(
            checkpoint=CheckpointStepperConfig(checkpoint_path="x"),
            init_from_checkpoint=True,
            parameter_init=ParameterInitializationConfig(weights_path="y"),
        )


def test_duplicate_component_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        ComponentPoolConfig(
            transforms={"shared": TransformConfig(_sfno_selector(), ["a"], ["z"])},
            backbones={"shared": BackboneConfig(stepper=_stepper_config(["z"], ["z"]))},
        )

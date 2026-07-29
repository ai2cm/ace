import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import nn

from fme.ace.stepper.parameter_init import (
    FrozenParameterConfig,
    ParameterClassification,
    ParameterInitializationConfig,
)
from fme.ace.stepper.single_module import Stepper, StepperConfig
from fme.core.dataset_info import DatasetInfo
from fme.core.registry.module import Module, ModuleSelector
from fme.core.step import SingleModuleStepConfig, StepSelector
from fme.core.testing import get_dataset_info, trivial_network_and_loss_normalization
from fme.core.weight_ops import strip_leading_module
from fme.translate.components import ComponentPool, ComponentPoolConfig, TransformConfig
from fme.translate.cutpoint import SFNOCutPointConfig
from fme.translate.domains import DomainConfig, LatentChannels
from fme.translate.modules import TransformSelector

IMG_SHAPE = (8, 16)
EMBED_DIM = 4
DONOR_NAMES = ["a", "b"]
PARTS = ("encoder", "processor", "decoder")


def _sfno_config(**overrides: Any) -> dict[str, Any]:
    """A tiny noise-conditioned SFNO configuration, shared by donor and parts."""
    return {
        "embed_dim": EMBED_DIM,
        "num_layers": 2,
        "noise_embed_dim": 3,
        **overrides,
    }


def _latent_channels(sfno: Mapping[str, Any]) -> int:
    """The cut-point width: the latent, plus the big-skip residual."""
    if sfno.get("big_skip", True):
        return EMBED_DIM + len(DONOR_NAMES)
    return EMBED_DIM


def _dataset_info() -> DatasetInfo:
    return get_dataset_info(img_shape=IMG_SHAPE)


def _donor_stepper(sfno: Mapping[str, Any]) -> Stepper:
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="NoiseConditionedSFNO", config=dict(sfno)
                    ),
                    in_names=list(DONOR_NAMES),
                    out_names=list(DONOR_NAMES),
                    normalization=trivial_network_and_loss_normalization(DONOR_NAMES),
                )
            ),
        ),
    )
    return config.get_stepper(dataset_info=_dataset_info())


def _donor_checkpoint(tmp_path, sfno: Mapping[str, Any]) -> tuple[str, nn.Module]:
    """Save a donor stepper checkpoint; return its path and monolithic module."""
    stepper = _donor_stepper(sfno)
    path = str(tmp_path / "donor.ckpt")
    torch.save({"stepper": stepper.get_state()}, path)
    return path, stepper.modules[0]


def _part_config(
    part: str, sfno: Mapping[str, Any], **overrides: Any
) -> TransformSelector:
    return TransformSelector(
        type="sfno_cut_point",
        config={"part": part, "sfno": dict(sfno), **overrides},
    )


def _build_part(
    part: str,
    sfno: Mapping[str, Any],
    donor_checkpoint: str | None = None,
    for_load: bool = False,
) -> Module:
    latent = _latent_channels(sfno)
    n_in, n_out = {
        "encoder": (len(DONOR_NAMES), latent),
        "processor": (latent, latent),
        "decoder": (latent, len(DONOR_NAMES)),
    }[part]
    selector = _part_config(part, sfno, donor_checkpoint=donor_checkpoint)
    build = selector.build_for_load if for_load else selector.build
    info = _dataset_info()
    return build(
        n_in_channels=n_in,
        n_out_channels=n_out,
        in_dataset_info=info,
        out_dataset_info=info,
    )


def _build_parts(
    sfno: Mapping[str, Any], donor_checkpoint: str | None = None
) -> dict[str, Module]:
    return {part: _build_part(part, sfno, donor_checkpoint) for part in PARTS}


def _compose(parts: Mapping[str, Module], x: torch.Tensor) -> torch.Tensor:
    return parts["decoder"](parts["processor"](parts["encoder"](x)))


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"normalize_big_skip": True}, id="normalized_big_skip"),
        pytest.param(
            {"normalize_big_skip": True, "affine_norms": True}, id="affine_norms"
        ),
        pytest.param({"big_skip": False}, id="no_big_skip"),
        pytest.param(
            {"big_skip": False, "normalize_big_skip": True}, id="no_big_skip_normalized"
        ),
        pytest.param({"filter_output": True}, id="filter_output"),
        pytest.param(
            {"filter_residual": True, "normalize_big_skip": True}, id="filter_residual"
        ),
        pytest.param({"pos_embed": False}, id="no_pos_embed"),
        pytest.param({"encoder_layers": 2}, id="encoder_layers"),
        pytest.param({"noise_type": "isotropic"}, id="isotropic_noise"),
        pytest.param(
            {"global_layer_norm": True, "normalize_big_skip": True},
            id="global_layer_norm",
        ),
        pytest.param(
            {"context_pos_embed_dim": 2, "pos_embed": False}, id="context_pos_embed"
        ),
        # The parts mirror the monolith's checkpointing levels (>=1 wraps the
        # encoder and decoder stacks, >=3 each block), so each level is a
        # separate path through the mirrored forward.
        pytest.param({"checkpointing": 1}, id="checkpointing_encoder_decoder"),
        pytest.param({"checkpointing": 3}, id="checkpointing_blocks"),
    ],
)
def test_composed_parts_reproduce_the_donor(tmp_path, overrides):
    """encoder -> processor -> decoder is the monolithic net, to the bit.

    This is the property that makes the decomposition a decomposition: it holds
    only because every context-conditioned operation (the blocks and, when
    configured, the big-skip normalization) sits in the processor, so the
    composed path draws noise exactly once, as the monolith does.
    """
    sfno = _sfno_config(**overrides)
    checkpoint, donor = _donor_checkpoint(tmp_path, sfno)
    parts = _build_parts(sfno, donor_checkpoint=checkpoint)
    x = torch.randn(3, len(DONOR_NAMES), *IMG_SHAPE)

    torch.manual_seed(0)
    expected = donor(x)
    torch.manual_seed(0)
    result = _compose(parts, x)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_composed_parts_accept_an_ensemble_dimension(tmp_path):
    """The parts flatten leading dimensions as ``NoiseConditionedModel`` does."""
    sfno = _sfno_config()
    checkpoint, donor = _donor_checkpoint(tmp_path, sfno)
    parts = _build_parts(sfno, donor_checkpoint=checkpoint)
    x = torch.randn(3, 2, len(DONOR_NAMES), *IMG_SHAPE)

    torch.manual_seed(0)
    expected = donor(x)
    torch.manual_seed(0)
    result = _compose(parts, x)

    assert result.shape == (6, len(DONOR_NAMES), *IMG_SHAPE)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_parts_partition_the_donor_parameters(tmp_path):
    """Every donor parameter lands in exactly one part."""
    sfno = _sfno_config(normalize_big_skip=True)
    _, donor = _donor_checkpoint(tmp_path, sfno)
    donor_names = {name.removeprefix("module.") for name, _ in donor.named_parameters()}
    part_names = {
        part: {name for name, _ in module.torch_module.named_parameters()}
        for part, module in _build_parts(sfno).items()
    }

    assert set().union(*part_names.values()) == donor_names
    for first, second in [
        ("encoder", "processor"),
        ("encoder", "decoder"),
        ("processor", "decoder"),
    ]:
        assert not part_names[first] & part_names[second]


@pytest.mark.parametrize("part", PARTS)
def test_donor_checkpoint_initializes_the_part(tmp_path, part):
    sfno = _sfno_config(normalize_big_skip=True)
    checkpoint, donor = _donor_checkpoint(tmp_path, sfno)
    donor_state = strip_leading_module(donor.state_dict())

    initialized = _build_part(part, sfno, donor_checkpoint=checkpoint)
    for name, parameter in initialized.torch_module.named_parameters():
        torch.testing.assert_close(parameter, donor_state[name], rtol=0, atol=0)

    # Without the donor the part is randomly initialized, so the check above
    # is not vacuous.
    fresh = _build_part(part, sfno)
    assert any(
        not torch.equal(parameter, donor_state[name])
        for name, parameter in fresh.torch_module.named_parameters()
    )


def test_build_for_load_ignores_the_donor_checkpoint(tmp_path):
    """A saved component reloads without its donor checkpoint still existing."""
    sfno = _sfno_config()
    checkpoint, _ = _donor_checkpoint(tmp_path, sfno)
    (tmp_path / "donor.ckpt").unlink()

    with pytest.raises(FileNotFoundError):
        _build_part("processor", sfno, donor_checkpoint=checkpoint)
    _build_part("processor", sfno, donor_checkpoint=checkpoint, for_load=True)


def test_donor_missing_part_weights_raises(tmp_path):
    """A part configured differently from the donor is not silently accepted."""
    checkpoint, _ = _donor_checkpoint(tmp_path, _sfno_config(num_layers=2))

    with pytest.raises(ValueError, match="no weights for"):
        _build_part("processor", _sfno_config(num_layers=3), checkpoint)


@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_donor_shape_mismatch_raises(tmp_path, part):
    """A too-wide cut-point is not silently loaded a leading slice at a time.

    ``overwrite_weights`` copies the initial slice when the destination axis is
    longer, so without a shape check a cut-point domain declaring more than
    ``embed_dim`` plus the donor's input channels would build a part that is
    only partly the donor's, with no error.
    """
    sfno = _sfno_config()
    checkpoint, _ = _donor_checkpoint(tmp_path, sfno)
    # One channel more than the donor has, declared self-consistently so that
    # _validate_channels passes and only the donor's shapes disagree.
    wide_physical = len(DONOR_NAMES) + 1
    n_in, n_out = {
        "encoder": (wide_physical, EMBED_DIM + wide_physical),
        "decoder": (EMBED_DIM + wide_physical, len(DONOR_NAMES)),
    }[part]
    info = _dataset_info()

    with pytest.raises(ValueError, match="differently-shaped weights"):
        _part_config(part, sfno, donor_checkpoint=checkpoint).build(
            n_in_channels=n_in,
            n_out_channels=n_out,
            in_dataset_info=info,
            out_dataset_info=info,
        )


@pytest.mark.parametrize(
    "part, n_in_channels, n_out_channels, sfno_overrides, match",
    [
        pytest.param(
            "encoder",
            2,
            EMBED_DIM + 3,
            {},
            "plus its own input channels",
            id="encoder_out",
        ),
        pytest.param(
            "processor",
            EMBED_DIM + 2,
            EMBED_DIM + 3,
            {},
            "same number of channels",
            id="processor_asymmetric",
        ),
        pytest.param(
            "decoder",
            EMBED_DIM,
            2,
            {},
            "plus the donor's input channels",
            id="decoder_missing_skip",
        ),
        pytest.param(
            "decoder",
            EMBED_DIM + 2,
            2,
            {"big_skip": False},
            "exactly embed_dim",
            id="decoder_unexpected_skip",
        ),
    ],
)
def test_channel_mismatch_raises(
    part, n_in_channels, n_out_channels, sfno_overrides, match
):
    info = _dataset_info()
    with pytest.raises(ValueError, match=match):
        _part_config(part, _sfno_config(**sfno_overrides)).build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            in_dataset_info=info,
            out_dataset_info=info,
        )


def test_resolution_change_raises():
    """Cut-point parts stay on one grid; resizing is a separate transform."""
    with pytest.raises(ValueError, match="does not change resolution"):
        _part_config("encoder", _sfno_config()).build(
            n_in_channels=len(DONOR_NAMES),
            n_out_channels=EMBED_DIM + len(DONOR_NAMES),
            in_dataset_info=get_dataset_info(img_shape=IMG_SHAPE),
            out_dataset_info=get_dataset_info(img_shape=(4, 8)),
        )


@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_conditional_rejected_for_context_free_parts(part):
    with pytest.raises(ValueError, match="conditional=True is not meaningful"):
        _part_config(part, _sfno_config(), conditional=True)


def test_unknown_part_raises():
    with pytest.raises(ValueError, match="Unknown sfno_cut_point part"):
        SFNOCutPointConfig(part="processer")  # type: ignore[arg-type]


def test_latent_global_mean_clip_matches_the_donor(tmp_path):
    """The clip envelope tracks and applies as it does in the monolith."""
    sfno = _sfno_config(clip_latent_global_means=True)
    checkpoint, donor = _donor_checkpoint(tmp_path, sfno)
    parts = _build_parts(sfno, donor_checkpoint=checkpoint)
    modules = [donor, *(part.torch_module for part in parts.values())]

    training = torch.randn(4, len(DONOR_NAMES), *IMG_SHAPE)
    for module in modules:
        module.train()
    torch.manual_seed(0)
    donor(training)
    torch.manual_seed(0)
    _compose(parts, training)

    # The same decomposition with clipping off, to show the clip is not a no-op
    # for an input far outside the envelope the training forward established.
    unclipped = _build_parts(_sfno_config(), donor_checkpoint=checkpoint)
    x = torch.randn(2, len(DONOR_NAMES), *IMG_SHAPE) + 50.0
    for module in modules:
        module.eval()

    torch.manual_seed(1)
    expected = donor(x)
    torch.manual_seed(1)
    result = _compose(parts, x)
    torch.manual_seed(1)
    without_clip = _compose(unclipped, x)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert not torch.allclose(result, without_clip)


def _cut_point_pool_config(
    checkpoint: str | None = None,
    freeze_processor: bool = False,
) -> ComponentPoolConfig:
    """A pool holding one full decomposition: physical -> latent -> physical."""
    sfno = _sfno_config()
    frozen = (
        ParameterInitializationConfig(
            parameters=[
                ParameterClassification(frozen=FrozenParameterConfig(include=["*"]))
            ]
        )
        if freeze_processor
        else ParameterInitializationConfig()
    )
    return ComponentPoolConfig(
        domains={
            "physical": DomainConfig(channels=list(DONOR_NAMES)),
            "latent": DomainConfig(
                channels=[LatentChannels(name="z", channels=_latent_channels(sfno))],
                grid_like="physical",
            ),
        },
        transforms={
            "encoder": TransformConfig(
                _part_config("encoder", sfno, donor_checkpoint=checkpoint),
                "physical",
                "latent",
            ),
            "processor": TransformConfig(
                _part_config("processor", sfno, donor_checkpoint=checkpoint),
                "latent",
                "latent",
                parameter_init=frozen,
            ),
            "decoder": TransformConfig(
                _part_config("decoder", sfno, donor_checkpoint=checkpoint),
                "latent",
                "physical",
            ),
        },
    )


def test_pool_builds_a_decomposition_with_a_frozen_processor(tmp_path):
    """The latent-splice shape: a frozen donor processor between trained parts."""
    checkpoint, donor = _donor_checkpoint(tmp_path, _sfno_config())
    pool = _cut_point_pool_config(checkpoint, freeze_processor=True).build(
        {"physical": _dataset_info()}
    )

    donor_state = strip_leading_module(donor.state_dict())
    processor = pool.transforms["processor"].torch_module
    assert not any(parameter.requires_grad for parameter in processor.parameters())
    for name, parameter in strip_leading_module(processor.state_dict()).items():
        torch.testing.assert_close(parameter, donor_state[name], rtol=0, atol=0)
    for name in ("encoder", "decoder"):
        assert all(
            parameter.requires_grad
            for parameter in pool.transforms[name].torch_module.parameters()
        )


def test_pool_state_round_trips_without_the_donor(tmp_path):
    checkpoint, _ = _donor_checkpoint(tmp_path, _sfno_config())
    pool = _cut_point_pool_config(checkpoint).build({"physical": _dataset_info()})
    state = pool.get_state()
    (tmp_path / "donor.ckpt").unlink()

    reloaded = ComponentPool.from_state(state)

    x = torch.randn(2, len(DONOR_NAMES), *IMG_SHAPE)
    torch.manual_seed(0)
    expected = _compose(pool.transforms, x)
    torch.manual_seed(0)
    result = _compose(reloaded.transforms, x)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_pool_set_epoch_resets_the_clip_envelope():
    """A cut-point encoder in the pool gets the per-epoch envelope reset."""
    sfno = _sfno_config(clip_latent_global_means=True)
    encoder = _build_part("encoder", sfno)
    pool = ComponentPool(
        config=ComponentPoolConfig(),
        transforms={"encoder": encoder},
        backbones={},
        dataset_info={"physical": _dataset_info()},
    )
    part = encoder.torch_module.conditional_model

    encoder.torch_module.train()
    encoder(torch.randn(2, len(DONOR_NAMES), *IMG_SHAPE))
    assert torch.isfinite(part._gm_max).all()

    pool.set_epoch(1)
    assert part._gm_reset_pending
    encoder(torch.randn(2, len(DONOR_NAMES), *IMG_SHAPE))
    assert not part._gm_reset_pending

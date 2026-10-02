import dataclasses
import pathlib

import pytest
import torch
import yaml

import fme
from fme.ace.registry.two_track_sfno import TwoTrackSFNOBuilder
from fme.ace.step.two_track import TwoTrackStep, TwoTrackStepConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig
from fme.core.step.args import StepArgs
from fme.core.step.step import StepSelector
from fme.core.testing import get_dataset_info, trivial_network_and_loss_normalization

IMG_SHAPE = (16, 32)
EXAMPLE_CONFIG = (
    pathlib.Path(__file__).parent / "example_configs" / "two_track_baseline.yaml"
)


def _builder(embed_dim=8, local_embed_dim=3, **kwargs) -> TwoTrackSFNOBuilder:
    return TwoTrackSFNOBuilder(
        embed_dim=embed_dim,
        local_embed_dim=local_embed_dim,
        noise_embed_dim=4,
        num_layers=2,
        pos_embed=True,
        **kwargs,
    )


def _config(**overrides) -> TwoTrackStepConfig:
    names = ["g_in", "l_in", "shared", "g_out", "l_out"]
    base = TwoTrackStepConfig(
        builder=_builder(),
        global_in_names=["g_in", "shared"],
        local_in_names=["l_in"],
        global_out_names=["g_out", "shared"],
        local_out_names=["l_out"],
        normalization=trivial_network_and_loss_normalization(names),
    )
    if overrides:
        # dataclasses.replace re-runs __post_init__ so overrides are validated.
        base = dataclasses.replace(base, **overrides)
    return base


def _selector(config: TwoTrackStepConfig) -> StepSelector:
    return StepSelector(type="two_track", config=dataclasses.asdict(config))


def _get_step(config: TwoTrackStepConfig) -> TwoTrackStep:
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE, device=fme.get_device())
    step = _selector(config).get_step(dataset_info, lambda _: None)
    assert isinstance(step, TwoTrackStep)
    return step


def _tensor_dict(names, n_samples=3):
    device = fme.get_device()
    return {n: torch.rand(n_samples, *IMG_SHAPE, device=device) for n in names}


# ---------------------------------------------------------------------------
# post-init validation: a variable may not be on both tracks on the same side
# ---------------------------------------------------------------------------
def test_raises_when_input_variable_on_both_tracks():
    with pytest.raises(ValueError, match="both global_in_names"):
        _config(global_in_names=["x", "shared"], local_in_names=["x"])


def test_raises_when_output_variable_on_both_tracks():
    with pytest.raises(ValueError, match="both global_out_names"):
        _config(global_out_names=["g_out", "y"], local_out_names=["y"])


def test_same_variable_on_different_sides_is_allowed():
    # A variable global on input and local on output (different sides) is fine.
    config = _config(
        global_in_names=["g_in", "shared"],
        local_in_names=["l_in"],
        global_out_names=["g_out"],
        local_out_names=["shared"],
    )
    assert "shared" in config.global_in_names
    assert "shared" in config.local_out_names


# ---------------------------------------------------------------------------
# pack / unpack: global inputs precede local inputs; outputs split the same way
# ---------------------------------------------------------------------------
def test_in_out_packer_order_is_global_then_local():
    config = _config()
    step = _get_step(config)
    assert step.in_packer.names == ["g_in", "shared", "l_in"]
    assert step.out_packer.names == ["g_out", "shared", "l_out"]


def test_step_produces_all_outputs_with_correct_shape():
    torch.manual_seed(0)
    step = _get_step(_config())
    output = step.step(
        args=StepArgs(
            input=_tensor_dict(step.input_names),
            next_step_input_data=_tensor_dict(step.next_step_input_names),
            labels=None,
        ),
    ).output
    assert set(output) == {"g_out", "shared", "l_out"}
    for name in output:
        assert output[name].shape == (3, *IMG_SHAPE)


def test_step_unpacks_network_output_in_declared_order(monkeypatch):
    # The network output tensor is [global_out | local_out]; the step must
    # unpack it into (global_out_names + local_out_names) in that channel order.
    config = _config()
    step = _get_step(config)
    n = 2
    device = fme.get_device()

    # Replace the network with one returning per-channel constant tensors so we
    # can assert each output name maps to its packed channel index.
    n_out = len(config.global_out_names) + len(config.local_out_names)

    class ConstantChannels:
        """Mimics the Module wrapper boundary: wrap_module then (input, labels)."""

        def wrap_module(self, wrapper):
            return self

        def __call__(self, input_tensor, labels=None):
            batch = input_tensor.shape[0]
            channels = [
                torch.full((batch, 1, *IMG_SHAPE), float(i), device=device)
                for i in range(n_out)
            ]
            return torch.cat(channels, dim=-3)

    monkeypatch.setattr(step, "module", ConstantChannels())
    output = step.step(
        args=StepArgs(
            input=_tensor_dict(step.input_names, n),
            next_step_input_data=_tensor_dict(step.next_step_input_names, n),
            labels=None,
        ),
    ).output
    ordered = config.global_out_names + config.local_out_names
    for i, name in enumerate(ordered):
        # normalization is trivial (mean 0, std 1), residual prediction off, so
        # the denormalized output equals the network's constant channel value.
        assert torch.allclose(
            output[name], torch.full_like(output[name], float(i))
        ), name


# ---------------------------------------------------------------------------
# builder: single-tensor build interface is unsupported (needs per-track counts)
# ---------------------------------------------------------------------------
def test_builder_single_tensor_build_raises():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE, device=fme.get_device())
    with pytest.raises(NotImplementedError, match="build_two_track"):
        _builder().build(n_in_channels=3, n_out_channels=2, dataset_info=dataset_info)


# ---------------------------------------------------------------------------
# builder: local_embed_dim=0 (single-track-equivalent) is grid-restricted, and
# local_embed_dim must be nonzero iff the local track carries channels
# ---------------------------------------------------------------------------
def test_builder_rejects_zero_local_embed_dim_with_equiangular():
    # The single-track-equivalent config (local_embed_dim=0) is byte-for-byte
    # backwards compatible only on legendre-gauss; equiangular is rejected loudly
    # instead of silently loading a wrong-output checkpoint.
    with pytest.raises(ValueError, match="equiangular"):
        _builder(local_embed_dim=0, data_grid="equiangular")


def test_builder_allows_zero_local_embed_dim_with_legendre_gauss():
    builder = _builder(local_embed_dim=0, data_grid="legendre-gauss")
    assert builder.local_embed_dim == 0


def test_build_two_track_requires_nonzero_local_embed_dim_when_local_present():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE, device=fme.get_device())
    with pytest.raises(ValueError, match="local_embed_dim must be > 0"):
        _builder(local_embed_dim=0).build_two_track(
            global_in_channels=2,
            local_in_channels=1,
            global_out_channels=2,
            local_out_channels=0,
            dataset_info=dataset_info,
        )


def test_build_two_track_requires_zero_local_embed_dim_when_no_local():
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE, device=fme.get_device())
    with pytest.raises(ValueError, match="local_embed_dim must be 0"):
        _builder(local_embed_dim=3).build_two_track(
            global_in_channels=2,
            local_in_channels=0,
            global_out_channels=2,
            local_out_channels=0,
            dataset_info=dataset_info,
        )


def test_builder_threads_spectral_ratio_to_net_config():
    # An out-of-range spectral_ratio is rejected via _net_config, proving the
    # builder passes it through to the network config (validated against the
    # global width).
    with pytest.raises(ValueError, match="spectral_ratio must be in"):
        _builder(spectral_ratio=1.5)


# ---------------------------------------------------------------------------
# config load round trip
# ---------------------------------------------------------------------------
def test_config_state_round_trip():
    config = _config(residual_prediction=True)
    restored = TwoTrackStepConfig.from_state(config.get_state())
    assert restored.global_in_names == config.global_in_names
    assert restored.local_out_names == config.local_out_names
    assert restored.residual_prediction is True


# ---------------------------------------------------------------------------
# documented baseline example config: loads, builds, and runs a train iteration
# ---------------------------------------------------------------------------
def _load_example_config() -> TwoTrackStepConfig:
    with open(EXAMPLE_CONFIG) as f:
        data = yaml.safe_load(f)
    return TwoTrackStepConfig.from_state(data)


def test_example_baseline_config_loads_with_options_off():
    config = _load_example_config()
    assert config.builder.feed_global_to_local is False
    assert config.builder.parallel_conv1x1 is False
    assert config.builder.per_track_layer_norm is False
    # some variables are assigned to the local track
    assert len(config.local_in_names) > 0
    assert len(config.local_out_names) > 0
    assert isinstance(config.normalization, NetworkAndLossNormalizationConfig)


def test_example_baseline_config_smoke_train_step():
    torch.manual_seed(0)
    config = _load_example_config()
    step = _get_step(config)
    step.train()
    output = step.step(
        args=StepArgs(
            input=_tensor_dict(step.input_names, n_samples=2),
            next_step_input_data=_tensor_dict(step.next_step_input_names, n_samples=2),
            labels=None,
        ),
    ).output
    loss = sum(v.sum() for v in output.values())
    loss.backward()
    # every trainable module received gradients
    grads = [
        p.grad is not None
        for module in step.modules
        for p in module.parameters()
        if p.requires_grad
    ]
    assert grads and all(grads)


def test_example_baseline_state_round_trip_matches_output():
    torch.manual_seed(0)
    config = _load_example_config()
    step1 = _get_step(config)
    step2 = _get_step(config)
    step2.load_state(step1.get_state())
    step1.eval()
    step2.eval()
    args = StepArgs(
        input=_tensor_dict(step1.input_names, n_samples=1),
        next_step_input_data=_tensor_dict(step1.next_step_input_names, n_samples=1),
        labels=None,
    )
    with torch.no_grad():
        out1 = step1.step(args=args).output
        out2 = step2.step(args=args).output
    for name in out1:
        torch.testing.assert_close(out1[name], out2[name])

import dataclasses

import pytest
import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.registry.module import ModuleConfig, ModuleSelector
from fme.core.registry.separated_module import (
    LegacyModuleAdapter,
    LegacyWrapper,
    SeparatedModule,
    SeparatedModuleConfig,
    SeparatedModuleSelector,
)


class SimpleSeparatedModule(nn.Module):
    """A trivial module for testing the separated interface."""

    def __init__(self, n_forcing, n_prognostic, n_diagnostic):
        super().__init__()
        n_in = n_forcing + n_prognostic
        self.prog_linear = nn.Linear(n_in, n_prognostic, bias=False)
        self.diag_linear = nn.Linear(n_in, n_diagnostic, bias=False)

    def forward(
        self, forcing: torch.Tensor, prognostic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([forcing, prognostic], dim=-3)
        b, c, h, w = combined.shape
        flat = combined.permute(0, 2, 3, 1).reshape(-1, c)
        prog_out = self.prog_linear(flat).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        diag_out = self.diag_linear(flat).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return prog_out, diag_out


@SeparatedModuleSelector.register("test_simple")
@dataclasses.dataclass
class SimpleSeparatedBuilder(SeparatedModuleConfig):
    def build(
        self,
        n_forcing_channels,
        n_prognostic_channels,
        n_diagnostic_channels,
        dataset_info,
    ):
        return SimpleSeparatedModule(
            n_forcing_channels, n_prognostic_channels, n_diagnostic_channels
        )


class SimpleLinearModule(nn.Module):
    """Legacy-style module: single tensor in, single tensor out."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        out = self.linear(flat).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return out


@ModuleSelector.register("test_simple_linear")
@dataclasses.dataclass
class SimpleLinearBuilder(ModuleConfig):
    """Registered in old ModuleSelector for testing."""

    def build(self, n_in_channels, n_out_channels, dataset_info):
        return SimpleLinearModule(n_in_channels, n_out_channels)


# --- Tests for SeparatedModuleSelector ---


def test_register_and_build():
    selector = SeparatedModuleSelector(type="test_simple", config={})
    dataset_info = DatasetInfo(img_shape=(4, 8))
    module = selector.build(
        n_forcing_channels=2,
        n_prognostic_channels=3,
        n_diagnostic_channels=1,
        dataset_info=dataset_info,
    )
    assert isinstance(module, SeparatedModule)
    assert isinstance(module.torch_module, SimpleSeparatedModule)


def test_separated_module_forward():
    inner = SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module = SeparatedModule(inner, label_encoding=None)

    forcing = torch.randn(2, 2, 4, 8)
    prognostic = torch.randn(2, 3, 4, 8)
    prog_out, diag_out = module(forcing, prognostic)

    assert prog_out.shape == (2, 3, 4, 8)
    assert diag_out.shape == (2, 1, 4, 8)


def test_separated_module_get_and_load_state():
    inner = SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module = SeparatedModule(inner, label_encoding=None)

    state = module.get_state()
    assert "label_encoding" in state
    assert state["label_encoding"] is None

    inner2 = SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module2 = SeparatedModule(inner2, label_encoding=None)
    module2.load_state(state)

    for p1, p2 in zip(inner.parameters(), inner2.parameters()):
        assert torch.equal(p1, p2)


def test_separated_module_wrap_module():
    inner = SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module = SeparatedModule(inner, label_encoding=None)

    wrapped = module.wrap_module(lambda m: m)
    assert isinstance(wrapped, SeparatedModule)

    forcing = torch.randn(1, 2, 4, 8)
    prognostic = torch.randn(1, 3, 4, 8)
    out1 = module(forcing, prognostic)
    out2 = wrapped(forcing, prognostic)
    assert torch.equal(out1[0], out2[0])
    assert torch.equal(out1[1], out2[1])


def test_separated_module_labels_not_allowed_for_unconditional():
    inner = SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module = SeparatedModule(inner, label_encoding=None)

    forcing = torch.randn(1, 2, 4, 8)
    prognostic = torch.randn(1, 3, 4, 8)

    # Should not raise with no labels
    module(forcing, prognostic)

    from fme.core.labels import BatchLabels

    labels = BatchLabels(torch.zeros(1, 2), ["a", "b"])
    with pytest.raises(TypeError, match="Labels are not allowed"):
        module(forcing, prognostic, labels=labels)


# --- Tests for LegacyWrapper ---


def test_legacy_wrapper_forward_splits_correctly():
    """LegacyWrapper concatenates input and splits output at the right place."""
    torch.manual_seed(42)
    n_forcing, n_prognostic, n_diagnostic = 2, 3, 1
    n_in = n_forcing + n_prognostic
    n_out = n_prognostic + n_diagnostic

    inner = SimpleLinearModule(n_in, n_out)
    wrapper = LegacyWrapper(inner, n_prognostic_channels=n_prognostic)

    forcing = torch.randn(2, n_forcing, 4, 8)
    prognostic = torch.randn(2, n_prognostic, 4, 8)

    prog_out, diag_out = wrapper(forcing, prognostic)

    # Compare with direct call
    combined = torch.cat([forcing, prognostic], dim=-3)
    direct_out = inner(combined)

    assert torch.allclose(prog_out, direct_out[:, :n_prognostic])
    assert torch.allclose(diag_out, direct_out[:, n_prognostic:])


def test_legacy_wrapper_state_dict_prefixes_inner():
    """LegacyWrapper.state_dict keys are prefixed with 'inner.'."""
    inner = SimpleLinearModule(3, 2)
    wrapper = LegacyWrapper(inner, n_prognostic_channels=1)

    inner_state = inner.state_dict()
    wrapper_state = wrapper.state_dict()

    for key in inner_state:
        assert f"inner.{key}" in wrapper_state
        assert torch.equal(inner_state[key], wrapper_state[f"inner.{key}"])


def test_legacy_wrapper_load_state_dict():
    """LegacyWrapper.load_state_dict loads into the inner module."""
    torch.manual_seed(0)
    inner1 = SimpleLinearModule(3, 2)

    torch.manual_seed(1)
    inner2 = SimpleLinearModule(3, 2)
    wrapper = LegacyWrapper(inner2, n_prognostic_channels=1)

    assert not torch.equal(list(inner1.parameters())[0], list(inner2.parameters())[0])

    # State dict needs 'inner.' prefix for LegacyWrapper
    state = {"inner." + k: v for k, v in inner1.state_dict().items()}
    wrapper.load_state_dict(state)

    for p1, p2 in zip(inner1.parameters(), inner2.parameters()):
        assert torch.equal(p1, p2)


def test_legacy_wrapper_zero_diagnostics():
    """LegacyWrapper works when there are no diagnostic channels."""
    torch.manual_seed(42)
    n_forcing, n_prognostic = 2, 3
    n_in = n_forcing + n_prognostic
    n_out = n_prognostic  # no diagnostics

    inner = SimpleLinearModule(n_in, n_out)
    wrapper = LegacyWrapper(inner, n_prognostic_channels=n_prognostic)

    forcing = torch.randn(2, n_forcing, 4, 8)
    prognostic = torch.randn(2, n_prognostic, 4, 8)

    prog_out, diag_out = wrapper(forcing, prognostic)
    assert prog_out.shape == (2, n_prognostic, 4, 8)
    assert diag_out.shape == (2, 0, 4, 8)


# --- Tests for LegacyModuleAdapter ---


def test_legacy_adapter_build():
    torch.manual_seed(42)
    selector = SeparatedModuleSelector(
        type="legacy",
        config={
            "legacy_builder": {"type": "test_simple_linear", "config": {}},
        },
    )

    dataset_info = DatasetInfo(img_shape=(4, 8))
    module = selector.build(
        n_forcing_channels=2,
        n_prognostic_channels=2,
        n_diagnostic_channels=1,
        dataset_info=dataset_info,
    )

    forcing = torch.randn(2, 2, 4, 8)
    prognostic = torch.randn(2, 2, 4, 8)
    prog_out, diag_out = module(forcing, prognostic)

    assert prog_out.shape == (2, 2, 4, 8)
    assert diag_out.shape == (2, 1, 4, 8)


def test_legacy_adapter_numerical_equivalence():
    """LegacyWrapper output matches direct legacy module call."""
    torch.manual_seed(42)
    n_forcing, n_prognostic, n_diagnostic = 2, 3, 1

    # Build via legacy adapter
    adapter = LegacyModuleAdapter(
        legacy_builder=ModuleSelector(type="test_simple_linear", config={})
    )
    dataset_info = DatasetInfo(img_shape=(4, 8))
    wrapper_module = adapter.build(
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        dataset_info=dataset_info,
    )

    # Build the same inner module directly
    torch.manual_seed(42)
    direct_module = SimpleLinearModule(
        n_forcing + n_prognostic, n_prognostic + n_diagnostic
    )

    # Load same weights (add 'inner.' prefix for LegacyWrapper)
    state = {"inner." + k: v for k, v in direct_module.state_dict().items()}
    wrapper_module.load_state_dict(state)

    forcing = torch.randn(2, n_forcing, 4, 8)
    prognostic = torch.randn(2, n_prognostic, 4, 8)

    prog_out, diag_out = wrapper_module(forcing, prognostic)

    combined = torch.cat([forcing, prognostic], dim=-3)
    direct_out = direct_module(combined)

    assert torch.allclose(prog_out, direct_out[:, :n_prognostic])
    assert torch.allclose(diag_out, direct_out[:, n_prognostic:])

import torch

from fme.core.dataset_info import DatasetInfo
from fme.core.registry.separated_module import (
    SeparatedModule,
    SeparatedModuleSelector,
    _SimpleSeparatedModule,
    register_test_types,
)

register_test_types()


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
    assert isinstance(module.torch_module, _SimpleSeparatedModule)


def test_separated_module_forward():
    inner = _SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module = SeparatedModule(inner, label_encoding=None)

    forcing = torch.randn(2, 2, 4, 8)
    prognostic = torch.randn(2, 3, 4, 8)
    prog_out, diag_out = module(forcing, prognostic)

    assert prog_out.shape == (2, 3, 4, 8)
    assert diag_out.shape == (2, 1, 4, 8)


def test_separated_module_get_and_load_state():
    inner = _SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module = SeparatedModule(inner, label_encoding=None)

    state = module.get_state()
    assert "label_encoding" in state
    assert state["label_encoding"] is None

    inner2 = _SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module2 = SeparatedModule(inner2, label_encoding=None)
    module2.load_state(state)

    for p1, p2 in zip(inner.parameters(), inner2.parameters()):
        assert torch.equal(p1, p2)


def test_separated_module_wrap_module():
    inner = _SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module = SeparatedModule(inner, label_encoding=None)

    wrapped = module.wrap_module(lambda m: m)
    assert isinstance(wrapped, SeparatedModule)

    forcing = torch.randn(1, 2, 4, 8)
    prognostic = torch.randn(1, 3, 4, 8)
    out1 = module(forcing, prognostic)
    out2 = wrapped(forcing, prognostic)
    assert torch.equal(out1[0], out2[0])
    assert torch.equal(out1[1], out2[1])


def test_separated_module_accepts_none_labels():
    inner = _SimpleSeparatedModule(n_forcing=2, n_prognostic=3, n_diagnostic=1)
    module = SeparatedModule(inner, label_encoding=None)

    forcing = torch.randn(1, 2, 4, 8)
    prognostic = torch.randn(1, 3, 4, 8)

    # Should work with no labels
    prog_out, diag_out = module(forcing, prognostic)
    assert prog_out.shape == (1, 3, 4, 8)
    assert diag_out.shape == (1, 1, 4, 8)

    # Should also work with explicit None
    prog_out, diag_out = module(forcing, prognostic, labels=None)
    assert prog_out.shape == (1, 3, 4, 8)

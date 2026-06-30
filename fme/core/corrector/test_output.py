import torch

from fme import get_device
from fme.core.corrector.output import (
    CorrectorDiagnostics,
    CorrectorOutput,
    build_corrector_diagnostics,
)

DEVICE = get_device()
IMG_SHAPE = (4, 5)


def test_corrector_diagnostics_defaults_to_empty_delta():
    diagnostics = CorrectorDiagnostics()
    assert diagnostics.delta == {}


def test_corrector_output_defaults():
    corrected = {"a": torch.ones(IMG_SHAPE, device=DEVICE)}
    output = CorrectorOutput(corrected=corrected)
    # corrected is carried through verbatim
    assert output.corrected is corrected
    # defaults: empty diagnostics, no corrector state
    assert output.diagnostics.delta == {}
    assert output.corrector_state is None


def test_build_corrector_diagnostics_basic_delta():
    snapshot = {"a": torch.full(IMG_SHAPE, 2.0, device=DEVICE)}
    corrected = {"a": torch.full(IMG_SHAPE, 5.0, device=DEVICE)}
    diagnostics = build_corrector_diagnostics(snapshot, corrected, ["a"])
    assert set(diagnostics.delta) == {"a"}
    # delta = corrected - snapshot = 5 - 2 = 3
    torch.testing.assert_close(
        diagnostics.delta["a"], torch.full(IMG_SHAPE, 3.0, device=DEVICE)
    )
    # raw network value recoverable as corrected - delta == snapshot
    torch.testing.assert_close(corrected["a"] - diagnostics.delta["a"], snapshot["a"])


def test_build_corrector_diagnostics_ignores_untouched_names():
    snapshot = {
        "a": torch.full(IMG_SHAPE, 1.0, device=DEVICE),
        "b": torch.full(IMG_SHAPE, 1.0, device=DEVICE),
    }
    corrected = {
        "a": torch.full(IMG_SHAPE, 4.0, device=DEVICE),
        "b": torch.full(IMG_SHAPE, 9.0, device=DEVICE),
    }
    # "b" is present in the inputs but not declared touched, so it must not appear
    diagnostics = build_corrector_diagnostics(snapshot, corrected, ["a"])
    assert set(diagnostics.delta) == {"a"}
    torch.testing.assert_close(
        diagnostics.delta["a"], torch.full(IMG_SHAPE, 3.0, device=DEVICE)
    )


def test_build_corrector_diagnostics_cumulative_delta():
    # Simulate two enabled corrections touching the same field: an offset of +3
    # followed by an offset of -1, so the corrector's exit value carries the
    # cumulative net effect (+2) against its entry snapshot.
    snapshot = {"a": torch.full(IMG_SHAPE, 10.0, device=DEVICE)}
    after_first = snapshot["a"] + 3.0
    after_second = after_first - 1.0
    corrected = {"a": after_second}
    diagnostics = build_corrector_diagnostics(snapshot, corrected, ["a"])
    # cumulative delta = (+3) + (-1) = +2
    torch.testing.assert_close(
        diagnostics.delta["a"], torch.full(IMG_SHAPE, 2.0, device=DEVICE)
    )
    # corrected - delta recovers the entry snapshot exactly
    torch.testing.assert_close(corrected["a"] - diagnostics.delta["a"], snapshot["a"])


def test_build_corrector_diagnostics_empty_touched_names():
    snapshot = {"a": torch.full(IMG_SHAPE, 2.0, device=DEVICE)}
    corrected = {"a": torch.full(IMG_SHAPE, 5.0, device=DEVICE)}
    diagnostics = build_corrector_diagnostics(snapshot, corrected, [])
    assert diagnostics.delta == {}

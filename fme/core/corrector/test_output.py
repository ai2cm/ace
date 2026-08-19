import pytest
import torch

from fme import get_device
from fme.core.corrector.output import (
    CorrectorDiagnostics,
    CorrectorOutput,
    build_corrector_diagnostics,
)
from fme.core.corrector.state import CorrectorState
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.spatial_masking import NullSpatialMasking

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


def test_apply_output_masking_masks_delta_and_returns_new_object():
    mask = torch.ones(IMG_SHAPE, device=DEVICE)
    mask[0, 0] = 0.0
    masking = SpatialMaskProvider({"mask_2d": mask}).build_output_spatial_masker()
    original = torch.full(IMG_SHAPE, 2.0, device=DEVICE)
    diagnostics = CorrectorDiagnostics(delta={"a": original.clone()})

    masked = diagnostics.apply_output_masking(masking)
    assert masked is not diagnostics
    # input unmutated
    torch.testing.assert_close(diagnostics.delta["a"], original)
    # NaN exactly off-mask, unchanged on-mask
    expected = original.clone()
    expected[0, 0] = float("nan")
    torch.testing.assert_close(masked.delta["a"], expected, equal_nan=True)

    # the no-op masking preserves values
    null_masked = diagnostics.apply_output_masking(NullSpatialMasking())
    torch.testing.assert_close(null_masked.delta["a"], original)


# ---- modified_names property ----


def test_modified_names_empty_when_no_corrections():
    output = CorrectorOutput(corrected={"a": torch.zeros(2)})
    assert output.modified_names == set()


def test_modified_names_from_delta():
    output = CorrectorOutput(
        corrected={"a": torch.zeros(2)},
        diagnostics=CorrectorDiagnostics(delta={"a": torch.ones(2)}),
    )
    assert output.modified_names == {"a"}


def test_modified_names_from_pre_diagnosis():
    output = CorrectorOutput(
        corrected={"a": torch.zeros(2)},
        diagnostics=CorrectorDiagnostics(
            pre_diagnosis_fields={"a": torch.ones(2)}
        ),
    )
    assert output.modified_names == {"a"}


def test_modified_names_union_of_delta_and_pre_diagnosis():
    output = CorrectorOutput(
        corrected={"a": torch.zeros(2), "b": torch.zeros(2)},
        diagnostics=CorrectorDiagnostics(
            delta={"a": torch.ones(2)},
            pre_diagnosis_fields={"b": torch.ones(2)},
        ),
    )
    assert output.modified_names == {"a", "b"}


# ---- with_state ----


def test_with_state_returns_new_output_with_given_state():
    original = CorrectorOutput(
        corrected={"a": torch.zeros(2)},
        corrector_state=None,
    )
    state = CorrectorState(global_dry_air_mass=torch.tensor([1.0]))
    updated = original.with_state(state)
    assert updated is not original
    assert updated.corrector_state is state
    assert updated.corrected is original.corrected
    assert updated.diagnostics is original.diagnostics


def test_with_state_preserves_seed_state():
    seed = {"a": torch.full((2,), 10.0)}
    output = CorrectorOutput(corrected={"a": torch.zeros(2)})
    after = output.apply_correction(diagnosed={"a": torch.ones(2)}, deltas={})
    # after has _seed_state set; with_state must carry it through
    state = CorrectorState(global_dry_air_mass=torch.tensor([1.0]))
    updated = after.with_state(state)
    assert updated.corrector_state is state
    assert updated._seed_state is after._seed_state


# ---- apply_correction: delta ----


def test_apply_correction_delta_accumulates():
    corrected = {"a": torch.full((2,), 10.0, device=DEVICE)}
    output = CorrectorOutput(corrected=corrected)
    after1 = output.apply_correction(diagnosed={}, deltas={"a": torch.ones(2)})
    torch.testing.assert_close(after1.corrected["a"], torch.full((2,), 11.0))
    after2 = after1.apply_correction(diagnosed={}, deltas={"a": torch.ones(2)})
    torch.testing.assert_close(after2.corrected["a"], torch.full((2,), 12.0))
    # delta accumulates
    torch.testing.assert_close(after2.diagnostics.delta["a"], torch.full((2,), 2.0))


def test_apply_correction_delta_on_absent_field_raises():
    output = CorrectorOutput(corrected={"a": torch.zeros(2)})
    with pytest.raises(ValueError, match="not present in corrected"):
        output.apply_correction(diagnosed={}, deltas={"b": torch.ones(2)})


# ---- apply_correction: diagnosis ----


def test_apply_correction_diagnosis_records_seed_state():
    seed_val = torch.full((2,), 10.0, device=DEVICE)
    output = CorrectorOutput(corrected={"a": seed_val.clone()})
    new_val = torch.full((2,), 99.0, device=DEVICE)
    after = output.apply_correction(diagnosed={"a": new_val}, deltas={})
    # corrected has the diagnosed value
    torch.testing.assert_close(after.corrected["a"], new_val)
    # pre_diagnosis_fields records the seed state
    torch.testing.assert_close(
        after.diagnostics.pre_diagnosis_fields["a"], seed_val
    )


def test_apply_correction_diagnosis_on_absent_field_raises():
    output = CorrectorOutput(corrected={"a": torch.zeros(2)})
    with pytest.raises(ValueError, match="not present in corrected"):
        output.apply_correction(diagnosed={"b": torch.ones(2)}, deltas={})


def test_apply_correction_diagnosis_after_delta_raises():
    output = CorrectorOutput(corrected={"a": torch.zeros(2)})
    after_delta = output.apply_correction(
        diagnosed={}, deltas={"a": torch.ones(2)}
    )
    with pytest.raises(ValueError, match="cannot diagnose"):
        after_delta.apply_correction(
            diagnosed={"a": torch.full((2,), 5.0)}, deltas={}
        )


def test_apply_correction_later_diagnosis_supersedes_earlier():
    output = CorrectorOutput(
        corrected={"a": torch.full((2,), 10.0, device=DEVICE)}
    )
    first = output.apply_correction(
        diagnosed={"a": torch.full((2,), 20.0)}, deltas={}
    )
    second = first.apply_correction(
        diagnosed={"a": torch.full((2,), 30.0)}, deltas={}
    )
    # corrected reflects the later diagnosis
    torch.testing.assert_close(second.corrected["a"], torch.full((2,), 30.0))
    # pre_diagnosis_fields still records the SEED state (10.0), not the
    # intermediate diagnosed value (20.0)
    torch.testing.assert_close(
        second.diagnostics.pre_diagnosis_fields["a"],
        torch.full((2,), 10.0, device=DEVICE),
    )


def test_apply_correction_delta_after_diagnosis_allowed():
    output = CorrectorOutput(
        corrected={"a": torch.full((2,), 10.0, device=DEVICE)}
    )
    diagnosed = output.apply_correction(
        diagnosed={"a": torch.full((2,), 50.0)}, deltas={}
    )
    after_delta = diagnosed.apply_correction(
        diagnosed={}, deltas={"a": torch.ones(2)}
    )
    # corrected = diagnosed value + delta
    torch.testing.assert_close(after_delta.corrected["a"], torch.full((2,), 51.0))
    # delta records only the additive part
    torch.testing.assert_close(
        after_delta.diagnostics.delta["a"], torch.ones(2)
    )
    # pre_diagnosis_fields retains the seed
    torch.testing.assert_close(
        after_delta.diagnostics.pre_diagnosis_fields["a"],
        torch.full((2,), 10.0, device=DEVICE),
    )


# ---- apply_correction: seed state ----


def test_apply_correction_seed_state_is_initial_corrected():
    """The seed state for pre_diagnosis_fields is the very first ``corrected``
    dict, not the current (mid-fold) corrected dict."""
    seed_a = torch.full((2,), 1.0, device=DEVICE)
    seed_b = torch.full((2,), 2.0, device=DEVICE)
    output = CorrectorOutput(corrected={"a": seed_a.clone(), "b": seed_b.clone()})
    # Delta on "a" changes the current corrected["a"] but not the seed
    after_delta = output.apply_correction(
        diagnosed={}, deltas={"a": torch.full((2,), 100.0)}
    )
    # Now diagnose "b" -- pre_diagnosis_fields["b"] should be the seed (2.0),
    # not the current corrected["b"] (which is still 2.0 here, but the seed
    # tracking path is what matters).
    after_diag = after_delta.apply_correction(
        diagnosed={"b": torch.full((2,), 99.0)}, deltas={}
    )
    torch.testing.assert_close(
        after_diag.diagnostics.pre_diagnosis_fields["b"], seed_b
    )


# ---- aliasing invariant ----


def test_aliasing_mutate_corrected_does_not_affect_pre_diagnosis():
    """Mutating the returned ``corrected`` dict in-place must not affect
    ``pre_diagnosis_fields`` -- they reference the seed state, not the
    new corrected dict."""
    seed = torch.full((2,), 5.0, device=DEVICE)
    output = CorrectorOutput(corrected={"a": seed.clone()})
    after = output.apply_correction(
        diagnosed={"a": torch.full((2,), 99.0, device=DEVICE)}, deltas={}
    )
    # Mutate the returned corrected tensor in-place
    after.corrected["a"].fill_(0.0)
    # pre_diagnosis_fields must be untouched
    torch.testing.assert_close(
        after.diagnostics.pre_diagnosis_fields["a"],
        torch.full((2,), 5.0, device=DEVICE),
    )


def test_aliasing_mutate_corrected_does_not_affect_seed():
    """Mutating the returned ``corrected`` dict must not affect the seed
    state used for future diagnoses in the same fold."""
    seed = torch.full((2,), 5.0, device=DEVICE)
    output = CorrectorOutput(corrected={"a": seed.clone(), "b": seed.clone()})
    after_diag_a = output.apply_correction(
        diagnosed={"a": torch.full((2,), 99.0, device=DEVICE)}, deltas={}
    )
    # Mutate the returned corrected in-place
    after_diag_a.corrected["a"].fill_(0.0)
    # Diagnose "b" next -- the seed for "b" should be 5.0 (the original
    # corrected value), not affected by the mutation above.
    after_diag_b = after_diag_a.apply_correction(
        diagnosed={"b": torch.full((2,), 77.0, device=DEVICE)}, deltas={}
    )
    torch.testing.assert_close(
        after_diag_b.diagnostics.pre_diagnosis_fields["b"],
        torch.full((2,), 5.0, device=DEVICE),
    )


# ---- corrector_state threading ----


def test_apply_correction_preserves_corrector_state():
    state = CorrectorState(global_dry_air_mass=torch.tensor([1.0]))
    output = CorrectorOutput(
        corrected={"a": torch.zeros(2)}, corrector_state=state
    )
    after = output.apply_correction(diagnosed={}, deltas={"a": torch.ones(2)})
    assert after.corrector_state is state


def test_apply_correction_is_not_mutating():
    """apply_correction returns a new CorrectorOutput; the original is
    unchanged."""
    original_tensor = torch.full((2,), 10.0, device=DEVICE)
    output = CorrectorOutput(corrected={"a": original_tensor.clone()})
    _ = output.apply_correction(
        diagnosed={}, deltas={"a": torch.ones(2, device=DEVICE)}
    )
    # original corrected untouched
    torch.testing.assert_close(output.corrected["a"], original_tensor)
    assert output.diagnostics.delta == {}

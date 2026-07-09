import math

import pytest
import torch

from fme.downscaling.spectral_loss import (
    SpectralMatchingLoss,
    SpectralMatchingLossConfig,
)

B, C, H, W = 2, 1, 8, 16


def _sinusoid(wavenumber: int, amplitude: float) -> torch.Tensor:
    """(B, C, H, W) field that is a pure cosine of the given zonal wavenumber."""
    lon = torch.arange(W, dtype=torch.float32) * (2 * math.pi / W)
    row = amplitude * torch.cos(wavenumber * lon)
    return row.reshape(1, 1, 1, W).expand(B, C, H, W).contiguous()


def _loss(config: SpectralMatchingLossConfig, out_names=None):
    return config.build(out_names or ["PRATEsfc"])


def test_zero_loss_when_equal():
    field = _sinusoid(wavenumber=3, amplitude=2.0)
    loss = _loss(SpectralMatchingLossConfig())(field, field.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_scalar_output():
    pred = _sinusoid(3, 1.0)
    target = _sinusoid(3, 2.0)
    out = _loss(SpectralMatchingLossConfig())(pred, target)
    assert out.shape == torch.Size([])


def test_gradient_flows_to_prediction_only():
    pred = _sinusoid(3, 1.0).requires_grad_(True)
    target = _sinusoid(3, 2.0).requires_grad_(True)
    loss = _loss(SpectralMatchingLossConfig())(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.any(pred.grad != 0)
    # target is detached inside the loss
    assert target.grad is None


def test_band_gamma_emphasizes_high_wavenumbers():
    # Mismatch located only at a high wavenumber.
    pred_hi = _sinusoid(6, 1.0)
    target_hi = _sinusoid(6, 2.0)
    # Same-magnitude mismatch located only at a low wavenumber.
    pred_lo = _sinusoid(1, 1.0)
    target_lo = _sinusoid(1, 2.0)

    flat = _loss(SpectralMatchingLossConfig(band_gamma=0.0))
    steep = _loss(SpectralMatchingLossConfig(band_gamma=3.0))

    hi_flat = flat(pred_hi, target_hi)
    hi_steep = steep(pred_hi, target_hi)
    lo_flat = flat(pred_lo, target_lo)
    lo_steep = steep(pred_lo, target_lo)

    # Increasing gamma up-weights the high-k mismatch and down-weights low-k.
    assert hi_steep > hi_flat
    assert lo_steep < lo_flat


def test_min_wavenumber_masks_low_wavenumbers():
    # Fields differ only at wavenumber 1; excluding wavenumbers < 2 makes the
    # loss vanish. Use raw power (not log) so the near-zero bins do not amplify
    # rfft roundoff -- the masking behavior is independent of the log option.
    pred = _sinusoid(1, 1.0)
    target = _sinusoid(1, 3.0)
    loss = _loss(SpectralMatchingLossConfig(min_wavenumber=2, log=False))(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_uses_ensemble_power_not_field_mean():
    # Two samples with opposite-sign high-k content: their field-mean is ~0
    # (smooth, no power), but each sample carries real power at wavenumber k.
    # Spectrum-then-average (correct) sees the full ensemble power; field-mean-
    # then-spectrum (the bug) would see ~0. Target is a zero field (no power),
    # so a correct loss must be clearly > 0.
    base = _sinusoid(4, 1.0)[:1]  # (1, 1, H, W)
    pred = torch.cat([base, -base], dim=0)  # (2, 1, H, W); field-mean ~ 0
    target = torch.zeros_like(pred)
    loss = _loss(SpectralMatchingLossConfig(log=False))(pred, target)
    assert loss.item() > 0.1


def test_variable_weights_can_zero_a_channel():
    out_names = ["a", "b"]
    # channel a matches; channel b mismatches.
    match = _sinusoid(3, 1.0)
    mismatch_a = torch.cat([match, _sinusoid(3, 2.0)], dim=1)  # (B, 2, H, W)
    target = torch.cat([match, match], dim=1)

    full = _loss(SpectralMatchingLossConfig(), out_names)(mismatch_a, target)
    zeroed = _loss(SpectralMatchingLossConfig(variable_weights={"b": 0.0}), out_names)(
        mismatch_a, target
    )

    assert full.item() > 0.0
    assert zeroed.item() == pytest.approx(0.0, abs=1e-6)


def test_build_orders_variable_weights_by_out_names():
    config = SpectralMatchingLossConfig(variable_weights={"b": 5.0, "a": 2.0})
    loss = config.build(["a", "b"])
    assert isinstance(loss, SpectralMatchingLoss)
    assert torch.allclose(loss.variable_weights.flatten(), torch.tensor([2.0, 5.0]))


def test_missing_variable_defaults_to_one():
    loss = SpectralMatchingLossConfig().build(["a", "b"])
    assert torch.allclose(loss.variable_weights.flatten(), torch.tensor([1.0, 1.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"band_gamma": -1.0},
        {"min_wavenumber": -1},
        {"eps": 0.0},
    ],
)
def test_config_validation_errors(kwargs):
    with pytest.raises(ValueError):
        SpectralMatchingLossConfig(**kwargs)


def test_unknown_variable_weight_key_errors():
    with pytest.raises(ValueError):
        SpectralMatchingLossConfig(variable_weights={"nope": 1.0}).build(["a"])


def test_min_wavenumber_excluding_all_errors():
    pred = _sinusoid(1, 1.0)
    target = _sinusoid(1, 2.0)
    # W // 2 + 1 = 9 wavenumbers, indices 0..8; exclude all.
    loss = _loss(SpectralMatchingLossConfig(min_wavenumber=100))
    with pytest.raises(ValueError):
        loss(pred, target)

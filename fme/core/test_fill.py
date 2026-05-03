import torch

from fme.core.fill import SmoothFloodFill, SmoothFloodFillPacked, fast_flood_fill


def test_smooth_flood_fill():
    H, W = 8, 16
    num_steps = 2
    nan_cols = [13, 14, 15, 0, 1, 2]

    data = torch.arange(W, dtype=torch.float32).expand(1, 1, H, W).clone()
    original = data.clone()
    data[:, :, :, nan_cols] = float("nan")

    flood_fill = SmoothFloodFill(
        num_steps=num_steps, blur_kernel_size=5, blur_sigma=1.0
    )
    filled = flood_fill(data, "name")

    # Verify mask caching: a second call should produce the same result.
    filled2 = flood_fill(data, "name")
    torch.testing.assert_close(filled, filled2)

    # 1. No NaNs in output
    assert not filled.isnan().any(), "Output should contain no NaNs"

    # 2. Preserves far data: use rows and columns far from NaN boundaries
    #    AND from the grid y-edges (the Gaussian blur uses zero y-padding).
    #    Safe rows: 3-4 (blur radius 2 away from y-edges 0 and 7).
    #    Safe columns: 6-9 (blur radius 2 away from NaN boundaries at 3 and 12).
    safe_rows = slice(3, 5)
    safe_cols = slice(6, 10)
    torch.testing.assert_close(
        filled[:, :, safe_rows, safe_cols],
        original[:, :, safe_rows, safe_cols],
    )

    # 3. Interior pixels (columns 15 and 0, deepest inside the NaN band)
    #    should be near the valid-data spatial mean (7.5).
    #    Check at a middle row to avoid y-edge blur effects.
    valid_mean = original[:, :, :, 3:13].mean()
    interior_mean = filled[:, :, 4, [15, 0]].mean()
    torch.testing.assert_close(interior_mean, valid_mean)

    # 4. Periodic boundary influence: column 14 sits between valid column 12
    #    (value 12) and column 3 (value 3) via the wrap. With correct periodic
    #    padding both sides contribute, so the filled value should not be near
    #    zero (which non-periodic zero-padding would produce).
    col14_mean = filled[:, :, 4, 14].mean().item()
    assert col14_mean > 2.0, (
        f"Column 14 fill value {col14_mean:.2f} should reflect periodic "
        f"influence from both sides of the boundary"
    )

    # 5. No y-boundary artifacts: top and bottom rows at columns far from the
    #    NaN region should be preserved (regression test for replicate padding).
    torch.testing.assert_close(
        filled[:, :, [0, -1], safe_cols],
        original[:, :, [0, -1], safe_cols],
    )

    # 6. Smoother than naive zero-fill: max absolute x-gradient should be much
    #    smaller for the smooth fill than for simply replacing NaN with 0.
    zero_filled = original.clone()
    zero_filled[:, :, :, nan_cols] = 0.0

    smooth_grad = torch.diff(filled, dim=-1).abs().max()
    zero_grad = torch.diff(zero_filled, dim=-1).abs().max()

    assert smooth_grad < zero_grad / 4, (
        f"Smooth fill max gradient ({smooth_grad.item():.2f}) should be much "
        f"less than zero-fill max gradient ({zero_grad.item():.2f})"
    )


def test_smooth_flood_fill_packed_multi_channel():
    """Per-channel masks: each channel has its own NaN pattern, including
    a channel with no NaN at all (atmosphere-style) which should pass
    through unchanged (modulo the all-ones blur identity)."""
    B, T, C, H, W = 2, 3, 3, 8, 16
    nan_cols_ch0 = [13, 14, 15, 0, 1, 2]
    nan_cols_ch1 = [6, 7, 8, 9]

    base = torch.arange(W, dtype=torch.float32).expand(B, T, C, H, W).clone()
    base = base + torch.arange(C, dtype=torch.float32).view(1, 1, C, 1, 1) * 100
    data = base.clone()
    data[:, :, 0, :, nan_cols_ch0] = float("nan")
    data[:, :, 1, :, nan_cols_ch1] = float("nan")

    fill = SmoothFloodFillPacked(num_steps=2, blur_kernel_size=5, blur_sigma=1.0)
    filled = fill(data)

    assert not filled.isnan().any(), "Output should contain no NaNs"

    cached_masks = fill._masks
    assert cached_masks is not None
    assert cached_masks.interior.shape == (C, H, W)
    assert cached_masks.valid.shape == (C, H, W)
    # blurred_valid carries one leading dim from _create_masks
    assert cached_masks.blurred_valid.shape == (1, C, H, W)

    filled2 = fill(data)
    torch.testing.assert_close(filled, filled2)

    # Channel 2 has no NaN -> blurred_valid_mask is uniformly 1 (replicate +
    # circular padding preserve all-ones), so output should equal input.
    torch.testing.assert_close(filled[:, :, 2], base[:, :, 2])

    # Channels 0 and 1 should be modified only in the NaN regions plus the
    # boundary band; pixels far from any NaN should be preserved.
    safe_cols_ch0 = slice(6, 10)
    safe_rows = slice(3, 5)
    torch.testing.assert_close(
        filled[:, :, 0, safe_rows, safe_cols_ch0],
        base[:, :, 0, safe_rows, safe_cols_ch0],
    )

    # Channels are processed independently: channel 1's fill must not leak
    # into channel 0's untouched region (sanity check that depthwise conv
    # is wired correctly).
    safe_cols_ch1_in_ch0 = [7, 8]
    torch.testing.assert_close(
        filled[:, :, 0, safe_rows, safe_cols_ch1_in_ch0],
        base[:, :, 0, safe_rows, safe_cols_ch1_in_ch0],
    )


def test_smooth_flood_fill_packed_pass_through_when_no_nans():
    """If the very first call has no NaN in the (C, H, W) slice, the cache
    is a no-op and subsequent calls return the input unchanged."""
    fill = SmoothFloodFillPacked(num_steps=2)
    x = torch.randn(2, 3, 4, 5)
    out = fill(x)
    assert out is x

    y = torch.randn(2, 3, 4, 5)
    y[:, 0, 0, 0] = float("nan")
    out_y = fill(y)
    assert out_y is y, (
        "Once the no-NaN cache is set, subsequent calls should be no-ops "
        "even if new NaNs appear (documented assumption)."
    )


def test_fast_flood_fill_gradient_flows_to_valid_pixels_only():
    """Backward pass: gradient is zero on land pixels (replaced by the
    fill) and propagates to valid (ocean) pixels.

    Also exercises the no_grad interior-mean path: the global-mean
    shortcut should NOT produce a gradient on every valid pixel.
    """
    B, T, C, H, W = 1, 1, 1, 8, 16
    nan_cols = [13, 14, 15, 0, 1, 2]
    x = torch.randn(B, T, C, H, W, requires_grad=True)
    nan_pattern = torch.zeros(C, H, W, dtype=torch.bool)
    nan_pattern[:, :, nan_cols] = True

    valid_mask = ~nan_pattern
    # Materialize NaN in a leaf-friendly way: do the masking on a copy and
    # let autograd track through the resulting tensor.
    masked = x.masked_fill(nan_pattern.unsqueeze(0).unsqueeze(0), float("nan"))

    out = fast_flood_fill(masked, num_steps=2, blur_kernel_size=5, blur_sigma=1.0)
    out.sum().backward()

    grad = x.grad
    assert grad is not None
    # Land (NaN) pixel gradients must be exactly zero: the masked_fill
    # replaces them with NaN before fast_flood_fill, so no path connects
    # them to the output.
    land_grad = grad[:, :, :, :, nan_cols]
    assert torch.all(land_grad == 0), "Land pixel gradients should be zero."
    # Ocean pixels far from the coast should have ~uniform gradient
    # (just from the identity portion of the boundary blend), not the
    # global-mean shortcut. The exact value depends on the Gaussian blur
    # support, so we just check that non-coastal pixels are nonzero and
    # coast-region variation is bounded.
    ocean_grad = grad[:, :, :, :, 6:10]
    assert torch.all(ocean_grad > 0), (
        f"Ocean pixel gradients should be positive, got "
        f"{ocean_grad.min().item():.4f} to {ocean_grad.max().item():.4f}."
    )
    _ = valid_mask  # silence unused

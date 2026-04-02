import torch

from fme.core.fill import SmoothFloodFill


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

from pathlib import Path

import pytest
import torch

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor_dict
from fme.downscaling.modules.preconditioners import EDMPrecond
from fme.downscaling.modules.unets import SongUNet


def test_edmprecond_regression():
    """Regression test for EDMPrecond output values."""
    device = get_device()
    if device.type == "cuda":
        pytest.skip("Skipping regression test on CUDA to avoid non-determinism issues")

    torch.manual_seed(0)

    # Create inner model (SongUNet)
    inner_model = SongUNet(
        img_resolution=16,
        in_channels=6,  # 3 for x + 3 for condition (concatenated)
        out_channels=3,
        model_channels=4,
        channel_mult=[1, 2],
        num_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
    )

    # Wrap with EDMPrecond
    model = EDMPrecond(
        model=inner_model,
        sigma_data=1.0,
    ).to(device)
    model.eval()

    # Deterministic inputs
    B, C, H, W = 2, 3, 16, 16
    x = (
        torch.arange(B * C * H * W, device=device, dtype=torch.float32).reshape(
            B, C, H, W
        )
        / 1000.0
    )
    condition = (
        torch.arange(B * C * H * W, device=device, dtype=torch.float32).reshape(
            B, C, H, W
        )
        / 500.0
    )
    sigma = (
        torch.arange(B, device=device, dtype=torch.float32).reshape(B, 1, 1, 1) / 10.0
        + 0.1
    )

    with torch.no_grad():
        output = model(x, condition, sigma)

    assert output.shape == (B, C, H, W)

    # Regression check
    reg_dir = Path(__file__).parent / "testdata"
    reg_dir.mkdir(parents=True, exist_ok=True)

    validate_tensor_dict(
        {"output": output},
        str(reg_dir / "test_edmprecond_regression.pt"),
    )

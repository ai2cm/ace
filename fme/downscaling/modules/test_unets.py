from pathlib import Path

import pytest
import torch

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor_dict
from fme.downscaling.modules.unets import (
    SongUNet,
    check_level_compatibility,
    validate_shape,
)


@pytest.mark.parametrize(
    "channel_mult, attn_res, passes",
    [
        pytest.param([1], [], True, id="no_division"),
        pytest.param([1] * 3, [], True, id="max_division"),
        pytest.param([1] * 4, [], False, id="too_many_division"),
        pytest.param([1, 1], [], True, id="no_attn_res"),
        pytest.param([1, 1], [2], True, id="single_attn_res"),
        pytest.param([1, 1], [2, 4], True, id="multiple_attn_res"),
        pytest.param([1, 1], [1], False, id="single_attn_res_missing"),
        pytest.param([1, 1], [1, 2], False, id="multiple_attn_res_missing"),
        pytest.param([1, 1], [1, 1], False, id="duplicated_attn"),
    ],
)
def test_check_level_compatibility(channel_mult, attn_res, passes):
    img_res = 4
    if passes:
        check_level_compatibility(img_res, channel_mult, attn_res)
    else:
        with pytest.raises(ValueError):
            check_level_compatibility(img_res, channel_mult, attn_res)


@pytest.mark.parametrize("shape", [(6, 6), (6, 8), (8, 6)])
def test_validate_shape(shape):
    # UNet only downsamples for depths 2 and greater
    # 6 -> 6 -> 3, any more than 3 levels should fail
    validate_shape(shape, 1)
    validate_shape(shape, 2)

    with pytest.raises(ValueError):
        validate_shape(shape, 3)


def test_songunet_regression():
    """Regression test for SongUNet output values."""
    device = get_device()
    if device.type == "cuda":
        pytest.skip("Skipping regression test on CUDA to avoid non-determinism issues")

    torch.manual_seed(0)

    model = SongUNet(
        img_resolution=16,
        in_channels=3,
        out_channels=3,
        model_channels=4,
        channel_mult=[1, 2, 2],
        num_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
    ).to(device)
    model.eval()

    # Deterministic inputs
    B, C, H, W = 2, 3, 16, 32
    x = (
        torch.arange(B * C * H * W, device=device, dtype=torch.float32).reshape(
            B, C, H, W
        )
        / 1000.0
    )
    noise_labels = torch.arange(B, device=device, dtype=torch.float32) / 10.0
    class_labels = torch.zeros(B, 0, device=device)

    with torch.no_grad():
        output = model(x, noise_labels, class_labels)

    assert output.shape == (B, C, H, W)

    # Regression check
    reg_dir = Path(__file__).parent / "testdata"
    reg_dir.mkdir(parents=True, exist_ok=True)

    validate_tensor_dict(
        {"output": output},
        str(reg_dir / "test_songunet_regression.pt"),
    )

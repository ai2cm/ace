from pathlib import Path

import pytest
import torch

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor_dict
from fme.downscaling.modules.physicsnemo_unets_v2.unets import SongUNetv2
from fme.downscaling.modules.physicsnemo_unets_v3.unets import (
    SongUNetv3,
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


def test_songunetv3_regression():
    """Regression test for SongUNetv3 output values."""
    device = get_device()
    if device.type == "cuda":
        pytest.skip("Skipping regression test on CUDA to avoid non-determinism issues")

    torch.manual_seed(0)

    model = SongUNetv3(
        img_resolution=16,
        in_channels=3,
        out_channels=3,
        model_channels=4,
        channel_mult=[1, 2, 2],
        num_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
        use_apex_gn=False,
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
        str(reg_dir / "test_songunetv3_regression.pt"),
    )


def test_songunetv3_matches_songunetv2():
    """Verify SongUNetv3 produces the same outputs as SongUNetv2 given same weights.

    This validates that the optimizations in v3 are numerically equivalent to v2.
    """
    device = get_device()
    if device.type == "cuda":
        pytest.skip("Skipping equivalence test on CUDA to avoid non-determinism issues")

    torch.manual_seed(42)

    kwargs = dict(
        img_resolution=16,
        in_channels=3,
        out_channels=3,
        model_channels=4,
        channel_mult=[1, 2, 2],
        num_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
        use_apex_gn=False,
    )

    model_v2 = SongUNetv2(**kwargs).to(device)
    model_v2.eval()

    # Create v3 model and copy weights from v2
    torch.manual_seed(42)
    model_v3 = SongUNetv3(**kwargs).to(device)
    model_v3.eval()

    # Copy state dict from v2 to v3 (they should have compatible keys)
    v2_state = model_v2.state_dict()
    v3_state = model_v3.state_dict()

    # Filter out buffers that differ due to pre-computation (tiled_resample_filter)
    # and pre-computed frequency buffers. Copy only shared parameter keys.
    shared_keys = set(v2_state.keys()) & set(v3_state.keys())
    load_dict = {k: v2_state[k] for k in shared_keys}
    # For keys only in v3, keep the existing values (pre-computed buffers)
    for k in v3_state.keys():
        if k not in load_dict:
            load_dict[k] = v3_state[k]
    model_v3.load_state_dict(load_dict)

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
        output_v2 = model_v2(x, noise_labels, class_labels)
        output_v3 = model_v3(x, noise_labels, class_labels)

    torch.testing.assert_close(output_v3, output_v2, atol=1e-5, rtol=1e-5)

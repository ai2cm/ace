import pytest
import torch

from fme.core.device import get_device
from fme.downscaling.modules.physicsnemo_unets_v2.unets import SongUNetv2
from fme.downscaling.modules.physicsnemo_unets_v3.unets import SongUNetv3


_COMMON_KWARGS = dict(
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


def test_songunetv3_without_compile():
    """SongUNetv3(compile_model=False) produces the same output as SongUNetv2."""
    device = get_device()
    if device.type == "cuda":
        pytest.skip(
            "Skipping equivalence test on CUDA to avoid non-determinism issues"
        )

    torch.manual_seed(42)
    model_v2 = SongUNetv2(**_COMMON_KWARGS).to(device)
    model_v2.eval()

    torch.manual_seed(42)
    model_v3 = SongUNetv3(**_COMMON_KWARGS, compile_model=False).to(device)
    model_v3.eval()

    # v3 inherits v2 exactly, so state dicts should be identical
    model_v3.load_state_dict(model_v2.state_dict())

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


@pytest.mark.slow
def test_songunetv3_with_compile():
    """SongUNetv3(compile_model=True) runs without error.

    Compilation can be slow on CPU, so this test is marked slow.
    We use a very small model to keep compile time manageable.
    """
    device = get_device()

    torch.manual_seed(0)
    model = SongUNetv3(
        img_resolution=8,
        in_channels=2,
        out_channels=2,
        model_channels=4,
        channel_mult=[1, 2],
        num_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
        use_apex_gn=False,
        compile_model=True,
    ).to(device)
    model.eval()

    B, C, H, W = 1, 2, 8, 8
    x = torch.randn(B, C, H, W, device=device)
    noise_labels = torch.rand(B, device=device)
    class_labels = torch.zeros(B, 0, device=device)

    with torch.no_grad():
        output = model(x, noise_labels, class_labels)

    assert output.shape == (B, C, H, W)

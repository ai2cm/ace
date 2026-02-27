import torch
from torch import nn

from fme.core.models.conditional_sfno.lora import LoRAConv2d


def test_lora_conv2d_load_conv2d_checkpoint():
    conv = nn.Conv2d(8, 16, 3, padding=1)
    lora = LoRAConv2d(8, 16, 3, padding=1)  # default should not use/require lora

    lora.load_state_dict(conv.state_dict(), strict=True)

    x = torch.randn(2, 8, 32, 32)
    with torch.no_grad():
        y0 = conv(x)
        y1 = lora(x)
    torch.testing.assert_close(
        y0,
        y1,
        atol=1e-6,
        rtol=0,
        msg="Outputs do not match after loading Conv2d checkpoint",
    )

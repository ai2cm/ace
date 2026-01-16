import torch

from fme.core.gridded_ops import LatLonOperations
from fme.core.models.conditional_sfno.s2convolutions import SpectralConvS2


def test_spectral_conv_s2_lora():
    in_channels = 8
    out_channels = in_channels
    n_lat = 12
    n_lon = 24
    operations = LatLonOperations(
        area_weights=torch.ones(n_lat, n_lon),
        grid="legendre-gauss",
    )
    sht = operations.get_real_sht()
    isht = operations.get_real_isht()

    conv1 = SpectralConvS2(
        forward_transform=sht,
        inverse_transform=isht,
        in_channels=in_channels,
        out_channels=out_channels,
        operator_type="dhconv",
        use_tensorly=False,
    )
    assert conv1.lora_A is None
    assert conv1.lora_B is None
    conv2 = SpectralConvS2(
        forward_transform=sht,
        inverse_transform=isht,
        in_channels=in_channels,
        out_channels=out_channels,
        operator_type="dhconv",
        use_tensorly=False,
        lora_rank=4,
        lora_alpha=8,
    )
    assert conv2.lora_A is not None
    assert conv2.lora_B is not None

    conv2.load_state_dict(conv1.state_dict(), strict=False)
    x = torch.randn(2, in_channels, n_lat, n_lon)
    y1, residual1 = conv1(x)
    y2, residual2 = conv2(x)

    # initial outputs should be identical since LoRA starts at 0
    assert torch.allclose(y1, y2, atol=1e-6)
    assert torch.allclose(residual1, residual2, atol=1e-6)

import pytest
import torch

from fme.downscaling.modules.physicsnemo_unets_v2.layers import Conv2d
from fme.downscaling.modules.physicsnemo_unets_v2.unets import SongUNetv2


def test_periodic_conv2d_is_circular_shift_equivariant_in_w():
    conv = Conv2d(in_channels=2, out_channels=2, kernel=3, periodic=True)
    x = torch.randn(1, 2, 6, 8)
    x_rolled = torch.roll(x, shifts=3, dims=-1)

    y = conv(x)
    y_from_rolled = conv(x_rolled)

    torch.testing.assert_close(y_from_rolled, torch.roll(y, shifts=3, dims=-1))


def test_periodic_conv2d_up_down_resample_guard():
    """The main (odd-kernel) conv branch implements circular padding, but the
    up/down resample-filter branches do not. With the default 2-tap
    resample_filter ([1, 1], f_pad=0) that branch never pads, so the gap is
    inert -- this locks in the guard that fails loudly if a wider filter
    (f_pad > 0) is ever combined with periodic=True, instead of silently
    reintroducing a zero-padding discontinuity at the antimeridian.
    """
    conv = Conv2d(
        in_channels=2,
        out_channels=2,
        kernel=3,
        down=True,
        resample_filter=[1, 3, 3, 1],
        periodic=True,
    )
    with pytest.raises(NotImplementedError):
        conv(torch.randn(1, 2, 8, 8))


def test_periodic_songunet_is_circular_shift_equivariant_in_w():
    """End-to-end check that setting periodic=True on every Conv2d in a
    SongUNetv2 (as VideoSongUNet does) makes the whole network -- including
    its up/down-sampling blocks -- circular-shift equivariant in longitude,
    not just the main conv branch tested in isolation above.
    """
    torch.manual_seed(0)
    net = SongUNetv2(
        img_resolution=[16, 16],
        in_channels=3,
        out_channels=3,
        model_channels=8,
        channel_mult=[1, 2],
        num_blocks=1,
        attn_resolutions=[],
        use_apex_gn=False,
    )
    for m in net.modules():
        if isinstance(m, Conv2d):
            m.periodic = True
    net.eval()

    x = torch.randn(1, 3, 16, 16)
    shift = 4  # multiple of the deepest stride (2) to stay phase-aligned
    x_rolled = torch.roll(x, shifts=shift, dims=-1)

    with torch.no_grad():
        y = net(x, noise_labels=torch.zeros(1), class_labels=None)
        y_from_rolled = net(x_rolled, noise_labels=torch.zeros(1), class_labels=None)

    torch.testing.assert_close(
        y_from_rolled, torch.roll(y, shifts=shift, dims=-1), atol=1e-5, rtol=1e-5
    )

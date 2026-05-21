import dataclasses
import datetime
import logging

import numpy as np
import pytest
import torch as th
import torch.nn as nn

from fme.ace.models.healpix.healpix_activations import CappedGELUConfig
from fme.ace.models.healpix.healpix_blocks import (
    ConvBlockConfig,
    DealiasedDownsample,
    DownsamplingBlockConfig,
    SmoothedInterpolateConv,
    UpsamplingBlockConfig,
)
from fme.ace.models.healpix.healpix_decoder import UNetDecoder
from fme.ace.models.healpix.healpix_encoder import UNetEncoder
from fme.ace.models.healpix.healpix_layers import (
    HEALPixLayer,
    HEALPixPadding,
    HEALPixPaddingIsolatitude,
    HEALPixPaddingv2,
    have_earth2grid,
)
from fme.ace.models.healpix.healpix_paddings import (
    isolatitude_pad_folded,
    make_hpx_padding_layer,
)
from fme.ace.models.healpix.healpix_unet import HEALPixUNet
from fme.ace.registry.hpx import UNetDecoderConfig, UNetEncoderConfig
from fme.ace.stepper import StepperConfig
from fme.core.coordinates import HEALPixCoordinates, HybridSigmaPressureCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector

TIMESTEP = datetime.timedelta(hours=6)
logger = logging.getLogger("__name__")


def fix_random_seeds(seed=0):
    """Fix random seeds for reproducibility"""
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def conv_next_block_config(in_channels=3, out_channels=1):
    activation_block_config = CappedGELUConfig(cap_value=10)
    conv_next_block_config = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation_block_config,
        kernel_size=3,
        dilation=1,
        upscale_factor=4,
        block_type="ConvNeXtBlock",
    )
    return conv_next_block_config


def down_sampling_block_config():
    return DownsamplingBlockConfig(pooling=2, block_type="AvgPool")


def encoder_config(
    conv_next_block_config, down_sampling_block_config, n_channels=[136, 68, 34]
):
    return UNetEncoderConfig(
        conv_block=conv_next_block_config,
        down_sampling_block=down_sampling_block_config,
        n_channels=n_channels,
        dilations=[1, 2, 4],
    )


def up_sampling_block_config(in_channels=3, out_channels=1):
    activation_block_config = CappedGELUConfig(cap_value=10)
    return UpsamplingBlockConfig(
        block_type="TransposedConvUpsample",
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation_block_config,
        stride=2,
    )


def output_layer_config(in_channels=3, out_channels=2):
    conv_block_config = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        dilation=1,
        n_layers=1,
        block_type="BasicConvBlock",
    )
    return conv_block_config


def decoder_config(
    conv_next_block_config,
    up_sampling_block_config,
    output_layer_config,
    n_channels=[34, 68, 136],
):
    decoder_config = UNetDecoderConfig(
        conv_block=conv_next_block_config,
        up_sampling_block=up_sampling_block_config,
        output_layer=output_layer_config,
        n_channels=n_channels,
        dilations=[4, 2, 1],
    )
    return decoder_config


def _nside_levels(shallow: int, n_levels: int) -> list[int]:
    """Face height/width per UNet level, shallowest to deepest."""
    return [max(1, shallow // (2**i)) for i in range(n_levels)]


def _hpx_unet_configs(
    img: int = 16,
    encoder_n_channels: list[int] | None = None,
    decoder_n_channels: list[int] | None = None,
    output_channels: int = 4,
    padding_mode: str = "karlbauer",
):
    """Build minimal encoder/decoder configs for HEALPixUNet tests."""
    if encoder_n_channels is None:
        encoder_n_channels = [8, 16]
    if decoder_n_channels is None:
        decoder_n_channels = list(reversed(encoder_n_channels))
    n_levels = len(encoder_n_channels)
    levels = _nside_levels(img, n_levels)

    enc_conv = ConvBlockConfig(
        block_type="ConvNeXtBlock",
        latent_channels=4,
        hpx_padding_mode=padding_mode,
        nside=img,
    )
    down = DownsamplingBlockConfig(
        block_type="AvgPool",
        pooling=2,
        hpx_padding_mode=padding_mode,
        nside=img,
    )
    enc = UNetEncoderConfig(
        conv_block=enc_conv,
        down_sampling_block=down,
        input_channels=3,
        n_channels=encoder_n_channels,
        n_layers=[1] * len(encoder_n_channels),
        nside=levels,
        hpx_padding_mode=padding_mode,
    )
    dec_conv = ConvBlockConfig(
        block_type="ConvNeXtBlock",
        latent_channels=4,
        hpx_padding_mode=padding_mode,
        nside=img,
    )
    dec = UNetDecoderConfig(
        conv_block=dec_conv,
        up_sampling_block=UpsamplingBlockConfig(
            block_type="TransposedConvUpsample",
            stride=2,
            hpx_padding_mode=padding_mode,
            nside=img // 2,
        ),
        output_layer=ConvBlockConfig(
            block_type="BasicConvBlock",
            n_layers=1,
            kernel_size=1,
            out_channels=output_channels,
            hpx_padding_mode=padding_mode,
            nside=img,
        ),
        n_channels=decoder_n_channels,
        n_layers=[1] * len(decoder_n_channels),
        output_channels=output_channels,
        hpx_padding_mode=padding_mode,
        nside=levels,
    )
    return enc, dec


def _test_data():
    # create dummy data
    def generate_test_data(batch_size=8, time_dim=1, channels=7, img_size=16):
        device = get_device()
        test_data = th.randn(batch_size, 12, time_dim * channels, img_size, img_size)
        return test_data.to(device)

    return generate_test_data


def constant_data():
    # create dummy data
    def generate_constant_data(channels=2, img_size=16):
        device = get_device()
        constants = th.randn(12, channels, img_size, img_size)

        return constants.to(device)

    return generate_constant_data


def insolation_data():
    # create dummy data
    def generate_insolation_data(batch_size=8, time_dim=1, img_size=16):
        device = get_device()
        insolation = th.randn(batch_size, 12, time_dim, img_size, img_size)

        return insolation.to(device)

    return generate_insolation_data


@pytest.mark.parametrize("shape", [pytest.param((8, 16))])
def test_hpx_init(shape):
    device = get_device()

    conv_next_block = conv_next_block_config()
    down_sampling_block = down_sampling_block_config()
    encoder = encoder_config(conv_next_block, down_sampling_block)
    up_sampling_block = up_sampling_block_config()
    output_layer = output_layer_config()
    decoder = decoder_config(conv_next_block, up_sampling_block, output_layer)

    hpx_config_data = {
        "encoder": dataclasses.asdict(encoder),
        "decoder": dataclasses.asdict(decoder),
    }

    horizontal_coordinates = HEALPixCoordinates(
        th.arange(12), th.arange(8), th.arange(8)
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=th.arange(7), bk=th.arange(7)
    ).to(device)
    stepper_config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="HEALPixUNet", config=hpx_config_data
                    ),
                    in_names=["x"],
                    out_names=["x"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={"x": float(np.random.randn(1).item())},
                            stds={"x": float(np.random.randn(1).item())},
                        ),
                    ),
                ),
            ),
        ),
    )
    stepper = stepper_config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=horizontal_coordinates,
            vertical_coordinate=vertical_coordinate,
            timestep=TIMESTEP,
        ),
    )
    assert len(stepper.modules) == 1
    assert type(stepper.modules[0].module) is HEALPixUNet


# pragma mark - encoder


def test_UNetEncoder_initialize():
    device = get_device()
    channels = 2
    n_channels = (16, 32, 64)

    # Dicts for block configs used by encoder
    conv_block_config = ConvBlockConfig(
        in_channels=channels,
        block_type="ConvNeXtBlock",
    )
    down_sampling_block_config = DownsamplingBlockConfig(
        pooling=2, block_type="MaxPool"
    )

    encoder = UNetEncoder(
        conv_block=conv_block_config,
        down_sampling_block=down_sampling_block_config,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)
    assert isinstance(encoder, UNetEncoder)

    # with dilations
    encoder = UNetEncoder(
        conv_block=conv_block_config,
        down_sampling_block=down_sampling_block_config,
        n_channels=n_channels,
        input_channels=channels,
        dilations=[1, 1, 1],
    ).to(device)
    assert isinstance(encoder, UNetEncoder)


def test_UNetEncoder_forward():
    channels = 2
    hw_size = 16
    b_size = 12
    n_channels = (16, 32, 64)
    device = get_device()

    # block configs used by encoder
    conv_block_config = ConvBlockConfig(
        in_channels=channels,
        block_type="ConvNeXtBlock",
    )
    down_sampling_block_config = DownsamplingBlockConfig(
        pooling=2, block_type="MaxPool"
    )
    encoder = UNetEncoder(
        conv_block=conv_block_config,
        down_sampling_block=down_sampling_block_config,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)

    tensor_size = [b_size, channels, hw_size, hw_size]
    invar = th.rand(tensor_size).to(device)
    outvar = encoder(invar)

    # outvar is a module list
    for idx, out_tensor in enumerate(outvar):
        # verify the channels and h dim are correct
        assert out_tensor.shape[1] == n_channels[idx]
        # default behaviour is to half the h/w size after first
        assert out_tensor.shape[2] == tensor_size[2] // (2**idx)


def test_UNetDecoder_initilization():
    in_channels = 2
    out_channels = 1
    n_channels = (64, 32, 16)
    device = get_device()

    # Dicts for block configs used by decoder
    conv_block_config = ConvBlockConfig(
        in_channels=in_channels, out_channels=out_channels, block_type="ConvNeXtBlock"
    )
    up_sampling_block_config = UpsamplingBlockConfig(
        block_type="TransposedConvUpsample",
        in_channels=in_channels,
        out_channels=out_channels,
        stride=2,
    )

    output_layer_config = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        dilation=1,
        n_layers=1,
        block_type="ConvNeXtBlock",
    )

    decoder = UNetDecoder(
        conv_block=conv_block_config,
        up_sampling_block=up_sampling_block_config,
        output_layer=output_layer_config,
        n_channels=n_channels,
    ).to(device)

    assert isinstance(decoder, UNetDecoder)

    decoder = UNetDecoder(
        conv_block=conv_block_config,
        up_sampling_block=up_sampling_block_config,
        output_layer=output_layer_config,
        n_channels=n_channels,
        dilations=[1, 1, 1],
    ).to(device)
    assert isinstance(decoder, UNetDecoder)


def test_UNetDecoder_forward():
    in_channels = 2
    out_channels = 1
    hw_size = 32
    b_size = 12
    n_channels = (64, 32, 16)
    device = get_device()

    # Dicts for block configs used by decoder
    conv_block_config = ConvBlockConfig(
        in_channels=in_channels, out_channels=out_channels, block_type="ConvNeXtBlock"
    )
    up_sampling_block_config = UpsamplingBlockConfig(
        block_type="TransposedConvUpsample",
        in_channels=in_channels,
        out_channels=out_channels,
        stride=2,
    )
    output_layer_config = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        dilation=1,
        n_layers=1,
        block_type="BasicConvBlock",
    )
    decoder = UNetDecoder(
        conv_block=conv_block_config,
        up_sampling_block=up_sampling_block_config,
        output_layer=output_layer_config,
        n_channels=n_channels,
    ).to(device)

    output_2_size = th.Size([b_size, out_channels, hw_size, hw_size])

    # build the list of tensors for the decoder
    invars = []
    # decoder has an algorithm that goes back to front
    for idx in range(len(n_channels) - 1, -1, -1):
        tensor_size = [b_size, n_channels[idx], hw_size, hw_size]
        invars.append(th.rand(tensor_size).to(device))
        hw_size = hw_size // 2

    outvar = decoder(invars)
    assert outvar.shape == output_2_size

    outvar_repeat = decoder(invars)
    assert compare_output(outvar, outvar_repeat)

    decoder = UNetDecoder(
        conv_block=conv_block_config,
        up_sampling_block=up_sampling_block_config,
        output_layer=output_layer_config,
        n_channels=n_channels,
        dilations=[1, 1, 1],
    ).to(device)

    outvar = decoder(invars)
    assert outvar.shape == output_2_size


def compare_output(
    output_1: th.Tensor | tuple[th.Tensor, ...],
    output_2: th.Tensor | tuple[th.Tensor, ...],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Compares model outputs and returns if they are the same

    Args
        output_1: First item to compare
        output_2: Second item to compare
        rtol: Relative tolerance of error allowed, by default 1e-5
        atol: Absolute tolerance of error allowed, by default 1e-5

    Returns:
        If outputs are the same
    """
    # Output of tensor
    if isinstance(output_1, th.Tensor):
        return th.allclose(output_1, output_2, rtol, atol)
    # Output of tuple of tensors
    elif isinstance(output_1, tuple):
        # Loop through tuple of outputs
        for i, (out_1, out_2) in enumerate(zip(output_1, output_2)):
            # If tensor use allclose
            if isinstance(out_1, th.Tensor):
                if not th.allclose(out_1, out_2, rtol, atol):
                    logger.warning(f"Failed comparison between outputs {i}")
                    logger.warning(f"Max Difference: {th.amax(th.abs(out_1 - out_2))}")
                    logger.warning(f"Difference: {out_1 - out_2}")
                    return False
            # Otherwise assume primative
            else:
                if not out_1 == out_2:
                    return False
    elif isinstance(output_1, list | tuple) and isinstance(output_2, list | tuple):
        if len(output_1) != len(output_2):
            print(
                f"Length mismatch: output_1 {len(output_1)}, output_2 {len(output_2)}"
            )
            return False
        for a, e in zip(output_1, output_2):
            if not compare_output(a, e):
                return False
        return True
    elif isinstance(output_1, dict) and isinstance(output_2, dict):
        if output_1.keys() != output_2.keys():
            print(
                f"Keys mismatch: output_1 keys {output_1.keys()}, ",
                f"output_2 keys {output_2.keys()}",
            )
            return False
        for key in output_1:
            if not compare_output(output_1[key], output_2[key]):
                return False
        return True
    # Unsupported output type
    else:
        logger.error(
            "Model returned invalid type for unit test, \
            should be th.Tensor or Tuple[th.Tensor]"
        )
        return False
    return True


@pytest.fixture
def mock_data():
    """
    Create a mock tensor with known values for 12 HEALPix faces.
    Shape: [B=1, F=12, C=1, H=4, W=4] - a single batch with 12 faces, one channel,
        4x4 grid
    """
    return th.arange(1, 12 * 4 * 4 + 1, dtype=th.float32).reshape((12, 1, 4, 4))


@pytest.fixture
def healpix_padding():
    padding = 2

    """Instantiate HEALPixPadding with the specified padding."""
    return HEALPixPadding(padding=padding, enable_nhwc=False)


def test_healpix_padding_pn(healpix_padding):
    """
    Test north hemisphere padding.
    This test checks if each northern face is padded correctly with its neighbors.
    """
    padding = 2

    # Mock the neighbor faces, assuming they are 4x4 for simplicity.
    face = th.ones((1, 1, 4, 4))
    top = face
    top_left = face * 2
    left = face * 3
    bottom_left = face * 4
    bottom = face * 5
    bottom_right = face * 6
    right = face * 7
    top_right = face * 8

    # Run the pn function directly for controlled testing
    padded = healpix_padding.pn(
        face, top, top_left, left, bottom_left, bottom, bottom_right, right, top_right
    )

    # Check if padding applied matches expected size: 4 + 2*padding in each dimension
    assert padded.shape[-2:] == (4 + 2 * padding, 4 + 2 * padding)
    # North padding (top two rows) should match `top`
    assert th.all(padded[..., :padding, padding:-padding] == top[..., -padding:, :])
    # Northwest corner padding should match `top_left`
    assert th.all(
        padded[..., :padding, :padding] == top_left[..., -padding:, -padding:]
    )
    # Northeast corner padding should match `top_right`
    assert th.all(
        padded[..., :padding, -padding:] == top_right[..., -padding:, :padding]
    )


def test_healpix_padding_pe(healpix_padding):
    """
    Test equatorial face padding to ensure proper padding alignment with
    equatorial neighbors.
    """
    padding = 2

    # Mock the neighbor faces, assuming they are 4x4 for simplicity.
    face = th.ones((1, 1, 4, 4))
    top = face
    top_left = face * 2
    left = face * 3
    bottom_left = face * 4
    bottom = face * 5
    bottom_right = face * 6
    right = face * 7
    top_right = face * 8

    # Run the pe function directly for controlled testing
    padded = healpix_padding.pe(
        face, top, top_left, left, bottom_left, bottom, bottom_right, right, top_right
    )

    # Check if padding applied matches expected size: 4 + 2*padding in each dimension
    assert padded.shape[-2:] == (4 + 2 * padding, 4 + 2 * padding)
    # Left padding (left two columns) should match `left`
    assert th.all(padded[..., padding:-padding, :padding] == left[..., :, -padding:])
    # Right padding (right two columns) should match `right`
    assert th.all(padded[..., padding:-padding, -padding:] == right[..., :, :padding])


def test_healpix_padding_ps(healpix_padding):
    """
    Test south hemisphere padding.
    This test checks if each southern face is padded correctly with its neighbors.
    """
    padding = 2

    # Mock the neighbor faces, assuming they are 4x4 for simplicity.
    face = th.ones((1, 1, 4, 4))
    top = face
    top_left = face * 2
    left = face * 3
    bottom_left = face * 4
    bottom = face * 5
    bottom_right = face * 6
    right = face * 7
    top_right = face * 8

    # Run the ps function directly for controlled testing
    padded = healpix_padding.ps(
        face, top, top_left, left, bottom_left, bottom, bottom_right, right, top_right
    )

    # Check if padding applied matches expected size: 4 + 2*padding in each dimension
    assert padded.shape[-2:] == (4 + 2 * padding, 4 + 2 * padding)
    # South padding (bottom two rows) should match `bottom`
    assert th.all(padded[..., -padding:, padding:-padding] == bottom[..., :padding, :])
    # Southwest corner padding should match `bottom_left`
    assert th.all(
        padded[..., -padding:, :padding] == bottom_left[..., :padding, -padding:]
    )
    # Southeast corner padding should match `bottom_right`
    assert th.all(
        padded[..., -padding:, -padding:] == bottom_right[..., :padding, :padding]
    )


def test_healpix_padding_forward(healpix_padding, mock_data):
    """
    Full integration test for HEALPixPadding's forward method.
    Checks that all faces receive the correct padding.
    """
    padding = 2

    padded_data = healpix_padding(mock_data)

    expected_height = mock_data.shape[-2] + 2 * padding
    expected_width = mock_data.shape[-1] + 2 * padding
    assert padded_data.shape[-2:] == (expected_height, expected_width)

    for face_idx in range(mock_data.shape[0]):
        face_data = padded_data[face_idx, 0]

        # the region where original data should reside
        data_region = face_data[padding : padding + 4, padding : padding + 4]
        original_data = mock_data[face_idx, 0]  # Original face data
        assert (
            data_region == original_data
        ).all(), f"Data region on face {face_idx} has been altered."


# --- HEALPix padding modes + dealias / smoothed blocks ---


def test_HEALPixPaddingIsolatitude_initialization():
    pad = HEALPixPaddingIsolatitude(padding=2, nside=16)
    assert isinstance(pad, HEALPixPaddingIsolatitude)

    with pytest.raises(ValueError, match="invalid value for 'padding'"):
        HEALPixPaddingIsolatitude(padding=0, nside=16)
    with pytest.raises(ValueError, match="nside must be a positive int"):
        HEALPixPaddingIsolatitude(padding=1, nside=0)


@pytest.mark.parametrize("padding", [1, 2, 3, 4, 5])
def test_HEALPixPaddingIsolatitude_forward_shape_cpu(padding: int):
    """Folded layout [B*12, C, H, H] -> [B*12, C, H+2p, H+2p]."""
    num_faces = 12
    batch_size = 2
    hw = 16
    c = 4
    if 2 * padding > hw:
        pytest.skip("face size too small for padding (isolatitude corner synthesis)")

    pad_mod = HEALPixPaddingIsolatitude(padding=padding, nside=hw)
    invar = th.rand(batch_size * num_faces, c, hw, hw)
    outvar = pad_mod(invar)
    hw_p = hw + 2 * padding
    assert outvar.shape == (batch_size * num_faces, c, hw_p, hw_p)


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("padding", [1, 2, 3])
def test_HEALPixPaddingIsolatitude_forward_shape_cuda(padding: int):
    num_faces = 12
    batch_size = 1
    hw = 16
    c = 2
    if 2 * padding > hw:
        pytest.skip("face size too small for padding")
    th.cuda.empty_cache()
    pad_mod = HEALPixPaddingIsolatitude(padding=padding, nside=hw).cuda()
    invar = th.rand(batch_size * num_faces, c, hw, hw, device="cuda")
    outvar = pad_mod(invar)
    hw_p = hw + 2 * padding
    assert outvar.shape == (batch_size * num_faces, c, hw_p, hw_p)


@pytest.mark.parametrize("padding", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("hw", [16, 32, 64])
@pytest.mark.parametrize("enable_nhwc", [False, True])
def test_healpix_padding_isolatitude_matches_folded_reference(
    padding: int, hw: int, enable_nhwc: bool
):
    """Gather-based HEALPixPaddingIsolatitude must match isolatitude_pad_folded."""
    if 2 * padding > hw:
        pytest.skip("face size too small for padding (isolatitude corner synthesis)")

    th.manual_seed(0)
    batch_size = 2
    num_faces = 12
    c = 3
    x = th.randn(batch_size * num_faces, c, hw, hw)

    ref = isolatitude_pad_folded(x, padding, enable_nhwc)
    y = HEALPixPaddingIsolatitude(
        padding=padding, nside=hw, enable_nhwc=enable_nhwc
    )(x)

    # Gather path uses 0.5 * (g0 + g1) in a form that can differ by ~1 ULP from the
    # reference on some output cells.
    th.testing.assert_close(y, ref, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("hw", [8, 16])
@pytest.mark.parametrize("enable_nhwc", [False, True])
def test_healpix_padding_isolatitude_gradcheck_cpu(
    padding: int, hw: int, enable_nhwc: bool
):
    """Analytic backward matches finite differences (double precision)."""
    if 2 * padding > hw:
        pytest.skip("face size too small for padding")

    pad = HEALPixPaddingIsolatitude(
        padding=padding, nside=hw, enable_nhwc=enable_nhwc
    ).double()

    batch_size = 1
    c = 2
    x = th.randn(
        batch_size * 12,
        c,
        hw,
        hw,
        dtype=th.double,
        requires_grad=True,
    )

    def fn(t: th.Tensor) -> th.Tensor:
        return pad(t).sum()

    assert th.autograd.gradcheck(
        fn,
        (x,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("hw", [8, 16])
def test_healpix_padding_isolatitude_gradcheck_cuda(padding: int, hw: int):
    if 2 * padding > hw:
        pytest.skip("face size too small for padding")
    th.cuda.empty_cache()
    pad = HEALPixPaddingIsolatitude(padding=padding, nside=hw).double().cuda()
    x = th.randn(
        1 * 12,
        2,
        hw,
        hw,
        dtype=th.double,
        device="cuda",
        requires_grad=True,
    )

    def fn(t: th.Tensor) -> th.Tensor:
        return pad(t).sum()

    assert th.autograd.gradcheck(
        fn,
        (x,),
        eps=1e-5,
        atol=1e-3,
        rtol=1e-2,
    )


def test_healpix_padding_isolatitude_backward_grad_matches_finite_diff():
    """Spot-check gradient vs central difference on a few entries (float32, CPU)."""
    padding, hw = 1, 16
    th.manual_seed(42)
    pad = HEALPixPaddingIsolatitude(padding=padding, nside=hw)
    x0 = th.randn(12, 2, hw, hw)
    x = x0.clone().requires_grad_(True)
    (g_analytic,) = th.autograd.grad(pad(x).sum(), x)

    eps = 1e-3
    indices = [(0, 0, 0, 0), (3, 1, 7, 7), (11, 0, 15, 15), (5, 1, 8, 8)]
    for b, c, h, w in indices:
        xp = x0.clone()
        xp[b, c, h, w] += eps
        xm = x0.clone()
        xm[b, c, h, w] -= eps
        g_fd_ij = (pad(xp).sum() - pad(xm).sum()) / (2 * eps)
        th.testing.assert_close(
            g_analytic[b, c, h, w],
            g_fd_ij,
            rtol=0.02,
            atol=0.02,
        )


def _folded_padding_dealias(
    batch: int = 2, channels: int = 3, h: int = 16, device=None, dtype=th.float32
):
    if device is None:
        device = th.device("cpu")
    return th.randn(batch * 12, channels, h, h, device=device, dtype=dtype)


@pytest.mark.parametrize("mode", ["karlbauer", "isolatitude"])
def test_healpix_layer_conv_same_geometry(mode):
    h = 16
    x = _folded_padding_dealias(h=h)
    kwargs = dict(
        layer=nn.Conv2d,
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding="same",
        enable_nhwc=False,
        hpx_padding_mode=mode,
    )
    if mode == "isolatitude":
        kwargs["nside"] = h
    layer = HEALPixLayer(**kwargs)
    y = layer(x)
    assert y.shape == x.shape
    assert th.isfinite(y).all()


@pytest.mark.skipif(not th.cuda.is_available(), reason="earth2grid requires CUDA")
@pytest.mark.skipif(not have_earth2grid, reason="earth2grid not installed")
def test_healpix_layer_earth2grid():
    h = 16
    x = _folded_padding_dealias(h=h, device=th.device("cuda"))
    layer = HEALPixLayer(
        nn.Conv2d,
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding="same",
        hpx_padding_mode="earth2grid",
    ).cuda()
    y = layer(x)
    assert y.shape == x.shape
    assert th.isfinite(y).all()


def test_make_hpx_padding_factory_types():
    p = make_hpx_padding_layer(1, "karlbauer", enable_nhwc=False)
    assert p is not None
    p2 = make_hpx_padding_layer(1, "isolatitude", enable_nhwc=False, nside=8)
    assert p2 is not None


def test_healpix_layer_uses_mode_selected_padding_class():
    k_layer = HEALPixLayer(
        nn.Conv2d,
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        hpx_padding_mode="karlbauer",
    )
    assert isinstance(k_layer.layers[0], HEALPixPadding)

    i_layer = HEALPixLayer(
        nn.Conv2d,
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        hpx_padding_mode="isolatitude",
        nside=16,
    )
    assert isinstance(i_layer.layers[0], HEALPixPaddingIsolatitude)

    if have_earth2grid and th.cuda.is_available():
        e_layer = HEALPixLayer(
            nn.Conv2d,
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            hpx_padding_mode="earth2grid",
        )
        assert isinstance(e_layer.layers[0], HEALPixPaddingv2)


@pytest.mark.parametrize("mode", ["karlbauer", "isolatitude"])
def test_dealiased_downsample_forward(mode):
    h = 16
    x = _folded_padding_dealias(h=h)
    kwargs = dict(
        in_channels=3,
        stride=2,
        hpx_padding_mode=mode,
    )
    if mode == "isolatitude":
        kwargs["nside"] = h
    m = DealiasedDownsample(**kwargs)
    y = m(x)
    assert y.shape[-2:] == (h // 2, h // 2)


@pytest.mark.parametrize("mode", ["karlbauer", "isolatitude"])
def test_smoothed_interpolate_conv_forward(mode):
    h = 16
    x = _folded_padding_dealias(h=h)
    kwargs = dict(
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        scale_factor=2,
        mode="nearest",
        hpx_padding_mode=mode,
    )
    if mode == "isolatitude":
        kwargs["nside"] = h
        kwargs["nside_after"] = h * 2
    m = SmoothedInterpolateConv(**kwargs)
    y = m(x)
    assert y.shape[-2:] == (h * 2, h * 2)
    assert y.shape[1] == 5


def test_dealiased_downsample_config_uses_stride():
    cfg = DownsamplingBlockConfig(
        block_type="DealiasedDownsample",
        in_channels=3,
        stride=4,
        resample_filter=(1.0, 2.0, 1.0),
        hpx_padding_mode="karlbauer",
    )
    assert cfg.downsample_spatial_factor() == 4
    module = cfg.build()
    assert isinstance(module, DealiasedDownsample)


def test_dealiased_downsample_config_accepts_list_filter():
    cfg = DownsamplingBlockConfig(
        block_type="DealiasedDownsample",
        in_channels=3,
        stride=2,
        resample_filter=[1.0, 2.0, 1.0],
        hpx_padding_mode="karlbauer",
    )
    module = cfg.build()
    assert isinstance(module, DealiasedDownsample)


def test_smoothed_interpolate_conv_config_builds():
    cfg = UpsamplingBlockConfig(
        block_type="SmoothedInterpolateConv",
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        stride=2,
        upsample_mode="nearest",
        hpx_padding_mode="karlbauer",
    )
    module = cfg.build()
    assert isinstance(module, SmoothedInterpolateConv)


def test_healpix_unet_dealias_smoothed():
    img = 16
    conv = ConvBlockConfig(
        block_type="ConvNeXtBlock",
        latent_channels=4,
        hpx_padding_mode="karlbauer",
        nside=img,
    )
    down = DownsamplingBlockConfig(
        block_type="DealiasedDownsample",
        stride=2,
        hpx_padding_mode="karlbauer",
        nside=img,
    )
    levels = _nside_levels(img, 2)
    enc = UNetEncoderConfig(
        conv_block=conv,
        down_sampling_block=down,
        input_channels=3,
        n_channels=[8, 16],
        n_layers=[1, 1],
        nside=levels,
        hpx_padding_mode="karlbauer",
    )
    up = UpsamplingBlockConfig(
        block_type="SmoothedInterpolateConv",
        stride=2,
        upsample_mode="nearest",
        hpx_padding_mode="karlbauer",
        nside=levels[1],
        nside_after=levels[0],
    )
    dec = UNetDecoderConfig(
        conv_block=conv,
        up_sampling_block=up,
        output_layer=ConvBlockConfig(
            block_type="BasicConvBlock",
            n_layers=1,
            kernel_size=1,
            out_channels=4,
            hpx_padding_mode="karlbauer",
            nside=img,
        ),
        n_channels=[16, 8],
        n_layers=[1, 1],
        output_channels=4,
        hpx_padding_mode="karlbauer",
        nside=levels,
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # Channel count matches prior stacked layout: 2*(3+1) prognostic+decoder + 1 constant
    in_ch = 9
    m = HEALPixUNet(
        encoder=enc,
        decoder=dec,
        input_channels=in_ch,
        output_channels=4,
        hpx_padding_mode="karlbauer",
        nside=_nside_levels(img, len(enc.n_channels)),
    ).to(device)
    b = 2
    x = th.randn(b, 12, 2 * 3, img, img, device=device)
    dec_in = th.randn(b, 12, 2, img, img, device=device)
    const = th.randn(b, 12, 1, img, img, device=device)
    inp = th.cat([x, dec_in, const], dim=2)
    out = m(inp)
    assert out.shape[0] == b
    assert th.isfinite(out).all()


def test_multi_symmetric_convnext_block_forward():
    cfg = ConvBlockConfig(
        block_type="Multi_SymmetricConvNeXtBlock",
        in_channels=3,
        out_channels=5,
        latent_channels=4,
        kernel_size=3,
        dilation=1,
        upscale_factor=4,
        n_layers=2,
        hpx_padding_mode="karlbauer",
        nside=16,
    )
    layer = cfg.build()
    x = _folded_padding_dealias(channels=3, h=16)
    y = layer(x)
    assert y.shape == (x.shape[0], 5, 16, 16)
    assert th.isfinite(y).all()


def test_multi_symmetric_isolatitude_forward():
    img = 16
    conv = ConvBlockConfig(
        block_type="Multi_SymmetricConvNeXtBlock",
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        n_layers=2,
        hpx_padding_mode="isolatitude",
        nside=img,
        activation=CappedGELUConfig(cap_value=10),
    ).build()
    x = _folded_padding_dealias(channels=3, h=img)
    y = conv(x)
    assert y.shape == x.shape
    assert th.isfinite(y).all()


def test_smoothed_interpolate_isolatitude_forward():
    img = 16
    x = _folded_padding_dealias(channels=3, h=img)
    up = UpsamplingBlockConfig(
        block_type="SmoothedInterpolateConv",
        in_channels=3,
        out_channels=3,
        stride=2,
        upsample_mode="nearest",
        activation=CappedGELUConfig(cap_value=10),
        hpx_padding_mode="isolatitude",
        nside=img,
        nside_after=img * 2,
    ).build()
    y_up = up(x)
    assert y_up.shape[-2:] == (img * 2, img * 2)
    assert th.isfinite(y_up).all()


def test_healpix_unet_isolatitude_nside_sequence():
    conv_cfg = ConvBlockConfig(
        block_type="ConvNeXtBlock",
        in_channels=3,
        out_channels=3,
        latent_channels=2,
        kernel_size=3,
        dilation=1,
        activation=CappedGELUConfig(cap_value=10),
    )
    encoder = UNetEncoderConfig(
        conv_block=conv_cfg,
        down_sampling_block=DownsamplingBlockConfig(block_type="AvgPool", pooling=2),
        input_channels=5,
        n_channels=[8, 8, 8],
        n_layers=[1, 1, 1],
        dilations=[1, 1, 1],
    )
    decoder = UNetDecoderConfig(
        conv_block=ConvBlockConfig(
            block_type="ConvNeXtBlock",
            in_channels=3,
            out_channels=3,
            latent_channels=2,
            kernel_size=3,
            dilation=1,
            activation=CappedGELUConfig(cap_value=10),
        ),
        up_sampling_block=UpsamplingBlockConfig(
            block_type="TransposedConvUpsample",
            in_channels=3,
            out_channels=3,
            stride=2,
            activation=CappedGELUConfig(cap_value=10),
        ),
        output_layer=ConvBlockConfig(
            block_type="BasicConvBlock",
            in_channels=3,
            out_channels=4,
            kernel_size=1,
            n_layers=1,
        ),
        n_channels=[8, 8, 8],
        n_layers=[1, 1, 1],
        output_channels=4,
        dilations=[1, 1, 1],
    )
    model = HEALPixUNet(
        encoder=encoder,
        decoder=decoder,
        input_channels=5,
        output_channels=4,
        hpx_padding_mode="isolatitude",
        nside=[64, 32, 16],
    )
    x = th.randn(1, 12, 5, 64, 64)
    y = model(x)
    assert y.shape == (1, 12, 4, 64, 64)
    assert th.isfinite(y).all()


# pragma mark - HEALPixUNet


def test_HEALPixUNet_initialize():
    img = 16
    in_channels = 5
    out_channels = 4
    enc, dec = _hpx_unet_configs(img=img, output_channels=out_channels)
    device = get_device()

    model = HEALPixUNet(
        encoder=enc,
        decoder=dec,
        input_channels=in_channels,
        output_channels=out_channels,
        hpx_padding_mode="karlbauer",
        nside=_nside_levels(img, len(enc.n_channels)),
    ).to(device)
    assert isinstance(model, HEALPixUNet)
    for layer in model.decoder.decoder:
        assert set(layer.keys()) == {"upsamp", "conv"}


def test_HEALPixUNet_forward_shape():
    img = 16
    in_channels = 5
    out_channels = 4
    batch = 2
    enc, dec = _hpx_unet_configs(img=img, output_channels=out_channels)
    device = get_device()

    model = HEALPixUNet(
        encoder=enc,
        decoder=dec,
        input_channels=in_channels,
        output_channels=out_channels,
        hpx_padding_mode="karlbauer",
        nside=_nside_levels(img, len(enc.n_channels)),
    ).to(device)

    x = th.randn(batch, 12, in_channels, img, img, device=device)
    y = model(x)
    assert y.shape == (batch, 12, out_channels, img, img)
    assert th.isfinite(y).all()


def test_HEALPixUNet_input_channel_validation():
    img = 16
    in_channels = 5
    out_channels = 4
    enc, dec = _hpx_unet_configs(img=img, output_channels=out_channels)
    device = get_device()

    model = HEALPixUNet(
        encoder=enc,
        decoder=dec,
        input_channels=in_channels,
        output_channels=out_channels,
        hpx_padding_mode="karlbauer",
        nside=_nside_levels(img, len(enc.n_channels)),
    ).to(device)

    bad_input = th.randn(1, 12, in_channels + 1, img, img, device=device)
    with pytest.raises(ValueError, match=f"Expected input to have {in_channels} channels"):
        model(bad_input)

    bad_ndim = th.randn(1, 12, in_channels, img, img, img, device=device)
    with pytest.raises(ValueError, match="5D input"):
        model(bad_ndim)


@pytest.mark.parametrize("mode", ["karlbauer", "isolatitude"])
def test_HEALPixUNet_forward_padding_mode(mode):
    img = 16
    in_channels = 5
    out_channels = 4
    batch = 2
    enc, dec = _hpx_unet_configs(
        img=img, output_channels=out_channels, padding_mode=mode
    )

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = HEALPixUNet(
        encoder=enc,
        decoder=dec,
        input_channels=in_channels,
        output_channels=out_channels,
        hpx_padding_mode=mode,
        nside=_nside_levels(img, len(enc.n_channels)),
    ).to(device)

    x = th.randn(batch, 12, in_channels, img, img, device=device)
    y = model(x)
    assert y.shape == (batch, 12, out_channels, img, img)
    assert th.isfinite(y).all()


def test_HEALPixUNet_in_stepper():
    """End-to-end build of a HEALPixUNet through the stepper config."""
    in_channels = 3
    out_channels = 3
    img = 8
    encoder = encoder_config(
        conv_next_block_config(),
        down_sampling_block_config(),
        n_channels=[16, 32, 64],
    )
    decoder = UNetDecoderConfig(
        conv_block=conv_next_block_config(),
        up_sampling_block=up_sampling_block_config(),
        output_layer=output_layer_config(),
        n_channels=[64, 32, 16],
        dilations=[4, 2, 1],
    )

    hpx_unet_config_data = {
        "encoder": dataclasses.asdict(encoder),
        "decoder": dataclasses.asdict(decoder),
        "nside": _nside_levels(img, len(encoder.n_channels)),
    }

    horizontal_coordinates = HEALPixCoordinates(
        th.arange(12), th.arange(img), th.arange(img)
    )
    device = get_device()
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=th.arange(in_channels), bk=th.arange(in_channels)
    ).to(device)
    stepper_config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="HEALPixUNet", config=hpx_unet_config_data
                    ),
                    in_names=["x"],
                    out_names=["x"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={"x": float(np.random.randn(1).item())},
                            stds={"x": float(np.random.randn(1).item())},
                        ),
                    ),
                ),
            ),
        ),
    )
    stepper = stepper_config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=horizontal_coordinates,
            vertical_coordinate=vertical_coordinate,
            timestep=TIMESTEP,
        ),
    )
    assert len(stepper.modules) == 1
    assert type(stepper.modules[0].module) is HEALPixUNet

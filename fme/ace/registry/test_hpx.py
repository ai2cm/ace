import dataclasses
import datetime
import logging

import numpy as np
import pytest
import torch as th

from fme.ace.models.healpix.healpix_activations import (
    CappedGELUConfig,
    DownsamplingBlockConfig,
)
from fme.ace.models.healpix.healpix_blocks import ConvBlockConfig, RecurrentBlockConfig
from fme.ace.models.healpix.healpix_decoder import UNetDecoder
from fme.ace.models.healpix.healpix_encoder import UNetEncoder
from fme.ace.models.healpix.healpix_layers import HEALPixPadding
from fme.ace.models.healpix.healpix_recunet import HEALPixRecUNet
from fme.ace.registry.hpx import UNetDecoderConfig, UNetEncoderConfig
from fme.ace.stepper import SingleModuleStepperConfig
from fme.core.coordinates import HEALPixCoordinates, HybridSigmaPressureCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.normalizer import NormalizationConfig

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
    transposed_conv_upsample_block_config = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation_block_config,
        stride=2,
        block_type="TransposedConvUpsample",
    )
    return transposed_conv_upsample_block_config


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


def recurrent_block_config(in_channels=3):
    recurrent_block_config = RecurrentBlockConfig(
        in_channels=in_channels,
        kernel_size=1,
        block_type="ConvGRUBlock",
    )
    return recurrent_block_config


def decoder_config(
    conv_next_block_config,
    up_sampling_block_config,
    output_layer_config,
    recurrent_block_config,
    n_channels=[34, 68, 136],
):
    decoder_config = UNetDecoderConfig(
        conv_block=conv_next_block_config,
        up_sampling_block=up_sampling_block_config,
        recurrent_block=recurrent_block_config,
        output_layer=output_layer_config,
        n_channels=n_channels,
        dilations=[4, 2, 1],
    )
    return decoder_config


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
    in_channels = 7
    out_channels = 7
    prognostic_variables = min(in_channels, out_channels)
    n_constants = 1
    decoder_input_channels = 1
    input_time_size = 2
    output_time_size = 4
    device = get_device()

    conv_next_block = conv_next_block_config()
    down_sampling_block = down_sampling_block_config()
    recurrent_block = recurrent_block_config()
    encoder = encoder_config(conv_next_block, down_sampling_block)
    up_sampling_block = up_sampling_block_config()
    output_layer = output_layer_config()
    decoder = decoder_config(
        conv_next_block, up_sampling_block, output_layer, recurrent_block
    )

    hpx_config_data = {
        "type": "HEALPixRecUNet",
        "config": {
            "encoder": dataclasses.asdict(encoder),
            "decoder": dataclasses.asdict(decoder),
            "prognostic_variables": prognostic_variables,
            "n_constants": n_constants,
            "decoder_input_channels": decoder_input_channels,
            "input_time_size": input_time_size,
            "output_time_size": output_time_size,
        },
    }

    stepper_config_data = {
        "builder": hpx_config_data,
        "in_names": ["x"],
        "out_names": ["x"],
        "normalization": dataclasses.asdict(
            NormalizationConfig(
                means={"x": float(np.random.randn(1).item())},
                stds={"x": float(np.random.randn(1).item())},
            )
        ),
    }
    horizontal_coordinates = HEALPixCoordinates(
        th.arange(12), th.arange(8), th.arange(8)
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=th.arange(7), bk=th.arange(7)
    ).to(device)
    stepper_config = SingleModuleStepperConfig.from_state(stepper_config_data)
    stepper = stepper_config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=horizontal_coordinates,
            vertical_coordinate=vertical_coordinate,
            timestep=TIMESTEP,
        ),
    )
    assert len(stepper.modules) == 1
    assert type(stepper.modules[0].module) is HEALPixRecUNet


@pytest.mark.parametrize(
    "in_channels, out_channels, n_constants, decoder_input_channels, input_time_size, \
    output_time_size, couplings, expected_exception, expected_message",
    [
        (7, 7, 1, 1, 2, 4, None, None, None),  # Valid case
        (
            7,
            7,
            1,
            1,
            2,
            3,
            None,
            ValueError,
            "'output_time_size' must be a multiple of 'input_time_size'",
        ),  # Bad input and output time dims
        (
            7,
            7,
            0,
            2,
            2,
            3,
            ["t2m", "v10m"],
            NotImplementedError,
            "support for coupled models with no constant field",
        ),  # Couplings with no constants
        (
            7,
            7,
            2,
            0,
            2,
            3,
            ["t2m", "v10m"],
            NotImplementedError,
            "support for coupled models with no decoder",
        ),  # Couplings with no decoder input channels
        (
            7,
            7,
            0,
            0,
            2,
            3,
            None,
            ValueError,
            "'output_time_size' must be a multiple of 'input_time_size'",
        ),  # No constant fields and no decoder
    ],
)
def test_HEALPixRecUNet_initialize(
    in_channels,
    out_channels,
    n_constants,
    decoder_input_channels,
    input_time_size,
    output_time_size,
    couplings,
    expected_exception,
    expected_message,
):
    prognostic_variables = min(out_channels, in_channels)
    conv_next_block = conv_next_block_config()
    up_sampling_block = up_sampling_block_config()
    output_layer = output_layer_config()
    recurrent_block = recurrent_block_config()
    encoder = encoder_config(conv_next_block, down_sampling_block_config())
    decoder = decoder_config(
        conv_next_block, up_sampling_block, output_layer, recurrent_block
    )
    device = get_device()

    if expected_exception:
        with pytest.raises(expected_exception, match=expected_message):
            model = HEALPixRecUNet(
                encoder=encoder,
                decoder=decoder,
                input_channels=in_channels,
                output_channels=out_channels,
                prognostic_variables=prognostic_variables,
                n_constants=n_constants,
                decoder_input_channels=decoder_input_channels,
                input_time_size=input_time_size,
                output_time_size=output_time_size,
                couplings=couplings,
            ).to(device)
    else:
        model = HEALPixRecUNet(
            encoder=encoder,
            decoder=decoder,
            input_channels=in_channels,
            output_channels=out_channels,
            prognostic_variables=prognostic_variables,
            n_constants=n_constants,
            decoder_input_channels=decoder_input_channels,
            input_time_size=input_time_size,
            output_time_size=output_time_size,
            couplings=couplings,
        ).to(device)
        assert isinstance(model, HEALPixRecUNet)


def test_HEALPixRecUNet_integration_steps():
    in_channels = 2
    out_channels = 2
    prognostic_variables = min(out_channels, in_channels)
    n_constants = 1
    decoder_input_channels = 0
    input_time_size = 2
    output_time_size = 4
    device = get_device()

    conv_next_block = conv_next_block_config()
    up_sampling_block = up_sampling_block_config()
    output_layer = output_layer_config()
    recurrent_block = recurrent_block_config()
    encoder = encoder_config(conv_next_block, down_sampling_block_config())
    decoder = decoder_config(
        conv_next_block, up_sampling_block, output_layer, recurrent_block
    )

    model = HEALPixRecUNet(
        encoder=encoder,
        decoder=decoder,
        input_channels=in_channels,
        output_channels=out_channels,
        prognostic_variables=prognostic_variables,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_size=input_time_size,
        output_time_size=output_time_size,
    ).to(device)

    assert model.integration_steps == output_time_size // input_time_size


def test_HEALPixRecUNet_reset(very_fast_only: bool):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    # create a smaller version of the dlwp healpix model
    in_channels = 3
    out_channels = 3
    prognostic_variables = min(out_channels, in_channels)
    n_constants = 2
    decoder_input_channels = 1
    input_time_size = 2
    output_time_size = 4
    size = 16
    device = get_device()

    conv_next_block = conv_next_block_config()
    up_sampling_block = up_sampling_block_config()
    output_layer = output_layer_config()
    recurrent_block = recurrent_block_config()
    encoder = encoder_config(conv_next_block, down_sampling_block_config())
    decoder = decoder_config(
        conv_next_block, up_sampling_block, output_layer, recurrent_block
    )

    fix_random_seeds(seed=42)
    x = _test_data()(time_dim=input_time_size, channels=in_channels, img_size=size)
    decoder_inputs = insolation_data()(time_dim=input_time_size, img_size=size)
    constants = constant_data()(channels=n_constants, img_size=size)
    batch_size = x.shape[0]
    constants = constants.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    inputs = th.concat(
        (x, decoder_inputs, constants), dim=-3
    )  # [x, decoder_inputs, constants]

    model = HEALPixRecUNet(
        encoder=encoder,
        decoder=decoder,
        input_channels=in_channels,
        output_channels=out_channels,
        prognostic_variables=prognostic_variables,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_size=input_time_size,
        output_time_size=output_time_size,
        enable_healpixpad=False,
        delta_time="6h",
    ).to(device)

    out_var = model(inputs)
    model.reset()

    assert compare_output(out_var, model(inputs))


# Checks the model can perform a forward class on various input configurations
# [full inputs, no decoder inputs, no constant inputs]
@pytest.mark.parametrize(
    "inputs_config, in_channels, decoder_input_channels, \
        out_channels, input_time_size, output_time_size, n_constants, size",
    [
        ([0, 1, 2], 3, 1, 3, 2, 4, 2, 16),  # full inputs
        ([0, 2], 3, 0, 3, 2, 4, 2, 16),  # no decoder inputs
        ([0, 1], 3, 1, 3, 2, 4, 0, 16),  # no constant inputs
    ],
)
def test_HEALPixRecUNet_forward(
    inputs_config,
    in_channels,
    decoder_input_channels,
    out_channels,
    input_time_size,
    output_time_size,
    n_constants,
    size,
    very_fast_only: bool,
):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    prognostic_variables = min(out_channels, in_channels)
    device = get_device()
    conv_next_block = conv_next_block_config()
    up_sampling_block = up_sampling_block_config()
    output_layer = output_layer_config()
    recurrent_block = recurrent_block_config()
    encoder = encoder_config(conv_next_block, down_sampling_block_config())
    decoder = decoder_config(
        conv_next_block, up_sampling_block, output_layer, recurrent_block
    )

    fix_random_seeds(seed=42)
    x = _test_data()(time_dim=input_time_size, channels=in_channels, img_size=size)
    batch_size = x.shape[0]

    if decoder_input_channels > 0:
        decoder_inputs = insolation_data()(time_dim=input_time_size, img_size=size)
    else:
        decoder_inputs = insolation_data()(time_dim=0, img_size=size)
    constants = constant_data()(channels=n_constants, img_size=size)
    constants = constants.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

    all_inputs = [x, decoder_inputs, constants]
    inputs = th.concat(all_inputs, dim=-3)

    model = HEALPixRecUNet(
        encoder=encoder,
        decoder=decoder,
        input_channels=in_channels,
        output_channels=out_channels,
        prognostic_variables=prognostic_variables,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_size=input_time_size,
        output_time_size=output_time_size,
        enable_healpixpad=False,
        delta_time="6h",
    ).to(device)
    model(inputs)


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

    # doesn't do anything
    encoder.reset()

    # outvar is a module list
    for idx, out_tensor in enumerate(outvar):
        # verify the channels and h dim are correct
        assert out_tensor.shape[1] == n_channels[idx]
        # default behaviour is to half the h/w size after first
        assert out_tensor.shape[2] == tensor_size[2] // (2**idx)


def test_UNetEncoder_reset():
    channels = 2
    n_channels = (16, 32, 64)
    device = get_device()

    # Dicts for block configs used by encoder
    conv_block_config = ConvBlockConfig(
        in_channels=channels,
        block_type="ConvNeXtBlock",
    )
    down_sampling_block_config = DownsamplingBlockConfig(
        pooling=2,
        block_type="MaxPool",
    )
    encoder = UNetEncoder(
        conv_block=conv_block_config,
        down_sampling_block=down_sampling_block_config,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)

    # doesn't do anything
    encoder.reset()
    assert isinstance(encoder, UNetEncoder)


def test_UNetDecoder_initilization():
    in_channels = 2
    out_channels = 1
    n_channels = (64, 32, 16)
    device = get_device()

    # Dicts for block configs used by decoder
    conv_block_config = ConvBlockConfig(
        in_channels=in_channels, out_channels=out_channels, block_type="ConvNeXtBlock"
    )
    up_sampling_block_config = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=2,
        block_type="TransposedConvUpsample",
    )

    output_layer_config = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        dilation=1,
        n_layers=1,
        block_type="ConvNeXtBlock",
    )

    recurrent_block_config = RecurrentBlockConfig(
        in_channels=2,
        kernel_size=1,
        block_type="ConvGRUBlock",
    )

    decoder = UNetDecoder(
        conv_block=conv_block_config,
        up_sampling_block=up_sampling_block_config,
        output_layer=output_layer_config,
        recurrent_block=recurrent_block_config,
        n_channels=n_channels,
    ).to(device)

    assert isinstance(decoder, UNetDecoder)

    # without the recurrent block and with dilations
    decoder = UNetDecoder(
        conv_block=conv_block_config,
        up_sampling_block=up_sampling_block_config,
        output_layer=output_layer_config,
        recurrent_block=None,
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
    up_sampling_block_config = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=2,
        block_type="TransposedConvUpsample",
    )
    output_layer_config = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        dilation=1,
        n_layers=1,
        block_type="BasicConvBlock",
    )
    recurrent_block_config = RecurrentBlockConfig(
        in_channels=2,
        kernel_size=1,
        block_type="ConvGRUBlock",
    )

    decoder = UNetDecoder(
        conv_block=conv_block_config,
        up_sampling_block=up_sampling_block_config,
        output_layer=output_layer_config,
        recurrent_block=recurrent_block_config,
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

    # make sure history is taken into account with ConvGRU
    outvar_hist = decoder(invars)
    assert not compare_output(outvar, outvar_hist)

    # check with no recurrent
    decoder = UNetDecoder(
        conv_block=conv_block_config,
        up_sampling_block=up_sampling_block_config,
        output_layer=output_layer_config,
        recurrent_block=None,
        n_channels=n_channels,
        dilations=[1, 1, 1],
    ).to(device)

    outvar = decoder(invars)
    assert outvar.shape == output_2_size


def test_UNetDecoder_reset():
    in_channels = 2
    out_channels = 1
    hw_size = 32
    b_size = 12
    n_channels = (64, 32, 16)
    device = get_device()

    # Dicts for block configs used by decoder
    conv_block = ConvBlockConfig(in_channels=in_channels, block_type="ConvNeXtBlock")
    up_sampling_block = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        block_type="TransposedConvUpsample",
    )
    output_layer = ConvBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        dilation=1,
        n_layers=1,
        block_type="BasicConvBlock",
    )

    recurrent_block = RecurrentBlockConfig(
        in_channels=2, kernel_size=1, block_type="ConvLSTMBlock"
    )

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=recurrent_block,
        n_channels=n_channels,
    ).to(device)

    # build the list of tensors for the decoder
    invars = []
    # decoder has an algorithm that goes back to front
    for idx in range(len(n_channels) - 1, -1, -1):
        tensor_size = [b_size, n_channels[idx], hw_size, hw_size]
        invars.append(th.rand(tensor_size).to(device))
        hw_size = hw_size // 2

    outvar = decoder(invars)

    # make sure history is taken into account with ConvGRU
    outvar_hist = decoder(invars)
    assert not compare_output(outvar, outvar_hist)

    # make sure after reset we get the same result
    decoder.reset()
    outvar_reset = decoder(invars)
    assert compare_output(outvar, outvar_reset)

    # test reset without recurrent block
    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
    ).to(device)

    outvar = decoder(invars)

    # without the recurrent block should be the same
    outvar_hist = decoder(invars)
    assert compare_output(outvar, outvar_hist)

    # make sure after reset we get the same result
    decoder.reset()
    outvar_reset = decoder(invars)
    assert compare_output(outvar, outvar_reset)


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

    # TODO: check each face to confirm padding values in each region
    for face_idx in range(mock_data.shape[0]):
        face_data = padded_data[face_idx, 0]

        # the region where original data should reside
        data_region = face_data[padding : padding + 4, padding : padding + 4]
        original_data = mock_data[face_idx, 0]  # Original face data
        assert (
            data_region == original_data
        ).all(), f"Data region on face {face_idx} has been altered."

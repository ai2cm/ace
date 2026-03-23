import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor

from .ankur import AnkurLocalNetConfig
from .hybrid import HybridNetConfig, get_lat_lon_hybridnet
from .localnet import LocalNetConfig
from .sfnonet import SFNONetConfig

DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize(
    "learn_residual",
    [True, False],
)
def test_can_call_hybridnet_with_ankur_local(learn_residual: bool):
    n_forcing = 3
    n_prognostic = 2
    n_diagnostic = 4
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    config = HybridNetConfig(
        backbone=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="makani-linear"),
        local=AnkurLocalNetConfig(embed_dim=16),
        learn_residual=learn_residual,
    )
    model = get_lat_lon_hybridnet(
        params=config,
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        img_shape=img_shape,
    ).to(device)
    forcing = torch.randn(n_samples, n_forcing, *img_shape, device=device)
    prognostic = torch.randn(n_samples, n_prognostic, *img_shape, device=device)
    prog_out, diag_out = model(forcing, prognostic)
    assert prog_out.shape == (n_samples, n_prognostic, *img_shape)
    assert diag_out.shape == (n_samples, n_diagnostic, *img_shape)


def test_can_call_hybridnet_with_localnet():
    n_forcing = 3
    n_prognostic = 2
    n_diagnostic = 4
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    config = HybridNetConfig(
        backbone=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="makani-linear"),
        local=LocalNetConfig(
            embed_dim=16,
            block_types=["disco", "disco"],
        ),
    )
    model = get_lat_lon_hybridnet(
        params=config,
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        img_shape=img_shape,
    ).to(device)
    forcing = torch.randn(n_samples, n_forcing, *img_shape, device=device)
    prognostic = torch.randn(n_samples, n_prognostic, *img_shape, device=device)
    prog_out, diag_out = model(forcing, prognostic)
    assert prog_out.shape == (n_samples, n_prognostic, *img_shape)
    assert diag_out.shape == (n_samples, n_diagnostic, *img_shape)


def test_can_call_hybridnet_with_localnet_conv1x1():
    n_forcing = 3
    n_prognostic = 2
    n_diagnostic = 4
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    config = HybridNetConfig(
        backbone=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="makani-linear"),
        local=LocalNetConfig(
            embed_dim=16,
            block_types=["conv1x1", "conv1x1"],
        ),
    )
    model = get_lat_lon_hybridnet(
        params=config,
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        img_shape=img_shape,
    ).to(device)
    forcing = torch.randn(n_samples, n_forcing, *img_shape, device=device)
    prognostic = torch.randn(n_samples, n_prognostic, *img_shape, device=device)
    prog_out, diag_out = model(forcing, prognostic)
    assert prog_out.shape == (n_samples, n_prognostic, *img_shape)
    assert diag_out.shape == (n_samples, n_diagnostic, *img_shape)


def test_hybridnet_with_labels():
    n_forcing = 3
    n_prognostic = 2
    n_diagnostic = 4
    img_shape = (9, 18)
    n_samples = 4
    embed_dim_labels = 5
    device = get_device()
    config = HybridNetConfig(
        backbone=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="makani-linear"),
        local=AnkurLocalNetConfig(embed_dim=16),
    )
    model = get_lat_lon_hybridnet(
        params=config,
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        img_shape=img_shape,
        embed_dim_labels=embed_dim_labels,
    ).to(device)
    forcing = torch.randn(n_samples, n_forcing, *img_shape, device=device)
    prognostic = torch.randn(n_samples, n_prognostic, *img_shape, device=device)
    labels = torch.randn(n_samples, embed_dim_labels, device=device)
    prog_out, diag_out = model(forcing, prognostic, labels=labels)
    assert prog_out.shape == (n_samples, n_prognostic, *img_shape)
    assert diag_out.shape == (n_samples, n_diagnostic, *img_shape)


def test_learn_residual_adds_prognostic_input():
    """Verify learn_residual adds the prognostic input to the backbone output."""
    torch.manual_seed(0)
    n_forcing = 2
    n_prognostic = 3
    n_diagnostic = 2
    img_shape = (9, 18)
    n_samples = 2
    device = get_device()

    config_no_residual = HybridNetConfig(
        backbone=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="makani-linear"),
        local=AnkurLocalNetConfig(embed_dim=16),
        learn_residual=False,
    )
    config_residual = HybridNetConfig(
        backbone=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="makani-linear"),
        local=AnkurLocalNetConfig(embed_dim=16),
        learn_residual=True,
    )

    model_no = get_lat_lon_hybridnet(
        params=config_no_residual,
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        img_shape=img_shape,
    ).to(device)
    model_yes = get_lat_lon_hybridnet(
        params=config_residual,
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        img_shape=img_shape,
    ).to(device)

    # Copy weights from model_no to model_yes
    model_yes.load_state_dict(model_no.state_dict())

    forcing = torch.randn(n_samples, n_forcing, *img_shape, device=device)
    prognostic = torch.randn(n_samples, n_prognostic, *img_shape, device=device)

    with torch.no_grad():
        prog_no, diag_no = model_no(forcing, prognostic)
        prog_yes, diag_yes = model_yes(forcing, prognostic)

    # Diagnostic outputs should be identical
    torch.testing.assert_close(diag_no, diag_yes)
    # Prognostic output with residual = without residual + prognostic input
    torch.testing.assert_close(prog_yes, prog_no + prognostic)


def test_backward_pass():
    """Test that gradients flow through both sub-networks."""
    n_forcing = 2
    n_prognostic = 3
    n_diagnostic = 2
    img_shape = (9, 18)
    n_samples = 2
    device = get_device()
    config = HybridNetConfig(
        backbone=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="makani-linear"),
        local=AnkurLocalNetConfig(embed_dim=16),
        learn_residual=True,
    )
    model = get_lat_lon_hybridnet(
        params=config,
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        img_shape=img_shape,
    ).to(device)
    forcing = torch.randn(n_samples, n_forcing, *img_shape, device=device)
    prognostic = torch.randn(n_samples, n_prognostic, *img_shape, device=device)
    prog_out, diag_out = model(forcing, prognostic)
    loss = prog_out.sum() + diag_out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_ankur_disco_encoder():
    """Test HybridNet with AnkurLocalNet using DISCO encoder."""
    n_forcing = 3
    n_prognostic = 2
    n_diagnostic = 4
    img_shape = (9, 18)
    n_samples = 2
    device = get_device()
    config = HybridNetConfig(
        backbone=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="makani-linear"),
        local=AnkurLocalNetConfig(embed_dim=16, use_disco_encoder=True, pos_embed=True),
    )
    model = get_lat_lon_hybridnet(
        params=config,
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        img_shape=img_shape,
    ).to(device)
    forcing = torch.randn(n_samples, n_forcing, *img_shape, device=device)
    prognostic = torch.randn(n_samples, n_prognostic, *img_shape, device=device)
    prog_out, diag_out = model(forcing, prognostic)
    assert prog_out.shape == (n_samples, n_prognostic, *img_shape)
    assert diag_out.shape == (n_samples, n_diagnostic, *img_shape)


def setup_hybridnet():
    n_forcing = 3
    n_prognostic = 2
    n_diagnostic = 4
    img_shape = (9, 18)
    n_samples = 4
    embed_dim_labels = 3
    device = get_device()
    config = HybridNetConfig(
        backbone=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="linear"),
        local=AnkurLocalNetConfig(embed_dim=16),
        learn_residual=True,
    )
    model = get_lat_lon_hybridnet(
        params=config,
        n_forcing_channels=n_forcing,
        n_prognostic_channels=n_prognostic,
        n_diagnostic_channels=n_diagnostic,
        img_shape=img_shape,
        embed_dim_labels=embed_dim_labels,
    ).to(device)
    # Initialize on CPU for reproducibility, then move to device
    forcing = torch.randn(n_samples, n_forcing, *img_shape).to(device)
    prognostic = torch.randn(n_samples, n_prognostic, *img_shape).to(device)
    labels = torch.randn(n_samples, embed_dim_labels).to(device)
    return model, forcing, prognostic, labels


def test_hybridnet_output_is_unchanged():
    torch.manual_seed(0)
    model, forcing, prognostic, labels = setup_hybridnet()
    with torch.no_grad():
        prog_out, diag_out = model(forcing, prognostic, labels=labels)
    validate_tensor(
        prog_out,
        os.path.join(DIR, "testdata/test_hybridnet_prognostic_output.pt"),
    )
    validate_tensor(
        diag_out,
        os.path.join(DIR, "testdata/test_hybridnet_diagnostic_output.pt"),
    )

import dataclasses

import torch

import fme
from fme.ace.registry.patch_discriminator import PatchDiscriminatorConfig
from fme.core.coordinates import LatLonCoordinates
from fme.core.registry import ModuleSelector
from fme.core.testing.dataset_info import get_dataset_info

IMG_SHAPE = (12, 24)


def _get_dataset_info() -> fme.core.dataset_info.DatasetInfo:
    device = fme.get_device()
    lat = torch.linspace(-torch.pi / 2, torch.pi / 2, IMG_SHAPE[0], device=device)
    lon = torch.linspace(0, 2 * torch.pi, IMG_SHAPE[1], device=device)
    return get_dataset_info(
        img_shape=IMG_SHAPE,
        horizontal_coordinate=LatLonCoordinates(lat=lat, lon=lon),
    )


def test_is_registered():
    assert "PatchDiscriminator" in ModuleSelector.get_available_types()


def test_output_shape():
    n_in, n_out = 5, 1
    dataset_info = _get_dataset_info()
    config = PatchDiscriminatorConfig(hidden_dim=32)
    module = config.build(n_in, n_out, dataset_info)
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    # Two 3x3 convs with valid lat padding each shrink lat by 2
    expected_h = IMG_SHAPE[0] - 4
    expected_w = IMG_SHAPE[1]
    assert out.shape == (2, n_out, expected_h, expected_w)


def test_circular_lon_padding():
    """Shifting input circularly in lon should circularly shift the output."""
    n_in, n_out = 3, 1
    dataset_info = _get_dataset_info()
    config = PatchDiscriminatorConfig(hidden_dim=16)
    module = config.build(n_in, n_out, dataset_info)
    module.eval()

    torch.manual_seed(42)
    x = torch.randn(1, n_in, *IMG_SHAPE, device=fme.get_device())
    out_base = module(x)

    shift = 5
    x_shifted = torch.roll(x, shifts=shift, dims=-1)
    out_shifted = module(x_shifted)
    out_base_shifted = torch.roll(out_base, shifts=shift, dims=-1)

    torch.testing.assert_close(out_shifted, out_base_shifted, atol=1e-5, rtol=1e-5)


def test_spectral_norm_applied():
    """All three conv layers should have spectral normalization hooks."""
    n_in, n_out = 4, 1
    dataset_info = _get_dataset_info()
    config = PatchDiscriminatorConfig(hidden_dim=16)
    module = config.build(n_in, n_out, dataset_info)

    for name in ["conv1", "conv2", "conv3"]:
        layer = getattr(module, name)
        assert hasattr(
            layer, "weight_orig"
        ), f"{name} missing spectral norm (no weight_orig)"


def test_positional_channels():
    """Latitude information should affect the output: identical input fields
    at different latitudes should produce different outputs."""
    n_in, n_out = 3, 1
    device = fme.get_device()
    n_lat, n_lon = IMG_SHAPE

    # Build two modules with different latitude grids
    lat_a = torch.linspace(-torch.pi / 2, torch.pi / 2, n_lat, device=device)
    lat_b = torch.linspace(0, torch.pi / 4, n_lat, device=device)

    info_a = get_dataset_info(
        img_shape=IMG_SHAPE,
        horizontal_coordinate=LatLonCoordinates(
            lat=lat_a, lon=torch.zeros(n_lon, device=device)
        ),
    )
    info_b = get_dataset_info(
        img_shape=IMG_SHAPE,
        horizontal_coordinate=LatLonCoordinates(
            lat=lat_b, lon=torch.zeros(n_lon, device=device)
        ),
    )

    config = PatchDiscriminatorConfig(hidden_dim=16)
    module_a = config.build(n_in, n_out, info_a)
    module_b = config.build(n_in, n_out, info_b)

    # Copy only conv weights (not the pos_encoding buffer) so the
    # positional encoding is the only difference between the two modules.
    state_a = {k: v for k, v in module_a.state_dict().items() if k != "pos_encoding"}
    module_b.load_state_dict(state_a, strict=False)

    torch.manual_seed(0)
    x = torch.randn(1, n_in, *IMG_SHAPE, device=device)
    out_a = module_a(x)
    out_b = module_b(x)

    assert not torch.allclose(
        out_a, out_b, atol=1e-6
    ), "Outputs should differ when latitude grids differ"


def test_via_module_selector():
    """Build through the ModuleSelector registry round-trip."""
    dataset_info = _get_dataset_info()
    selector = ModuleSelector(
        type="PatchDiscriminator",
        config=dataclasses.asdict(PatchDiscriminatorConfig(hidden_dim=32)),
    )
    module = selector.build(
        n_in_channels=4, n_out_channels=1, dataset_info=dataset_info
    )
    module = module.to(fme.get_device())
    x = torch.randn(2, 4, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    expected_h = IMG_SHAPE[0] - 4
    assert out.shape == (2, 1, expected_h, IMG_SHAPE[1])

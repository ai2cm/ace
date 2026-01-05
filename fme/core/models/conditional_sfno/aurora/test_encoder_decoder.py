from fme.core.device import get_device
from fme.core.models.conditional_sfno.aurora.encoder import Perceiver3DEncoder
from fme.core.models.conditional_sfno.aurora.decoder import Perceiver3DDecoder
from fme.core.models.conditional_sfno.aurora.batch import Batch, Metadata
import torch
from datetime import datetime, timedelta


def get_batch(
    surf_names,
    static_names,
    atmos_names,
    n_lat,
    n_lon,
    n_time,
    n_levels,
    batch_size,
    device="cpu",
):
    """Create a dummy batch for testing."""
    surf_vars = {}
    static_vars = {}
    atmos_vars = {}
    lat = torch.linspace(-90 + (180 / n_lat), 90 - (180 / n_lat), n_lat, device=device)
    lon = torch.linspace(0, 360 - (360 / n_lon), n_lon, device=device)
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    time = tuple(datetime(2000, 1, 1) for _ in range(batch_size))
    atmos_levels = tuple(range(n_levels))
    rollout_step = 0
    metadata = Metadata(
        lat=lat,
        lon=lon,
        time=time,
        atmos_levels=atmos_levels,
        rollout_step=rollout_step,
    )

    for name in surf_names:
        surf_vars[name] = torch.randn(
            batch_size, n_time, n_lat, n_lon, device=device
        )
    for name in static_names:
        static_vars[name] = torch.randn(
            batch_size, n_time, n_lat, n_lon, device=device
        )
    for name in atmos_names:
        atmos_vars[name] = torch.randn(
            batch_size, n_time, n_levels, n_lat, n_lon, device=device
        )

    return Batch(
        surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata
    )

def test_can_call_round_trip_encoder_decoder():
    surf_names = ["surface_temp", "surface_pressure"]
    static_names = ["orography"]
    atmos_names = ["temp", "humidity", "u_wind", "v_wind"]
    n_lat = 16
    n_lon = 32
    n_time = 2
    n_levels = 3
    batch_size = 4
    batch = get_batch(
        surf_names=surf_names,
        static_names=static_names,
        atmos_names=atmos_names,
        n_lat=n_lat,
        n_lon=n_lon,
        n_time=n_time,
        n_levels=n_levels,
        batch_size=batch_size,
        device=get_device(),
    )
    lead_time = timedelta(hours=6)

    encoder = Perceiver3DEncoder(
        surf_vars = surf_names,
        static_vars = static_names,
        atmos_vars = atmos_names,
        patch_size=2,
        latent_levels=4,
        embed_dim=16,
        num_heads=4,
        head_dim=4,
        dynamic_vars=True,
    ).to(get_device())

    decoder = Perceiver3DDecoder(
        surf_vars=surf_names,
        atmos_vars=atmos_names,
        patch_size=2,
        embed_dim=16,
        num_heads=4,
        head_dim=4,
    ).to(get_device())

    H, W = batch.spatial_shape
    patch_res = (
        encoder.latent_levels,
        H // encoder.patch_size,
        W // encoder.patch_size,
    )

    latents = encoder(batch=batch, lead_time=lead_time)
    output = decoder(x=latents, batch=batch, patch_res=patch_res, lead_time=lead_time)
    assert isinstance(output, Batch)
    assert output.metadata.rollout_step == batch.metadata.rollout_step + 1
    torch.testing.assert_close(output.metadata.lat, batch.metadata.lat)
    torch.testing.assert_close(output.metadata.lon, batch.metadata.lon)
    for name in surf_names:
        assert name in output.surf_vars
        assert output.surf_vars[name].shape == (
            batch_size,
            n_lat,
            n_lon,
        )
    for name in atmos_names:
        assert name in output.atmos_vars
        assert output.atmos_vars[name].shape == (
            batch_size,
            n_levels,
            n_lat,
            n_lon,
        )


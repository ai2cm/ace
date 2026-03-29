"""Run the shallow water model and save a video of the evolution.

Initializes a Gaussian height perturbation at mid-latitudes and
integrates the linearized shallow water equations on the sphere.
Saves an mp4 video showing the height field and velocity vectors.

Usage:
    python scripts/run_shallow_water.py [--output OUTPUT.mp4]
        [--steps STEPS] [--shape NLAT NLON]
"""

import argparse
import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from fme.core.disco._quadrature import precompute_latitudes, precompute_longitudes
from fme.core.shallow_water import ShallowWaterStepper


def gaussian_bump(nlat, nlon, lat0_deg=30.0, lon0_deg=180.0, sigma_deg=10.0):
    """Create a Gaussian height perturbation."""
    colats = precompute_latitudes(nlat)[0].float()
    lons = precompute_longitudes(nlon).float()
    lat = (math.pi / 2.0 - colats).unsqueeze(1)
    lon = lons.unsqueeze(0)

    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sigma = math.radians(sigma_deg)

    dlat = lat - lat0
    dlon = lon - lon0
    dist2 = dlat**2 + (dlon * torch.cos(lat)) ** 2
    return torch.exp(-dist2 / (2 * sigma**2))


def make_grid(nlat, nlon):
    """Create lat/lon arrays in degrees for plotting."""
    colats = precompute_latitudes(nlat)[0].numpy()
    lons = precompute_longitudes(nlon).numpy()
    lat_deg = np.degrees(np.pi / 2.0 - colats)
    lon_deg = np.degrees(lons)
    return lon_deg, lat_deg


def run(
    nlat=48,
    nlon=96,
    n_steps=2000,
    dt=0.01,
    amplitude=0.01,
    omega=0.5,
    lat0=30.0,
    lon0=90.0,
    frames_per_step=10,
    output="shallow_water.gif",
):
    from fme.core.device import get_device

    device = get_device()
    shape = (nlat, nlon)
    stepper = ShallowWaterStepper(shape=shape, omega=omega).to(device)

    h = amplitude * gaussian_bump(
        nlat, nlon, lat0_deg=lat0, lon0_deg=lon0, sigma_deg=12.0
    )
    h = h.unsqueeze(0).unsqueeze(0).to(device)
    uv = torch.zeros(1, 1, nlat, nlon, 2, device=device)

    # Collect snapshots
    n_frames = n_steps // frames_per_step
    h_frames = []
    u_frames = []
    v_frames = []

    for i in range(n_steps):
        if i % frames_per_step == 0:
            h_frames.append(h[0, 0].detach().cpu().numpy().copy())
            u_frames.append(uv[0, 0, :, :, 0].detach().cpu().numpy().copy())
            v_frames.append(uv[0, 0, :, :, 1].detach().cpu().numpy().copy())
        h, uv = stepper.step(h, uv, dt)

    lon_deg, lat_deg = make_grid(nlat, nlon)
    LON, LAT = np.meshgrid(lon_deg, lat_deg)

    # Determine color scale from all frames
    h_all = np.stack(h_frames)
    vmax = max(abs(h_all.min()), abs(h_all.max()))
    if vmax < 1e-10:
        vmax = 1.0

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_aspect("auto")

    # Initial plot
    mesh = ax.pcolormesh(
        LON,
        LAT,
        h_frames[0],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    fig.colorbar(mesh, ax=ax, label="h'", shrink=0.8)

    # Subsample vectors for readability
    skip_lat = max(1, nlat // 16)
    skip_lon = max(1, nlon // 24)
    quiver = ax.quiver(
        LON[::skip_lat, ::skip_lon],
        LAT[::skip_lat, ::skip_lon],
        u_frames[0][::skip_lat, ::skip_lon],
        v_frames[0][::skip_lat, ::skip_lon],
        scale=0.2,
        color="k",
        alpha=0.6,
    )
    title = ax.set_title(f"t = 0.00")

    def update(frame):
        mesh.set_array(h_frames[frame].ravel())
        quiver.set_UVC(
            u_frames[frame][::skip_lat, ::skip_lon],
            v_frames[frame][::skip_lat, ::skip_lon],
        )
        t = frame * frames_per_step * dt
        title.set_text(f"t = {t:.2f}")
        return mesh, quiver, title

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=50, blit=False
    )
    if output.endswith(".mp4"):
        anim.save(output, writer="ffmpeg", fps=20, dpi=100)
    else:
        anim.save(output, writer="pillow", fps=20, dpi=100)
    plt.close(fig)
    print(f"Saved {n_frames} frames to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run shallow water model and save video"
    )
    parser.add_argument(
        "--output", default="shallow_water.gif", help="Output file (.gif or .mp4)"
    )
    parser.add_argument("--steps", type=int, default=2000, help="Number of time steps")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[48, 96],
        help="Grid shape (nlat nlon)",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--omega", type=float, default=0.5, help="Rotation rate")
    parser.add_argument(
        "--lat0", type=float, default=30.0, help="Bump center latitude (degrees)"
    )
    parser.add_argument(
        "--lon0", type=float, default=90.0, help="Bump center longitude (degrees)"
    )
    args = parser.parse_args()
    run(
        nlat=args.shape[0],
        nlon=args.shape[1],
        n_steps=args.steps,
        dt=args.dt,
        omega=args.omega,
        lat0=args.lat0,
        lon0=args.lon0,
        output=args.output,
    )

import logging
import sys

# need to import initial conditions first to avoid unfortunate
# GLIBC version conflict when importing xarray. There are some unfortunate
# issues with the environment.
from fcn_mip import initial_conditions

import torch
import numpy as np
import xarray

from networks import get_model
import argparse
from datetime import datetime
from netCDF4 import Dataset as DS
from fcn_mip.ensemble_utils import generate_noise_correlated, draw_noise
from modulus.distributed.manager import DistributedManager
import json
import os
import shutil
from fcn_mip import weather_events
from fcn_mip.netcdf import initialize_netcdf, update_netcdf, finalize_netcdf


def run_ensembles(
    *,
    n_steps: int,
    weather_event,
    model,
    nc,
    domains,
    ds,
    n_ensemble: int,
    batch_size: int,
    device: str,
    rank: int,
    noise_amplitude: float,
    cuda_graphs: bool,
    autocast_fp16: bool,
    output_frequency: int,
    perturbation_strategy: str,
    noise_reddening: float,
    date_obj: datetime,
):
    # TODO infer this from the model
    ds = ds.astype(np.float32)
    assert not np.any(np.isnan(ds))

    diagnostics = initialize_netcdf(
        nc, domains, model.grid, ds.lat.values, ds.lon.values, n_ensemble, device
    )
    time_units = ds.time[-1].dt.strftime("hours since %Y-%m-%d %H:%M:%S").item()
    nc["time"].units = time_units
    nc["time"].calendar = "standard"

    for batch_id in range(0, n_ensemble, batch_size):
        batch_size = min(batch_size, n_ensemble - batch_id)
        x = torch.from_numpy(ds.values)[None].to(device)
        x = model.normalize(x)
        x = x.repeat(batch_size, 1, 1, 1, 1)

        shape = x.shape
        if perturbation_strategy == "gaussian":
            noise = noise_amplitude * torch.normal(
                torch.zeros(shape), torch.ones(shape)
            ).to(device)
        elif perturbation_strategy == "correlated":
            noise = generate_noise_correlated(
                shape,
                reddening=noise_reddening,
                device=device,
                noise_amplitude=noise_amplitude,
            )
        elif perturbation_strategy == "gp":
            corr = torch.load("/lustre/fsw/sw_climate_fno/ensemble_init_stats/corr.pth")
            length_scales = torch.load(
                "/lustre/fsw/sw_climate_fno/ensemble_init_stats/length_scales.pth"
            ).to(device)
            noise = noise_amplitude * draw_noise(
                corr, spreads=None, length_scales=length_scales, device=device
            ).to(device)
            noise = noise.repeat(batch_size, 1, 1, 1, 1)

        if rank == 0 and batch_id == 0:  # first ens-member is deterministic
            noise[0, :, :, :, :] = 0
        x += noise

        time_count = 0
        for k, data in enumerate(
            model.run_steps(
                x,
                n_steps,
                normalize=False,
                cuda_graphs=cuda_graphs,
                autocast_fp16=autocast_fp16,
            )
        ):
            # Saving the output
            sys.stderr.write(f"{k} ")
            sys.stderr.flush()
            if k % output_frequency == 0:
                hour = (k + 1) * 6
                if (batch_id == 0) and rank == 0:
                    nc["time"][time_count] = hour
                update_netcdf(
                    data,
                    diagnostics,
                    domains,
                    batch_id,
                    time_count,
                    model,
                    ds,
                )
                time_count += 1
    finalize_netcdf(diagnostics, nc, domains, weather_event, model.channel_set)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("inference")

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--fcn_model", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    def getarg(key):
        return getattr(args, key) or config[key]

    n_ensemble_global = config["ensemble_members"]
    noise_amplitude = config["noise_amplitude"]
    n_steps = config["simulation_length"]
    io_freq = config["output_frequency"]
    fcn_model = getarg("fcn_model")
    cuda_graphs = config["use_cuda_graphs"]
    seed = config["seed"]
    ensemble_batch_size = config["ensemble_batch_size"]
    autocast_fp16 = config["autocast_fp16"]
    perturbation_strategy = config["perturbation_strategy"]
    noise_reddening = config["noise_reddening"]

    # Set up parallel
    DistributedManager.initialize()
    dist = DistributedManager()
    n_ensemble = n_ensemble_global // dist.world_size
    if n_ensemble == 0:
        logger.warning("MPI world size is larger than global number of ensembles")
        n_ensemble = n_ensemble_global

    # Set random seed
    torch.manual_seed(seed + dist.rank)
    np.random.seed(seed + dist.rank)

    model = get_model(fcn_model)

    if "forecast_name" in config:
        forecast_name = config["forecast_name"]
        weather_event = weather_events.read(forecast_name)
    else:
        obj = config["weather_event"]
        weather_event = weather_events.WeatherEvent.parse_obj(obj)

    name = weather_event.properties.name
    model = get_model(fcn_model)

    if weather_event.properties.netcdf:
        ds = xarray.open_dataset(weather_event.properties.netcdf)["fields"]
    else:
        date_obj = weather_event.properties.start_time
        ds = initial_conditions.ic(
            n_history=model.n_history,
            grid=model.grid,
            time=date_obj,
            channel_set=model.channel_set,
            source=weather_event.properties.initial_condition_source,
        )

    date_obj = datetime.fromisoformat(np.datetime_as_string(ds.time[-1], "s"))
    date_str = "{:%Y_%m_%d_%H_%M_%S}".format(date_obj)[5:]

    if "output_dir" in config:
        output_dir = config["output_dir"]
        output_path = output_dir + "Output." + fcn_model + "." + name + "." + date_str
    else:
        output_path = config["output_path"]

    if not os.path.exists(output_path):
        # Avoid race condition across ranks
        os.makedirs(output_path, exist_ok=True)

    if dist.rank == 0:
        # Only rank 0 copies config files over
        shutil.copyfile(args.config, os.path.join(output_path, "config.json"))
        shutil.copyfile(
            "weather_events.json", os.path.join(output_path, "weather_events.json")
        )

    model.to(dist.device)
    output_file_path = output_path + f"/ensemble_out_{dist.rank}.nc"
    print("weather event: ", weather_event.properties.name)
    print("ensemble members: ", n_ensemble)
    print("model: ", fcn_model)
    print("n_steps: ", n_steps)
    print("noise_amplitude: ", noise_amplitude)

    with DS(output_file_path, "w", format="NETCDF4") as nc:
        # assign global attributes
        nc.model = fcn_model
        nc.config = json.dumps(config)
        nc.weather_event = weather_event.json()
        nc.date_created = datetime.now().isoformat()
        nc.history = " ".join(sys.argv)
        nc.institution = "NVIDIA"
        nc.Conventions = "CF-1.10"

        run_ensembles(
            weather_event=weather_event,
            model=model,
            nc=nc,
            domains=weather_event.domains,
            ds=ds,
            n_ensemble=n_ensemble,
            noise_amplitude=noise_amplitude,
            n_steps=n_steps,
            output_frequency=io_freq,
            batch_size=ensemble_batch_size,
            rank=dist.rank,
            device=dist.device,
            cuda_graphs=cuda_graphs,
            autocast_fp16=autocast_fp16,
            perturbation_strategy=perturbation_strategy,
            noise_reddening=noise_reddening,
            date_obj=date_obj,
        )

    print(
        "\n", "=====> Ensemble forecast finshed, saved to:", output_file_path, " <====="
    )


if __name__ == "__main__":
    main()

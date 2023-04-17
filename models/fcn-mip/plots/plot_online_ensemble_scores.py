import numpy as np
import matplotlib.pyplot as plt
import argparse
import netCDF4 as nc
import json
import xarray

# import pathlib
from fcn_mip import geometry, weather_events


def _open(f, domain, chunks={"time": 1}):
    return xarray.open_dataset(f, chunks=chunks, group=domain)


def lat_weighted(var, lat):
    lats = np.tile(np.expand_dims(lat, axis=1), var.shape[2:])
    area_weights = np.cos(np.deg2rad(lats))
    return np.mean(area_weights * var, axis=(1, 2))


def contour_metric(
    metric,
    metric_name,
    land_mask,
    lat,
    lon,
    time,
    output_path,
    channel,
    model_name,
    start_time,
    config,
):
    n_ensemble = config["ensemble_members"]
    perturbation_strategy = config["perturbation_strategy"]
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(bottom=0.3)
    plt.title(metric_name + " " + channel)
    ax.contourf(lon, lat, metric)
    fig.colorbar(ax.contourf(lon, lat, metric))
    ax.contour(
        lon[0:1440],
        lat[0:720],
        np.subtract(1.0, land_mask[:-1, :]),
        [0.25],
        cmap="gray",
        linewidths=0.5,
    )
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    fig.text(0.1, 0.17, "model: " + model_name)
    fig.text(0.1, 0.12, "intial time: " + start_time)
    fig.text(0.1, 0.07, "ensemble members: " + str(n_ensemble))
    fig.text(0.1, 0.02, "perturbation strategy: " + perturbation_strategy)
    plt.savefig(output_path + channel + "_online_" + metric_name + "_map.png")

    plt.close()


def main():

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="full path to the ensemble simulation directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="full path to the ensemble score output directory",
    )
    parser.add_argument(
        "--time_index",
        type=int,
        nargs="+",
        help="a list of time indices for which contour plots are produced",
    )
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = args.input_path
    if args.time_index is None:
        args.time_index = -1
    output_path = args.output_path
    time_index = args.time_index
    input_path = args.input_path
    output_path = args.output_path
    data = nc.Dataset(input_path + "ensemble_out_0.nc", "r")
    model_name = data.model
    config = json.loads(data.config)
    perturbation_strategy = config["perturbation_strategy"]
    n_ensemble = config["ensemble_members"]
    weather_event_obj = json.loads(data.weather_event)
    weather_event = weather_events.WeatherEvent.parse_obj(weather_event_obj)
    lead_time = np.array(data.variables["time"])
    start_time = weather_event_obj["properties"]["start_time"]
    for domain in weather_event.domains:
        if domain.type != "Window":
            continue
        lat = np.array(data.groups[domain.name].variables["lat"])
        lon = np.array(data.groups[domain.name].variables["lon"])
        lat_sl, lon_sl = geometry.get_bounds_window(domain, lat, lon)
        land_mask_global = np.load("./land_sea_mask.npy")
        land_mask = land_mask_global
        for channel in domain.diagnostics[0].channels:
            spread = np.sqrt(data["global"]["ensemble_variance"][channel])
            skill = np.array(data["global"]["skill"][channel])
            adjusted_spread = spread * ((n_ensemble + 1) / n_ensemble)
            contour_metric(
                skill[time_index, :, :],
                "skill",
                land_mask,
                lat,
                lon,
                lead_time,
                output_path,
                channel,
                model_name,
                start_time,
                config,
            )
            contour_metric(
                spread[time_index, :, :],
                "spread",
                land_mask,
                lat,
                lon,
                lead_time,
                output_path,
                channel,
                model_name,
                start_time,
                config,
            )
            weight_mean_adjusted_spread = lat_weighted(adjusted_spread, lat)
            weight_mean_skill = lat_weighted(skill, lat)
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(bottom=0.3)
            ax.plot(lead_time, weight_mean_skill)
            ax.plot(lead_time, weight_mean_adjusted_spread, "--")
            plt.title(channel + " skill / spread")
            plt.xlabel("time [h]")
            plt.ylabel("skill (solid) spread (dashed) for " + channel)
            fig.text(0.1, 0.15, "model:" + model_name)
            fig.text(0.1, 0.10, "intial time: " + start_time)
            fig.text(0.1, 0.05, "ensemble members: " + str(n_ensemble))
            fig.text(0.1, 0.01, "perturbation strategy: " + perturbation_strategy)
            plt.savefig(output_path + channel + "_online_spread_skill_ratio.png")
            plt.close()
    return


if __name__ == "__main__":
    main()

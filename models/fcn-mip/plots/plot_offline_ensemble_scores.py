import numpy as np
import matplotlib.pyplot as plt
import argparse
import netCDF4 as nc
import json


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
    input_path,
    channel_name,
    model_name,
    start_time,
    config,
):
    n_ensemble = config["ensemble_members"]
    perturbation_strategy = config["perturbation_strategy"]
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(bottom=0.3)
    plt.title(metric_name + " " + channel_name)
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
    plt.savefig(input_path + channel_name + "_" + metric_name + "_map.png")

    plt.close()


def main():
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="path to the simulation output directory",
    )
    parser.add_argument(
        "--time_index",
        type=int,
        nargs="+",
        help="a list of time indices for which contour plots are produced",
    )
    args = parser.parse_args()
    if args.time_index is None:
        args.time_index = -1
    input_path = args.input_path
    time_index = args.time_index
    data = nc.Dataset(input_path + "ensemble_scores.nc", "r")
    weather_event = json.loads(data.weather_event)
    config = json.loads(data.config)
    n_ensemble = config["ensemble_members"]
    perturbation_strategy = config["perturbation_strategy"]
    start_time = weather_event["properties"]["start_time"]
    model_name = data.model
    time = np.array(data.variables["time"])
    n_ensemble = config["ensemble_members"]
    for group_name in data.groups:
        lat = np.array(data.groups[group_name].variables["lat"])
        lon = np.array(data.groups[group_name].variables["lon"])
        land_mask_global = np.load("./land_sea_mask.npy")
        land_mask = land_mask_global
        channels = []
        for variable_name in data.groups[group_name].variables:
            if variable_name == "lat" or variable_name == "lon":
                continue
            channels = channels + [variable_name.split("_")[1]]
        channels = set(channels)
        for channel_name in channels:
            crps = np.array(data.groups[group_name].variables["crps_" + channel_name])
            spread = np.array(
                data.groups[group_name].variables["spread_" + channel_name]
            )
            skill = np.array(data.groups[group_name].variables["rmse_" + channel_name])
            adjusted_spread = spread * ((n_ensemble + 1) / n_ensemble)
            contour_metric(
                crps[time_index, :, :],
                "crps",
                land_mask,
                lat,
                lon,
                time,
                input_path,
                channel_name,
                model_name,
                start_time,
                config,
            )
            contour_metric(
                skill[time_index, :, :],
                "skill",
                land_mask,
                lat,
                lon,
                time,
                input_path,
                channel_name,
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
                time,
                input_path,
                channel_name,
                model_name,
                start_time,
                config,
            )
            weight_mean_crps = lat_weighted(crps, lat)
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(bottom=0.3)
            ax.plot(time, weight_mean_crps)
            plt.title(channel_name + " mean CRPS")
            plt.xlabel("time [h]")
            plt.ylabel(channel_name + " mean CRPS")
            fig.text(0.1, 0.15, "model:" + model_name)
            fig.text(0.1, 0.10, "intial time: " + start_time)
            fig.text(0.1, 0.05, "ensemble members: " + str(n_ensemble))
            fig.text(0.1, 0.01, "perturbation strategy: " + perturbation_strategy)
            plt.savefig(input_path + channel_name + "_crps_mean.png")
            plt.close()

            weight_mean_adjusted_spread = lat_weighted(adjusted_spread, lat)
            weight_mean_skill = lat_weighted(skill, lat)
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(bottom=0.3)
            ax.plot(time, weight_mean_skill)
            ax.plot(time, weight_mean_adjusted_spread, "--")
            plt.title(channel_name + " skill / spread")
            plt.xlabel("time [h]")
            plt.ylabel("skill (solid) spread (dashed) for " + channel_name)
            fig.text(0.1, 0.15, "model:" + model_name)
            fig.text(0.1, 0.10, "intial time: " + start_time)
            fig.text(0.1, 0.05, "ensemble members: " + str(n_ensemble))
            fig.text(0.1, 0.01, "perturbation strategy: " + perturbation_strategy)
            plt.savefig(input_path + channel_name + "_spread_skill_ratio.png")
            plt.close()
    return


if __name__ == "__main__":
    main()

import numpy as np
import pylab as plt
import json
import argparse
from fcn_mip.ensemble_utils import (
    extended_best_track_reader,
    tropical_cyclone_tracker,
    compute_vorticity,
    exceedance_probability,
    emanuel_damage_function,
)
import matplotlib as mpl
import xarray
import pathlib


def main():
    argparse.ArgumentParser(description="Flip maps")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    full_path = args.path
    config = json.loads(open(full_path + "config.in").read())
    forecast_name = config["forecast_name"]
    path = pathlib.Path(full_path)
    ensemble_files = list(path.glob("*_?.nc"))
    data = xarray.concat(
        [xarray.open_dataset(f, group="Somewhere") for f in ensemble_files],
        dim="ensemble",
    )

    imin = data.variables["imin"][0]
    imax = data.variables["imax"][0]
    jmin = data.variables["jmin"][0]
    jmax = data.variables["jmax"][0]

    t_init = np.array(data.variables["time"])[0]
    time = data.variables["time"] - t_init
    t_finit = time[-1]
    lat = data.variables["lat"]
    lon = data.variables["lon"]
    u850 = data.variables["u850"]
    v850 = data.variables["v850"]
    z850 = data.variables["z850"]
    z250 = data.variables["z250"]

    try:
        (
            ebt_time,
            ebt_lat,
            ebt_lon,
            ebt_mslp,
            ebt_vmax,
            ebt_rmax,
        ) = extended_best_track_reader(forecast_name, t_init, t_finit)
    except Exception:
        print(forecast_name + " is not a named tropical cyclone")
    if len(time) > len(ebt_time):
        time = time[0 : len(ebt_time)]
        u850 = u850[:, 0 : len(ebt_time), :, :]
        v850 = v850[:, 0 : len(ebt_time), :, :]
        z850 = z850[:, 0 : len(ebt_time), :, :]
        z250 = z250[:, 0 : len(ebt_time), :, :]

    msl = np.divide(data.variables["msl"], 100.0)
    [Lon, Lat] = np.meshgrid(lon, lat)
    idxs = np.shape(z850)
    max_vorticity = np.zeros((idxs[0], idxs[1]))
    max_v850 = np.zeros((idxs[0], idxs[1]))
    min_z850 = np.zeros((idxs[0], idxs[1]))
    min_msl = np.zeros((idxs[0], idxs[1]))
    stormLat = np.zeros((idxs[0], idxs[1]))
    stormLon = np.zeros((idxs[0], idxs[1]))
    biased_ensemble_variance_estimate = np.zeros(idxs[1])
    ensemble_mean_squared_error = np.zeros(idxs[1])
    P_hfw = np.zeros_like(u850)  # probability of hurricane force winds
    f_damage = np.zeros_like(u850)
    stormLat_mean = np.zeros(idxs[1])
    stormLon_mean = np.zeros(idxs[1])
    for j in range(idxs[1]):  # time
        for i in range(idxs[0]):  # ensemble
            V = np.sqrt(
                np.add(np.power(u850[i, j, :, :], 2), np.power(v850[i, j, :, :], 2))
            )
            P_hfw[i, j, :, :] = exceedance_probability(V, 33.0)
            f_damage[i, j, :, :] = emanuel_damage_function(V)
            vorticity = compute_vorticity(u850[i, j, :, :], v850[i, j, :, :])
            max_vorticity[i, j] = np.max(vorticity)
            max_v850[i, j] = np.max(V)
            min_z850[i, j] = np.min(z850[i, j, :, :])
            min_msl[i, j] = np.min(msl[i, j, :, :])
            cyclone_i, cyclone_j = tropical_cyclone_tracker(
                z850[i, j, :, :], z250[i, j, :, :], u850[i, j, :, :], v850[i, j, :, :]
            )
            try:
                stormLat[i, j] = Lat[cyclone_i, cyclone_j]
                stormLon[i, j] = Lon[cyclone_i, cyclone_j]
            except Exception:
                stormLat[i, j] = np.nan
                stormLon[i, j] = np.nan
        stormLat_mean[j] = np.nanmean(stormLat[:, j])
        stormLon_mean[j] = np.nanmean(stormLon[:, j])
        ensemble_mean_squared_error[j] = np.add(
            np.power(np.subtract(stormLat_mean[j], ebt_lat[j]), 2),
            np.power(np.subtract(stormLon_mean[j], ebt_lon[j]), 2),
        )
        squared_euclidian_distance_to_ens_mean = np.add(
            np.power(np.subtract(stormLat[:, j], stormLat_mean[j]), 2),
            np.power(np.subtract(stormLon[:, j], stormLon_mean[j]), 2),
        )
        biased_ensemble_variance_estimate[j] = np.nanmean(
            squared_euclidian_distance_to_ens_mean
        )

    max_v850[np.where(max_v850 == 0)] = np.nan
    min_z850[np.where(min_z850 == 0)] = np.nan
    stormLat[np.where(stormLat == 0)] = np.nan
    stormLon[np.where(stormLon == 0)] = np.nan
    max_v850_mean = np.nanmean(max_v850, axis=0)
    min_msl_mean = np.nanmean(min_msl, axis=0)
    rms_spread = biased_ensemble_variance_estimate * idxs[0] / (idxs[0] - 1)
    rms_error = ensemble_mean_squared_error * idxs[0] / (idxs[0] + 1)
    land_mask_global = np.load("./land_sea_mask.npy")
    land_mask = land_mask_global[imin:imax, jmin:jmax]
    P_hfw_mean = np.mean(np.mean(P_hfw, axis=1), axis=0)
    f_damage_sum = np.clip(np.sum(np.sum(f_damage, axis=1), axis=0), 0.0, 1.0)

    plt.figure("trajectory spread-reliability")
    plt.plot(rms_spread, rms_error, "o")
    max_val = np.max([np.max(rms_spread), np.max(rms_error)])
    plt.plot([0, max_val], [0, max_val], "k--")
    plt.xlabel("RMS spread")
    plt.ylabel("RMS error")
    plt.savefig(
        full_path + forecast_name + "_trajectory_spread_reliability.pdf", format="pdf"
    )

    cmap = mpl.colors.ListedColormap(
        [
            "white",
            "lightskyblue",
            "deepskyblue",
            "dodgerblue",
            "goldenrod",
            "darkorange",
            "orangered",
            "seagreen",
            "teal",
            "crimson",
        ]
    )
    plt.figure("exceedance_probability")
    plt.contourf(Lon, Lat, P_hfw_mean, cmap=cmap)
    plt.colorbar()
    plt.contour(Lon, Lat, np.subtract(1.0, land_mask), [0.25], cmap="gray")
    plt.plot(
        stormLon_mean,
        stormLat_mean,
        linewidth=1,
        marker=".",
        color="k",
        label="ens_mean",
    )
    plt.plot(ebt_lon, ebt_lat, linewidth=1, color="b", label="observed")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.xlim([np.min(lon), np.max(lon)])
    plt.ylim([np.min(lat), np.max(lat)])
    plt.legend()
    plt.savefig(full_path + forecast_name + "_exceedance_probability.pdf", format="pdf")

    plt.figure("damage")
    plt.contourf(Lon, Lat, f_damage_sum, cmap=cmap)
    plt.colorbar()
    plt.contour(Lon, Lat, np.subtract(1.0, land_mask), [0.25], cmap="gray")
    plt.plot(
        stormLon_mean,
        stormLat_mean,
        linewidth=1,
        marker=".",
        color="k",
        label="ens_mean",
    )
    plt.plot(ebt_lon, ebt_lat, linewidth=1, color="b", label="observed")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.xlim([np.min(lon), np.max(lon)])
    plt.ylim([np.min(lat), np.max(lat)])
    plt.legend()
    plt.savefig(full_path + forecast_name + "_damage.pdf", format="pdf")

    plt.figure("trajectories")
    plt.contour(Lon, Lat, np.subtract(1.0, land_mask), [0.25], cmap="gray")
    for i in range(1, idxs[0]):
        plt.plot(stormLon[i, :], stormLat[i, :], linewidth=0.5, color="lightgrey")
        plt.plot(stormLon[i, 0], stormLat[i, 0], ".", markersize=5, color="lightgrey")
    plt.plot(
        stormLon[0, :],
        stormLat[0, :],
        linewidth=2,
        marker=".",
        color="limegreen",
        label="deterministic",
    )
    plt.plot(stormLon[0, 0], stormLat[0, 0], ".", markersize=10, color="limegreen")
    plt.plot(
        stormLon_mean,
        stormLat_mean,
        linewidth=1,
        marker=".",
        color="k",
        label="ens_mean",
    )
    plt.plot(stormLon_mean[0], stormLat_mean[0], ".", markersize=5, color="k")
    plt.plot(ebt_lon, ebt_lat, linewidth=1, marker=".", color="b", label="observed")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.legend()
    plt.xlim([np.min(lon), np.max(lon)])
    plt.ylim([np.min(lat), np.max(lat)])
    plt.savefig(full_path + forecast_name + "_trajectories.pdf", format="pdf")

    plt.figure("minimum sea level pressure")
    for i in range(idxs[0]):
        plt.plot(time, min_msl[i, :], linewidth=0.5, color="lightgrey")
    plt.plot(time, min_msl[0, :], linewidth=2, color="limegreen", label="deterministic")
    plt.plot(time, min_msl_mean, linewidth=1, color="k", label="ens_mean")
    plt.plot(ebt_time, ebt_mslp, linewidth=1, color="b", label="observed")
    plt.xlabel("time [h]")
    plt.ylabel("mslp [m]")
    plt.xlim([0, 80])
    plt.savefig(full_path + forecast_name + "_mslp.pdf", format="pdf")

    plt.figure("max surface winds")
    for i in range(idxs[0]):
        plt.plot(time, max_v850[i, :], linewidth=0.5, color="lightgrey")
    plt.plot(time, max_v850[0, :], linewidth=2, color="limegreen")
    plt.plot(time, max_v850_mean, linewidth=1, color="k")
    plt.plot(ebt_time, ebt_vmax, linewidth=1, color="b", label="observed")
    plt.xlabel("time [h]")
    plt.ylabel("wind speed [m/s]")
    plt.xlim([0, 80])
    plt.savefig(full_path + forecast_name + "_max_wind.pdf", format="pdf")
    return


if __name__ == "__main__":
    main()

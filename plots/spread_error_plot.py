import pylab as plt
import json
import argparse
import glob
import xarray
from datetime import datetime
from datasets import era5
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Path of the ensemble")
    parser.add_argument(
        "--save_path", "-s", required=True, help="Path where to save figure"
    )
    parser.add_argument(
        "--era5_path",
        "-e",
        default="/lustre/fsw/sw_climate_fno/34Vars/out_of_sample/2018.h5",
        help="Path of the era5 data",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    ensemble_files = glob.glob(args.path + "/ensemble*")

    config_file = glob.glob(args.path + "/config*.in")

    config = json.loads(open(config_file[0]).read())

    weather_events = json.loads(open(args.path + "/weather_events.json").read())
    forecast_name = config["forecast_name"]
    forecast = weather_events[forecast_name]

    ensemble_ds = xarray.concat(
        [xarray.open_dataset(f, group="Somewhere") for f in ensemble_files],
        dim="ensemble",
    )

    era5_ds = era5.open_34_vars(args.era5_path)

    n_points = 0
    if forecast["diagnostics"]["geometry"][0]["type"] == "MultiPoint":
        print("Filtering points using multipoint")
        multipoint = True
        lats = ensemble_ds["lat"][0]
        lons = ensemble_ds["lon"][0]
        coords = [coord for coord in zip(lats, lons)]
        n_points = len(lats)
        target_lon = xarray.DataArray(lons, dims="lon")
        target_lat = xarray.DataArray(lats, dims="lat")
        era5_points = era5_ds.sel(lon=target_lon, lat=target_lat, method="nearest")
        era5_ds = era5_points

    n_ensembles = len(ensemble_ds["t2m"])
    n_tsteps = len(ensemble_ds["time"])

    # generate list of data vars
    ch_list = []
    for key in ensemble_ds.keys():
        if "time" in ensemble_ds[key].dims:
            ch_list.append(key)
    n_channels = len(ch_list)

    # date filtering
    print("Filtering dates")
    start_time = forecast["properties"]["start_time"]
    date_obj = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    date_str = "{:%Y-%m-%dT%H:%M:%S}".format(date_obj)
    start_date = np.datetime64(date_str)
    time_delta = np.timedelta64(6, "h")
    end_date = start_date + time_delta * n_tsteps
    dates = np.arange(start_date, end_date, time_delta, dtype="datetime64")

    # might reactivate later, xarray doesn't drop elements from time dim,
    # need to check later
    # target_dates = xarray.DataArray(dates, dims='time')
    # era5_dates = era5_ds.sel(time=target_dates, method='nearest', drop=True)
    # era5_ds = era5_dates

    # start_time_idx = np.where(era5_ds.loc[:,'t2m']['time'] == start_date )
    # end_time_idx = np.where(era5_ds.loc[:,'t2m']['time'] == end_date )

    biased_ensemble_variance_estimate_spread = np.zeros(n_ensembles)
    ensemble_mean_squared_error_spread = np.zeros(n_ensembles)
    # n_ensembles = 1000

    for ensemble in range(n_ensembles):
        if ensemble % 100 == 0:
            print(f"ensemble {ensemble}")
        if multipoint:
            ens_data = ensemble_ds.sel(npoints=0, ensemble=ensemble)
            ens_mean = ens_data.mean()
            for idx in range(1, n_points):
                ens_data = ensemble_ds.sel(npoints=idx, ensemble=ensemble)
                ens_mean += ens_data.mean()
            ens_mean = ens_mean / n_points
            ens_mean = ens_mean.drop_vars(["i", "j", "lat", "lon"])

        print("ens_mean")
        print(ens_mean)
        ens_se = ens_mean.copy(deep=True)
        for var in ens_se.variables:
            ens_se[var] = 0.0

        ens_diff = ens_mean.copy(deep=True)
        for var in ens_diff.variables:
            ens_diff[var] = 0.0

        for tstep in range(n_tsteps):
            for channel in ch_list:
                if multipoint:
                    for idx, coord in enumerate(coords):
                        era5_data = era5_ds.sel(
                            lat=coord[0],
                            lon=coord[1],
                            time=dates[tstep],
                            channel=channel,
                        )
                        ens_data = ensemble_ds.sel(
                            npoints=idx, time=tstep, ensemble=ensemble
                        )[channel]
                        ens_se[channel] += np.power(ens_data - era5_data.values, 2)

                        ens_diff[channel] += np.power(ens_data - ens_mean[channel], 2)

        # sum everything together to get a single variance and se per ensemble
        ens_diff_sum = 0
        for var in ens_diff.variables:
            ens_diff_sum += ens_diff[var]
        ens_var = ens_diff_sum / (n_tsteps * n_channels * n_points)
        biased_ensemble_variance_estimate_spread[ensemble] = (
            ens_var * n_ensembles / (n_ensembles - 1)
        )

        ens_se_sum = 0
        for var in ens_se.variables:
            ens_se_sum += ens_se[var]
        ens_mse = ens_se_sum / (n_tsteps * n_channels * n_points)
        ensemble_mean_squared_error_spread[ensemble] = (
            ens_mse * n_ensembles / (n_ensembles + 1)
        )

    plt.figure("trajectory spread-reliability")
    plt.plot(
        biased_ensemble_variance_estimate_spread,
        ensemble_mean_squared_error_spread,
        "o",
    )
    max_val = np.max(
        [
            np.max(ensemble_mean_squared_error_spread),
            np.max(biased_ensemble_variance_estimate_spread),
        ]
    )
    plt.plot([0, max_val], [0, max_val], "k--")
    plt.xlabel("RMS spread")
    plt.ylabel("RMS error")
    plt.savefig(args.save_path)

    return


if __name__ == "__main__":
    main()

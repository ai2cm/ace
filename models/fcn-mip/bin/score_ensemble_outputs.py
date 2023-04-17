import sys
import numpy as np
import json
import argparse
import netCDF4 as nc
import xarray
import pathlib
from datetime import datetime, timedelta
from networks import get_model
from fcn_mip import initial_conditions, geometry, weather_events
from properscoring import crps_ensemble


def _open(f, domain, chunks={"time": 1}):
    return xarray.open_dataset(f, chunks=chunks, group=domain)


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
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = args.input_path
    input_path = args.input_path
    output_path = args.output_path
    path = pathlib.Path(input_path)
    ensemble_files = list(path.glob("*_?.nc"))
    data_nc4 = nc.Dataset(input_path + "ensemble_out_0.nc", "r")
    config = json.loads(data_nc4.config)
    weather_event_obj = json.loads(data_nc4.weather_event)
    weather_event = weather_events.WeatherEvent.parse_obj(weather_event_obj)
    ic_source = weather_event.properties.initial_condition_source
    date_obj = datetime.fromisoformat(weather_event_obj["properties"]["start_time"])
    lead_time = np.array(data_nc4.variables["time"])
    fcn_model = get_model(data_nc4.model)
    output = nc.Dataset(output_path + "ensemble_scores.nc", "w", format="NETCDF4")
    output.model = data_nc4.model
    output.config = json.dumps(config)
    output.weather_event = json.dumps(weather_event_obj)
    output.date_created = datetime.now().isoformat()
    output.history = " ".join(sys.argv)
    output.institution = "NVIDIA"
    output.Conventions = "CF-1.10"
    output.createDimension("one", 1)
    output.createDimension("time", lead_time.size)
    output.createVariable("time", np.float32, ("time"))
    output.variables["time"][:] = lead_time
    for domain in weather_event.domains:
        if domain.type != "Window":
            continue
        output.createGroup(domain.name)
        lat = np.array(data_nc4.groups[domain.name].variables["lat"])
        lon = np.array(data_nc4.groups[domain.name].variables["lon"])
        lat_sl, lon_sl = geometry.get_bounds_window(domain, lat, lon)
        data = xarray.concat(
            [_open(f, domain.name) for f in ensemble_files], dim="ensemble"
        )
        output.groups[domain.name].createDimension("lat", lat.size)
        output.groups[domain.name].createDimension("lon", lon.size)
        output.groups[domain.name].createVariable("lat", np.float32, ("lat"))
        output.groups[domain.name].createVariable("lon", np.float32, ("lon"))
        output.groups[domain.name].variables["lat"][:] = lat
        output.groups[domain.name].variables["lon"][:] = lon
        for diagnostic in domain.diagnostics:
            if diagnostic.type != "raw":
                continue
            for c, channel in enumerate(diagnostic.channels):
                output.groups[domain.name].createVariable(
                    "crps_" + channel, np.float32, ("time", "lat", "lon")
                )
                output.groups[domain.name].createVariable(
                    "rmse_" + channel, np.float32, ("time", "lat", "lon")
                )
                output.groups[domain.name].createVariable(
                    "spread_" + channel, np.float32, ("time", "lat", "lon")
                )
                crps = np.zeros((len(lead_time), len(lat), len(lon)))
                rmse = np.zeros((len(lead_time), len(lat), len(lon)))
                spread = np.array(np.sqrt(data[channel].var(["ensemble"])))
                ensemble_mean = np.array(data[channel].mean(["ensemble"]))
                for it, time in enumerate(lead_time):
                    date_obj_now = date_obj + timedelta(hours=6 * (it + 1))
                    target_data = initial_conditions.ic(
                        n_history=0,
                        grid=fcn_model.grid,
                        time=date_obj_now,
                        channel_set=fcn_model.channel_set,
                        source=ic_source,
                    )
                    target_channel = np.squeeze(
                        np.asarray(target_data.sel(channel=channel))
                    )
                    crps[it, :, :] = crps_ensemble(
                        target_channel[lat_sl, lon_sl],
                        data[channel][:, it, :, :],
                        axis=0,
                    )
                    rmse[it, :, :] = np.abs(
                        target_channel[lat_sl, lon_sl] - ensemble_mean[it, :, :]
                    )
                output.groups[domain.name].variables["crps_" + channel][:, :, :] = crps
                output.groups[domain.name].variables["rmse_" + channel][:, :, :] = rmse
                output.groups[domain.name].variables["spread_" + channel][
                    :, :, :
                ] = spread
    output.close()
    return


if __name__ == "__main__":
    main()

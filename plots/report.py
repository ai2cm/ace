import glob
from typing import List

import xarray
import numpy as np
import os
import matplotlib.pyplot as plt
import datapane
import logging
from fcn_mip.report import scorecard

logging.basicConfig(level=logging.DEBUG)


def plot_with_timedelta(ax, x, dim="lead_time", **kwargs):
    dt = np.timedelta64(1, "h")
    y = x.assign_coords({dim: x[dim] / dt})
    y[dim].attrs["units"] = "hours"
    return y.plot(ax=ax, **kwargs)


def get_metric(files, metric, channel):
    data = {}
    for f in files:
        ds = xarray.open_dataset(f)
        try:
            var = ds[metric].sel(channel=channel)
            data[ds.attrs["model"]] = var
        except KeyError:
            pass
    return data


def plot_metrics(files, metric, channels):
    for v in channels:
        fig, ax = plt.subplots()
        data = get_metric(files, metric, v)

        for key in data:
            plot_with_timedelta(ax, data[key], label=key)

        ax.legend()
        yield datapane.Plot(fig, label=v, responsive=False)
        plt.close(fig)


acc_text = """## Anomaly Correlation Coefficient

Note that the ACCs for 34 channel and 73 channel models are not strictly
comparable. The time-mean of the 73 channel data is computed over a longer
interval of time than for the 34 channel data. Please use the RMSE plots to
compare models trained with different channel sets, for now.
"""


def acc_section(models, baseline_model):
    files = glob.glob("34Vars/acc/*.nc")

    def _is_in_models(path):
        with xarray.open_dataset(path) as ds:
            model = ds.model
            return model in models

    if models:
        files = [f for f in files if _is_in_models(f)]

    models = models + [baseline_model]

    with xarray.open_dataset(files[1]) as data:
        variables = data.channel.values.tolist()
        df = data.initial_times.to_dataframe()
        num_times = len(data.initial_times)

    blocks = []

    scorecard_data = scorecard.read(files, scorecard.CHANNELS, scorecard.LEAD_TIMES)
    table = datapane.Table(scorecard.view(scorecard_data, baseline_model))
    text = datapane.Text(
        f""" Determinstic Scorecard
Shows the fractional change in RMSE relative to **{baseline_model}**. Lower is better.
"""
    )
    blocks += [
        text,
        table,
    ]

    acc_figs = list(plot_metrics(files, "acc", variables))
    rmse_figs = list(plot_metrics(files, "rmse", variables))

    blocks += [
        datapane.Text("## RMSE"),
        datapane.Select(blocks=rmse_figs, type=datapane.SelectType.DROPDOWN),
        datapane.Text(acc_text),
        datapane.Select(blocks=acc_figs, type=datapane.SelectType.DROPDOWN),
        datapane.Text(f"## Timesteps in plot\n\nNumber of times: {num_times}"),
        datapane.DataTable(df),
    ]

    return [
        datapane.Page(
            blocks=blocks,
            title="Medium-range Forecast Accuracy",
        )
    ]


def plot_tcwv_weekly(ds):
    tcwv = ds.tcwv
    tcwv.attrs["long_name"] = "total column water vapor"
    tcwv.attrs["units"] = "mm"
    tcwv.resample(time="7D").nearest().plot(
        col="time", col_wrap=3, vmax=80, vmin=0, rasterized=True
    )


def open_file(path, group):
    root = xarray.open_dataset(path)
    group = xarray.open_dataset(path, group=group)
    return root.merge(group)


def long_term():
    files = glob.glob("34Vars/long/*.nc")

    if not files:
        return

    def plot(f):
        print(f"Plotting {f}")
        ds = open_file(f, group="global")
        plot_tcwv_weekly(ds)
        fig = plt.gcf()
        return datapane.Plot(fig, label=f)

    plots = [plot(f) for f in files]
    yield datapane.Page(
        blocks=[datapane.Text("## TCWV"), datapane.Select(blocks=plots)],
        title="9-week roll-outs",
    )


def id_from_path(f):
    basename = os.path.basename(f)
    id_, _ = os.path.splitext(basename)
    return id_


def multi_year():
    files = glob.glob("34Vars/2year/*.zarr")
    for f in files:
        id_ = id_from_path(f)
        ds = xarray.open_zarr(f, mask_and_scale=False)
        ds = ds.fields.to_dataset("channel")

        for key in ds:
            kwargs = dict(vmax=80, vmin=0) if key == "tcwv" else {}
            monthly = ds[key].resample(time="3MS").nearest()
            deg1 = monthly.coarsen(lat=4, lon=4).mean()
            fig = plt.figure()
            deg1.plot(col="time", col_wrap=4, rasterized=True, **kwargs)
            output_file = f"report/2year/{id_}/{key}.png"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close(fig)


def list_6month():
    try:
        runs_6month = os.listdir("report/6month")
    except FileNotFoundError:
        return []

    models = [os.path.basename(run) for run in runs_6month]
    return models


def get_6month_image(model, field):
    model_dir = os.path.join("report/6month", model)
    image = os.path.join(model_dir, field) + ".png"
    return image


def time_average(field):
    def _to_image(model):
        image_path = get_6month_image(model, field)
        return datapane.Media(file=image_path, label=model, caption=model)

    models = [_to_image(model) for model in list_6month()]
    if not models:
        return []
    elif len(models) == 1:
        models = models[0]
    else:
        models = datapane.Select(blocks=models)

    label = datapane.Text(f"## {field}")
    return [datapane.Group(blocks=[label, models])]


def climate_section():
    yield datapane.Page(
        blocks=[
            datapane.Text("# Time Averages"),
            *time_average("tcwv"),
            *time_average("t2m"),
        ],
        title="Time Averages",
    )


def get_sections(models: List[str], baseline_model: str):
    yield from acc_section(models, baseline_model)
    yield from long_term()
    yield from climate_section()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        type=str,
        help="If provided, limit the medium-range weather page to just these models.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="IFS",
        help="Model to use as baseline for the scorecard. Defaults to IFS.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="index.html",
        help="The html file will be saved at this location. Defaults to index.html.",
    )
    args = parser.parse_args()

    sections = get_sections(args.models, args.baseline_model)
    report = datapane.App(*sections)
    report.save(args.output)

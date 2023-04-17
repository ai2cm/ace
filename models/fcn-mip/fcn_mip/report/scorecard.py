from typing import List
import xarray
import pandas
import numpy as np
import datapane
from matplotlib import colors
import sys

import datetime

import typer

CHANNELS = [
    "z850",
    "t850",
    "r850",
    "z500",
    "t500",
    "r500",
    "z200",
    "t200",
    "r200",
    "tcwv",
    "t2m",
    "u10m",
    "v10m",
]


LEAD_TIMES = [
    datetime.timedelta(days=3),
    datetime.timedelta(days=7),
    datetime.timedelta(days=14),
]


def get_cmap():
    NVIDIA_GREEN = "#76b900"
    BLUE = "#0072CE"
    white = "#ffffff"
    return colors.LinearSegmentedColormap.from_list(
        "nvidia", colors=[NVIDIA_GREEN, white, BLUE]
    )


def read(files, channels, lead_times):
    score_card_data = {}
    for f in files:
        with xarray.open_dataset(f) as ds:
            score_card_data[ds.model] = {}
            for c in channels:
                if c in ds.channel:
                    score_card_data[ds.model][c] = {}
                    for lead_time in lead_times:
                        scalar = ds.rmse.sel(channel=c, lead_time=lead_time)
                        score_card_data[ds.model][c][lead_time] = scalar.item()
    return score_card_data


def view(score_card_data, baseline_model):

    baseline_scores = score_card_data.pop(baseline_model)
    normalized_scores = []
    for model in score_card_data:
        for c in set(score_card_data[model]).intersection(baseline_scores):
            for lead_time in baseline_scores[c]:
                base_score = baseline_scores[c][lead_time]
                model_score = score_card_data[model][c][lead_time]
                relative_score = model_score / base_score - 1
                normalized_scores.append(
                    {
                        "model": model,
                        "channel": c,
                        "rmse": relative_score,
                        "lead_time": lead_time,
                    }
                )
    df = pandas.DataFrame.from_records(normalized_scores)

    unstacked = df.set_index(["model", "channel", "lead_time"]).unstack("model")
    unstacked = unstacked.droplevel(axis=1, level=0)
    unstacked = unstacked.sort_values(by=[("z500", "3d")], axis=1, ascending=False)

    def rotate(_):
        return "writing-mode: vertical-rl; text-align: end;"
        return "width: 50px; overflow-wrap: break-word;"

    def colorna(i):
        if np.isnan(i):
            return "color: white; background-color: white;"

    styled = (
        unstacked.style.background_gradient(cmap=get_cmap(), vmin=-0.5, vmax=0.5)
        .format(precision=2)
        .applymap_index(rotate, axis=1)
        .applymap(colorna)
        .set_properties(**{"font-size": "small"})
    )

    return styled


def main(files: List[str], output: str, baseline: str = "baseline_afno_26"):
    read(files, CHANNELS, LEAD_TIMES)
    scorecard_data = read(files, CHANNELS, LEAD_TIMES)
    styler = view(scorecard_data, baseline)

    cmd = " ".join(sys.argv)

    footer = datapane.Text(
        f"""
    To reproduce:
    ```
    {cmd}
    ```
    """
    )

    app = datapane.App(
        datapane.Text(f"# Scorecard\n RMSE relative to {baseline}. Lower is better."),
        datapane.Table(styler),
        footer,
    )

    app.save(output)


if __name__ == "__main__":
    typer.run(main)
    pass

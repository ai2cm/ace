# Diagnostic: is the cascade-infill-then-sr SR stage sensitive to inputs
# that differ subtly from real dense truth, or is the stage-1 temporal
# infill itself already poor at track 789?
#
# Computes min-SLP-in-window time series (same track/window/timesteps as
# tc_min_pressure_timeseries.py) for the two 100km-resolution sources:
#   - "100km truth": the real dense 100km truth that st-singlestage-flat
#     conditions on (2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr)
#   - "100km infill": the stage-1 temporal-infill model's own output
#     (ensemble member 0, the same member propagated into the cascade's
#     SR stage) -- video-pmd-5ch-per-channel-kernel-global-1degree-24to3-v1
#
# If "100km infill" tracks "100km truth" closely (including at 00Z), but
# the downstream 25km cascade-infill-then-sr line still diverges sharply
# from st-singlestage-flat at 00Z (see the pre-computed CSV from
# tc_min_pressure_timeseries.py), that localizes the problem to the SR
# stage's sensitivity to slightly-off-distribution conditioning, not to
# the infill stage being bad at reconstruction.
import re

import cftime
import numpy as np
import pandas as pd
import xarray as xr

TRUTH_100KM = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr"
)
INFILL_100KM = (
    "/climate-default/2026-06-25-temporal-diffusion/inference/"
    "video-pmd-5ch-per-channel-kernel-global-1degree-24to3-v1/"
    "test-2023-2024-ens32.zarr"
)
KNOWN_TRACKS_CSV = "/known_tracks/known_tracks_2023_filtered_25km.csv"
TRACK_ID = 789
END_TIME = "2023-06-01 00:00:00"
WINDOW_DEG = 2.5
OUT_CSV = "/results/tc_100km_infill_vs_truth_track789.csv"


def parse_cftime(time_str: str) -> cftime.DatetimeJulian:
    m = re.match(r"(\d+)-(\d+)-(\d+)[ T](\d+):(\d+):(\d+)", time_str)
    assert m is not None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    return cftime.DatetimeJulian(y, mo, d, h, mi, s)


def load_prmsl(path, ens_idx):
    ds = xr.open_zarr(path)
    da = ds["PRMSL"]
    if ens_idx is not None and "ensemble" in da.dims:
        da = da.isel(ensemble=ens_idx)
    return da


def main():
    df = pd.read_csv(KNOWN_TRACKS_CSV)
    track = df[df.track_id == TRACK_ID].sort_values("time")
    track = track[track["time"] <= END_TIME]
    times = track["time"].tolist()
    lats = track["lat"].tolist()
    lons = track["lon"].tolist()
    print(f"{len(times)} frames: {times[0]} -> {times[-1]}")

    fields = {
        "100km truth": load_prmsl(TRUTH_100KM, None),
        "100km infill": load_prmsl(INFILL_100KM, 0),
    }
    row_names = list(fields.keys())

    records: dict[str, list[float]] = {name: [] for name in row_names}
    for i, (t_str, lat0, lon0) in enumerate(zip(times, lats, lons)):
        t = parse_cftime(t_str)
        lat_win = slice(lat0 - WINDOW_DEG, lat0 + WINDOW_DEG)
        lon_win = slice(lon0 - WINDOW_DEG, lon0 + WINDOW_DEG)
        for name in row_names:
            da = (
                fields[name]
                .sel(time=t, method="nearest")
                .sel(latitude=lat_win, longitude=lon_win)
            )
            vals = da.values
            if np.nanmean(vals) > 2000:
                vals = vals / 100.0
            records[name].append(np.nanmin(vals) if vals.size else np.nan)
        if (i + 1) % 10 == 0 or i == len(times) - 1:
            print(f"frame {i + 1}/{len(times)} done")

    out = pd.DataFrame({"time": times, **records})
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()

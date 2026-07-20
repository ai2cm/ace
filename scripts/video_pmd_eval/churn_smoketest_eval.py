# One-off scoring for the churn=20 smoke test (see
# ../../configs/experiments/2026-07-20-video-pmd-bb-pcn-churn20-smoketest/).
# Compares bb-pcn at churn=0 (the original full-year inference output) vs.
# churn=20 (the smoke test's 10-day output) on the exact same shared
# sub-window (2023-01-01..2023-01-04) so the two are directly comparable --
# crps_eval.py's own 4-season-pooled numbers aren't a clean baseline here
# since they average over dates the churn=20 run never generated.
import cftime
import numpy as np
import pandas as pd
import xarray as xr

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

TRUTH_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2025-07-25-X-SHiELD-AMIP-FME-3h.zarr"
)
CHURN0_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/inference/"
    "video-pmd-bb-pcn-global-1degree-24to3-v1/test-2023-2024-ens32.zarr"
)
CHURN20_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/inference/"
    "video-pmd-bb-pcn-churn20-global-1degree-24to3-v1/"
    "smoketest-jan2023-ens32.zarr"
)
CHANNELS = [
    "eastward_wind_at_ten_meters",
    "northward_wind_at_ten_meters",
    "PRMSL",
    "PRATEsfc",
]
UNITS = {
    "eastward_wind_at_ten_meters": "m/s",
    "northward_wind_at_ten_meters": "m/s",
    "PRMSL": "mb",
    "PRATEsfc": "kg/m2/s",
}
WINDOW = (cftime.DatetimeJulian(2023, 1, 1), cftime.DatetimeJulian(2023, 1, 4))


def crps_fair(ens, truth_arr):
    M = ens.shape[-1]
    sorted_ens = np.sort(ens, axis=-1)
    k = np.arange(1, M + 1)
    weighted_sum = np.tensordot(sorted_ens, (2 * k - M - 1), axes=([-1], [0]))
    term2 = weighted_sum / (M * (M - 1))
    term1 = np.abs(ens - truth_arr[..., None]).mean(axis=-1)
    return term1 - term2


def area_weighted_mean(arr, area_weight, lat_axis):
    shape = [1] * arr.ndim
    shape[lat_axis] = len(area_weight)
    w = area_weight.reshape(shape)
    w = np.broadcast_to(w, arr.shape)
    return np.sum(arr * w) / np.sum(w)


def spread_skill(ens, truth_arr, area_weight, lat_axis, member_axis=-1):
    M = ens.shape[member_axis]
    ens_mean = ens.mean(axis=member_axis)
    rmse = np.sqrt(area_weighted_mean((ens_mean - truth_arr) ** 2, area_weight, lat_axis))
    var = ens.var(axis=member_axis, ddof=1)
    spread = np.sqrt(area_weighted_mean(var, area_weight, lat_axis)) * np.sqrt((M + 1) / M)
    return spread, rmse, spread / rmse


def score(label, pred_zarr, truth_raw):
    pred_full = xr.open_zarr(pred_zarr)
    truth_full = truth_raw.sel(
        latitude=pred_full.latitude, longitude=pred_full.longitude, method="nearest"
    )
    p_win = pred_full.sel(time=slice(*WINDOW))
    t_win = truth_full.sel(time=slice(*WINDOW))
    interior_mask = p_win["frame_source"].values == 1
    lat = pred_full["latitude"].values
    area_weight = np.cos(np.radians(lat))

    print(f"{label:12s} {p_win.sizes['time']:3d} timesteps, "
          f"{int(interior_mask.sum()):2d} interior, window {WINDOW[0]}..{WINDOW[1]}")

    rows = []
    for name in CHANNELS:
        p = p_win[name].isel(time=interior_mask).transpose(
            "time", "latitude", "longitude", "ensemble").values
        t = t_win[name].isel(time=interior_mask).transpose(
            "time", "latitude", "longitude").values
        crps_val = area_weighted_mean(crps_fair(p, t), area_weight, lat_axis=1)
        spread, rmse, ratio = spread_skill(p, t, area_weight, lat_axis=1)
        rows.append({
            "channel": name, "units": UNITS[name], "n_frames": p.shape[0],
            "CRPS": crps_val, "spread": spread, "RMSE (ens mean)": rmse,
            "spread/skill ratio": ratio,
        })
    return pd.DataFrame(rows).set_index("channel")


def main():
    truth_raw = xr.open_zarr(TRUTH_ZARR)
    churn0 = score("churn=0", CHURN0_ZARR, truth_raw)
    churn20 = score("churn=20", CHURN20_ZARR, truth_raw)

    print("\n--- churn=0 (bb-pcn as originally run) ---")
    print(churn0)
    print("\n--- churn=20 (smoke test) ---")
    print(churn20)

    combined = pd.concat({"churn=0": churn0, "churn=20": churn20}, names=["model"]).reset_index()
    combined = combined[["model", "channel", "units", "n_frames", "CRPS",
                          "spread", "RMSE (ens mean)", "spread/skill ratio"]]
    print("\n--- Combined (same 2023-01-01..2023-01-04 window, both models) ---")
    print(combined.set_index(["channel", "model"]).sort_index())
    combined.to_csv("/results/churn_smoketest_comparison.csv", index=False)
    print("\nSaved /results/churn_smoketest_comparison.csv")


if __name__ == "__main__":
    main()

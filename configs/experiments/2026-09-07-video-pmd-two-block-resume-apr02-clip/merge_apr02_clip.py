"""Copy the 8-frame 2023-04-02 clip from the scratch RESUME store into the
main two-block inference zarr, in place (zarr mode="r+"), WITHOUT touching
anything else. Run from a weka-mounted Beaker session.

Safety rails:
  - matches frames by timestamp, not a hard-coded index
  - asserts the 8 timestamps are contiguous in the main store
  - asserts the target region in the main store is currently ALL-NaN
    (i.e. really is the gap) before writing -- aborts otherwise
  - asserts lat/lon grids match between the two stores
  - verifies the region is fully finite afterwards
"""
import numpy as np
import xarray as xr
import zarr

INF = "/climate-default/2026-06-25-temporal-diffusion/inference/"
MAIN = INF + ("video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-"
              "coarse-endpoints-flat/test-2023-2024-ens4-global.zarr")
SCRATCH = INF + ("video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-"
                 "coarse-endpoints-flat/RESUME-2023-04-02-clip.zarr")
VARS = ["eastward_wind_at_ten_meters", "northward_wind_at_ten_meters",
        "PRMSL", "PRATEsfc", "air_temperature_at_two_meters"]
N_FRAMES = 8  # Apr 2 00,03,06,09,12,15,18,21 Z -- a non-last tumbling clip


def main():
    m = xr.open_zarr(MAIN)
    s = xr.open_zarr(SCRATCH)
    print(f"MAIN   time {m.time.values[0]} .. {m.time.values[-1]}  "
          f"shape {dict(m.sizes)}", flush=True)
    print(f"SCRATCH time {s.time.values[0]} .. {s.time.values[-1]}  "
          f"shape {dict(s.sizes)}", flush=True)

    assert np.array_equal(m.latitude.values, s.latitude.values), "lat grid mismatch"
    assert np.array_equal(m.longitude.values, s.longitude.values), "lon grid mismatch"
    assert m.sizes["ensemble"] == s.sizes["ensemble"], "ensemble size mismatch"

    scr_times = s.time.values[:N_FRAMES]
    main_times = list(m.time.values)
    idx = [main_times.index(t) for t in scr_times]
    assert idx == list(range(idx[0], idx[0] + N_FRAMES)), \
        f"target frames not contiguous in MAIN: {idx}"
    t0 = idx[0]
    print(f"target MAIN time indices [{t0}:{t0 + N_FRAMES}]  "
          f"({main_times[t0]} .. {main_times[t0 + N_FRAMES - 1]})", flush=True)

    zg_main = zarr.open_group(MAIN, mode="r+")
    zg_scr = zarr.open_group(SCRATCH, mode="r")

    for v in VARS:
        before = zg_main[v][t0:t0 + N_FRAMES]
        n_nan = int(np.isnan(before).sum())
        assert n_nan == before.size, (
            f"{v}: target region is NOT all-NaN before write "
            f"({before.size - n_nan}/{before.size} finite) -- ABORTING, "
            "the gap analysis is wrong")
        src = zg_scr[v][0:N_FRAMES]
        assert np.isfinite(src).all(), f"{v}: scratch clip has non-finite values"
        assert src.shape == before.shape, f"{v}: shape {src.shape} vs {before.shape}"
        zg_main[v][t0:t0 + N_FRAMES] = src
        after = zg_main[v][t0:t0 + N_FRAMES]
        assert np.isfinite(after).all(), f"{v}: region still has NaN after write"
        print(f"  {v:32s} wrote {src.shape}  "
              f"range [{np.nanmin(after):.3g}, {np.nanmax(after):.3g}]", flush=True)

    # final: whole-store NaN scan on PRMSL to confirm no gaps remain
    prmsl = zg_main["PRMSL"]
    per_t_nan = np.isnan(prmsl[:]).reshape(prmsl.shape[0], -1).any(axis=1)
    bad = np.where(per_t_nan)[0]
    if len(bad):
        print(f"WARNING: {len(bad)} time steps still have NaN in PRMSL: "
              f"{[str(main_times[i]) for i in bad[:10]]}", flush=True)
    else:
        print("PRMSL: no NaN anywhere in the store -- gap filled.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

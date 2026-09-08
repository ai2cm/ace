"""Copy the 8-frame 2023-04-02 clip from the scratch RESUME store into the
main two-block inference zarr, in place (zarr mode="r+"), WITHOUT touching
anything else. Run from a weka-mounted Beaker session.

Safety rails:
  - matches frames by timestamp, not a hard-coded index
  - asserts the 8 timestamps are contiguous in the main store
  - asserts the target region in the main store is currently the GAP FILL
    -- all-zero OR all-NaN (the ZarrWriter fill value is 0.0, so an
    unwritten clip reads back as zeros, not NaN) -- aborts otherwise
  - asserts lat/lon grids match between the two stores
  - verifies the region is non-trivial afterwards
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

    def looks_generated(a):
        """A real generated clip: finite, and NOT all-zero. Precip is
        legitimately >90% exact zeros, so don't use a nonzero-fraction test --
        just require some spatial variance / a nonzero max."""
        return (np.isfinite(a).all() and np.nanmax(np.abs(a)) > 1e-9
                and float(np.nanstd(a)) > 0.0)

    for v in VARS:
        before = zg_main[v][t0:t0 + N_FRAMES]
        src = zg_scr[v][0:N_FRAMES]
        assert src.shape == before.shape, f"{v}: shape {src.shape} vs {before.shape}"
        assert looks_generated(src), f"{v}: scratch clip is empty/degenerate"
        gap = np.isnan(before) | (before == 0.0)
        if gap.all():
            zg_main[v][t0:t0 + N_FRAMES] = src
            after = zg_main[v][t0:t0 + N_FRAMES]
            assert looks_generated(after), f"{v}: region still empty after write"
            print(f"  {v:32s} WROTE  range [{np.nanmin(after):.3g}, {np.nanmax(after):.3g}]",
                  flush=True)
        elif np.allclose(before, src, equal_nan=True):
            print(f"  {v:32s} already merged (matches scratch) -- skip", flush=True)
        else:
            raise AssertionError(
                f"{v}: target region is neither the gap nor equal to the scratch "
                f"clip ({int((~gap).sum())}/{before.size} real, "
                f"max|diff|={np.nanmax(np.abs(before - src)):.3g}) -- ABORTING")

    # final: whole-store gap scan, all channels, confirm no all-zero/NaN frames
    ok = True
    for v in VARS:
        arr = zg_main[v]
        empty = []
        for s in range(0, arr.shape[0], 32):
            c = arr[s:s + 32].reshape(min(32, arr.shape[0] - s), -1)
            for k in range(c.shape[0]):
                if np.isnan(c[k]).any() or (c[k] == 0.0).all():
                    empty.append(s + k)
        if empty:
            ok = False
            print(f"WARNING {v}: {len(empty)} empty frames: "
                  f"{[str(main_times[i]) for i in empty[:8]]}", flush=True)
        else:
            print(f"  {v}: no empty frames", flush=True)
    print("ALL CHANNELS CLEAN -- gap filled." if ok else "STILL GAPS -- see above",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

# Combine a 4-region-tiled stage-2 model's output into ONE global zarr, for
# a single ensemble member -- detect_tc_tracks.py needs one contiguous zarr
# path, not 4 region stores. Lazy (dask-backed) concat + .to_zarr(), so the
# ~1-year x 720x1440 x 5-channel combined store streams to disk in chunks
# rather than materializing in memory (same OOM lesson as crps_eval.py's
# _load_pred_window, applied to a write instead of a read).
import argparse

import xarray as xr

PATCHED_MODELS = {
    # HiRO-ACE-style spatial downscaling baseline: plain single-frame SR,
    # zero temporal conditioning. n_ens=4 test-inference (see
    # crps_eval.py's PATCHED_MODELS comment for the full provenance note).
    # Not on weka -- mounted at /hiro_result only when "hiro" is requested.
    "hiro": "/hiro_result/test-2023-2024-ens4.zarr",
    # Single-stage coarse-endpoints (v2 of the single-stage architecture),
    # global patch-tiled inference -- ONE contiguous global zarr, so a
    # plain str path (see crps_eval.py's PATCHED_MODELS comment for the
    # full architecture/provenance note).
    "st-singlestage-coarse-endpoints-flat": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat/"
        "test-2023-2024-ens4-global.zarr"
    ),
    "st-singlestage-coarse-endpoints-ou": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-ou/"
        "test-2023-2024-ens4-global.zarr"
    ),
    # v2 retrains of st-flat/st-ou after the endpoint-only-conditioning fix
    # (see crps_eval.py's PATCHED_MODELS comment for the full caveat --
    # epoch 41/200, preliminary/undertrained).
    "st-flat-v2": {
        "mid_west": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v2/"
            "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr"
        ),
        "mid_east": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v2/"
            "test-2023-2024-ens4-region-lat-44to44-lon180to360.zarr"
        ),
        "north_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v2/"
            "test-2023-2024-ens4-region-lat44to88-lon0to360.zarr"
        ),
        "south_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v2/"
            "test-2023-2024-ens4-region-lat-88to-44-lon0to360.zarr"
        ),
    },
    "st-ou-v2": {
        "mid_west": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2/"
            "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr"
        ),
        "mid_east": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2/"
            "test-2023-2024-ens4-region-lat-44to44-lon180to360.zarr"
        ),
        "north_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2/"
            "test-2023-2024-ens4-region-lat44to88-lon0to360.zarr"
        ),
        "south_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2/"
            "test-2023-2024-ens4-region-lat-88to-44-lon0to360.zarr"
        ),
    },
    "st-singlestage-flat": {
        "mid_west": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr"
        ),
        "mid_east": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat-44to44-lon180to360.zarr"
        ),
        "north_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat44to88-lon0to360.zarr"
        ),
        "south_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat-88to-44-lon0to360.zarr"
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine a 4-region-tiled model's output into one global zarr"
    )
    parser.add_argument("--model", default="st-singlestage-flat", choices=sorted(PATCHED_MODELS))
    parser.add_argument("--ensemble-member", type=int, default=0)
    parser.add_argument("--out-zarr", required=True)
    parser.add_argument("--chunk-time", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = PATCHED_MODELS[args.model]
    if not isinstance(spec, dict):
        # Already a single global zarr (e.g. divide_generation patch-tiled
        # inference output) -- just apply the ensemble-member selection,
        # no region merge needed.
        combined = xr.open_zarr(spec).isel(ensemble=args.ensemble_member, drop=True)
    else:
        parts = {
            region: xr.open_zarr(path).isel(ensemble=args.ensemble_member, drop=True)
            for region, path in spec.items()
        }
        mid_band = xr.concat([parts["mid_west"], parts["mid_east"]], dim="longitude")
        combined = xr.concat([parts["south_cap"], mid_band, parts["north_cap"]], dim="latitude")
    combined = combined.chunk({"time": args.chunk_time, "latitude": -1, "longitude": -1})
    # Each source region zarr's own chunk encoding (e.g. frame_source's
    # single-chunk-covering-the-original-multi-year-axis metadata) survives
    # the concat and conflicts with the new dask chunking above -- drop it so
    # zarr derives chunks purely from the dask array instead.
    for var in combined.variables.values():
        var.encoding.pop("chunks", None)
    combined.encoding.pop("chunks", None)

    print(f"Combined shape: { {k: v for k, v in combined.sizes.items()} }")
    print(f"Writing to {args.out_zarr} ...")
    combined.to_zarr(args.out_zarr, mode="w")
    print("done.")


if __name__ == "__main__":
    main()

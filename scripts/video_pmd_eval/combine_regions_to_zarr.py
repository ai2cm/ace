# Combine a 4-region-tiled stage-2 model's output into ONE global zarr, for
# a single ensemble member -- detect_tc_tracks.py needs one contiguous zarr
# path, not 4 region stores. Lazy (dask-backed) concat + .to_zarr(), so the
# ~1-year x 720x1440 x 5-channel combined store streams to disk in chunks
# rather than materializing in memory (same OOM lesson as crps_eval.py's
# _load_pred_window, applied to a write instead of a read).
import argparse

import xarray as xr

PATCHED_MODELS = {
    # No-temporal-attention ablation of st-singlestage-coarse-endpoints-flat
    # (see crps_eval.py's PATCHED_MODELS comment for the full provenance
    # note). Global patch-tiled inference, ONE contiguous global zarr on
    # weka -- plain str path.
    "st-singlestage-coarse-endpoints-flat-no-temporal-attn": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat-no-temporal-attn/"
        "test-2023-2024-ens4-global.zarr"
    ),
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
    # Cascade: 100km temporal infill -> 25km spatial SR (same "hiro"
    # checkpoint, conditioned on the infill output instead of real dense
    # truth). See crps_eval.py's PATCHED_MODELS comment for the full
    # pipeline rationale.
    "cascade-infill-then-sr": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "hiro-downscaling-25km-100km-global-5ch-v6-cascade-infill-then-sr/"
        "test-2023-2024-ens4.zarr"
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
    parser.add_argument(
        "--model", default="st-singlestage-flat", choices=sorted(PATCHED_MODELS)
    )
    parser.add_argument("--ensemble-member", type=int, default=0)
    parser.add_argument("--out-zarr", required=True)
    parser.add_argument("--chunk-time", type=int, default=256)
    return parser.parse_args()


# Models produced via fme.downscaling.inference's ZarrWriter (spatial-only
# SR, not the video trainer): "hiro" and its cascade variant, which reuses
# hiro's exact checkpoint. Both need two fixes their video-PMD siblings
# don't:
#   1. PRMSL is natively stored in hPa (mean ~1010) here, unlike every
#      video-PMD PATCHED_MODELS entry, whose PRMSL is in Pa (matching
#      known_tracks_2023_filtered.csv's ~100000 Pa scale). CRPS/MSE scoring
#      is unaffected (crps_eval.py's own truth zarr is ALSO in hPa, so it's
#      internally unit-consistent), but detect_tc_tracks.py's TempestExtremes
#      thresholds are calibrated for Pa and find zero candidate storm nodes
#      against raw hPa values -- a real, silent unit mismatch, not a
#      "no storms" result.
#   2. The source zarr's own chunk encoding uses zarr v3 sharding (ZarrWriter
#      writes with shards) -- popping only "chunks" isn't enough; the
#      leftover encoding["shards"] tuple still has as many entries as the
#      pre-ensemble-isel array and conflicts with the new dask chunking
#      below, raising "zip() argument 3 is shorter than arguments 1-2" on
#      write. Clearing each variable's FULL .encoding dict (not just
#      "chunks") avoids this.
HPA_SHARDED_MODELS = {"hiro", "cascade-infill-then-sr"}


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
        combined = xr.concat(
            [parts["south_cap"], mid_band, parts["north_cap"]], dim="latitude"
        )
    if args.model in HPA_SHARDED_MODELS and "PRMSL" in combined:
        combined["PRMSL"] = combined["PRMSL"] * 100.0
    combined = combined.chunk(
        {"time": args.chunk_time, "latitude": -1, "longitude": -1}
    )
    # Each source zarr's own chunk encoding survives the concat/isel and
    # conflicts with the new dask chunking above -- clear it so zarr derives
    # chunks purely from the dask array instead. Popping only "chunks" isn't
    # enough for a zarr v3 sharded store (see HPA_SHARDED_MODELS comment
    # above) -- clear the full encoding dict for those models; for everyone
    # else, popping "chunks" (prior behavior) is sufficient and lower-risk.
    for var in combined.variables.values():
        if args.model in HPA_SHARDED_MODELS:
            var.encoding = {}
        else:
            var.encoding.pop("chunks", None)
    combined.encoding.pop("chunks", None)

    print(f"Combined shape: { {k: v for k, v in combined.sizes.items()} }")
    print(f"Writing to {args.out_zarr} ...")
    combined.to_zarr(args.out_zarr, mode="w")
    print("done.")


if __name__ == "__main__":
    main()

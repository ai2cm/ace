"""Operational initial conditions for a coupled SamudrACE (GLO12 + ERA5) forecast.

    python make_ic.py --date 2026-10-16 --out /path/to/ic

For an issue day D this (1) checks that D is an analysed GLO12 day and that
ERA5 (final or ERA5T) covers the instant D+1 00Z, (2) downloads the five GLO12
daily means D-4..D and processes each into the training store's 4-degree
layout, (3) downloads one ERA5 instant plus the day's hourly TOA insolation
and builds the ACE2S daily sample, (4) writes the forcing datasets for the
forecast horizon and a ready-to-run ``fme.coupled.inference`` config.

Outputs in ``--out``:

    validation.json                 what was checked and chosen
    glo12_daily_4deg_<D-4>_<D>.zarr the five processed GLO12 states
    ocean_ic_<D>.nc                 the ocean initial condition (one state)
    atmos_ic_<D>.nc                 the atmosphere initial condition
    atmos_forcing_<D>.zarr          daily exogenous atmosphere forcing
    ocean_forcing_<D>.zarr          5-daily static ocean forcing
    inference_config.yaml           fme.coupled.inference config (local paths)
    manifest.json                   files, sizes, timings

Everything downstream of the download reuses the GLORYS and ERA5 pipeline
code (see ocean_ic.py, atmos_ic.py). ``--dry-run`` stops after validation.
"""

import argparse
import datetime
import json
import logging
import pathlib
import time

import atmos_ic
import make_forcing
import ocean_ic
import pandas as pd
import xarray as xr
import yaml
from common import (
    REFERENCE_ATMOS_STORE,
    REFERENCE_OCEAN_STORE,
    REGRID_WEIGHTS_URL,
    set_time_encoding,
)

log = logging.getLogger("operational_ic")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--date",
        required=True,
        help="issue day D (YYYY-MM-DD): the GLO12 daily mean of D is the ocean state",
    )
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--ocean-days",
        type=int,
        default=5,
        help="GLO12 days to download and process, ending at D (default 5)",
    )
    p.add_argument(
        "--ocean-ic",
        choices=["day", "mean5"],
        default="day",
        help="ocean IC: the day-D state (training convention) or the mean "
        "of the five states",
    )
    p.add_argument(
        "--atmos-source",
        choices=["auto", "final", "era5t"],
        default="auto",
        help="ERA5 model levels: analysis-ready final store, cloud-optimized "
        "ERA5T, or auto",
    )
    p.add_argument(
        "--atmos-full",
        action="store_true",
        help="also build the diagnostic (output-only) variables of the daily "
        "store (~3 GB more)",
    )
    p.add_argument(
        "--forcing-steps",
        type=int,
        default=72,
        help="coupled (5-day) steps of forcing to write; 0 skips the forcing "
        "and config",
    )
    p.add_argument(
        "--n-ensemble",
        type=int,
        default=8,
        help="n_ensemble_per_ic written into the inference config",
    )
    p.add_argument("--reference-ocean-store", default=REFERENCE_OCEAN_STORE)
    p.add_argument("--reference-atmos-store", default=REFERENCE_ATMOS_STORE)
    p.add_argument("--regrid-weights", default=REGRID_WEIGHTS_URL)
    p.add_argument(
        "--sea-ice-volume",
        choices=["thickness", "per_area"],
        default="thickness",
        help="training store convention is 'thickness' (sea_ice_volume = HI)",
    )
    p.add_argument(
        "--no-placeholders",
        action="store_true",
        help="keep GLO12's UI/VI and omit the zero placeholder channels",
    )
    p.add_argument("--skip-ocean", action="store_true")
    p.add_argument("--skip-atmos", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="validate the date and stop")
    p.add_argument(
        "--today",
        default=None,
        help="override today's date (YYYY-MM-DD) for the GLO12 analysis cutoff",
    )
    p.add_argument(
        "--zarr-concurrency",
        type=int,
        default=48,
        help="concurrent chunk fetches (CloudFerro gives ~16 MB/s with 32-48)",
    )
    return p.parse_args()


def write_json(path: pathlib.Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")


def inference_config(
    out: pathlib.Path, day: pd.Timestamp, n_steps: int, n_ensemble: int
) -> dict:
    d = str(day.date())
    return {
        "experiment_dir": str(out / "forecast"),
        "n_coupled_steps": n_steps,
        # must divide n_coupled_steps
        "coupled_steps_in_memory": max(k for k in range(1, 9) if n_steps % k == 0),
        "n_ensemble_per_ic": n_ensemble,
        "checkpoint_path": "/ckpt.tar",
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": False,
            "log_to_file": True,
            "project": "ace-samudra-coupled-cm4",
            "entity": "ai2cm",
        },
        "initial_condition": {
            "ocean": {"path": str(out / f"ocean_ic_{d}.nc"), "engine": "netcdf4"},
            "atmosphere": {"path": str(out / f"atmos_ic_{d}.nc"), "engine": "netcdf4"},
        },
        "forcing_loader": {
            "num_data_workers": 1,  # the loader rejects 0 (persistent workers)
            "atmosphere": {
                "dataset": {
                    "data_path": str(out),
                    "file_pattern": f"atmos_forcing_{d}.zarr",
                    "engine": "zarr",
                }
            },
            "ocean": {
                "dataset": {
                    "data_path": str(out),
                    "file_pattern": f"ocean_forcing_{d}.zarr",
                    "engine": "zarr",
                }
            },
        },
        "data_writer": {
            "atmosphere": {
                "names": [
                    "PRATEsfc",
                    "TMP2m",
                    "PRESsfc",
                    "UGRD10m",
                    "VGRD10m",
                    "h500",
                    "TMP850",
                    "total_frozen_precipitation_rate",
                    "surface_temperature",
                ],
                "save_prediction_files": False,
                "save_monthly_files": True,
            },
            "ocean": {
                "names": ["sst", "thetao_0", "thetao_5", "ocean_sea_ice_fraction"],
                "save_prediction_files": True,
                "save_monthly_files": True,
            },
        },
        "aggregator": {"log_global_mean_time_series": True},
    }


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    logging.getLogger("operational_ic").setLevel(logging.INFO)
    import zarr

    zarr.config.set({"async.concurrency": args.zarr_concurrency})
    day = pd.Timestamp(args.date)
    today = datetime.date.fromisoformat(args.today) if args.today else None
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = str(day.date())
    days = pd.date_range(day - pd.Timedelta(days=args.ocean_days - 1), day, freq="1D")
    timings: dict = {}
    manifest: dict = {"date": d, "args": vars(args), "files": {}}

    # ---- 1. validation
    t0 = time.time()
    validation: dict = {"date": d}
    if not args.skip_ocean:
        validation["ocean"] = ocean_ic.validate_ocean_days(days, today)
    if not args.skip_atmos:
        validation["atmosphere"] = atmos_ic.validate_atmos_day(day, args.atmos_source)
    problems = [
        p
        for k in ("ocean", "atmosphere")
        for p in validation.get(k, {}).get("problems", [])
    ]
    warnings = [
        w
        for k in ("ocean", "atmosphere")
        for w in validation.get(k, {}).get("warnings", [])
    ]
    validation["ok"] = not problems
    write_json(out / "validation.json", validation)
    timings["validation"] = time.time() - t0
    for w in warnings:
        log.warning("%s", w)
    if problems:
        for p in problems:
            log.error("%s", p)
        raise SystemExit(
            f"{d}: not a usable issue day ({len(problems)} problem(s), "
            f"see {out / 'validation.json'})"
        )
    log.info(
        "%s validated: ocean days %s..%s, atmosphere source %s",
        d,
        days[0].date(),
        days[-1].date(),
        validation.get("atmosphere", {}).get("source"),
    )
    if args.dry_run:
        print(json.dumps(validation, indent=1, default=str))
        return

    # ---- 2. ocean
    ocean_ic_ds = None
    if not args.skip_ocean:
        t0 = time.time()
        proc = ocean_ic.OceanProcessor(
            reference_store=args.reference_ocean_store,
            regrid_weights=args.regrid_weights,
            sea_ice_volume=args.sea_ice_volume,
            placeholders=not args.no_placeholders,
        )
        timings["ocean_setup"] = time.time() - t0
        t0 = time.time()
        states = proc.process_days(days)
        daily_path = out / f"glo12_daily_4deg_{days[0].date()}_{days[-1].date()}.zarr"
        set_time_encoding(states).to_zarr(daily_path, mode="w", consolidated=True)
        ocean_ic_ds = ocean_ic.initial_condition(states, args.ocean_ic)
        ic_path = out / f"ocean_ic_{d}.nc"
        set_time_encoding(ocean_ic_ds).to_netcdf(ic_path)
        timings["ocean_processing"] = time.time() - t0
        manifest["files"]["glo12_daily"] = str(daily_path)
        manifest["files"]["ocean_ic"] = str(ic_path)
        ocean_statics = proc.statics
        log.info("ocean IC written: %s (%s)", ic_path, args.ocean_ic)
    else:
        ocean_statics = ocean_ic.load_ocean_statics(args.reference_ocean_store)

    # ---- 3. atmosphere
    atmos_ic_ds = None
    if not args.skip_atmos:
        t0 = time.time()
        if ocean_ic_ds is not None:
            sic = ocean_ic_ds["ocean_sea_ice_fraction"]
        else:
            ref = xr.open_dataset(
                args.reference_ocean_store,
                engine="zarr",
                chunks=None,
                storage_options={"token": "cloud"}
                if args.reference_ocean_store.startswith("gs://")
                else None,
            )
            sic = ref["ocean_sea_ice_fraction"].sel(time=day, method="nearest").load()
            log.warning(
                "--skip-ocean: sea ice fraction for the atmosphere taken from "
                "the reference store at %s",
                str(sic.time.values)[:10],
            )
        atmos_ic_ds = atmos_ic.build_atmosphere_sample(
            day,
            validation["atmosphere"]["source"],
            ocean_statics,
            sic,
            reference_store=args.reference_atmos_store,
            full=args.atmos_full,
        )
        ic_path = out / f"atmos_ic_{d}.nc"
        set_time_encoding(atmos_ic_ds).to_netcdf(ic_path)
        timings["atmosphere"] = time.time() - t0
        manifest["files"]["atmos_ic"] = str(ic_path)
        log.info(
            "atmosphere IC written: %s (ERA5 %s)",
            ic_path,
            validation["atmosphere"]["source"],
        )

    # ---- 4. forcing + config
    if args.forcing_steps > 0 and atmos_ic_ds is not None:
        t0 = time.time()
        af = make_forcing.atmosphere_forcing(
            day, args.forcing_steps, atmos_ic_ds, args.reference_atmos_store
        )
        af_path = out / f"atmos_forcing_{d}.zarr"
        af.to_zarr(af_path, mode="w", consolidated=True)
        of = make_forcing.ocean_forcing(day, args.forcing_steps, ocean_statics)
        of_path = out / f"ocean_forcing_{d}.zarr"
        of.to_zarr(of_path, mode="w", consolidated=True)
        cfg = inference_config(out, day, args.forcing_steps, args.n_ensemble)
        (out / "inference_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        timings["forcing"] = time.time() - t0
        manifest["files"].update(
            {
                "atmos_forcing": str(af_path),
                "ocean_forcing": str(of_path),
                "inference_config": str(out / "inference_config.yaml"),
            }
        )
        log.info(
            "forcing for %d coupled steps written; config %s",
            args.forcing_steps,
            out / "inference_config.yaml",
        )

    manifest["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
    manifest["validation"] = validation
    write_json(out / "manifest.json", manifest)
    log.info("done: %s", json.dumps(manifest["timings_s"]))


if __name__ == "__main__":
    main()

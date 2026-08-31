#!/usr/bin/env python3
"""Generate the all-in-one arm: one Samudra that predicts its own atmosphere.

Transform of the corrected pretrain run config (Troy's
01KW2BQ83EGZ90WZ74CZ4TJATN lineage), per Troy's decisions:

- The ten flux/stress fields Samudra previously consumed as next-step forcing
  become prognostic: the model predicts them and feeds them back, so a free
  rollout needs no atmosphere model at all.
- New prognostics from the 5-day snapshot zarr: h500, TMP850, jet-level winds
  (ACE layer 2, 126-247 hPa) and PRESsfc.
- The only remaining forcings are the truly exogenous ones: DSWRFtoa and
  carbon_dioxide (scalar, broadcast by the loader).
- surface_energy_flux_correction is dropped: its "prescribed" method computes
  the net flux from *forcing* fields, which no longer exist. The heat-content
  corrector stays: it reads the model's own predicted hfds_total_area
  (first-priority branch of _force_conserve_ocean_heat_content), so the ocean
  still conserves heat against its own predicted flux.
- Plain full-field loss, unchanged from the lineage (Troy's call: the arm's
  question is architectural; tendency loss is tendloss's question).
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE / "corrected-pretrain-run-config.yaml"
OUT = HERE / "wave1_configs" / "allinone-pretrain.yaml"

SNAPSHOT_ZARR = "2026-08-31-cm4-1pctCO2-140yr-atmos-5day-snapshots.zarr"

FLUXES_TO_PROGNOSE = [
    "DLWRFsfc",
    "DSWRFsfc",
    "ULWRFsfc",
    "USWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "eastward_surface_wind_stress",
    "northward_surface_wind_stress",
    "total_frozen_precipitation_rate",
]
NEW_PROGNOSTICS = [
    "h500",
    "TMP850",
    "eastward_wind_2",
    "northward_wind_2",
    "PRESsfc",
]
NEW_FORCINGS = ["DSWRFtoa", "carbon_dioxide"]


def add_snapshot_zarr(loader: dict) -> None:
    merge = loader["dataset"]["merge"]
    entry = {
        "data_path": "/climate-default",
        "file_pattern": SNAPSHOT_ZARR,
        "engine": "zarr",
    }
    if "subset" in merge[0]:
        entry["subset"] = copy.deepcopy(merge[0]["subset"])
    merge.append(entry)


def main() -> None:
    with open(BASE) as f:
        c = yaml.safe_load(f)

    add_snapshot_zarr(c["inference"]["loader"])
    add_snapshot_zarr(c["train_loader"])
    add_snapshot_zarr(c["validation"]["loader"])

    stepper = c["stepper"]
    cfg = stepper["step"]["config"]

    assert cfg["next_step_forcing_names"] == FLUXES_TO_PROGNOSE
    cfg["next_step_forcing_names"] = list(NEW_FORCINGS)

    for name in NEW_PROGNOSTICS + NEW_FORCINGS:
        assert name not in cfg["in_names"]
        cfg["in_names"].append(name)
    for name in FLUXES_TO_PROGNOSE + NEW_PROGNOSTICS:
        assert name not in cfg["out_names"]
        cfg["out_names"].append(name)

    excludes = stepper["input_masking"]["exclude_names_and_prefixes"]
    for name in NEW_PROGNOSTICS + NEW_FORCINGS:
        assert name not in excludes
        excludes.append(name)

    corrector = cfg["corrector"]["config"]
    del corrector["surface_energy_flux_correction"]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(yaml.safe_dump(c, sort_keys=False))
    print(f"wrote {OUT}")
    print(
        f"in={len(cfg['in_names'])} out={len(cfg['out_names'])} "
        f"next_step_forcings={cfg['next_step_forcing_names']}"
    )


if __name__ == "__main__":
    main()

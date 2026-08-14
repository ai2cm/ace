#!/usr/bin/env python3
"""Generate per-arm, per-year coupled configs for the prescribe-from-truth probe.

The coupled rollout loses ENSO phase information while the wind-stress bridge
measures healthy (gain ratio 0.98, CI [0.89, 1.08]) and the interface noise
amplitude is correct. So the loss is somewhere the scalar bridge audit cannot
see. This probe localizes it by holding one part of the coupled state at CM4
truth during an otherwise free rollout and asking how much Nino3.4 skill comes
back.

``prescribed_prognostic_names`` overwrites a variable's predicted value with
the value from forcing (truth) data at every step. It requires only that the
name be in that stepper's ``out_names``, so it works for the atmosphere's
*diagnostic* coupling fields (wind stress, surface fluxes) as well as for ocean
prognostics.

Arms (each vs the existing free-running baseline, same ICs):

  free          nothing prescribed; the baseline, rerun only to write the ocean
                interior that no previous eval saved
  subsurface    thetao_1..18 + zos held at truth -> does the rollout corrupt its
                own slow memory?
  sst           sst held at truth -> is the leak the surface field the
                atmosphere sees?
  currents      uo_*/vo_*/ssu/ssv held at truth -> is it the advective feedback?
  windstress    atmosphere's wind stress held at truth -> does the bridge fail
                in a way the scalar gain missed (pattern, curl, phase)?
  fluxes        atmosphere's surface energy/freshwater fluxes held at truth ->
                is the thermodynamic coupling the leak?

Every arm writes the same small monthly-mean output: SST (for skill, directly
comparable to the existing curves) plus zos and four thermocline levels, so the
warm-water-volume / recharge behaviour is visible for the first time.

Usage:
    python make_year_configs_prescribe.py --arm subsurface --years 233 246 250
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from make_year_configs_ar_sst import make_config, monthly_ic_times

# Ocean interior levels to save. thetao_0 is the surface; 2/4/6/8 span roughly
# the thermocline in this 19-level configuration. Kept small on purpose: these
# are monthly means and the point is the warm-water-volume signal, not detail.
INTERIOR_NAMES = ["zos", "thetao_2", "thetao_4", "thetao_6", "thetao_8"]
OCEAN_WRITE_NAMES = ["sst"] + INTERIOR_NAMES

SUBSURFACE = [f"thetao_{k}" for k in range(1, 19)] + ["zos"]
CURRENTS = (
    [f"uo_{k}" for k in range(19)] + [f"vo_{k}" for k in range(19)] + ["ssu", "ssv"]
)
WIND_STRESS = ["eastward_surface_wind_stress", "northward_surface_wind_stress"]
FLUXES = [
    "DLWRFsfc",
    "DSWRFsfc",
    "ULWRFsfc",
    "USWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
]

# arm -> (ocean prescribed names, atmosphere prescribed names)
ARMS: dict[str, tuple[list[str], list[str]]] = {
    "free": ([], []),
    "subsurface": (SUBSURFACE, []),
    "sst": (["sst"], []),
    "currents": (CURRENTS, []),
    "windstress": ([], WIND_STRESS),
    "fluxes": ([], FLUXES),
}


def make_prescribe_config(times: list[str], arm: str) -> dict:
    ocean_names, atmos_names = ARMS[arm]
    config = make_config(times)
    config["data_writer"] = {
        "ocean": {
            "save_prediction_files": False,
            "save_monthly_files": True,
            "names": OCEAN_WRITE_NAMES,
        },
        "atmosphere": {
            "save_prediction_files": False,
            "save_monthly_files": False,
        },
    }
    if ocean_names:
        config["ocean_stepper_override"] = {"prescribed_prognostic_names": ocean_names}
    if atmos_names:
        config["atmosphere_stepper_override"] = {
            "prescribed_prognostic_names": atmos_names
        }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[233, 246, 250],
        help="IC years; must be a subset of the 0231-0250 window used by the "
        "existing evals so the skill curves stay directly comparable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prescribe_year_configs",
    )
    args = parser.parse_args()

    for year in args.years:
        if not 231 <= year <= 250:
            raise SystemExit(f"year {year} is outside the 0231-0250 eval window")

    out = args.output_dir / args.arm
    out.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        cfg = make_prescribe_config(monthly_ic_times(year), args.arm)
        path = out / f"yr{year:04d}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

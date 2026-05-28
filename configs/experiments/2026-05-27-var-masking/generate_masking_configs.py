"""Generate var-masking training configs from the base SFNO config.

Produces one config per (mask_rate, masking_type, gmr, rp) combination:
  - mask_rates: 0.0, 0.2, 0.8  (applied as input_dropout.default_rate)
  - masking_types:
      uniform  - all variables share default_rate
      forcing  - forcing/static variables (land_fraction, ocean_fraction,
                 sea_ice_fraction, DSWRFtoa, HGTsfc) are excluded (rate=0.0)
  - gmr: gmron (global mean removal enabled) / gmroff (disabled)
  - rp:  rpon  (residual_prediction=true)   / rpoff (residual_prediction=false)
"""

import copy
import pathlib

import yaml

FORCING_VARS = [
    "land_fraction",
    "ocean_fraction",
    "sea_ice_fraction",
    "DSWRFtoa",
    "HGTsfc",
]

MASK_RATES = [0.0, 0.2, 0.8]

HERE = pathlib.Path(__file__).parent
BASE_CONFIG = HERE / "ace-train-config-4deg-AIMIP-sfno.yaml"


def build_input_dropout(mask_rate: float, exclude_forcing: bool) -> dict:
    cfg: dict = {"default_rate": mask_rate}
    if exclude_forcing:
        cfg["rates"] = {v: 0.0 for v in FORCING_VARS}
    return cfg


def main():
    with BASE_CONFIG.open() as f:
        base = yaml.safe_load(f)

    for mask_rate in MASK_RATES:
        for exclude_forcing in (False, True):
            if mask_rate == 0.0 and exclude_forcing:
                continue

            for gmr_on in (True, False):
                for rp_on in (True, False):
                    cfg = copy.deepcopy(base)
                    masking_type = "forcing" if exclude_forcing else "uniform"
                    gmr_suffix = "gmron" if gmr_on else "gmroff"
                    rp_suffix = "rpon" if rp_on else "rpoff"
                    name = (
                        f"ace-train-config-4deg-AIMIP-sfno"
                        f"-mask{mask_rate:.2f}-{masking_type}-{gmr_suffix}-{rp_suffix}"
                    )
                    out_path = HERE / f"{name}.yaml"

                    step_cfg = cfg["stepper"]["step"]["config"]
                    step_cfg["input_dropout"] = build_input_dropout(
                        mask_rate, exclude_forcing
                    )
                    step_cfg["residual_prediction"] = rp_on

                    if gmr_on:
                        step_cfg["global_mean_removal"] = {
                            "kind": "shared",
                            "append_as_input": True,
                        }
                    else:
                        step_cfg.pop("global_mean_removal", None)

                    with out_path.open("w") as f:
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

                    print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()

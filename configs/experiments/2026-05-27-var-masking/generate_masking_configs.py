"""Generate var-masking training configs from the base SFNO config.

Produces configs for two masking schemes:

Bernoulli (suffix -bernoulli):
  One config per (mask_rate, masking_type, gmr, rp):
  - mask_rates: 0.0, 0.2  (applied as input_dropout.default_rate)
  - masking_types:
      all      - all variables share default_rate
      noforcing - forcing/static variables (land_fraction, ocean_fraction,
                 sea_ice_fraction, DSWRFtoa, HGTsfc) are excluded (rate=0.0)
  - gmr: gmron (global mean removal enabled) / gmroff (disabled)
  - rp:  rpon  (residual_prediction=true)   / rpoff (residual_prediction=false)

Jeremy/uniform (suffix -uniform):
  One config per (max_vars, masking_type, gmr, rp):
  - max_vars: "all" (0 to all eligible) or an integer k (0 to k)
  - masking_types:
      all      - all variables are eligible for masking
      noforcing - forcing/static variables are excluded via ignore_vars
  - gmr / rp: same as above
  File names use mask{k} where k is "all" or the integer (e.g. mask10).
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

MASK_RATES = [0.0, 0.2, 0.4]
GMR_VALS = [True]
RP_VALS = [False]
EXCLUDE_FORCING = [False, True]
UNIFORM_MAX_VARS: dict[bool, list[int | str]] = {
    False: ["all", 17],  # all vars eligible: 0.20 × 43 ≈ 8.6 → max = 17
    True: ["all", 15],  # forcing vars excluded: 0.20 × 38 ≈ 7.6 → max = 15
}

HERE = pathlib.Path(__file__).parent
BASE_CONFIG_STEMS = [
    "ace-train-config-4deg-AIMIP-sfno",
    "ace-train-config-4deg-AIMIP-nc-sfno",
]


def build_bernoulli_input_dropout(mask_rate: float, exclude_forcing: bool) -> dict:
    cfg: dict = {"default_rate": mask_rate}
    if exclude_forcing:
        cfg["rates"] = {v: 0.0 for v in FORCING_VARS}
    return cfg


def build_uniform_input_dropout(
    exclude_forcing: bool, max_vars: int | str = "all"
) -> dict:
    cfg: dict = {
        "min_vars": "min",
        "max_vars": "max" if max_vars == "all" else max_vars,
    }
    if exclude_forcing:
        cfg["ignore_vars"] = list(FORCING_VARS)
    return cfg


def _apply_common_settings(
    step_cfg: dict, gmr_on: bool, rp_on: bool, input_dropout: dict
) -> None:
    step_cfg["input_dropout"] = input_dropout
    step_cfg["residual_prediction"] = rp_on
    step_cfg["include_channel_mask_inputs"] = True
    if gmr_on:
        step_cfg["global_mean_removal"] = {
            "kind": "shared",
            "append_as_input": True,
        }
    else:
        step_cfg.pop("global_mean_removal", None)


def generate_bernoulli_configs(base: dict, stem: str) -> None:
    for mask_rate in MASK_RATES:
        for exclude_forcing in EXCLUDE_FORCING:
            if mask_rate == 0.0 and exclude_forcing:
                continue
            for gmr_on in GMR_VALS:
                for rp_on in RP_VALS:
                    cfg = copy.deepcopy(base)
                    masking_type = "noforcing" if exclude_forcing else "all"
                    gmr_suffix = "gmron" if gmr_on else "gmroff"
                    rp_suffix = "rpon" if rp_on else "rpoff"
                    name = (
                        f"{stem}"
                        f"-mask{mask_rate:.2f}-{masking_type}"
                        f"-{gmr_suffix}-{rp_suffix}-bernoulli"
                    )
                    out_path = HERE / f"{name}.yaml"
                    step_cfg = cfg["stepper"]["step"]["config"]
                    _apply_common_settings(
                        step_cfg,
                        gmr_on,
                        rp_on,
                        build_bernoulli_input_dropout(mask_rate, exclude_forcing),
                    )
                    with out_path.open("w") as f:
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                    print(f"Wrote {out_path.name}")


def generate_uniform_configs(base: dict, stem: str) -> None:
    for exclude_forcing in EXCLUDE_FORCING:
        for max_vars in UNIFORM_MAX_VARS[exclude_forcing]:
            for gmr_on in GMR_VALS:
                for rp_on in RP_VALS:
                    cfg = copy.deepcopy(base)
                    masking_type = "noforcing" if exclude_forcing else "all"
                    gmr_suffix = "gmron" if gmr_on else "gmroff"
                    rp_suffix = "rpon" if rp_on else "rpoff"
                    k = "all" if max_vars == "all" else max_vars
                    name = (
                        f"{stem}"
                        f"-mask{k}-{masking_type}"
                        f"-{gmr_suffix}-{rp_suffix}-uniform"
                    )
                    out_path = HERE / f"{name}.yaml"
                    step_cfg = cfg["stepper"]["step"]["config"]
                    _apply_common_settings(
                        step_cfg,
                        gmr_on,
                        rp_on,
                        build_uniform_input_dropout(exclude_forcing, max_vars),
                    )
                    with out_path.open("w") as f:
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                    print(f"Wrote {out_path.name}")


def main():
    for stem in BASE_CONFIG_STEMS:
        base_config = HERE / f"{stem}.yaml"
        with base_config.open() as f:
            base = yaml.safe_load(f)
        generate_bernoulli_configs(base, stem)
        generate_uniform_configs(base, stem)


if __name__ == "__main__":
    main()

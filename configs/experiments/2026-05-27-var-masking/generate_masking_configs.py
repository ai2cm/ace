"""Generate var-masking training configs from the base SFNO config.

Produces configs for two masking schemes:

Bernoulli (suffix -bernoulli):
  One config per (mask_rate, masking_type, gmr, rp):
  - mask_rates: 0.0, 0.2, 0.4
    (applied as input_dropout.per_variable.default_rate)
  - masking_types:
      all      - all variables share per_variable.default_rate, except the
                 shared global_mean_removal reference when gmr is enabled
      noforcing - forcing/static variables (land_fraction, ocean_fraction,
                 sea_ice_fraction, DSWRFtoa, HGTsfc) are explicitly
                 handled with rate=0.0; the shared global_mean_removal
                 reference is also handled with rate=0.0 when gmr is enabled
  - gmr: gmron (global mean removal enabled) / gmroff (disabled)
  - rp:  rpon  (residual_prediction=true)   / rpoff (residual_prediction=false)

Jeremy/uniform (suffix -uniform):
  One config per (max_vars, masking_type, gmr, rp):
  - max_vars: "all" (0 to all eligible) or an integer k (0 to k)
  - masking_types:
      all      - all variables are eligible for masking, except the shared
                 global_mean_removal reference when gmr is enabled
      noforcing - forcing/static variables are excluded via uniform.ignore_vars
  - gmr / rp: same as above
  File names use mask{k} where k is "all" or the integer (e.g. mask10).

CO2 variants are generated for every Bernoulli and uniform config:
  - suffix -co2-mask: global_mean_co2 is masked the same way as other variables
  - suffix -co2-nomask: global_mean_co2 is excluded from input dropout
"""

import argparse
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
SHARED_GMR_REFERENCE_FIELD = "surface_temperature"
CO2_FIELD = "global_mean_co2"

MASK_RATES = [0.0, 0.2, 0.4]
GMR_VALS = [True]
RP_VALS = [False]
EXCLUDE_FORCING = [False, True]
CO2_MODES = ["mask", "nomask"]
UNIFORM_MAX_VARS: dict[bool, list[int | str]] = {
    False: ["all", 17],  # with GMR reference excluded: 0.20 × 42 ≈ 8.4 → max = 17
    True: ["all", 15],  # forcing + GMR reference excluded: 0.20 × 37 ≈ 7.4 → max = 15
}

HERE = pathlib.Path(__file__).parent
BASE_CONFIG_STEMS = [
    "ace-train-config-4deg-AIMIP-sfno",
    "ace-train-config-4deg-AIMIP-nc-sfno",
]


def build_bernoulli_input_dropout(mask_rate: float, exclude_forcing: bool) -> dict:
    per_variable: dict[str, float] = {"default_rate": mask_rate}
    if exclude_forcing:
        per_variable.update({v: 0.0 for v in FORCING_VARS})
    return {"per_variable": per_variable}


def build_uniform_input_dropout(
    exclude_forcing: bool, max_vars: int | str = "all"
) -> dict:
    uniform: dict = {
        "min_vars": "min",
        "max_vars": "max" if max_vars == "all" else max_vars,
    }
    if exclude_forcing:
        uniform["ignore_vars"] = list(FORCING_VARS)
    return {"uniform": uniform}


def _protect_shared_gmr_reference(input_dropout: dict) -> dict:
    """Prevent input dropout from masking the shared GMR reference field."""
    return _exclude_from_input_dropout(input_dropout, SHARED_GMR_REFERENCE_FIELD)


def _exclude_from_input_dropout(input_dropout: dict, name: str) -> dict:
    input_dropout = copy.deepcopy(input_dropout)
    if "uniform" in input_dropout:
        uniform = dict(input_dropout["uniform"])
        ignore_vars = list(uniform.get("ignore_vars", []))
        if name not in ignore_vars:
            ignore_vars.append(name)
        uniform["ignore_vars"] = ignore_vars
        input_dropout["uniform"] = uniform
        per_variable = dict(input_dropout.get("per_variable", {}))
        if per_variable.get("default_rate", 0.0) > 0.0:
            per_variable[name] = 0.0
            input_dropout["per_variable"] = per_variable
    else:
        per_variable = dict(input_dropout.get("per_variable", {}))
        per_variable[name] = 0.0
        input_dropout["per_variable"] = per_variable
    return input_dropout


def _add_co2(step_cfg: dict) -> None:
    for name_key in ["next_step_forcing_names", "in_names"]:
        names = list(step_cfg[name_key])
        if CO2_FIELD not in names:
            names.append(CO2_FIELD)
        step_cfg[name_key] = names


def _unmask_co2(input_dropout: dict) -> dict:
    return _exclude_from_input_dropout(input_dropout, CO2_FIELD)


def _co2_suffix(co2_mode: str | None) -> str:
    return "" if co2_mode is None else f"-co2-{co2_mode}"


def _apply_common_settings(
    step_cfg: dict,
    gmr_on: bool,
    rp_on: bool,
    input_dropout: dict,
    co2_mode: str | None = None,
) -> None:
    if co2_mode is not None:
        _add_co2(step_cfg)
        if co2_mode == "nomask":
            input_dropout = _unmask_co2(input_dropout)
        elif co2_mode != "mask":
            raise ValueError(f"Invalid CO2 mode: {co2_mode}")
    if gmr_on:
        input_dropout = _protect_shared_gmr_reference(input_dropout)
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


def _write_config(cfg: dict, out_path: pathlib.Path, existing_only: bool) -> None:
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return
    with out_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}")


def generate_bernoulli_configs(
    base: dict,
    stem: str,
    co2_mode: str | None = None,
    existing_only: bool = False,
) -> None:
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
                        f"{_co2_suffix(co2_mode)}"
                    )
                    out_path = HERE / f"{name}.yaml"
                    step_cfg = cfg["stepper"]["step"]["config"]
                    _apply_common_settings(
                        step_cfg,
                        gmr_on,
                        rp_on,
                        build_bernoulli_input_dropout(mask_rate, exclude_forcing),
                        co2_mode=co2_mode,
                    )
                    _write_config(cfg, out_path, existing_only)


def generate_uniform_configs(
    base: dict,
    stem: str,
    co2_mode: str | None = None,
    existing_only: bool = False,
) -> None:
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
                        f"{_co2_suffix(co2_mode)}"
                    )
                    out_path = HERE / f"{name}.yaml"
                    step_cfg = cfg["stepper"]["step"]["config"]
                    _apply_common_settings(
                        step_cfg,
                        gmr_on,
                        rp_on,
                        build_uniform_input_dropout(exclude_forcing, max_vars),
                        co2_mode=co2_mode,
                    )
                    _write_config(cfg, out_path, existing_only)


def generate_co2_variants(base: dict, stem: str, existing_only: bool = False) -> None:
    """Generate -co2-mask and -co2-nomask configs."""
    for co2_mode in CO2_MODES:
        generate_bernoulli_configs(
            base, stem, co2_mode=co2_mode, existing_only=existing_only
        )
        generate_uniform_configs(
            base, stem, co2_mode=co2_mode, existing_only=existing_only
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite generated YAML files that already exist.",
    )
    args = parser.parse_args()

    for stem in BASE_CONFIG_STEMS:
        base_config = HERE / f"{stem}.yaml"
        with base_config.open() as f:
            base = yaml.safe_load(f)
        # generate_bernoulli_configs(base, stem, existing_only=args.existing_only)
        # generate_uniform_configs(base, stem, existing_only=args.existing_only)
        generate_co2_variants(base, stem, existing_only=args.existing_only)


if __name__ == "__main__":
    main()

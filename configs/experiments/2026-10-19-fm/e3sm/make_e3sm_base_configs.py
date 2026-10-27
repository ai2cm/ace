"""Derive the -e3sm base configs from the cm4-xshield base configs.

For each stem in SOURCE_STEMS, base_configs/<stem>.yaml is copied to
base_configs/<stem with -cm4-xshield replaced by -cm4-xshield-e3sm>.yaml with
these edits:

- An E3SMv3 piControl entry (label `e3sm`) is appended to the train concat. The
  daily store keeps E3SM variable names, so the entry renames them to the fme
  names and converts PHIS geopotential (m2/s2) to HGTsfc (m). Its
  surface_precipitation_rate is labeled m/s but is already in kg/m2/s (global
  mean ~3.4e-5, consistent with its latent heat flux), so only its units
  metadata is corrected. The store has no TMP850 or h500.
- The builder sets `allow_missing_variables: true`, so TMP850 and h500 are
  NaN-filled for E3SM samples and excluded from the loss.
- Normalization points at the v3 pooled stats (which include E3SM) and gains
  an `e3sm` group.
- label_sampling weights, where present, are replaced by LABEL_WEIGHTS.

The configs are edited as text, as in create_fm_jobs.py, so that comments and
formatting are kept verbatim.

Usage:
    python make_e3sm_base_configs.py
"""

import pathlib

HERE = pathlib.Path(__file__).parent
BASE_CONFIGS_DIR = HERE.parent / "base_configs"

SOURCE_STEMS = (
    "ace-train-config-4deg-nc-sfno-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield",
    "ace-train-config-4deg-nc-sfno-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-balanced",
    "ace-train-config-4deg-nc-sfno-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-smoke-diag",
    "ace-train-config-4deg-nc-swin-v2.2-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield",
    "ace-train-config-4deg-nc-swin-v2.2-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-balanced",
)

OLD_STATS = "/climate-default/alexeyy/norm_ablation_0/2026-10-22-fm-all-datasets-v2"
NEW_STATS = "/climate-default/alexeyy/norm_ablation_0/2026-10-26-fm-all-datasets-v3"

# Temperature sampling p_label ~ n_label**ALPHA over per-label training steps
# (cm4 141,353; e3sm 38,690; c96 32,520; era5 24,836; xshield 4,020).
# ALPHA = 0.37 is the exponent that best reproduces the earlier cm4/c96/era5/
# xshield weights of 0.40/0.25/0.25/0.10 from the same counts.
OLD_WEIGHTS = """\
    weights:
      cm4: 0.40
      c96: 0.25
      era5: 0.25
      xshield: 0.10
"""
NEW_WEIGHTS = """\
    weights:
      cm4: 0.33
      e3sm: 0.21
      c96: 0.19
      era5: 0.18
      xshield: 0.09
"""

RENAMED_VARIABLES = {
    "PS": "PRESsfc",
    "TS": "surface_temperature",
    "TREFHT": "TMP2m",
    "QREFHT": "Q2m",
    "Uat10m": "UGRD10m",
    "Vat10m": "VGRD10m",
    "LANDFRAC": "land_fraction",
    "OCNFRAC": "ocean_fraction",
    "ICEFRAC": "sea_ice_fraction",
    "SOLIN": "DSWRFtoa",
    "PHIS": "HGTsfc",
    "LHFLX": "LHTFLsfc",
    "SHFLX": "SHTFLsfc",
    "surface_precipitation_rate": "PRATEsfc",
    "FLUS": "ULWRFsfc",
    "FLUT": "ULWRFtoa",
    "FLDS": "DLWRFsfc",
    "FSDS": "DSWRFsfc",
    "FSUS": "USWRFsfc",
    "FSUTOA": "USWRFtoa",
    "DTENDTTW": "tendency_of_total_water_path_due_to_advection",
    **{
        f"{e3sm}_{k}": f"{fme}_{k}"
        for e3sm, fme in (
            ("T", "air_temperature"),
            ("STW", "specific_total_water"),
            ("U", "eastward_wind"),
            ("V", "northward_wind"),
        )
        for k in range(8)
    },
}

E3SM_STORE = "2026-10-22-E3SMv3-piControl-atmosphere-105yr-4deg-daily"
E3SM_ENTRY = (
    f"    - data_path: /climate-default/{E3SM_STORE}\n"
    f"      file_pattern: {E3SM_STORE}.zarr\n"
    + """\
      engine: zarr
      labels:
      - e3sm
      renamed_variables:
"""
    + "".join(f"        {k}: {v}\n" for k, v in RENAMED_VARIABLES.items())
    + """\
      overwrite:
        multiply_scalar:
          HGTsfc: 0.10197162129779283  # 1 / 9.80665 m/s2
      variable_metadata:
        HGTsfc:
          units: m
        PRATEsfc:
          units: kg/m**2/s
"""
)

TRAIN_END_ANCHOR = "      labels:\n      - xshield\nvalidation:\n"
BUILDER_ANCHOR = "        conditional: true\n"


def _norm_group(label: str) -> str:
    group_dir = f"{OLD_STATS}/groups/{label}"
    return (
        f"            {label}:\n"
        f"              global_means_path: {group_dir}/centering.nc\n"
        f"              global_stds_path: {group_dir}/scaling-full-field.nc\n"
    )


XSHIELD_GROUP_ANCHOR = _norm_group("xshield")
E3SM_GROUP = _norm_group("e3sm")


def _replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise ValueError(f"expected one occurrence of {old!r}, found {n}")
    return text.replace(old, new)


def make_e3sm_config(text: str) -> str:
    if "e3sm" in text or "allow_missing_variables" in text:
        raise ValueError("source config already mentions e3sm or missing variables")
    text = _replace_once(
        text,
        TRAIN_END_ANCHOR,
        "      labels:\n      - xshield\n" + E3SM_ENTRY + "validation:\n",
    )
    text = _replace_once(
        text, BUILDER_ANCHOR, BUILDER_ANCHOR + "        allow_missing_variables: true\n"
    )
    text = _replace_once(text, XSHIELD_GROUP_ANCHOR, XSHIELD_GROUP_ANCHOR + E3SM_GROUP)
    if OLD_WEIGHTS in text:
        text = _replace_once(text, OLD_WEIGHTS, NEW_WEIGHTS)
    if OLD_STATS not in text:
        raise ValueError(f"expected normalization paths under {OLD_STATS}")
    return text.replace(OLD_STATS, NEW_STATS)


def main() -> None:
    for stem in SOURCE_STEMS:
        source = BASE_CONFIGS_DIR / f"{stem}.yaml"
        target = BASE_CONFIGS_DIR / (
            stem.replace("-cm4-xshield", "-cm4-xshield-e3sm") + ".yaml"
        )
        target.write_text(make_e3sm_config(source.read_text()))
        print(f"Wrote {target.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()

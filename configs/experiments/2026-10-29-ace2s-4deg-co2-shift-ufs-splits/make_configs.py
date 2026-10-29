"""Write the pretrain and fine-tune configs of the CO2-shift arms.

They come from the 4deg ACE2S recipe (configs/baselines/era5-ace2s-ablations/
ace2s-{pretrain,finetune}-4deg.yaml) by text substitution, so that only the split
and the CO2 shift differ from the recipe. Run from this directory.
"""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RECIPE = HERE.parents[1] / "baselines" / "era5-ace2s-ablations"
STORE = "2026-09-08-era5-4deg-8layer-daily-1940-2025.zarr"
ARMS = {
    "co2shift0": None,
    "co2shift10": 3652.5,
    "co2shift30": 10957.5,
}  # max shift in days

OLD_INLINE_TIMES = "".join(
    f"          - '1996-{md}T00:00:00'\n"
    for md in ("01-01", "02-15", "04-01", "05-15", "07-01", "08-15", "10-01", "11-15")
)
NEW_INLINE_TIMES = "".join(
    f"          - '{d}T00:00:00'\n"
    for d in ("2012-01-01", "2012-07-04", "2013-01-05", "2013-07-04")
)
OLD_TRAIN = f"""    concat:
    - data_path: /climate-default
      file_pattern: {STORE}
      engine: zarr
      subset:
        stop_time: '1995-12-31'
    - data_path: /climate-default
      file_pattern: {STORE}
      engine: zarr
      subset:
        start_time: '2011-01-01'
        stop_time: '2019-12-31'
    - data_path: /climate-default
      file_pattern: {STORE}
      engine: zarr
      subset:
        start_time: '2021-01-01'
"""
OLD_VAL = """        start_time: '1996-01-01'
        stop_time: '1997-12-31'
"""
NEW_VAL = """        start_time: '2022-01-01'
        stop_time: '2023-12-31'
"""


def new_train(max_shift_days):
    shift = (
        ""
        if max_shift_days is None
        else f"""      random_time_shift:
        names: [global_mean_co2]
        max_shift_days: {max_shift_days}
"""
    )
    return f"""    concat:
    - data_path: /climate-default
      file_pattern: {STORE}
      engine: zarr
      subset:
        stop_time: '2001-12-31'
{shift}    - data_path: /climate-default
      file_pattern: {STORE}
      engine: zarr
      subset:
        start_time: '2012-01-01'
        stop_time: '2021-12-31'
{shift}"""


def header(arm, max_shift_days, stage):
    if max_shift_days is None:
        shift = "no CO2 shift (the split-only control)"
    else:
        years = max_shift_days / 365.25
        shift = (
            "global_mean_co2 replaced, per training window, by its value at a\n"
            f"# time shifted uniformly within +/-{years:.0f} years (validation and\n"
            "# inline inference unshifted)"
        )
    return (
        f"# {arm} {stage}: the 4deg ACE2S recipe\n"
        f"# (configs/baselines/era5-ace2s-ablations/ace2s-{stage}-4deg.yaml) with the\n"
        "# UFS SamudrACE split: train 1940-2001 + 2012-2021, validation 2022-2023,\n"
        "# inline inference = 10-yr windows from 2012-01-01/2012-07-04/2013-01-05/\n"
        "# 2013-07-04 (the UFS coupled FT ICs), 2002-2011 and 2024-2025 held out;\n"
        f"# {shift}. Written by make_configs.py.\n"
    )


def convert(text, arm, max_shift_days, stage):
    for old, new in (
        (OLD_INLINE_TIMES, NEW_INLINE_TIMES),
        ("    n_forward_steps: 1825\n", "    n_forward_steps: 3650\n"),
        (OLD_TRAIN, new_train(max_shift_days)),
        (OLD_VAL, NEW_VAL),
    ):
        assert text.count(old) == 1, (stage, old[:60])
        text = text.replace(old, new)
    body = "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith("#") or line.startswith("# arg:")
    )
    return header(arm, max_shift_days, stage) + body + "\n"


for arm, shift in ARMS.items():
    for stage in ("pretrain", "finetune"):
        text = (RECIPE / f"ace2s-{stage}-4deg.yaml").read_text()
        if stage == "finetune":
            text = text.replace(
                "# arg: --dataset 01M38HBSRAQ02QY797HG3P7QP8:/weights\n", ""
            )
        (HERE / f"{arm}-{stage}.yaml").write_text(convert(text, arm, shift, stage))
print("written", sorted(p.name for p in HERE.glob("*.yaml")))

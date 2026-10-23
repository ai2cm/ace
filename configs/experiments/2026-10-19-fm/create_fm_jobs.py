"""Generate the seed x input-masking run configs for the FM training cells.

Each base config in BASE_CONFIG_STEMS is copied into run_configs/ once per
(seed, masking) pair, named `<stem>-s<seed>-mask<on|off>.yaml`:

- `seed:` is set to the given seed. The input-masking RNG is seeded from the
  same distributed seed, so masking draws differ across seeds too.
- Both arms set `include_channel_mask_inputs: true`, so the network always
  receives per-variable presence channels and the arms differ only in
  `input_dropout`.
- mask-on adds `input_dropout` with uniform masking of 0..MAX_MASKED_VARS
  channels drawn from the default pool (all packed input channels); mask-off
  has no `input_dropout`, so every presence channel is 1.

The base configs are edited as text rather than round-tripped through a YAML
library so that their comments and anchors are kept verbatim.

Usage:
    python create_fm_jobs.py
"""

import pathlib
import re

HERE = pathlib.Path(__file__).parent
BASE_CONFIGS_DIR = HERE / "base_configs"
RUN_CONFIGS_DIR = HERE / "run_configs"

BASE_CONFIG_STEMS = (
    "ace-train-config-4deg-nc-sfno-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield",
    "ace-train-config-4deg-nc-sfno-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-balanced",
    "ace-train-config-4deg-nc-swin-v2.2-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield",
    "ace-train-config-4deg-nc-swin-v2.2-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-balanced",
)
SEEDS = (0, 1, 2, 3, 4)
MAX_MASKED_VARS = 10

SEED_PATTERN = re.compile(r"^seed: \d+$", re.MULTILINE)
# The single_module step config; new step options are inserted right after
# its opening line, at the indentation of its other fields.
STEP_CONFIG_ANCHOR = "stepper:\n  step:\n    type: single_module\n    config:\n"
STEP_FIELD_INDENT = " " * 6


def _replace_once(text: str, pattern: re.Pattern, replacement: str) -> str:
    new_text, n = pattern.subn(replacement, text)
    if n != 1:
        raise ValueError(f"expected one match of {pattern.pattern!r}, found {n}")
    return new_text


def _insert_step_fields(text: str, lines: list[str]) -> str:
    if text.count(STEP_CONFIG_ANCHOR) != 1:
        raise ValueError("expected exactly one single_module stepper config")
    if "include_channel_mask_inputs" in text or "input_dropout" in text:
        raise ValueError("base config already sets input masking options")
    fields = "".join(f"{STEP_FIELD_INDENT}{line}\n" for line in lines)
    return text.replace(STEP_CONFIG_ANCHOR, STEP_CONFIG_ANCHOR + fields)


def make_run_config(base_text: str, seed: int, mask: bool) -> str:
    text = _replace_once(base_text, SEED_PATTERN, f"seed: {seed}")
    lines = ["include_channel_mask_inputs: true"]
    if mask:
        lines += [
            "input_dropout:",
            "  default:",
            f"    max_masked_vars: {MAX_MASKED_VARS}",
        ]
    return _insert_step_fields(text, lines)


def run_config_name(stem: str, seed: int, mask: bool) -> str:
    return f"{stem}-s{seed}-mask{'on' if mask else 'off'}.yaml"


def main() -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for stem in BASE_CONFIG_STEMS:
        base_text = (BASE_CONFIGS_DIR / f"{stem}.yaml").read_text()
        for seed in SEEDS:
            for mask in (True, False):
                path = RUN_CONFIGS_DIR / run_config_name(stem, seed, mask)
                path.write_text(make_run_config(base_text, seed, mask))
                print(f"Wrote {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()

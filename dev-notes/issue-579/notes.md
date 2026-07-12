# Working notes — issue #579

Reproduction and planning notes for [ai2cm/ace#579](https://github.com/ai2cm/ace/issues/579)
(unordered lists cast to/from `set` in stepper code). Full write-up lives in the
Contribution README: https://github.com/FaresIbrahim32/su26-ai301-contribution/blob/main/contribution-2-README.md

- `repro_current_behavior.py` — standalone snippet isolating the exact
  `list(set(a) - set(b))` / re-cast-to-`set` pattern from
  `fme/ace/stepper/single_module.py` (lines 561-562, 569, 693-695) and
  `fme/coupled/stepper.py` (lines 296-333, 343, 358), without the full
  torch/xarray dependency stack. Confirms with `mypy --strict` that the
  current `list[str]`-typed, cast-heavy version type-checks.
- `repro_proposed_fix.py` — same logic retyped as `set[str]` with casts
  removed. Also passes `mypy --strict`, confirming the fix is type-safe.

No source changes yet — this is Phase II (reproduction + plan). The actual
`fme/ace/stepper/single_module.py` / `fme/coupled/stepper.py` edits happen in
Phase III.

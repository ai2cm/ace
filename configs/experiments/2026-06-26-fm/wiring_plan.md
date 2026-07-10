# Plan: adopt checkpoint vertical coordinate in `parameter_init` (Option B)

## Context

The `2026-06-26-fm` cooldown configs cool the FM model down onto ERA5 only. The
base FM model was trained with the **SHiELD/c96** vertical coordinate (ak/bk):
the base `train_loader` concat lists a c96 dataset first and `strict: false`
swallows the ERA5 member's differing coordinate, so the recorded stepper
coordinate is c96 (used for all samples, including ERA5).

The cooldown restricts the `train_loader` to ERA5 only. Because
`parameter_init.weights_path` loads **weights only** and discards the
checkpoint's `DatasetInfo`, the cooldown stepper re-derives its coordinate from
the ERA5 loader → coordinate flips **c96 → ERA5**. So SHiELD-trained weights are
fine-tuned under an ERA5 corrector/derive grid — a coordinate discontinuity
confounded with the cooldown itself, and the re-added c96 inline-inference
entries then run against the wrong grid. ERA5 and c96 ak/bk genuinely differ
(companion notebooks `coordinate_mismatch_report.ipynb`, `fm-ak-bk.ipynb`).

**Goal:** let a cooldown train on ERA5 data while keeping the checkpoint's (c96)
vertical coordinate, so the coordinate is consistent from base training through
cooldown and inline inference. No config-only path exists today; this adds the
capability in `fme` and wires the cooldown generator to use it.

## Approach

Add an opt-in flag on `ParameterInitializationConfig`. When set, the training
stepper is built with the vertical coordinate loaded from `weights_path`'s
checkpoint instead of the one from `train_data.dataset_info`. The coordinate is
loaded in isolation (not through the shared `WeightsAndHistoryLoader` tuple), so
existing weight-loading and the `coupled/` steppers are untouched. Default is
off → zero change to existing runs/checkpoints.

Confirmed safe: no training-time assertion compares the stepper's
`training_dataset_info` against `train_data.dataset_info`
(`assert_compatible_with` only runs in the standalone inference/eval paths), so
building the stepper with a c96 coordinate over ERA5 data raises nothing. Inline
inference during cooldown reuses the training stepper, so it inherits the c96
coordinate automatically.

## Commit 1 — fme capability

### 1a. New config flag — `fme/ace/stepper/parameter_init.py`
- Add field to `ParameterInitializationConfig` (`:122` block):
  `override_vertical_coordinate_from_weights: bool = False`.
- Document it in the class docstring (`:106`).
- Validate in `__post_init__` (`:129`): if `True` and `weights_path is None`,
  raise `ValueError` (per AGENTS.md: validate in `__post_init__`, not at runtime).

### 1b. `DatasetInfo.update_vertical_coordinate` — `fme/core/dataset_info.py`
- Add a method mirroring `update_variable_metadata` (`:237-252`): return a new
  `DatasetInfo` with `vertical_coordinate` replaced and every other field passed
  through unchanged (preserves the `gridded_operations`/`img_shape` vs
  `horizontal_coordinates` mutual-exclusion by forwarding all as-is).

### 1c. Coordinate loader helper — `fme/ace/stepper/single_module.py`
- Add `load_vertical_coordinate(path: str) -> VerticalCoordinate` next to
  `load_weights_and_history` (`:78`). It loads the checkpoint the same way
  `load_stepper` does (`torch.load(..., weights_only=False)`) and returns
  `get_serialized_stepper_vertical_coordinate(checkpoint["stepper"])` — reusing
  the existing helper at `:1834` (which already handles the legacy
  `sigma_coordinates` / `dataset_info` fallbacks). Light load — no full `Stepper`
  build.

### 1d. Apply the override — `TrainStepperConfig.get_train_stepper` (`single_module.py:1504-1539`)
- After building `parameter_initializer`, before `stepper_config.get_stepper(...)`:
  if `self.parameter_init.override_vertical_coordinate_from_weights`, load the
  coordinate from `self.parameter_init.weights_path` via `load_vertical_coordinate`
  and set `dataset_info = dataset_info.update_vertical_coordinate(coord)`.
- Mirror the existing injectable-loader pattern: add an optional
  `load_vertical_coordinate_fn: Callable[[str], VerticalCoordinate] =
  load_vertical_coordinate` parameter so tests can inject, consistent with
  `load_weights_and_history_fn`.

### 1e. Tests — `fme/ace/stepper/test_parameter_init.py`
- Reuse `get_dataset_info` (`:34`) and the checkpoint-save pattern from
  `test_builder_with_weights_loads_same_state` (`:48`).
- New test: save a stepper checkpoint with coordinate A (c96-like ak/bk); call
  `TrainStepperConfig.get_train_stepper` with `dataset_info` carrying a *different*
  coordinate B and `override_vertical_coordinate_from_weights=True`; assert the
  resulting stepper's `training_dataset_info.vertical_coordinate == A` (and that
  weights still load). Add a contrast case with the flag off asserting the
  coordinate stays B.
- New validation test: `ParameterInitializationConfig(override_...=True,
  weights_path=None)` raises `ValueError`.

## Commit 2 — cooldown generator wiring (`configs/experiments/2026-06-26-fm/`)

- In `generate_cooldown_configs.py`, where `parameter_init.weights_path` is set
  (`:248-250`), also set
  `override_vertical_coordinate_from_weights: True`. Safe for all variants:
  single-dataset runs (e.g. `nc-sfno-c96`) have checkpoint coord == data coord, so
  the override is a no-op; multi-dataset FM runs get the c96 checkpoint coordinate.
- Regenerate configs: `python generate_cooldown_configs.py` (writes the flag into
  every `*-cooldown.yaml` / `*-bestinfcooldown.yaml`).
- Update the module docstring (`:1-17`) to note the cooldown now retains the
  checkpoint's vertical coordinate.

## Out of scope
- The `coupled/` stepper training path (not used by these cooldowns).
- The older `SingleModuleStepperConfig.parameter_init` (`single_module.py:122`) —
  inference-time init, does not rebuild `dataset_info` from a train loader, so the
  flag is inert there; leave it unhandled.

## Verification
- Unit: `python -m pytest fme/ace/stepper/test_parameter_init.py` (new + existing
  pass); `python -m pytest fme/core/test_dataset_info.py` if it exists for the new
  method.
- Behavioral end-to-end: build a `TrainStepperConfig` with the flag on pointing at
  a saved checkpoint whose coordinate differs from a dummy ERA5-like
  `dataset_info`; confirm `stepper.training_dataset_info.vertical_coordinate`
  equals the checkpoint's ak/bk (drive via the new test, or a short REPL using
  `get_dataset_info`).
- Config: after regeneration, confirm a generated cooldown YAML contains
  `stepper_training.parameter_init.override_vertical_coordinate_from_weights: true`
  alongside `weights_path`, and that inference still lists both era5 + c96 entries.
- `pre-commit run --all-files` (ruff, ruff-format, mypy) — watch for mypy on the
  new `Callable`/`VerticalCoordinate` annotations; no new `isinstance`/`type: ignore`.

## Risks / notes
- Forcing c96 coord means the ERA5 cooldown *data* is corrected with c96 ak/bk — a
  mismatch for ERA5, but identical to how base training already treated ERA5
  samples; it removes the base→cooldown flip, which is the intent.
- Inline inference in cooldown will now use c96 coord for all entries: c96 entries
  become coordinate-correct; era5 entries use c96 (consistent with base training).
- Confirm the checkpoint top-level key is `"stepper"` when implementing
  `load_vertical_coordinate` (match `load_stepper`'s `torch.load` + `checkpoint["stepper"]`).

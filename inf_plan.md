# Orography-swap eval config generator

## Context

`configs/experiments/2026-06-26-fm/generate_eval_configs.py` generates eval-suite configs (3 checkpoints per training run: besttrain/bestinf/lastepoch) by copying each training config's inline `inference` entries verbatim. We want a sibling script, `generate_orography_configs.py`, producing the same suites but with each inference entry's `loader.dataset` forced to source `HGTsfc` (surface height / orography, a static model input) from a specific grid — `era5` or `c96` — regardless of which grid the run actually trained on. Per user decision, **both** grid variants are generated unconditionally for every run (2 eval suites × 3 checkpoints per run), not just "swap to the opposite of native."

**Revised approach (superseding an earlier draft of this plan):** the earlier draft routed this through the generic `merge:` dataset mechanism (`MergeNoConcatDatasetConfig`), which required a new `restrict_names` apportionment field, a new `has_time_dependent_variables` property on the whole `DatasetABC` hierarchy (8 subclasses), and relaxing `MergedXarrayDataset`'s time-coordinate equality check. Investigation showed this is unnecessary:

- Confirmed on the real GCS stores (`gsutil cat .../HGTsfc/zarr.json`): `HGTsfc` has **no time dimension at all** in either store — shape `[45, 90]`, dims `latitude/longitude` (era5) or `grid_yt/grid_xt` (c96). Both are the same 4° target grid (same shape, positionally aligned — this is why the training config already mixes era5 and c96 samples in one `train_loader.dataset.concat`).
- `XarrayDataset.get_sample_by_time_slice` (`fme/core/dataset/xarray.py:938-949`) already loads time-invariant variables (including `HGTsfc`) by opening the *first file only* and broadcasting the result across the requested window — it never consults `time_slice` values for these variables, and no time-consistency check ever runs against them. So there's no time-alignment problem to solve at all for this use case.
- The two stores' *full* time coordinates do differ (era5: 31410 steps from epoch 1940-01-01T12:00; c96: 30027 steps from epoch 1939-10-01T06:00), but that's irrelevant once `HGTsfc` is handled independently of the windowing/merge machinery.

This lets us skip `fme/core/dataset/merged.py`, `dataset.py`, `concat.py`, `subset.py`, `dummy.py`, and `testing.py` entirely. The whole feature is one new field plus ~15 lines in `fme/core/dataset/xarray.py`.

## Part 1 — `fme/core/dataset/xarray.py` change

### 1a. `XarrayDataConfig.orography_override` (new field)

Add to `XarrayDataConfig` (after `labels: list[str] | None = None`, line 468):

```python
orography_override: "XarrayDataConfig | None" = None
```

Docstring bullet (`Parameters:` block, after `labels:`): "If set, `HGTsfc` is sourced from this dataset instead of from `self`. Used to evaluate a checkpoint with a different static orography field than it trained on (e.g. swapping between the era5 and c96 grids). Only the `HGTsfc` variable is read from the override dataset; all other requested variables continue to come from `self`."

`__post_init__` (xarray.py:500-511): raise `ValueError` if `self.orography_override is not None and self.orography_override.orography_override is not None` (no nested overrides — keeps the feature one level deep, matches the only real use case).

### 1b. `XarrayDataset.__init__` (xarray.py:544-624)

After the existing init body (once `self._time_invariant_names` etc. are known), if `config.orography_override is not None` and `"HGTsfc"` is in `names`:

```python
override_dataset = XarrayDataset(
    config.orography_override, ["HGTsfc"], n_timesteps
)
tensors, *_ = override_dataset[0]
self._orography_override_tensor = tensors["HGTsfc"][0]  # single spatial slice
```

Store `None` otherwise. This is a direct nested `XarrayDataset` construction (not `config.build()` / `XarraySubset` — `XarraySubset.get_sample_by_time_slice` raises `NotImplementedError`, and we don't need subset/properties machinery here). Built once at construction, not per-sample.

### 1c. `get_sample_by_time_slice` time-invariant loop (xarray.py:938-949)

```python
for name in self._time_invariant_names:
    if name == "HGTsfc" and self._orography_override_tensor is not None:
        tensors[name] = self._orography_override_tensor.unsqueeze(0).expand(
            total_steps, *self._orography_override_tensor.shape
        ).clone()
        continue
    variable = ds[name].variable
    ...  # existing code unchanged
```

### 1d. Tests — `fme/core/dataset/test_xarray.py`

Using existing temp-zarr/netcdf fixture helpers in that file (check first for reusable ones before writing new):
- Two small on-disk stores, A (native `HGTsfc` = some value) and B (`HGTsfc` = a different value, same spatial shape). Build `XarrayDataConfig` for A with `orography_override=XarrayDataConfig(B)`. Assert samples' `HGTsfc` equals B's value, not A's, across every timestep in the window.
- Assert all *other* requested variables still come from A (e.g. `PRESsfc` unaffected).
- `orography_override=None` (default): unchanged existing behavior, no regression.
- `__post_init__` test: nested override (`orography_override.orography_override` set) raises `ValueError`.
- Mismatched spatial shape between A and B: confirm it fails loudly (whatever error `.expand`/tensor assignment naturally raises) rather than silently broadcasting wrong data — no special-case handling needed, just confirm the natural failure mode is a clear error, not silent corruption.

## Part 2 — `generate_orography_configs.py`

### 2a. Minor additive refactor of `generate_eval_configs.py` (no behavior change to its own output)

- Extract the source-config glob from `main()` (lines 259-268) into `discover_source_configs(version) -> list[pathlib.Path]`, called from both scripts.
- Add optional `eval_run_name_base: str | None = None` param to `_write_config` (lines 146-174), used only for the wandb-existence-check run names (defaults to `source_run_name`, so the existing call site and behavior are unchanged). Needed because the orography script's actual eval run names (derived from its own output filenames, per `submit_eval_jobs.py`'s `config_to_jobs`) differ from the source training run name that belongs in the header comment.

### 2b. New file `configs/experiments/2026-06-26-fm/generate_orography_configs.py`

Imports from `generate_eval_configs`: `CONFIG_PREFIX`, `DEFAULT_CHECKPOINT_PATH`, `DEFAULT_SOURCE_MAP`, `HERE`, `_build_eval_suite_config`, `_fetch_wandb_run_names`, `_write_config`, `discover_source_configs`, `eval_suite_config_to_run_name`, `source_config_to_run_name`, and `EVAL_SUITE_CONFIG_PREFIX` (aliased) — reusing rather than duplicating (per AGENTS.md on consolidating duplication).

- `OROGRAPHY_EVAL_SUITE_CONFIG_PREFIX = f"{EVAL_SUITE_CONFIG_PREFIX}orog-"` — distinct output filenames from the plain generator; since it's a superset-prefix, `eval_suite_config_to_run_name` (imported unmodified) still correctly strips it and yields a distinct run name per grid, e.g. `ace2-fm-orog-era5-nc-sfno-v1`.
- `OROGRAPHY_SOURCES = {"era5": {...era5 zarr path/pattern...}, "c96": {...c96 zarr path/pattern...}}` — plain `data_path`/`file_pattern`/`engine` dicts, no `restrict_names` needed anymore. Paths confirmed from the real training configs (era5: `data_path=/climate-default`, `file_pattern=2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr`; c96: `data_path=/climate-default/2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset`, `file_pattern=ic_0001.zarr`; both `engine: zarr`).
- `_dataset_grid(dataset) -> str | None`: identify if a `loader.dataset` dict already natively matches one of `OROGRAPHY_SOURCES` by `data_path`/`file_pattern`.
- `_swap_hgtsfc_grid(dataset, grid) -> dict`: if `dataset` already natively matches `grid`, return a deep copy unchanged (no-op); otherwise return `{**deepcopy(dataset), "orography_override": deepcopy(OROGRAPHY_SOURCES[grid])}`; raise `ValueError` if `dataset` isn't a plain `data_path`-keyed dict (concat/merge inference datasets aren't supported by this script).
- `source_config_to_orography_eval_suite_config(config_filename, grid) -> str`: `f"{OROGRAPHY_EVAL_SUITE_CONFIG_PREFIX}{grid}-{suffix}.yaml"`.
- `generate_orography_eval_config(source_path, source_map, inference_names, checkpoint_path, existing_only, wandb_run_names=None)`: mirrors `generate_eval_config`, but loops `for grid in ("era5", "c96")`, builds `_build_eval_suite_config(...)` fresh per grid (already deep-copies internally, so safe to mutate), sets `entry["config"]["loader"]["dataset"] = _swap_hgtsfc_grid(...)` for every inference entry, writes via `_write_config(..., eval_run_name_base=eval_suite_config_to_run_name(out_path.name))`.
- `main()`: same flags as `generate_eval_configs.py` (`--inference-name`, `--checkpoint-path`, `--source-map`, `--existing-only`, `--delete-if-in-wandb`); no direction flag (always both grids); iterates `discover_source_configs(args.version)`.

**Known follow-up (explicitly out of scope):** `submit_eval_jobs.py`'s glob will match these new files (prefix superset) but skip them since their derived run names aren't in `wandb_to_beaker_map.json` — this is a safe no-op, not a crash, but actually submitting these jobs needs separate wiring (e.g. mapping `orog-{grid}-{run}` names to the same beaker dataset IDs). Not building this now, per user's "config generation only" decision.

## Critical files

- `fme/core/dataset/xarray.py` — `orography_override` field, `__post_init__`, `XarrayDataset.__init__` override construction, `get_sample_by_time_slice` override branch
- `fme/core/dataset/test_xarray.py` — new tests
- `configs/experiments/2026-06-26-fm/generate_eval_configs.py` — additive refactor (`discover_source_configs`, `_write_config` param)
- `configs/experiments/2026-06-26-fm/generate_orography_configs.py` — new file

## Verification

1. `python -m pytest fme/core/dataset/test_xarray.py -v` — new tests pass, all existing ones unmodified/passing.
2. `pre-commit run --files fme/core/dataset/xarray.py fme/core/dataset/test_xarray.py configs/experiments/2026-06-26-fm/generate_eval_configs.py configs/experiments/2026-06-26-fm/generate_orography_configs.py` — ruff/ruff-format/mypy clean.
3. Dry run without network/Beaker access: `cd configs/experiments/2026-06-26-fm && python generate_orography_configs.py --version v1` (no `--delete-if-in-wandb`) — inspect the two written files per run (e.g. `ace-eval-suite-config-4deg-AIMIP-orog-era5-nc-sfno-v1.yaml`, `...-orog-c96-nc-sfno-v1.yaml`): confirm the native-grid variant has an unmodified `loader.dataset`, the other has `orography_override:` set to the opposite grid's plain dataset dict; confirm filenames don't collide with plain `generate_eval_configs.py` output; confirm non-`loader` keys are unaffected (diff one inference entry's `config` against the plain script's output for the same run).
4. Parsing sanity check without touching real data mounts: `dacite.from_dict(InferenceDataLoaderConfig, yaml.safe_load(...)["inferences"][0]["config"]["loader"], config=dacite.Config(strict=True))` against a generated file's `loader` sub-dict — confirms `orography_override` round-trips into `XarrayDataConfig` correctly.
5. If mounts are available: build an `XarrayDataset` from one real orography-swapped inference entry and confirm the returned `HGTsfc` values match the override store's `HGTsfc` (not the native store's), and confirm every other requested variable is unaffected.

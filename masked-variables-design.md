# Masked Variables Feature Design

## Overview

This feature adds support for per-sample variable masking in ACE training, inference,
and data loading. A `data_mask` field on `BatchData` carries per-variable, per-sample
boolean masks that tell the system which variables are present for a given sample. This
enables two use cases:

1. **Dataset-level masking**: datasets that are missing some variables can indicate this
   via masks, allowing training and inference on heterogeneous data sources.
2. **Semi-supervised masking** (future): the `TrainStepper` can randomly mask
   input/output variables during training for semi-supervised learning.

This design covers dataset-level masking end-to-end: the masking infrastructure
(data representation, input zeroing, loss exclusion) and the actual dataset-level
auto-detection of missing variables, implemented as separable commits.

## User Stories

1. **As a researcher training on a single complete dataset**, I want ACE to work exactly
   as it does today when I don't configure variable masking. No new config is required.

2. **As a researcher with datasets missing some variables**, I want to set
   `allow_variable_masking: true` on my `ModuleSelector` so that the data loader
   gracefully handles missing variables instead of erroring, producing masks alongside
   the data and letting the training loop handle the rest.

3. **As a researcher**, when a variable is masked for a given sample:
   - **Inputs**: the masked input channels should be set to 0 in normalized space (i.e.
     filled with the climatological mean in physical space) before being fed to the
     model.
   - **Outputs**: the masked output variables should be excluded from the loss for
     that sample only. Other samples in the same batch that have that variable should
     still compute loss for it.

4. **As a researcher running inference**, I want to be able to run a model trained with
   variable masking on data sources that are missing some variables. The model should
   receive zeroed inputs for masked variables (same treatment as training) and produce
   all outputs. This lets me run inference using data sources from training, including
   ones that were missing certain inputs or outputs.

5. **As a developer**, the masking logic should be testable in isolation, with unit tests
   covering both masked and unmasked cases through `train_on_batch`.

6. **As a researcher**, when `allow_variable_masking` is `false` (the default), I want an
   error if my dataset is missing required variables, so I don't accidentally train with
   incomplete data.

## Technical Requirements

### Data Representation

- `BatchData` gets a new field: `data_mask: TensorMapping | None`
  - Keys are variable names (any of the model's in/out names).
  - Values are `[batch]` tensors of dtype `bool` where `True` means **present**
    (unmasked) and `False` means **masked/missing**.
  - When `None`, all variables are assumed present (backwards compatible).

- `data_mask` must be preserved through all `BatchData` operations:
  `to_device`, `to_cpu`, `scatter_spatial`, `broadcast_ensemble`, `subset_names`,
  `select_time_slice`, `remove_initial_condition`, `prepend`,
  `compute_derived_variables`, `from_sample_tuples`, `new_on_cpu`, `new_on_device`,
  `new_for_testing`, `pin_memory`.

- `PairedData` does **not** need `data_mask` because it operates at the
  prediction/reference level after masking has already been applied to inputs, and loss
  masking operates on the `data_mask` from the original `BatchData`.

### Feature Gating

- `ModuleSelector` gets a new config field: `allow_variable_masking: bool = False`.
  This is a safety gate: when `False` (default), the data loader errors if the dataset
  is missing required variables. When `True`, missing variables are handled gracefully
  with masks. This is similar to the existing `conditional: bool` flag for label support.
- The flag propagates through `SingleModuleStepConfig` (read from
  `self.builder.allow_variable_masking`), `StepConfigABC` / `StepSelector`, and into
  `DataRequirements` as `allow_variable_masking: bool = False`.
- All variable names (inputs and outputs) are maskable, not just prognostic variables.

### Data Loading

- When `allow_variable_masking` is `True`, datasets may yield samples with fewer
  variables than required. Missing variables are auto-detected: the data loader
  compares what variables exist in the data files against what `DataRequirements.names`
  asks for. Variables present in the files are loaded normally; variables absent from
  the files are omitted from the sample's `TensorDict`.
- The `CollateFn` handles heterogeneous variable sets across samples in a batch:
  it takes the union of all variable names, fills missing variables with a placeholder
  value (0), and constructs `data_mask` indicating which variables are present per
  sample.
- When `allow_variable_masking` is `False`, the current behavior is preserved:
  missing variables cause an error.

### Input Masking

In `SingleModuleStep.step()`, before the network call:
- If `data_mask` is provided, for each input variable that has a `False` mask entry
  for some batch members, set those batch members' input to 0 in normalized space.
- This is equivalent to replacing with the normalizer mean in physical space.
- The mask information is passed through `StepArgs` so that `SingleModuleStep`
  can access it.
- The code in `SingleModuleStep` always handles masks when present; no per-step-config
  flag is needed. The `allow_variable_masking` flag on `ModuleSelector` controls whether
  the data pipeline produces masks, not whether the step code supports them.

### Output / Loss Masking

In `TrainStepper._accumulate_loss()`:
- After getting `gen_step` and `target_step`, apply the `data_mask` to exclude
  masked output variables from the loss.
- For each output variable with masked samples, set both `gen_step[var]` and
  `target_step[var]` to the same constant (e.g. 0) for the masked batch members.
  This zeroes out their loss contribution, similar to the existing NaN masking in
  `WeightedMappingLoss`.
- This approach reuses the existing NaN masking infrastructure rather than changing
  the loss function signatures.

### Inference

- When a model was trained with `allow_variable_masking: true`, inference data loaders
  also support masked data. The `allow_variable_masking` flag is saved with the model
  checkpoint as part of `ModuleSelector` config.
- During inference, masked input variables are zeroed in normalized space (same as
  training). The model produces all outputs regardless of which inputs were masked.
- The inference data loader (`InferenceDataset`) uses the same auto-detection and
  collation logic as the training data loader.

## Implementation Plan

### Commit 1: Add `data_mask` field to `BatchData`

**Files changed:**
- `fme/ace/data_loading/batch_data.py`

**Changes:**
- Add `data_mask: TensorMapping | None = None` field to `BatchData`.
- Update `__post_init__` to validate that `data_mask` tensor shapes are `[n_samples]`
  and keys are a subset of `data` keys.
- Update all methods that return new `BatchData` instances to propagate `data_mask`:
  `to_device`, `to_cpu`, `scatter_spatial`, `broadcast_ensemble`, `subset_names`,
  `select_time_slice`, `remove_initial_condition`, `prepend`,
  `compute_derived_variables`, `new_on_cpu`, `new_on_device`, `new_for_testing`,
  `pin_memory`.
- Update `broadcast_ensemble` to `repeat_interleave` the mask tensors (same as labels).
- Add tests in `test_batch_data.py` to cover `data_mask` propagation.

### Commit 2: Add `allow_variable_masking` to `ModuleSelector` and wire through config

**Files changed:**
- `fme/core/registry/module.py` (ModuleSelector)
- `fme/core/step/single_module.py` (SingleModuleStepConfig)
- `fme/core/step/step.py` (StepConfigABC, StepSelector)
- `fme/ace/requirements.py` (DataRequirements)
- `fme/ace/stepper/single_module.py` (StepperConfig)

**Changes:**
- Add `allow_variable_masking: bool = False` to `ModuleSelector`.
- Add read-only property `allow_variable_masking` to `SingleModuleStepConfig` that
  delegates to `self.builder.allow_variable_masking`.
- Add property `allow_variable_masking` to `StepConfigABC` (default `False`),
  delegate in `StepSelector`.
- Add `allow_variable_masking: bool = False` to `DataRequirements`.
- Update `StepperConfig.get_evaluation_window_data_requirements` and
  `get_forcing_window_data_requirements` to set `allow_variable_masking` from
  `self.step.allow_variable_masking`.

### Commit 3: Add `data_mask` to `StepArgs` and pass through predict_generator

**Files changed:**
- `fme/core/step/args.py`
- `fme/ace/stepper/single_module.py` (Stepper.predict_generator, _accumulate_loss)

**Changes:**
- Add `data_mask: TensorMapping | None = None` to `StepArgs`.
- In `Stepper.predict_generator`, pass `data_mask` from the caller to `StepArgs`.
- In `TrainStepper._accumulate_loss`, pass `data_mask` from `BatchData` through to
  `predict_generator` -> `StepArgs`.
- The data_mask is constant across timesteps (a variable is either present or absent
  for a given sample across the entire window).

### Commit 4: Apply input masking in `SingleModuleStep` and output masking in loss

**Files changed:**
- `fme/core/step/single_module.py` (SingleModuleStep.step)
- `fme/ace/stepper/single_module.py` (TrainStepper._accumulate_loss)

**Changes:**
- In `SingleModuleStep.step()`, before the network call, apply input masking:
  for each input variable where `data_mask[var]` has `False` entries, zero out
  those batch members in normalized space.
- In `_accumulate_loss`, after getting `gen_step` and `target_step`, if `data_mask`
  is present, apply masking before calling `self._loss_obj`:
  - For each output variable in `data_mask`, expand the `[batch]` boolean mask to
    match the spatial/ensemble dims, then set both prediction and target to 0 where
    masked.
  - This makes the loss contribution zero for masked samples, consistent with the
    existing NaN masking approach in `WeightedMappingLoss`.
- Add unit tests:
  - Test that `train_on_batch` with all-unmasked `data_mask` produces the same loss as
    without `data_mask`.
  - Test that `train_on_batch` with a fully-masked output variable produces zero loss
    contribution for that variable (verify via `per_channel_losses`).
  - Test that `train_on_batch` with a partially-masked output variable (some samples
    masked, others not) correctly excludes only the masked samples.
  - Test that masked input variables are set to 0 in normalized space before the
    network call.

### Commit 5: Dataset-level auto-detection of missing variables

**Files changed:**
- `fme/core/dataset/xarray.py` (XarrayDataset)
- `fme/ace/data_loading/getters.py` (CollateFn, get_gridded_data)
- `fme/ace/data_loading/batch_data.py` (from_sample_tuples)

**Changes:**
- In `XarrayDataset`, when `allow_variable_masking` is `True`, only load variables
  that exist in the data files. Variables in the required `names` list that don't
  exist in the dataset are silently skipped. The resulting `TensorDict` from
  `__getitem__` only contains keys for variables that are actually present.
- `allow_variable_masking` is passed through `get_gridded_data` ->
  `DataLoaderConfig.get_dataset` -> dataset construction.
- Update `CollateFn` to handle heterogeneous variable sets across samples in a batch
  when `allow_variable_masking` is `True`:
  - Take the union of all variable names across samples.
  - For each variable, stack tensors from samples that have it and fill with 0 for
    samples that don't.
  - Construct `data_mask` with `True` where the sample had the variable and `False`
    where it was filled.
  - Uses a custom collation path instead of `default_collate` for the data dicts.
- When `allow_variable_masking` is `False`, behavior is unchanged: `default_collate`
  is used and missing variables cause an error.
- Add tests covering:
  - A dataset missing a subset of required variables produces correct masks.
  - Concatenated datasets with different variable sets produce per-sample masks.
  - Error when `allow_variable_masking=False` and variables are missing.

### Commit 6: Inference support for masked data

**Files changed:**
- `fme/ace/data_loading/inference.py` (InferenceDataset)
- `fme/ace/data_loading/getters.py` (get_inference_data)

**Changes:**
- `InferenceDataset` supports `allow_variable_masking`, auto-detecting missing
  variables in the inference data the same way as training.
- `get_inference_data` passes `allow_variable_masking` from requirements through
  to `InferenceDataset`.
- The inference stepper already handles `data_mask` through `StepArgs` (commit 3),
  and `SingleModuleStep` already applies input masking (commit 4), so no changes
  to the inference step/predict path are needed.
- Add tests:
  - Inference with masked data produces predictions for all output variables.
  - Masked inputs are zeroed in normalized space during inference.

## Design Decisions

### Why `TensorMapping | None` instead of a dedicated class like `BatchLabels`?

Labels use `BatchLabels` because they have complex encoding/decoding logic (one-hot
encoding, conforming to encodings across checkpoints). Variable masks are simple boolean
tensors keyed by variable name with no encoding needed, so a plain `TensorMapping`
(i.e. `Mapping[str, torch.Tensor]`) is sufficient and avoids unnecessary abstraction.

### Why the `allow_variable_masking` flag on `ModuleSelector`?

The flag is a safety gate. Without it, a dataset missing required variables would
silently produce masked training data, which could lead to unexpected model behavior.
The flag makes this opt-in, so researchers must explicitly acknowledge that their data
may be incomplete. It lives on `ModuleSelector` (not `SingleModuleStepConfig`) because
it's a property of the model configuration, analogous to `conditional` for labels.
The step code always handles masks when present; the flag only controls whether the
data pipeline is allowed to produce them.

### Why mask at the loss level instead of changing the loss function signature?

The existing `WeightedMappingLoss` already handles NaN masking by zeroing out both
prediction and target where the target is NaN. Applying the variable mask in the same
way (zeroing out both prediction and target for masked samples) reuses this established
pattern without requiring changes to the loss function interface. This keeps the loss
functions clean and composable.

### Why auto-detect missing variables instead of explicit config?

Auto-detection is more ergonomic: the researcher doesn't need to manually list which
variables each dataset has. The data loader compares what's in the files against what
the model requires, and masks the difference. The `allow_variable_masking` flag
prevents this from happening accidentally.

### Why handle masking at the collate level?

Individual datasets yield samples where variables are either present or not. The
`CollateFn` is the natural place to reconcile heterogeneous variable sets across
samples in a batch: it takes the union of keys, fills gaps, and constructs the mask.
This keeps the per-sample mask logic in one place and avoids threading mask metadata
through `DatasetItem`.

### Why produce all outputs during inference even when inputs are masked?

The model always produces predictions for all output variables, even when some inputs
were missing. This is the simplest approach and matches how the model was trained (it
learned to predict outputs from partial inputs). Filtering outputs based on input
availability is a possible future enhancement.

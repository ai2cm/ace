# Plan: remove `n_ensemble` from Step, apply input dropout as a late input mask

## Problem

`input_dropout` (variable masking) lives inside `SingleModuleStep.network_call`
(`fme/core/step/single_module.py:392-418`). It samples a per-channel mask on the
**folded** batch dimension (`n_base * n_ensemble`), so it needs `n_ensemble` to
`repeat_interleave` the mask and keep ensemble members of a base sample identical.
That requirement threaded `n_ensemble` through `StepArgs` → `step` →
`predict_generator` → `predict` → coupled stepper, breaking the intentional
"Step is ensemble-agnostic" contract (PR #1246 review, `fme/core/step/args.py:43`).

## Key insight

- `data_mask` is not just an input-zeroing mechanism. It means a variable is
  genuinely absent and is consumed by preprocessing (`global_mean_removal`) and by
  loss masking.
- Synthetic input dropout must therefore stay separate from real `data_mask`.
  Otherwise shared GMR can raise when its reference field is randomly "missing",
  per-channel GMR changes the generated extra channels, and loss can incorrectly
  skip randomly dropped target variables. The model output produced from a
  dropped input must still be scored against the normal target; only genuinely
  absent target variables may be excluded from loss.
- The Step can still remain ensemble-agnostic: generate a synthetic
  `input_dropout_mask` in the ensemble-aware training layer, already shaped on the
  folded batch dimension, then pass that concrete mask into `StepArgs`.
- `SingleModuleStep` applies `input_dropout_mask` **late**, inside
  `network_call`, after GMR has produced normalized inputs and GMR extra channels,
  and just before packing/passing tensors to the module. No `n_ensemble` is exposed
  to Step.
- The concrete mask's key space is the Step-owned packed input channel namespace
  (`self.in_packer.names`). Training code may request/supply a mask, but it must not
  reconstruct this order itself or reach into `SingleModuleStep` internals.

## Decisions (locked)

- **Approach:** ensemble-aware mask sampling above Step, concrete late input mask
  inside Step (reviewer's Option 1 "refactor into ensemble-aware layer"), not
  uniform-across-batch and not synthetic dropout-as-`data_mask`.
- **Mask cadence: per-window.** Mask sampled once per rollout (1 IC + N forward
  steps), held constant across all N steps. Matches inference reality (a missing
  input is missing for the whole rollout) and matches how the real absent-variable
  `data_mask` already behaves. Was per-step before.
- **Config ownership: keep `input_dropout` on `SingleModuleStepConfig`.** The
  Step owns the packed input channel order (`self.in_packer.names`, including GMR
  extras) and config validation. Training owns ensemble-aware repeat logic and is
  the only caller of the sampling hook. Serialized-but-inert inference config is
  acceptable; document this rationale in the PR.
- **Mode guard: dropout only in train mode.** Moving sampling above `Step` must
  preserve the current behavior that `input_dropout` is inactive when modules are
  in eval mode. The sampling hook returns `None` if the wrapped module is not
  training, even when `input_dropout` is configured. This matters for validation
  paths that call `train_on_batch` with `NullOptimization`, because
  `NullOptimization.set_mode` puts modules in eval mode.

---

## Maskable channel set

All packed network-input channels = `self.in_packer.names` = `in_names` + GMR
extra channels (`__gmr_extra__<source>`). **All of these must be maskable** (the
current in-step dropout already masks GMR extras, since it counts packed channels).
Sample the dropout mask over `self.in_packer.names`, not `self.in_names`.

NOT maskable: the channel-mask indicator channels (`include_channel_mask_inputs`)
— they are the mask signal itself, masking them is circular. Static/positional/grid
encodings live inside the torch module, outside the Step input contract — out of
scope.

## Commit 1 — concrete input-dropout mask API

- `fme/core/step/args.py`: add
  `input_dropout_mask: TensorMapping | None = None` to `StepArgs`, documented as a
  synthetic training-only input presence mask (`[batch]` bool, True = present),
  keyed by the receiving Step's packed input channel names. Clarify that
  `StepArgs.input_dropout_mask` is preserved through input processing but not
  transformed by it.
- `fme/core/step/step.py`: add a narrow hook such as
  `make_input_dropout_mask(self, batch_size, device) -> TensorMapping | None` to
  `StepABC`; base returns `None`. This hook samples a mask but does not apply it.
  Also add a separate non-random introspection hook,
  `has_input_dropout(self) -> bool`, defaulting to `False`. Do **not** use
  `make_input_dropout_mask` to detect configuration, because it is mode-dependent
  and intentionally returns `None` in eval mode.
- `fme/core/step/multi_call.py`: delegate both hooks to the wrapped step so wrapped
  `SingleModuleStep` configs keep working.
- `fme/ace/stepper/single_module.py`: expose the same narrow hooks on `Stepper`
  (delegating to `self._step_obj`) because `TrainStepper._accumulate_loss` holds a
  `Stepper`, not the raw `StepABC`. Coupled training does **not** call the random
  sampling hook (coupled `input_dropout` is unsupported — see Commit 2 fail-loud
  guard).
- `fme/core/step/single_module.py`: `SingleModuleStep` overrides the hook and
  samples a per-channel presence mask over **`self.in_packer.names`** (`in_names`
  + GMR extra sentinel names; `[batch_size]` bool, True = present) from
  `self._config.input_dropout`; returns `None` if `input_dropout` is None **or
  if `self.module.torch_module.training` is false**. **No ensemble arg.** Plain
  per-sample masks. GMR extra channels stay **independently maskable**.
  `has_input_dropout` returns whether `self._config.input_dropout is not None`,
  independent of module train/eval mode.
- Do **not** store synthetic dropout in `data_mask` and do not pass it to
  `global_mean_removal.forward_transform` or to the loss as an exclusion mask.
  Dropout is still active for the forward pass, so the loss evaluates predictions
  generated from masked inputs.
- `_build_channel_mask_dict`: add/adjust a helper path for channel indicators to
  combine real `data_mask` and late `input_dropout_mask` only inside
  `network_call`, by **logical AND** (a channel is present only if it is both
  really present and not synthetically dropped). The same combined mask drives
  both input zeroing (`_apply_input_mask`) and the indicator channel value.
  Per packed channel `name`:
  `real = data_mask.get(name, ones)` (for GMR extras fall back to the source
  field: `data_mask.get(source, ones)`); `synthetic =
  input_dropout_mask.get(name, ones)`; `combined = real & synthetic`.
  **Do not** use fallback-priority (pick one source). Priority would resurrect a
  genuinely-missing variable whenever dropout left it un-dropped:
  `data_mask=0, dropout=1` must yield present=0, but priority yields 1.
- Leave `fme/core/var_masking.py`'s current `sample_mask(..., n_ensemble=...)`
  API in place in this commit. The old in-step sampler still calls it until
  Commit 2 removes that sampler and the Step-facing `n_ensemble` plumbing
  together.
- Tests: `StepArgs` preserves `input_dropout_mask`; hook returns correct
  shape/dtype; `None` when unset; GMR receives only real `data_mask`.

## Commit 2 — generate in ensemble-aware layer, delete in-step dropout, remove Step `n_ensemble`

- `fme/ace/stepper/single_module.py` `_accumulate_loss` (~1678): before
  `broadcast_ensemble`, call `self._stepper.make_input_dropout_mask(...)` on the
  **pre-broadcast** `input_data`
  (`n_base` samples), then repeat-interleave that mask to the folded ensemble batch
  dimension. Pass the repeated mask to `predict_generator` as
  `input_dropout_mask`. The training code should call the hook unconditionally
  and use the returned value; do not check the config or sample directly in the
  training layer, because the hook also enforces eval-mode disablement. Keep
  `input_data.data_mask` unchanged for real missing variables and loss masking.
- Keep two mask variables through training code:
  `step_data_mask` / real `data_mask` for genuinely missing variables, and
  `input_dropout_mask` for synthetic input corruption. `StepLoss` receives only
  the real `data_mask`; this means dropped-input predictions are **not ignored**
  in the loss unless the corresponding target variable is truly absent.
- **Coupled training: out of scope, fail loud.** Coupled is a separate route
  (`CoupledTrainStepper.train_on_batch` → `CoupledTrainStepper._accumulate_loss`
  → `CoupledStepper.get_prediction_generator`). No coupled config sets
  `input_dropout` today (`input_dropout` exists only on `SingleModuleStepConfig`,
  no refs in `fme/coupled/`, and `ComponentTrainingConfig.input_dropout` is out
  of scope). So **do not** build the per-component mask-generation machinery
  (no `input_dropout_masks` carrier, no threading through
  `CoupledStepper.get_prediction_generator`) — that would be an unused feature
  reviewed in isolation (AGENTS.md).
- Instead, **fail loud**: after the in-step sampler is deleted, the coupled path
  never calls `make_input_dropout_mask`, so a configured `input_dropout` would
  silently do nothing. Make `CoupledTrainStepper` (or coupled stepper
  construction) raise a clear error if any component step reports configured
  `input_dropout` via the explicit `has_input_dropout` hook. Do **not** probe
  `make_input_dropout_mask` for this check; it is random and returns `None` in
  eval mode by design. Error must name the unsupported feature and point at
  uncoupled training.
- Delete the dropout block from `network_call` (`single_module.py:392-418` dropout
  sampling parts). Keep late deterministic application:
  `real data_mask -> GMR/normalization/extras -> _apply_input_mask(real data_mask)
  -> _apply_input_mask(input_dropout_mask) -> pack -> module`.
- `fme/core/step/args.py`: drop `n_ensemble` (lines 29-50, 63).
- `fme/ace/stepper/single_module.py`: revert `predict` / `predict_generator`
  `n_ensemble` additions (1066-1163, 1696), but add/pass through
  `input_dropout_mask` for training-owned calls. Keep inference-facing APIs clean:
  inference callers should not need to know this argument exists, with
  `input_dropout_mask=None` as the only public default if exposed. Resolves review
  comments `args.py:43` and `single_module.py:1066`.
- `fme/coupled/stepper.py`: revert the `n_ensemble` plumbing this branch added,
  restoring API signatures and calls to their state on `main`. **Do not blindly
  delete every `n_ensemble=` line:** keep constructor-level `BatchData` /
  `PrognosticState` / `CoupledBatchData` ensemble metadata whenever the wrapped
  data has already been ensemble-broadcasted. `BatchData.n_ensemble` defaults to
  1, so dropping those metadata assignments can silently make folded ensemble
  data look like singleton data. Remove `n_ensemble` only from the Step-facing
  and prediction-generator API plumbing; leave pre-existing uses such as config
  fields, `broadcast_ensemble` calls, attribute reads, and independently correct
  `BatchData.n_ensemble` preservation.
- If `include_channel_mask_inputs=True`, indicator channels reflect the
  AND-combined real presence and synthetic dropout presence
  (`real & synthetic`, per `_build_channel_mask_dict`), but that combined mask is
  local to `network_call`; it must not escape into BatchData or loss. The two
  sequential `_apply_input_mask` calls above zero inputs to the same effect.
- Delete tests that only validate the rejected `n_ensemble`-in-`StepArgs` API:
  - `test_get_prediction_generator_infers_n_ensemble_for_input_dropout`
  - `test_get_prediction_generator_preserves_n_ensemble`
  - `test_input_dropout_ensemble_members_share_mask` as written (it constructs
    `StepArgs(n_ensemble=...)`) — rewrite as a training-layer test (ensemble
    members of a base share the generated mask) per Commit 2 tests.
  - Do **not** blanket-delete core tests that configure
    `SingleModuleStepConfig(input_dropout=...)`: the config still lives on the
    step and the step still owns packed-channel sampling. Keep focused core tests
    for `make_input_dropout_mask` shape/dtype, train/eval mode guard,
    `has_input_dropout`, sampling over `self.in_packer.names` including GMR
    extras, and deterministic `input_dropout_mask` application/indicator behavior.
    Move only ensemble-sharing behavior to training-layer tests.
- Tests: `StepArgs` has no `n_ensemble` attribute; per-window invariance (mask
  constant across forward steps); ensemble
  members of a base share mask; GMR extra channels are independently maskable and
  their indicator channel agrees with their dropped value; shared/per-channel GMR
  still sees only the real `data_mask`; loss is computed for targets whose inputs
  were synthetically dropped, and only real missing-variable masks reduce loss
  counts; **AND-combine correctness**: a channel that is genuinely missing
  (`data_mask=0`) but not synthetically dropped (`dropout=1`) yields present=0 —
  both zeroed input and indicator=0 (guards against fallback-priority resurrecting
  a missing variable); **inference applies no dropout**: with an `input_dropout`
  config serialized on `SingleModuleStepConfig`, public `predict` /
  `predict_generator` (called with `input_dropout_mask=None`, hook never fired)
  produces identical output to a no-dropout run — guards against serialized
  training config leaking into a non-training path; **eval-mode training batch
  applies no dropout**: with `input_dropout` configured, `train_on_batch` using
  `NullOptimization` (or modules explicitly set to eval before `_accumulate_loss`)
  does not generate/pass an `input_dropout_mask` and matches the no-dropout
  result; **coupled + `input_dropout` raises**: constructing/running coupled
  training with a component whose `input_dropout` is configured raises a clear
  unsupported-feature error (guards against silent no-op).

### Helper design for Commit 2

- Add `_repeat_interleaved_tensor_mapping(mask, n_ensemble)` near training code to
  expand synthetic masks from base batch to folded ensemble batch without abusing
  `BatchData.data_mask`.
- Add `_combine_input_presence_masks(...)` inside the Step/network-call layer for
  `real & synthetic` input presence. This helper is only for input zeroing and
  channel-mask indicators; never feed its result to `StepLoss`.

### Adopted from Codex plan

- **`None` short-circuit in mask helpers** (per Codex `_merge_data_masks`): both
  `_repeat_interleaved_tensor_mapping` and `_combine_input_presence_masks` return
  `None` when their mask input is `None`, so the no-dropout / no-`data_mask` path
  stays a clean pass-through and never allocates an all-ones mask.
- **State the final network-input channel order explicitly** for the implementer
  (per Codex step 4). When `include_channel_mask_inputs=True` the module receives
  the packed data channels followed by one positional indicator channel per packed
  channel, built as
  `in_packer.pack(_build_channel_mask_dict(in_packer.names, data_mask, ...))` and
  concatenated onto the packed input (`single_module.py:410-421`). The indicator
  channels are **positional, not separately named** — there is no
  `__channel_mask__<name>` namespace. Dropout samples **only over the data half**
  (`self.in_packer.names`); the indicator half is never independently dropped —
  each indicator's value is the AND-combined `real & synthetic` presence of its
  data channel (the existing `mask_tensor * channel_mask` at
  `single_module.py:415-418`, generalized to the name-keyed mask).

## Commit 3 — clean up `VariableMaskingConfig` API

- Now that no Step-facing call site needs ensemble-aware sampling,
  `fme/core/var_masking.py` `sample_mask` returns `[batch, n_channels]` over
  `in_names` only (drop `n_ensemble` param, drop `_repeat_ensemble_mask` /
  `_get_base_batch_size`). Fold in open review nit: rename `max_vars` →
  `max_masked_vars`, drop `min_vars` so the minimum masked count is **0** (was
  default 1 for `UniformMaskingConfig` — a behavior change, see below).
- Backwards compatibility for this training-only variable-masking config API is
  not a concern/priority for this branch. Do not add alias fields or deprecation
  shims for `min_vars` / `max_vars`; update generated experiment configs,
  config generators, and tests to the new `max_masked_vars` schema.
- Update var-masking docstrings and tests to consistently say "masked variables"
  rather than ambiguous "variables" or "channels" when describing sampled counts.
- Update `fme/core/test_var_masking.py` for the new API and the zero-masked
  minimum behavior.
- Existing tests confirm green.

## Commit 4 — dist coordination + parallel test

- Random mask is per-sample. Under spatial/model parallelism all ranks hold the
  same samples but `torch.rand` diverges → tiles of one sample get different
  channels masked → corruption. (Current in-step code has this bug too.)
- Sample on all spatial ranks, then `dist.broadcast` the model-parallel group
  root's value across the group. This intentionally advances the RNG on every
  spatial co-rank while still making the actual dropout mask identical across
  tiles. Do this in the sampling hook before the mask is returned to training.
- **No `broadcast` exists on `Distributed` today** (only `reduce_*`, `gather*`,
  `scatter*`, `spatial_reduce_sum`, `is_root`), so add the primitive:
  - Add `Distributed.broadcast_spatial(tensor)` delegating to the backend.
  - `non_distributed`: return the tensor unchanged.
  - `torch_distributed` (data-parallel only, no tiles): return unchanged — each
    rank already holds distinct samples, so no agreement is needed.
  - `model_torch_distributed`: `torch.distributed.broadcast(tensor, src,
    group=self._spatial_group)` where `src = torch.distributed.get_global_rank(
    self._spatial_group, 0)`. Broadcast over `_spatial_group` only, **never the
    whole world** — data-parallel ranks must keep different masks (different
    samples).
  - In the hook: every rank samples the full `[batch, n_channels]` mask, then
    calls `broadcast_spatial` before returning so the spatial root's sampled
    mask overwrites the co-ranks. The mask has no spatial dim, so it can't be
    sliced per-tile; it must be made identical across the tile group.
- Add a `@pytest.mark.parallel` test (per AGENTS.md): generate `.pt` baseline from a
  single-rank `python -m pytest` run, verify under `torchrun`.

---

## Behavior changes to note in PR description

- Dropout is now **per-window** not per-step (matches inference + real `data_mask`).
- `UniformMaskingConfig` minimum masked count is now **0** (was 1): `min_vars`
  dropped. A window may now have no channels dropped.
- GMR extra channels remain **independently maskable** (no change — both the
  dropped input and its indicator channel already behaved this way).
- **Channel-mask indicator channels are not maskable** (explicit decision, kept).
  When `include_channel_mask_inputs=True`, dropout samples only over
  `in_packer.names` (data channels), never the positional indicator channels
  (the second half of the packed input, concatenated by `network_call`) —
  masking the presence signal itself is circular. An indicator
  channel's value reflects its data channel's combined `real & synthetic`
  presence; the indicator is never independently dropped.
- Synthetic dropout no longer uses `data_mask`; real missing-variable masks remain
  the only masks that exclude targets from loss. Predictions generated with
  synthetic dropout still contribute to the objective.

## Considered and rejected: move `input_dropout` to `TrainStepperConfig`

Codex proposed removing `input_dropout` from `SingleModuleStepConfig` and adding
`input_dropout: VariableMaskingConfig | None` to `TrainStepperConfig`, with the
training layer sampling the mask via a new step interface
(`get_maskable_input_channel_names()`). Rationale given: dropout is training-only,
so it need not be serialized into the inference step-reconstruction config.

**Rejected.** This plan keeps `input_dropout` in `SingleModuleStepConfig` and
samples through the `make_input_dropout_mask` hook (Commit 1). Reasons:

- Encapsulation: the step owns `self.in_packer.names` (the packed channel order,
  incl. GMR extras). Sampling inside the step avoids leaking that channel order
  out through a `get_maskable_input_channel_names()` accessor plus a separate
  training-side sampling helper.
- Lower churn: `input_dropout` already lives in `SingleModuleStepConfig` on this
  branch; relocating it is a config-compat change for an inference-reconstructed
  config, which AGENTS.md flags as critical to keep stable.
- The field is inactive at inference either way (the hook is only called from the
  training `_accumulate_loss` path); serialized-but-inert is acceptable.

Coupled training relocation (`ComponentTrainingConfig.input_dropout`) is out of
scope — same hook-in-step approach applies per component.

## Validation commands

Run focused tests first:

```bash
python -m pytest fme/core/test_var_masking.py
python -m pytest fme/core/step/test_args.py fme/core/step/test_step.py
python -m pytest fme/ace/stepper/test_single_module.py
python -m pytest fme/coupled/test_stepper.py
```

If a parallel test is added:

```bash
FME_FORCE_CPU=1 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 FME_DISTRIBUTED_W=1 \
  torchrun --nproc-per-node 2 -m pytest -m parallel fme/core/distributed/parallel_tests/test_step.py
```

Before finalizing:

```bash
pre-commit run --all-files
```

# Plan: Keep Variable Dropout Out of Step Ensemble Semantics

## Objective

Address the PR review concern that `Step` should not know about ensemble layout.
Variable/input dropout is a training-time data augmentation, so it should be
implemented in a layer that still has `BatchData.n_ensemble`, not by adding
`n_ensemble` to `StepArgs` or by teaching `Step` that batch members are
repeat-interleaved ensembles.

## Current Issue

The PR currently:

- Adds `n_ensemble` to `fme.core.step.args.StepArgs`.
- Threads `n_ensemble` through uncoupled and coupled prediction generator APIs.
- Samples the dropout mask inside `fme.core.step.single_module.SingleModuleStep.step`.
- Assumes the packed batch dimension is ordered as `base_sample.repeat_interleave(n_ensemble)`.

That violates the existing contract: `Step` receives tensor mappings and should
not interpret ensemble structure hidden inside the batch dimension.

## Recommended Direction

Use the hard version of reviewer Option 1:

- Move dropout mask generation above `Step`, into training/stepper code that has
  access to `BatchData.n_ensemble`.
- Use the existing `data_mask` pathway for real missing-variable masks and for
  mask-indicator construction where it fits, but represent synthetic input
  dropout as a concrete channel mask when it targets packed or appended network
  input channels.
- Keep `StepArgs` free of ensemble metadata.
- Avoid a broad refactor to make all `Step` tensor mappings carry an explicit
  ensemble dimension.

## Implementation Plan

1. Remove `n_ensemble` from core step arguments.

   - Revert the `n_ensemble` parameter and attribute in
     `fme/core/step/args.py`.
   - Remove propagation from `StepArgs.apply_input_process_func`.
   - Remove the added assertion from `fme/core/step/test_args.py`.

2. Remove dropout sampling from `SingleModuleStep.step`.

   - In `fme/core/step/single_module.py`, delete the random `channel_mask`
     sampling block.
   - If the chosen design passes a concrete precomputed mask to `Step`, keep only
     deterministic application of that mask; `SingleModuleStep` must not call the
     random sampler or interpret ensemble grouping.
   - Keep `_apply_input_mask` and `_build_channel_mask_dict`; these already
     express per-sample variable presence without exposing ensemble structure.
   - Do not pass `n_ensemble` to `VariableMaskingConfig.sample_mask`.

3. Move the dropout config to training-owned configuration.

   Preferred shape:

   - Add `input_dropout: VariableMaskingConfig | None = None` to
     `TrainStepperConfig`.
   - For coupled training, add the same field to `ComponentTrainingConfig`, so
     atmosphere and ocean can be configured independently.
   - Remove `input_dropout` from `SingleModuleStepConfig`.

   Rationale: this feature is disabled during inference and does not need to be
   serialized as part of inference step reconstruction.

4. Mask full network-input channels, not only user variable names.

   Do not implement synthetic dropout solely as a `data_mask` keyed by
   `step.input_names`. That would miss synthetic channels that are appended after
   the user variables, especially global-mean-removal input channels.

   For `SingleModuleStep`, the maskable data-channel list should come from the
   actual packed channel order:

   - `self.in_packer.names`
   - this already includes `config.in_names`
   - this also includes `global_mean_removal.extra_channel_names`, such as
     `__gmr_extra__<source_field>`

   If `include_channel_mask_inputs=True`, the final tensor sent to the module is
   doubled by appending one mask-indicator channel for each packed input channel.
   Decide explicitly whether those indicator channels are independently maskable
   or always tied to their corresponding data channel. If the desired semantics
   are that all network-input channels have the option to be masked, the
   maskable final channel order should be:

   - `self.in_packer.names`
   - plus `__channel_mask__<packed_name>` for each packed name when
     `include_channel_mask_inputs=True`

   The mask-indicator value for a data channel should reflect real
   missing-variable masks and synthetic dropout of that data channel. If the
   indicator channel itself is independently dropped, that is a separate final
   channel mask applied after indicator construction.

5. Generate synthetic dropout masks in the training layer.

   For uncoupled training:

   - In `TrainStepper._accumulate_loss`, generate dropout masks while `BatchData`
     and `n_ensemble` are still available. This can happen after
     `input_ensemble_data` and `forcing_ensemble_data` are created with
     `broadcast_ensemble`, or inside a training-only per-step mask provider
     called by the prediction loop.
   - Sample over the full maskable network-input channel list, not just
     user-facing variable names.
   - Keep real missing-variable masks (`BatchData.data_mask`) separate from
     synthetic channel dropout. This is important because `data_mask` is also
     used by `StepLoss` to exclude missing output variables from the loss; random
     input dropout must not remove target variables from the training loss.
   - When a synthetic dropout affects a real input variable, combine it with the
     real `data_mask` only for the `Step` call and mask-indicator construction,
     using logical AND so existing missing-variable behavior is preserved.

   For coupled training:

   - In `CoupledTrainStepper._accumulate_loss`, after component data has been
     broadcast to `n_ensemble`, generate separate component masks from each
     component's `ComponentTrainingConfig.input_dropout`.
   - Preserve existing component `data_mask` values by logical AND for the step
     call.
   - Keep the original component `data_mask` available for loss computation; do
     not let synthetic input dropout mask coupled target losses.
   - Use each component stepper's own maskable channel list so atmosphere and
     ocean components handle their own GMR extras and optional mask-indicator
     channels correctly.

6. Apply precomputed masks without exposing ensemble semantics to `Step`.

   A pure `BatchData.data_mask` implementation is not enough if all packed or
   final channels must be independently maskable, because some network-input
   channels are synthetic and do not exist in `BatchData.data`, and because
   `data_mask` also has loss-masking semantics.

   Preferred design:

   - Training code samples a concrete per-step channel mask after ensemble
     broadcast, while `BatchData.n_ensemble` is still available.
   - `Step` receives only the concrete mask, never `n_ensemble`.
   - `SingleModuleStep` applies the provided mask after it has constructed the
     complete network input tensor, including GMR extras and optional
     mask-indicator channels.
   - This keeps ensemble grouping outside `Step`; `Step` only applies a mask that
     is already shaped for its batch and channel order.

   If this requires extending `StepArgs`, use a field whose semantics are a
   concrete mask, such as `input_channel_mask`, rather than an ensemble metadata
   field. The mask should be keyed by stable final-channel names or supplied in
   the exact final channel order exposed by the step. Update
   `StepArgs.apply_input_process_func` to preserve this field.

   Because `TrainStepper` holds a generic `Stepper`, expose the maskable channel
   order through a narrow step interface rather than by reaching into
   `SingleModuleStep` internals. For `MultiCallStep`, this should delegate to the
   wrapped step.

7. Use one dropout mask per rollout (no per-step resampling).

   The current implementation samples dropout inside each `Step.step` call, which
   produces a fresh mask every forecast step. This refactor intentionally changes
   that: sample the dropout mask once per rollout and reuse it for every forecast
   step.

   Rationale:

   - Sampling once keeps the dropout mask out of the per-step generator loop, so
     no per-step mask-provider callback or per-step iterable needs to be threaded
     through `predict_generator`.
   - The mask is generated in the training layer after ensemble broadcast (step
     5/6) and passed once into the rollout.

   Approach:

   - In `_accumulate_loss`, sample the concrete channel mask once before the
     prediction loop, while `BatchData.n_ensemble` is still available.
   - Pass that single mask into the rollout so each `Step.step` call applies the
     same mask.
   - Keep public inference-oriented `get_prediction_generator` signatures clean;
     the mask is supplied through the training-internal mask field only.
   - Document this behavior change in the PR: dropout is now resampled per
     training batch/rollout, not per forecast step.

8. Remove the `n_ensemble` API plumbing.

   Revert the new `n_ensemble` parameters and validation from:

   - `fme/ace/stepper/single_module.py::get_prediction_generator`
   - `fme/ace/stepper/single_module.py::predict_generator`
   - `fme/coupled/stepper.py::get_prediction_generator`
   - coupled calls into component `get_prediction_generator`

   Keep any unrelated fixes that preserve `BatchData.n_ensemble` when constructing
   new `BatchData` objects, if those are independently correct.

9. Update `VariableMaskingConfig` API.

   Address the separate review comment in `fme/core/var_masking.py`:

   - Rename `UniformMaskingConfig.max_vars` to `max_masked_vars`.
   - Remove configurable `min_vars`.
   - Always allow zero masked variables.
   - Update docstrings and tests to say "masked variables" explicitly.

10. Update tests.

   Remove tests that exist only to validate the rejected API:

   - `test_get_prediction_generator_infers_n_ensemble_for_input_dropout`
   - `test_get_prediction_generator_preserves_n_ensemble`
   - `test_input_dropout_ensemble_members_share_mask` as currently written, since
     it constructs `StepArgs(n_ensemble=...)`.
   - Any `fme/core/step/test_step.py` test that configures
     `SingleModuleStepConfig(input_dropout=...)` directly; these need to move to
     training-layer tests or become tests for deterministic application of a
     concrete `input_channel_mask`.

   Add or rewrite tests for the new behavior:

   - `StepArgs` has no `n_ensemble` attribute and propagates any concrete
     `input_channel_mask` field if one is added.
   - Variable dropout creates a synthetic channel mask in the training layer.
   - Existing real `data_mask` and overlapping synthetic dropout are combined by
     logical AND for the step call, while loss masking still uses only the real
     `data_mask`.
   - All ensemble members of one base sample receive the same generated mask.
   - GMR appended channels (`__gmr_extra__...`) are included in the maskable
     channel set and can be masked independently of their source fields.
   - When `include_channel_mask_inputs=True`, the appended mask-indicator
     channels are included in the maskable channel set if the chosen semantics
     require every final network-input channel to be independently maskable.
   - A single dropout mask is reused across all forecast steps of a rollout
     (not resampled per step).
   - Eval/inference paths do not apply input dropout.
   - `include_channel_mask_inputs=True` reflects synthetic dropout via the
     existing mask-input path.

11. Distributed testing.

   The mask is now sampled once per rollout in training code from `BatchData`
   before spatial scatter, so it is shared across forecast steps but generated
   on a single rank's view of the batch. Add the reviewer-requested
   `@pytest.mark.parallel` coverage to prove the per-rollout mask does not depend
   on spatial decomposition and stays consistent across spatial co-ranks.

## Suggested Helper Design

Add small helpers near training code for sampling, and keep mask application in
the step as a concrete tensor operation:

```python
def _merge_data_masks(
    base_mask: TensorMapping | None,
    dropout_mask: TensorMapping | None,
) -> TensorMapping | None:
    ...
```

Use this helper only for masks passed to `Step`, not for the loss mask.

Add a helper that samples a concrete mask in the final module-input channel
order:

```python
def _sample_input_dropout_channel_mask(
    config: VariableMaskingConfig,
    channel_names: list[str],
    batch_size: int,
    n_ensemble: int,
    device: torch.device,
) -> TensorDict:
    ...
```

The sampling helper can call `VariableMaskingConfig.sample_mask(...)`, but the
`n_ensemble` argument should remain local to this training helper, not be exposed
through `StepArgs`.

`SingleModuleStep` should expose or otherwise provide the stable maskable channel
order it will use at the module boundary:

```python
def get_maskable_input_channel_names(self) -> list[str]:
    ...
```

For `include_channel_mask_inputs=True`, this list should include both packed data
channels and the appended mask-indicator channels if the implementation supports
independent masking of every final channel.

## Review Response Summary

The intended response to the main review thread:

- Agree that passing `n_ensemble` into `StepArgs` leaks ensemble layout into
  `Step`.
- Refactor dropout into the training/stepper layer, where `BatchData.n_ensemble`
  is still explicit.
- Keep `data_mask` for real per-sample variable presence, and pass synthetic
  dropout as a concrete precomputed channel mask when it targets packed or
  appended network-input channels.
- Avoid the larger explicit-ensemble-dimension refactor because this feature is
  training-specific and unnecessary for post-training inference.

## Validation Commands

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

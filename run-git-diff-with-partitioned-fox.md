# Plan: Move Variable Masking to TrainStepper + Multi-step IC + Next-step Input

## Context

The current `VariableMaskingConfig` lives inside `AugmentationConfig` (data loading). Masking is applied by injecting NaN into IC timesteps before the data reaches the stepper; `_build_ic_channel_mask()` then re-detects those NaNs to reconstruct the mask. This indirect approach means data loading knows about training augmentation details, and NaN propagation through physics corrections requires a special workaround in `step_with_adjustments`.

Moving masking into `TrainStepperConfig` makes it an explicit training concern and enables:
1. Configurable IC look-back window (`n_ic_timesteps > 1`)
2. Per-step, per-variable IC masking (finer than current per-variable-across-all-IC)
3. Masked ground-truth observations at the TARGET timestep as additional input (assimilation-style), absent during inference
4. Separate masking configs for IC vs. next-step observations

---

## Architecture Overview

### Input to the model at training step k (predicting t+k+1):
```
[IC: n_ic_timesteps prognostic steps, each independently maskable]
+ [Next-step obs: ground truth at t+k+1, optionally masked]
+ [Forcing: current-step external forcing, single step]
→ predicts t+k+1
```

### Input at inference (autoregressive rollout):
```
[IC: rolling buffer of n_ic_timesteps model predictions]
+ [Next-step obs: ABSENT (zeros + channel mask indicator = 0)]
+ [Forcing: known external forcing]
→ predicts next step
```

---

## Step 1: Remove Variable Masking from AugmentationConfig

**Why:** Data loading should not know about training augmentation. The current NaN-injection approach is indirect and forces a special-case fix in `step_with_adjustments` to prevent NaN propagation through physics corrections. Moving masking to TrainStepper means we own the mask explicitly — no NaN injection, no NaN detection, no corrector workaround.

**Files:** `fme/ace/data_loading/augmentation.py`, `fme/ace/data_loading/getters.py`, `fme/ace/requirements.py`

- Remove `variable_masking: VariableMaskingConfig | None` field from `AugmentationConfig`
- Revert `AugmentationConfig.build_modifier()` to not accept `n_ic_timesteps` (back to rotate-only)
- Remove `n_ic_timesteps` field from `DataRequirements` — it was only threaded through to feed `build_modifier()`
- Remove `n_ic_timesteps=requirements.n_ic_timesteps` kwarg from `getters.py:151`
- Keep `VariableMaskingConfig` and `VariableMaskingModifier` classes in `augmentation.py` — they will be imported and reused by TrainStepper
- Keep `ComposedModifier` (still needed for rotate + any future modifiers)

---

## Step 2: Make n_ic_timesteps Configurable in SingleModuleStepConfig

**Why:** "How many input steps the model sees" is a model architecture decision — it changes the number of input channels the neural network is built with. It must live in `SingleModuleStepConfig`, not `TrainStepperConfig`, because the packer and channel count are determined at construction time.

**The core idea:** The neural network has no time dimension — it takes a flat `[batch, n_channels, lat, lon]` tensor. To give it 3 previous states instead of 1, we concatenate those states along the channel dimension (100 vars × 3 steps = 300 channels). The `Packer` doesn't need to change; we just expand the names it's built with.

**File:** `fme/core/step/single_module.py`

### Config change
Replace the hardcoded `n_ic_timesteps` property (currently always returns 1) with an actual field:
```python
n_ic_timesteps: int = 1
```

### Expanded packer
In `SingleModuleStep.__init__()`, build `in_packer` with timestep-expanded names for prognostic variables only. Prognostic variables are those in both `in_names` and `out_names` — they're the ones with a rolling history. Forcing variables (`in_names \ out_names`) are always single-step (current forcing only, no history buffer).

```python
ic_var_names = [n for n in config.in_names if n in set(config.out_names)]
forcing_names = [n for n in config.in_names if n not in set(config.out_names)]
expanded_ic = [f"{n}__t{i}" for n in ic_var_names for i in range(config.n_ic_timesteps)]
packer_names = expanded_ic + forcing_names
self.in_packer = Packer(packer_names)
# self.in_names still holds original names for normalization and masking
```

For `n_ic_timesteps=1` (default), `"temp__t0"` is the only copy — this is equivalent to the current `"temp"` behavior so old checkpoints remain loadable (though naming changes; see backward compat note below).

### Flattening IC in network_call
Before packing in `network_call()`, flatten the rolling buffer:
```python
# args.input["temp"] has shape [batch, n_ic, lat, lon] for prognostic vars
# args.input["solar"] has shape [batch, lat, lon] for forcing vars
flat_input = {}
for name in ic_var_names:
    for i in range(n_ic):
        flat_input[f"{name}__t{i}"] = args.input[name][:, i]  # [batch, lat, lon]
for name in forcing_names:
    flat_input[name] = args.input[name]
# then normalize and pack flat_input as before
```

### Rolling buffer in predict_generator
Replace `state = {k: ic_dict[k].squeeze(TIME_DIM) for k in ic_dict}` with a rolling buffer:
```python
# ic_dict[k]: [batch, n_ic, lat, lon]
ic_buffer = {k: ic_dict[k] for k in ic_dict}

# After each step, roll forward:
new_state_4d = {k: new_state[k].unsqueeze(1) for k in new_state}  # add time dim
ic_buffer = {
    k: torch.cat([ic_buffer[k][:, 1:], new_state_4d[k]], dim=1)
    for k in ic_buffer
}
```

### Masking with n_ic > 1
`_apply_input_mask()`: extend to handle `[batch, n_ic]` mask tensors when variable is `[batch, n_ic, lat, lon]`:
```python
# present: [batch, n_ic] → broadcast to [batch, n_ic, 1, 1]
broadcast = present.view(batch, n_ic, *([1] * spatial_dims))
```

`_build_channel_mask_dict()`: emit per-timestep indicators:
```python
# Variable "temp" with n_ic=3 and mask [batch, 3]:
# → {"temp__t0": [batch, lat, lon], "temp__t1": ..., "temp__t2": ...}
```

---

## Step 3: Add Next-step Prognostic Input Channels

**Why:** The model needs to be built knowing about these channels (they change the input channel count), so the config belongs in `SingleModuleStepConfig`. By always including them (even zeroed at inference), the model weight dimensions are identical between training and inference — the model just sees mask indicator = 0 for absent channels.

**Files:** `fme/core/step/single_module.py`, `fme/core/step/args.py`

### Config
Add to `SingleModuleStepConfig`:
```python
next_step_prognostic_input_names: list[str] = field(default_factory=list)
```
These are prognostic variables whose ground-truth observation at t+1 can be provided to the model as input during training.

### Packer expansion
Extend packer names further:
```python
next_step_channels = [f"{n}__next" for n in config.next_step_prognostic_input_names]
packer_names = expanded_ic + forcing_names + next_step_channels
```

### StepArgs
Add new field:
```python
next_step_prognostic_obs: TensorMapping | None = None
```
The dict contains `{name: [batch, lat, lon]}` tensors already masked by TrainStepper. When `None` (inference), those channels are filled with zeros and their mask indicators are 0.

### network_call
After flattening IC, add next-step channels:
```python
for name in self._config.next_step_prognostic_input_names:
    if args.next_step_prognostic_obs is not None and name in args.next_step_prognostic_obs:
        flat_input[f"{name}__next"] = normalize(args.next_step_prognostic_obs[name])
    else:
        flat_input[f"{name}__next"] = torch.zeros_like(ref_tensor)  # absent → zero
# channel mask indicators for {name}__next: 1.0 if present, 0.0 if absent
```

---

## Step 4: Move Masking Logic into TrainStepper

**Why:** The training loop owns the decision of which variables to mask, with what probability, and at which rollout steps. `TrainStepper.train_on_batch()` is the right place to sample masks, apply them, and pass them to `predict_generator`. The config fields belong in `TrainStepperConfig` (training behavior, not architecture).

**File:** `fme/ace/stepper/single_module.py`

### TrainStepperConfig additions
```python
ic_masking: VariableMaskingConfig | None = None
next_step_masking: VariableMaskingConfig | None = None
mask_all_rollout_steps: bool = False  # True: re-sample masks at every rollout step
                                       # False (default): mask only initial IC window
```

### New masking utility (replaces _build_ic_channel_mask)
```python
def _apply_variable_masking(
    data: dict[str, Tensor],   # [batch, n_ic, lat, lon] per variable
    config: VariableMaskingConfig,
    n_ic: int,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    # Returns: (masked_data, channel_mask)
    # masked_data: NaN-free; instead sets values to 0 in normalized space
    # channel_mask: {name: [batch, n_ic]} bool tensor, True=present
```
This replaces the indirect NaN-injection + NaN-detection cycle. The mask is constructed directly.

### predict_generator extended signature
```python
def predict_generator(
    ...
    ic_mask: dict[str, Tensor] | None = None,        # [batch, n_ic] per var, initial IC
    next_step_obs: dict[str, Tensor] | None = None,  # [batch, n_steps+1, lat, lon] ground truth
    next_step_mask_config: VariableMaskingConfig | None = None,
    mask_all_rollout_steps: bool = False,
) -> Generator[TensorDict, None, None]:
```

**channel_mask per step:** When `mask_all_rollout_steps=False` (default), use `ic_mask` only at step 0, fall back to `data_mask` for subsequent steps. When `True`, re-sample masks at each rollout step using `ic_masking` config.

**next_step_obs at each rollout step k (training only):** `next_step_obs` contains the full ground-truth batch `[batch, n_steps+1, lat, lon]`. At step k, extract the slice at `step+1`, apply `next_step_mask_config` sampling (so some variables/samples are randomly absent), and pass as `StepArgs.next_step_prognostic_obs`. This is only done when `next_step_obs is not None`, which is only the case during training — `TrainStepper` passes the ground truth from the batch; the inference `predict_generator` never receives it.

**next_step_obs during inference/evaluation:** `next_step_obs` is `None`. Inside `network_call`, every `{name}__next` channel is filled with zeros and its channel mask indicator is set to 0.0 (absent). The model handles this gracefully because during training it was regularly exposed to fully-absent next-step channels (whenever `next_step_masking` masked all variables for a sample). The autoregressive rollout therefore runs without any ground-truth leakage.

### TrainStepper.train_on_batch / _accumulate_loss
Replace current NaN-based channel_mask construction:
```python
# Old: detect NaN in IC → build channel_mask
# New: apply ic_masking config directly
n_ic = self.n_ic_timesteps
if self._config.ic_masking is not None:
    ic_data, ic_mask = _apply_variable_masking(
        {k: v[:, :n_ic] for k, v in data.items()},
        self._config.ic_masking,
        n_ic=n_ic,
    )
else:
    ic_data = {k: v[:, :n_ic] for k, v in data.items()}
    ic_mask = None
channel_mask = _combine_with_data_mask(ic_mask, data_mask)
```

---

## Step 5: Inference/Evaluation Behaviour

**Training path** (`TrainStepper.train_on_batch`): passes `next_step_obs` (ground truth from the batch) and `ic_mask` (sampled from `ic_masking` config) into `predict_generator`. Every rollout step gets masked ground-truth observations at the target timestep.

**Inference/evaluation path** (`Stepper.predict` / `predict_paired`): calls `predict_generator` with no `next_step_obs` and no `ic_mask`. Inside `network_call`, every `{name}__next` channel is zeros with indicator = 0. The IC buffer rolls forward using model predictions only — fully autoregressive, no ground-truth leakage.

**Why no code changes needed for inference:** The model was built with `include_channel_mask_inputs=True` and trained with `next_step_masking` sometimes fully masking all variables for a sample, so it already knows how to behave when those channels are absent. The `n_ic_timesteps > 1` rolling buffer is maintained in `predict_generator` for both paths.

---

## Critical Files

| File | Change |
|------|--------|
| `fme/ace/data_loading/augmentation.py` | Remove `variable_masking` from `AugmentationConfig`; keep `VariableMaskingConfig`/`VariableMaskingModifier` |
| `fme/ace/data_loading/getters.py` | Remove `n_ic_timesteps` kwarg from `build_modifier()` call |
| `fme/ace/requirements.py` | Remove `n_ic_timesteps` field |
| `fme/core/step/single_module.py` | `n_ic_timesteps` field in config; expanded packer; IC flattening in `network_call`; per-step masking; next-step channels |
| `fme/core/step/args.py` | Add `next_step_prognostic_obs` to `StepArgs` |
| `fme/ace/stepper/single_module.py` | `TrainStepperConfig` masking fields; rolling IC buffer; explicit masking in `train_on_batch`; remove `_build_ic_channel_mask` |
| `fme/core/step/test_step.py` | Tests for `n_ic_timesteps > 1`, next-step channels, per-step masking |
| `fme/ace/stepper/test_single_module.py` | Tests for new `TrainStepperConfig` masking fields |
| `fme/ace/data_loading/test_augmentation.py` | Remove tests for `variable_masking` in `AugmentationConfig` |

## Reused Utilities

- `VariableMaskingConfig`, `VariableMaskingModifier` — `fme/ace/data_loading/augmentation.py` (keep, import into stepper)
- `Packer` — `fme/core/packer.py` (unchanged; new expanded name lists passed to it)
- `_apply_input_mask`, `_build_channel_mask_dict` — `fme/core/step/single_module.py` (extend for n_ic > 1)
- `step_with_adjustments` — `fme/core/step/single_module.py` (remove NaN-fixing corrector workaround, no longer needed)

## Backward Compatibility Note

- Existing checkpoints with `n_ic_timesteps=1` and no next-step input: the packer will use `"temp__t0"` instead of `"temp"` for the single IC step. This changes the internal name but not the model weights (packer names are just metadata). `from_state` loading via `dacite` will need `n_ic_timesteps` defaulting to 1. **Confirm with user whether old checkpoint loading must be preserved before implementing.**
- Configs with `augmentation.variable_masking`: will fail at load time (clean break, confirmed).

---

## Verification

1. `make test_fast` passes after all changes
2. New unit tests cover:
   - `n_ic_timesteps=2`: forward pass produces correct channel count; rolling buffer advances correctly
   - Per-step IC masking: masked timestep is zeroed in normalized space, indicator=0
   - Next-step channels absent at inference: zeros + indicator=0 fed to network
   - `train_on_batch` with `ic_masking` config: loss is finite, channel masks reach the network
3. Integration smoke test: small training run with `n_ic_timesteps=2`, `ic_masking` configured, `next_step_prognostic_input_names` set — loss decreases over first few batches

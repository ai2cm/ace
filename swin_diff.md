# SwinTransformer vs ArchesWeather: Training Differences

## Differences that need to be addressed

### 1. **Scalar time conditioning (month + hour) — model gap** ✅

ArchesWeather embeds the forecast timestamp's calendar month and hour-of-day via `TimestepEmbedder` (DiT-style sinusoidal embedder), sums them to a `cond_emb` of shape `(B, cond_dim=256)`, and feeds it to every `CondBasicLayer` as an AdaLN modulation.

**Implemented**: `TimeConditionedSwinTransformerBuilder` in `fme/ace/registry/swin_transformer.py`. Adds `TimestepEmbedder` (sinusoidal + MLP) and `_TimeConditionedContextWrappedModule` which embeds month+hour from `StepArgs.forward_time` and passes the sum as `context.embedding_scalar`. Timestamp flows from `forcing_data.time` through `predict_generator` → `StepArgs.forward_time` → `Module.__call__` (gated by `has_time_conditioning`) → wrapper.

```yaml
module:
  type: TimeConditionedSwinTransformer
  config:
    embed_dim_scalar: 256
    # ... other swin fields
```

---

### 2. **Optimizer: differential weight decay** — training config ✅

ArchesWeather's `configure_optimizers` splits parameters into two groups:
- Weights (non-norm): `weight_decay=0.05`
- Biases + norm weights: `weight_decay=0`
- `lr=3e-4`, `betas=(0.9, 0.98)`

ACE's `OptimizationConfig` applies a single weight decay across all parameters. There's no built-in support for this split.

**Implemented**: Added `no_weight_decay_bias_and_norm: bool = False` to `OptimizationConfig` in `fme/core/optimization.py`. When `True`, `build()` partitions `named_parameters()` into two optimizer groups — params whose name contains `"weight"` but not `"norm"` get the configured `weight_decay`; all others (biases and norm-layer weights) get `weight_decay=0`. Flag defaults to `False` so existing configs are unaffected.

```yaml
optimization:
  lr: 3.0e-4
  weight_decay: 0.05
  betas: [0.9, 0.98]
  no_weight_decay_bias_and_norm: true
```

---

### 3. **LR schedule: cosine-with-warmup** — training config

ArchesWeather uses `diffusers.get_cosine_schedule_with_warmup` stepped per-iteration:
- 5000 warmup steps, 300k total steps, 0.5 cosine cycles (single decay to 0)

ACE has `SequentialSchedulerConfig` which could chain a `LinearLR` warmup with `CosineAnnealingLR` to approximate this. Currently typical ACE configs don't configure this for the SwinTransformer.

**Config Option:** Set `scheduler` inside `optimization` to a `SequentialSchedulerConfig` chaining `LinearLR` warmup into `CosineAnnealingLR`. Both sub-schedulers must have `step_each_iteration: true` so they are stepped per batch, not per epoch. The `milestones` value must match `total_iters` of the warmup scheduler.

```yaml
optimization:
  scheduler:
    schedulers:
      - type: LinearLR
        kwargs:
          start_factor: 0.0001   # near-zero initial LR
          end_factor: 1.0
          total_iters: 5000
        step_each_iteration: true
      - type: CosineAnnealingLR
        kwargs:
          T_max: 295000          # 300k total − 5k warmup
          eta_min: 0.0
        step_each_iteration: true
    milestones: [5000]
```

---

### 4. **Delta normalization in the loss** — training config

ArchesWeather's `loss_delta_normalization=True` rescales per-variable loss by `(var_std / delta_std)^2`, where `delta_std` is the estimated 6-hour tendency std (loaded from `pangu_norm_stats2_with_w.pt`). This is more aggressive weighting of slowly-changing variables than a raw MSE on normalized outputs.

ACE's loss uses `loss_normalization` (normalizer std), which is equivalent to ArchesWeather's *un-delta-normalized* mode. The delta normalization is a separate scaling on top.

**Config Option:** No code change required — `residual_normalization` in `StepperConfig` already does this. It overrides the loss normalizer's std for prognostic variables with per-variable residual stds from a separate stats file, while leaving the network normalizer unchanged. Precompute a per-variable 6-hour tendency std stats file from the training dataset (equivalent to ArchesWeather's hardcoded `delta_std` constants) and point `residual_normalization` at it:

```yaml
stepper_config:
  residual_normalization:
    global_means_path: /path/to/tendency_means.nc
    global_stds_path: /path/to/tendency_stds.nc  # per-variable 6h tendency stds
```

---

### 5. **Multistep curriculum** — training config

ArchesWeather starts at 2-step rollouts and increments by 1 every 2 epochs. ACE supports `TimeLengthSchedule` for `n_forward_steps` in `TrainStepperConfig`, so the infrastructure is there — it just needs to be configured.

**Config Option:** Set `n_forward_steps` in `stepper_training` to a `TimeLengthSchedule` with `start_value: 2` and milestones incrementing by 1 every 2 epochs up to the desired maximum. The data loader window is sized to the schedule's `max_n_forward_steps`, so set that to the intended ceiling (e.g. 8):

```yaml
stepper_training:
  n_forward_steps:
    start_value: 2
    milestones:
      - epoch: 2
        value: 3
      - epoch: 4
        value: 4
      - epoch: 6
        value: 5
      - epoch: 8
        value: 6
      - epoch: 10
        value: 7
      - epoch: 12
        value: 8
```

---

### 6. **2-step input (previous state concatenation)** — data / model config

ArchesWeather uses `use_prev=True` in its dataloader to load the state at `t - Δt` and `n_concatenated_states=1` in the embedder, which doubles the input channels by concatenating the previous state with the current one before patching.

In ACE, the stepper's `n_ic_timesteps` controls how many initial-condition timesteps are stacked into the channel dimension before the model sees them. Setting `n_ic_timesteps=2` gives the SwinTransformer the same two-frame input with no model code changes.

**Config Option:** `n_ic_timesteps` is currently hardcoded to 1 as a property on `SingleModuleStepConfig` (`fme/core/step/single_module.py:121`). A small code change is required to make it a configurable field (default 1); after that, set it under the step config:

```yaml
stepper_config:
  step:
    n_ic_timesteps: 2
```

No architecture changes required; `in_chans` doubles automatically because the stepper stacks IC timesteps into channels.

---

### 7. **4 independently trained ensemble members** — training strategy

ArchesWeather trains 4 deterministic models with different random seeds, then combines them at inference via `avg_with_modules` to form ArchesWeather-Mx4. These averaged predictions are also the backbone for ArchesWeatherGen.

ACE has no equivalent multi-model averaging infrastructure at inference time.

**Fix needed**: Train 4 SwinTransformer runs with different seeds. At inference, average their outputs. This requires either (a) running 4 separate jobs and averaging predictions externally, or (b) adding a lightweight `avg_with_modules`-style wrapper to the ACE inference path that loads N checkpoints and averages their forward passes.

---

### 8. **Daily-averaged input variables** — data preprocessing

From the paper: "Rather than using instantaneous state variables as in ERA5, we pre-process the dataset by computing daily averaged physical variables… Daily averages are generated from 6-hourly ERA5 using a rolling window of size 4 and stride 1."

This is an **offline preprocessing step** — the code just loads whatever files are on disk. The rolling average is applied to the ERA5 files before training, producing daily-mean snapshots that still step at 6-hourly intervals (stride-1 rolling window). The effect is to remove the diurnal cycle from the target, making long-range dynamics easier to learn.

**Fix needed**: Preprocess the training dataset by applying a rolling mean of window size 4 (24 h) over the 6-hourly ERA5 files before feeding them to ACE. No model code changes required.

---

### 9. **`axis_attn` / `LinVert`** — minor architecture note ✅

ArchesWeather's full config uses `axis_attn=True` (cross-level axial attention). ACE's 2D adaptation replaces this with `ChannelMixer` (a single linear mixing all channels once before the U-Net). This is a deliberate 2D simplification and not a bug, but it means the vertical/channel interaction is weaker than in ArchesWeather.

**Implemented**: `AxialAttentionMixer` in `fme/core/models/swin_transformer/swin_layers.py`. Zero-dependency port of the `axial-attention` library: replicates `AxialPositionalEmbedding(dim, shape=(H,W), emb_dim_index=-1)` (per-row and per-column learnable parameters) and `AxialAttention(num_dimensions=2, sum_axial_out=True)` (independent H- and W-axis self-attentions, outputs summed). Replaces `ColumnMixer` in every `SwinTransformerBlock` when `axis_attn=True`.

```yaml
module:
  type: TimeConditionedSwinTransformer  # or SwinTransformer / NoiseConditionedSwinTransformer
  config:
    axis_attn: true
    # ... other swin fields
```

---

## Summary

| Gap | Where |
|---|---|
| Month/hour time conditioning (embedding_scalar wrapper) ✅ | **Model / registry** |
| Differential weight decay (norm params get wd=0) ✅ | **Optimizer config** |
| Cosine schedule with warmup | **LR scheduler config** |
| Delta normalization in loss | **Loss config** |
| Multistep curriculum (2-step start, +1 per 2 epochs) | **Training loop config** |
| 2-step input (`n_ic_timesteps=2`) | **Stepper config** |
| 4-seed ensemble + averaging at inference | **Training strategy / inference** |
| Daily-averaged inputs (rolling window of 4 over 6-hourly ERA5) | **Data preprocessing** |
| `axis_attn` (2D axial self-attention replacing ColumnMixer) ✅ | **Model / registry** |

**Not found in ArchesWeather codebase**: monthly mean SST/sea ice forcings. This does not appear in the ArchesWeather codebase or documentation; ArchesWeather's surface variables are `[U10m, V10m, T2m, MSLP]` only.

**In paper but implemented as offline preprocessing**: daily variable averaging (see item 9 above).

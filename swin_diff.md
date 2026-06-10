# SwinTransformer vs ArchesWeather: Training Differences

## Differences that need to be addressed

### 1. **Scalar time conditioning (month + hour) — model gap** ✅

ArchesWeather embeds the forecast timestamp's calendar month and hour-of-day via `TimestepEmbedder` (DiT-style sinusoidal embedder), sums them to a `cond_emb` of shape `(B, cond_dim=256)`, and feeds it to every `CondBasicLayer` as an AdaLN modulation.

**Implemented**: `TimeConditionedSwinTransformerBuilder` in `fme/ace/registry/swin_transformer.py`. Adds `TimestepEmbedder` (sinusoidal + MLP) and `_TimeConditionedContextWrappedModule` which embeds month+hour from `StepArgs.forward_time` and passes the sum as `context.embedding_scalar`. Timestamp flows from `forcing_data.time` through `predict_generator` → `StepArgs.forward_time` → `Module.__call__` (gated by `has_time_conditioning`) → wrapper. Use `type: TimeConditionedSwinTransformer` with `embed_dim_scalar: 256` in config.

---

### 2. **Optimizer: differential weight decay** — training config ✅

ArchesWeather's `configure_optimizers` splits parameters into two groups:
- Weights (non-norm): `weight_decay=0.05`
- Biases + norm weights: `weight_decay=0`
- `lr=3e-4`, `betas=(0.9, 0.98)`

ACE's `OptimizationConfig` applies a single weight decay across all parameters. There's no built-in support for this split.

**Implemented**: Added `no_weight_decay_bias_and_norm: bool = False` to `OptimizationConfig` in `fme/core/optimization.py`. When `True`, `build()` partitions `named_parameters()` into two optimizer groups — params whose name contains `"weight"` but not `"norm"` get the configured `weight_decay`; all others (biases and norm-layer weights) get `weight_decay=0`. Flag defaults to `False` so existing configs are unaffected.

---

### 3. **LR schedule: cosine-with-warmup** — training config

ArchesWeather uses `diffusers.get_cosine_schedule_with_warmup` stepped per-iteration:
- 5000 warmup steps, 300k total steps, 0.5 cosine cycles (single decay to 0)

ACE has `SequentialSchedulerConfig` which could chain a `LinearLR` warmup with `CosineAnnealingLR` to approximate this. Currently typical ACE configs don't configure this for the SwinTransformer.

---

### 4. **Delta normalization in the loss** — training config

ArchesWeather's `loss_delta_normalization=True` rescales per-variable loss by `(var_std / delta_std)^2`, where `delta_std` is the estimated 6-hour tendency std (loaded from `pangu_norm_stats2_with_w.pt`). This is more aggressive weighting of slowly-changing variables than a raw MSE on normalized outputs.

ACE's loss uses `loss_normalization` (normalizer std), which is equivalent to ArchesWeather's *un-delta-normalized* mode. The delta normalization is a separate scaling on top.

**Fix needed**: No code change required — `residual_normalization` in `StepperConfig` already does this. It overrides the loss normalizer's std for prognostic variables with per-variable residual stds from a separate stats file, while leaving the network normalizer unchanged. What's needed is to precompute a per-variable 6-hour tendency std stats file from the training dataset (equivalent to ArchesWeather's hardcoded `delta_std` constants) and point `residual_normalization` at it in the YAML config.

---

### 5. **Multistep curriculum** — training config

ArchesWeather starts at 2-step rollouts and increments by 1 every 2 epochs. ACE supports `IntSchedule` for `n_forward_steps`, so the infrastructure is there — it just needs to be configured.

---

### 6. **2-step input (previous state concatenation)** — data / model config

ArchesWeather uses `use_prev=True` in its dataloader to load the state at `t - Δt` and `n_concatenated_states=1` in the embedder, which doubles the input channels by concatenating the previous state with the current one before patching.

In ACE, the stepper's `n_ic_timesteps` controls how many initial-condition timesteps are stacked into the channel dimension before the model sees them. Setting `n_ic_timesteps=2` gives the SwinTransformer the same two-frame input with no model code changes.

**Fix needed**: Set `n_ic_timesteps=2` in `StepperConfig`. No architecture changes required; `in_chans` doubles automatically because the stepper stacks IC timesteps into channels.

---

### 7. **4 independently trained ensemble members** — training strategy

ArchesWeather trains 4 deterministic models with different random seeds, then combines them at inference via `avg_with_modules` to form ArchesWeather-Mx4. These averaged predictions are also the backbone for ArchesWeatherGen.

ACE has no equivalent multi-model averaging infrastructure at inference time.

**Fix needed**: Train 4 SwinTransformer runs with different seeds. At inference, average their outputs. This requires either (a) running 4 separate jobs and averaging predictions externally, or (b) adding a lightweight `avg_with_modules`-style wrapper to the ACE inference path that loads N checkpoints and averages their forward passes.

---

### 9. **Daily-averaged input variables** — data preprocessing

From the paper: "Rather than using instantaneous state variables as in ERA5, we pre-process the dataset by computing daily averaged physical variables… Daily averages are generated from 6-hourly ERA5 using a rolling window of size 4 and stride 1."

This is an **offline preprocessing step** — the code just loads whatever files are on disk. The rolling average is applied to the ERA5 files before training, producing daily-mean snapshots that still step at 6-hourly intervals (stride-1 rolling window). The effect is to remove the diurnal cycle from the target, making long-range dynamics easier to learn.

**Fix needed**: Preprocess the training dataset by applying a rolling mean of window size 4 (24 h) over the 6-hourly ERA5 files before feeding them to ACE. No model code changes required.

---

### 10. **`axis_attn` / `LinVert`** — minor architecture note

ArchesWeather's full config uses `axis_attn=True` (cross-level axial attention). ACE's 2D adaptation replaces this with `ChannelMixer` (a single linear mixing all channels once before the U-Net). This is a deliberate 2D simplification and not a bug, but it means the vertical/channel interaction is weaker than in ArchesWeather.

---

## Summary

| Gap | Where |
|---|---|
| Month/hour time conditioning (embedding_scalar wrapper) | **Model / registry** |
| Differential weight decay (norm params get wd=0) | **Optimizer config** |
| Cosine schedule with warmup | **LR scheduler config** |
| Delta normalization in loss | **Loss config** |
| Multistep curriculum (2-step start, +1 per 2 epochs) | **Training loop config** |
| 2-step input (`n_ic_timesteps=2`) | **Stepper config** |
| 4-seed ensemble + averaging at inference | **Training strategy / inference** |
| Daily-averaged inputs (rolling window of 4 over 6-hourly ERA5) | **Data preprocessing** |

**Not found in ArchesWeather codebase**: monthly mean SST/sea ice forcings. This does not appear in the ArchesWeather codebase or documentation; ArchesWeather's surface variables are `[U10m, V10m, T2m, MSLP]` only.

**In paper but implemented as offline preprocessing**: daily variable averaging (see item 9 above).

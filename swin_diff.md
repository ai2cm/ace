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

ArchesWeather uses `diffusers.get_cosine_schedule_with_warmup` stepped **per-iteration** (`"interval": "step"`): 5000 warmup steps, 300k total steps, 0.5 cosine cycles (single decay to 0).

ACE's `SequentialSchedulerConfig` supports per-iteration stepping via `step_each_iteration: true` on each sub-scheduler. The config below uses the simpler **per-epoch** approximation (default `step_each_iteration: false`) with counts proportionally scaled to 120 epochs: 5000/300k ≈ 1.67% → 2 warmup epochs, 118 cosine epochs. To step per-iteration instead, add `step_each_iteration: true` to each sub-scheduler and replace `total_iters`/`T_max` with actual batch counts.

```yaml
optimization:
  scheduler:
    schedulers:
      - type: LinearLR
        kwargs:
          start_factor: 0.0001   # near-zero initial LR
          end_factor: 1.0
          total_iters: 2         # epochs (~1.67% of 120, matching ArchesWeather proportion)
      - type: CosineAnnealingLR
        kwargs:
          T_max: 118             # 120 total − 2 warmup epochs
          eta_min: 0.0
    milestones: [2]
```

---

### 4. **Delta normalization in the loss** — training config ✅

ArchesWeather's `loss_delta_normalization=True` rescales per-variable loss by `(var_std / delta_std)^2`, where `delta_std` values are **hardcoded constants** loaded from `pangu_norm_stats2_with_w.pt` (not computed from the training dataset). This upweights slowly-changing variables relative to a raw MSE on normalized outputs.

**Note**: Delta normalization is not described in either the ArchesWeather paper or the ArchesWeatherAIMIP paper — it exists only in the code (`ArchesWeather/geoarches/lightning_modules/forecast.py:77–93`), where it is explicitly called **"fake delta normalization"** in a comment (line 84). It is enabled by default with no ablation or justification in the papers.

**Implemented**: The equivalent in ACE is `normalization.residual` inside `step.config`, which makes the network predict residuals and normalises the loss by per-variable residual (tendency) stds from a pre-computed stats file. The current config already points this at `scaling-residual.nc`, which plays the role of ArchesWeather's hardcoded `delta_std` values.

```yaml
stepper:
  step:
    config:
      normalization:
        residual:
          global_means_path: /path/to/centering.nc
          global_stds_path: /path/to/scaling-residual.nc  # per-variable tendency stds
```

---

### 5. **Multi-step fine-tuning (optional Phase 3)** — training config

ArchesWeather uses **single-step training** (`rollout_iterations: 1`) for its full main training run (Phase 1 + Phase 2). Phase 3 is an optional separate fine-tuning stage applied after Phase 1/2: fixed-length rollouts of 2 days for 8k steps, then 3 days for 8k steps, then 4 days for 4k steps, summing MSE losses over all rollout steps.

The `on_train_epoch_start` progressive-curriculum code in `forecast.py` (lines 314-316) does exist but is never triggered for standard ArchesWeather training: the `dataset.multistep > 1` guard is never satisfied because the training config starts with `rollout_iterations: 1`.

**Config:** Use `n_forward_steps: 1` for the main training run. Multi-step fine-tuning would be a separate run starting from a Phase 1/2 checkpoint with a higher fixed `n_forward_steps`.

```yaml
stepper_training:
  n_forward_steps: 1
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

### 8. **Daily-averaged input variables** — data preprocessing ✅

From the ArchesWeatherAIMIP paper: "Rather than using instantaneous state variables as in ERA5, we pre-process the dataset by computing daily averaged physical variables… Daily averages are generated from 6-hourly ERA5 using a rolling window of size 4 and stride 1."

This is an **offline preprocessing step** — the code just loads whatever files are on disk. The rolling average produces daily-mean snapshots that still step at 6-hourly intervals (stride-1 rolling window), removing the diurnal cycle and making long-range dynamics easier to learn. There is no daily-averaging code in the ArchesWeather repository; it was applied externally before training.

**Implemented**: The ACE training config already uses `2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr`, which is a pre-computed daily-averaged ERA5 dataset. No further action required.

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

### 10. **SwiGLU MLP** — model config

ArchesWeather uses SwiGLU in all transformer blocks (`mlp_layer: swiglu`). To keep parameter count equal to a standard MLP at `mlp_ratio=4.0`, it internally adjusts the hidden dimension to `mlp_ratio * 2/3 = 2.667` before constructing the SwiGLU (timm's two-branch formulation). ACE's SwiGLU implementation doubles the hidden dimension internally (single `Linear(d, 2h)` then chunk), so to match ArchesWeather's parameter count, `mlp_ratio` must be set to `8/3 ≈ 2.6667`.

**Config:** Change `mlp_layer` and `mlp_ratio` in the module config:

```yaml
module:
  config:
    mlp_layer: swiglu
    mlp_ratio: 2.6667   # matches ArchesWeather param count; mlp_ratio=4.0 with standard MLP
```

---

### 11. **Gradient clipping** — training config

ArchesWeather uses `gradient_clip_val=1` (via PyTorch Lightning's `Trainer`). ACE's training loop has no gradient clipping support — no equivalent config field exists.

**Fix needed**: Add gradient clipping support to `fme/core/optimization.py` (e.g. a `gradient_clip_val: float | None = None` field that calls `torch.nn.utils.clip_grad_norm_` before the optimizer step), then set it to `1.0` in the config.

---

## Summary

| Gap | Where |
|---|---|
| Month/hour time conditioning (embedding_scalar wrapper) ✅ | **Model / registry** |
| Differential weight decay (norm params get wd=0) ✅ | **Optimizer config** |
| Cosine schedule with warmup (per-epoch approximation) | **LR scheduler config** |
| Delta normalization in loss ✅ | **Step config (`normalization.residual`)** |
| Multi-step fine-tuning (Phase 3, optional; single-step main training) | **Training loop config** |
| 2-step input (`n_ic_timesteps=2`) | **Stepper config** |
| 4-seed ensemble + averaging at inference | **Training strategy / inference** |
| Daily-averaged inputs ✅ | **Data preprocessing (zarr already daily)** |
| `axis_attn` (2D axial self-attention replacing ColumnMixer) ✅ | **Model / registry** |
| SwiGLU MLP (`mlp_layer: swiglu`, `mlp_ratio: 2.6667`) | **Model config** |
| Gradient clipping (`clip_val=1`) | **Training loop (needs code change)** |

**ArchesWeather (original) only**: surface variables are `[U10m, V10m, T2m, MSLP]` — no SST/SIC forcings.

**ArchesWeatherAIMIP addition**: monthly mean SST and sea ice cover (SIC) are concatenated to surface variables as additional conditioning inputs before embedding. ACE handles `surface_temperature` and `sea_ice_fraction` as regular instantaneous state variables, not as separate monthly-mean forcings — this is a fidelity gap for the AIMIP protocol but requires a code change to properly implement.

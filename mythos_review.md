# Swin/ArchesWeather Matching Review

Comprehensive comparison of the ACE swin implementation (`fme/core/models/swin_transformer/`,
`fme/ace/registry/swin_transformer.py`), the ArchesWeather code (`~/Git/ArchesWeather/geoarches/`),
both papers (ArchesWeather & ArchesWeatherGen; ArchesWeatherAIMIP) including appendices, the six
config variants in `configs/experiments/2026-06-11-swin/`, and the existing `swin_diff.md`.

## What already matches (verified, no action needed)

- **SwiGLU param parity**: `mlp_ratio: 2.6667` + chunk-style SwiGLU is exactly equivalent to
  timm's two-branch SwiGLU at Arches' `4.0 * 2/3` (swin_diff item 10 ✅).
- **2-frame input**: `n_ic_timesteps: 2` matches AIMIP's "conditioned on the two previous
  states" ✅.
- **Optimizer**: lr 3e-4, betas (0.9, 0.98), wd 0.05, `no_weight_decay_bias_and_norm` — matches
  ArchesWeather appendix A.1 exactly (batch: theirs is 4, ours 8; total sample exposure ~1.2M vs
  ~1.5M — close enough).
- **Scheduler**: 2-epoch warmup / 118 cosine ≈ their 5k/300k proportion ✅.
- **Delta normalization**: `normalization.residual` with `scaling-residual.nc` reproduces Arches'
  `(σ_var/σ_Δ)²` loss rescaling ✅.
- **U-Net skeleton**: depths `[2,6,6,2]×d`, shared deep drop-path schedule, PatchMerging
  (norm→linear 4C→2C), PatchExpanding (linear→pixelshuffle→norm→linear), LinVert↔ChannelMixer,
  per-layer shared AdaLN projection, `residual_prediction: false`, drop_path 0.2 — all faithful ✅.
- **Circular topology**: the `earth` padding (circular lon, polar reflection) before encoder *and*
  decoder reproduces AIMIP's "circular padding in the decoder" ablation winner ✅. Nice detail:
  45×90 + pad(2,1)/(3,3) = 48×96 is an exact multiple of 2×window, so no dead zero-padding.
- **Daily-averaged data, 24h step, 1979–2013 train / 2014 val** — matches the AIMIP split
  exactly ✅.

## Likely bugs / config gaps (actionable, ordered by importance)

### 1. Loss weights are applied to the *fields*, so coefficients enter the MSE squared

`VariableWeightingLoss.forward` (`fme/core/loss.py:459`) computes `loss(w·x, w·y)` → effective
squared-error weight is **w²**. ArchesWeather/GraphCast multiply the *squared error* linearly by
the coefficient. The config's pressure-proportional weights (0.16…3.24) therefore become (p/p̄)²,
and surface 0.1 becomes 0.01 — a much steeper tilt than Arches. To match, square roots are needed
(e.g. `air_temperature_7: 1.8`, surface `0.316`). Also note ACE's `total()` is a plain mean over
channels, whereas Arches sums a surface-group mean (weight 1.3/7.3) and a level-group mean (6/7.3)
— the aggregate surface:level balance differs too. Worth deliberately recomputing the weight table
against ACE's semantics.

### 2. `max_grad_norm: 1.0` is only in the two hybrid configs

ArchesWeather sets `gradient_clip_val=1` globally in its Trainer (`main_hydra.py:261`). The
`swin`, `swin-lr`, `tc-swin(-lr)`, `nc-swin(-lr)`, and sfno configs all lack it, even though
support now exists in `fme/core/optimization.py` (swin_diff item 11 is otherwise done). Likely
relevant to 3e-4 stability.

### 3. Time conditioning uses (month, hour); AIMIP uses (day-of-year, hour) — and ablates this

The "Month cond (vs. DOY)" ablation in AIMIP appendix B shows month conditioning is distinctly
worse (notably for SST/SIC and surface-temperature climatology RMSE).
`_TimeConditionedContextWrappedModule` embeds month+hour from `forward_time`
(`fme/ace/stepper/single_module.py:1151-1157`). Worse: with daily-averaged data the hour is
constant, so the hour embedder is a no-op and the total seasonal signal is 12 discrete month
values. Switching the embedder to day-of-year is a small change with paper-documented benefit.
(Conditioning on the *input* step's time matches both papers ✅.)

### 4. SST/SIC output supervision is effectively inverted vs. AIMIP

ArchesWeatherAIMIP adds daily SST and SIC as supervised outputs (masked to ocean), and the "Don't
supervise SST/SIC" ablation shows removing this hurts — they believe supervision teaches the model
to attend to the monthly forcings. In ACE, the ocean override runs *before* the loss
(`step_with_adjustments`, `fme/core/step/single_module.py:612-614`), so `surface_temperature`
gets zero gradient over ocean (only land skin temperature is supervised), and `sea_ice_fraction`
isn't an output at all. To mirror AIMIP, add daily SST/SIC as diagnostic outputs supervised over
ocean. Related protocol note (already in swin_diff's footer): AIMIP forcings are *monthly means*;
the ACE configs prescribe daily SST/sea-ice, which leaks higher-frequency ocean information.

### 5. `axis_attn: true` does not implement ArchesWeather's axis_attn — it adds something Arches doesn't have

In Arches, `axis_attn` is the Cross-Level Attention from the paper: 1D attention across the **8
vertical tokens** per spatial location (`AxialPositionalEmbedding(shape=(8,))`,
`num_dimensions=1`, `archesweather_layers.py:408-418`). ACE's `AxialAttentionMixer` instead
attends globally along the **H and W spatial axes**. In the 2D layout (levels stacked in
channels), cross-level interaction is already fully dense in every linear layer — the honest 2D
analog of CLA is the plain `ColumnMixer`, which was replaced. So the current configs add global
spatial attention with `randn`-initialized (std 1) positional embeddings injected into the
residual stream — a significant architectural and compute addition with no ArchesWeather
counterpart. swin_diff item 9's "zero-dependency port" description is misleading on this point.
Consider `axis_attn: false` for the matching experiments, or treat it as a deliberate ablation arm.

### 6. Shifted windows: Arches rolls with *no* attention mask; ACE masks

In `EarthSpecificBlock` the mask is permanently `None` (the `get_shift_window_mask` call is
commented out, `archesweather_layers.py:404-406`) — odd blocks roll and attention wraps freely,
making the network cylindrical in longitude (the paper §4.6 explicitly says they connect
−180°/+180° at every layer). ACE builds the standard Swin shift mask (`swin_layers.py:458`),
which forbids wrap attention. The earth padding partially restores seam locality for the
conv/window paths, but shifted-window cross-seam attention is still blocked. A config/flag to
skip the mask (at least in longitude) would match Arches more closely.

## Architectural divergences baked into the ACE port

Deliberate Swin-V2 modernizations, but they are *not* ArchesWeather:

- **Block topology**: Arches is pre-norm DiT-style — `norm1(x)` → AdaLN scale/shift → attention →
  gated residual; MLP input is modulated `norm2(x)`. ACE is res-post-norm (Swin V2) — attention
  runs on the *unnormalized* stream, norm is applied *after* attention/MLP, and AdaLN modulates
  the **outputs** rather than the inputs (`swin_layers.py:582-587`). Functionally trainable, but
  the conditioning mechanism is genuinely different: output modulation can only rescale what the
  block computed, never change what attention/MLP attend to.
- **Attention math**: Arches uses standard scaled dot-product plus a learned **earth position
  bias lookup table** (per relative position × latitude-band of window, trunc_normal init). ACE
  uses Swin-V2 cosine attention with per-head τ plus a **continuous CPB MLP** over log-spaced
  offsets with cos-lat scaling. The cos-lat CPB is a nice earth-aware analog, but it's a
  different inductive bias; if matching matters, this is the biggest remaining un-ablated
  difference in the attention itself.
- **Embedding/decoding**: Arches patch-embeds with stride 2 (3 at 1°) and decodes with
  ICNR-initialized pixel-shuffle deconv; ACE runs full-resolution with k3s1 convs. At 4° the
  token counts are comparable and k3s1 convs have no checkerboard issue, so this is reasonable —
  just structurally different.
- **Model size**: the config (embed 256, dm 4 → 64 blocks, decoder stage at dim 512) is roughly
  ~200M params vs ArchesWeather-M/AIMIP's 85M (192, dm 2, 32 blocks). Presumably intentional, but
  worth knowing when comparing training budgets and overfitting behavior (the Arches papers
  emphasize that deterministic-model overfitting is their main failure mode).

## Training-protocol differences not in swin_diff.md

- **No recent-past fine-tuning (RPFT)**: the ArchesWeather protocol (used unchanged for AIMIP) is
  Phase 1 (250k steps) + Phase 2 fine-tune on 2007–2018 for 50k steps, and Table 2 shows RPFT is
  worth roughly as much as CLA itself (Z500 50.6→49.3). The current runs are single-phase on
  1979–2013. This would be a cheap follow-on run from a finished checkpoint. (The
  `generate_finetuning_configs.py` in the experiment dir is for var-masking, not RPFT.)
- **EMA**: the configs use `ema: decay 0.999` + `validate_using_ema`; ArchesWeather's
  deterministic training has no EMA at all. Possibly beneficial, but it's an uncontrolled
  deviation.
- **Precision**: Arches trains in `16-mixed`; ACE configs have AMP off (fp32). Conservative, just
  slower — fine.
- **Mx4 ensembling detail**: the AIMIP paper averages the 4 seeds **at every autoregressive step**
  (not averaging four separate rollouts), and reports this reduces bias accumulation on climate
  timescales. If/when swin_diff item 7 is done, implement step-wise averaging.
- **Stochastic variants**: the `nc`/`hybrid` configs (CLN noise + CRPS 0.9/energy 0.1,
  n_ensemble 2) are an ACE-native stochastic recipe; ArchesWeatherGen is flow matching trained on
  *residuals of a frozen Mx4* with OOD fine-tuning on a held-out year and noise scaling 1.05.
  These aren't comparable methods — fine if intentional, but a "matching" claim shouldn't extend
  to those runs.

## Suggested priority

1. Add `max_grad_norm: 1.0` to all configs (one-liner).
2. Decide and fix the loss-weight semantics (sqrt the table, or document that w² is intended).
3. Switch the time embedder to day-of-year + hour (AIMIP-ablated win; hour is currently dead
   weight).
4. Re-evaluate `axis_attn: true` — it's not the ArchesWeather CLA; consider `false` or an
   explicit ablation.
5. Add SST/SIC as supervised ocean-masked outputs (AIMIP-ablated win).
6. Consider removing the shifted-window mask (cylindrical wrap) and/or trying pre-norm +
   input-side AdaLN to close the remaining block-structure gap.
7. Plan an RPFT pass (2007–2013 fine-tune) from the best checkpoints.

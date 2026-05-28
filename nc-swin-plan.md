# Plan: Noise-conditioned Swin Transformer

## Context

The deterministic Swin task is done: `ace-train-config-4deg-AIMIP-swin.yaml` trains the
plain `SwinTransformer` with MSE loss. The reserved `ace-train-config-4deg-AIMIP-nc-swin.yaml`
still points its builder at the deterministic `SwinTransformer` while using `EnsembleLoss`
(CRPS + energy score) with `n_ensemble: 2` — but the Swin has **no noise source**
(`SwinTransformerBuilder` hardcodes `embed_dim_noise=0`, `_ContextWrappedModule` passes
`noise=None`), so the two ensemble members are identical and the probabilistic loss is
degenerate.

This task makes the Swin genuinely stochastic so `nc-swin.yaml` trains like the reference
`ace-train-config-4deg-AIMIP-sfno.yaml` (`NoiseConditionedSFNO` + `EnsembleLoss`,
`n_ensemble: 2`, `optimize_last_step_only: true`). The mechanism mirrors the SFNO directly:
a noise field is generated per forward pass and injected through the **reused
`ConditionalLayerNorm` class** as a per-pixel scale/bias, so resampling the noise yields
distinct ensemble members and a meaningful CRPS spread.

The existing deterministic `swin.yaml` must keep working: its native AdaLN path stays
**byte-for-byte equivalent**.

**Naming:** `"SwinTransformer"` / `"NoiseConditionedSwinTransformer"` are builder *type strings*
(registry keys); the underlying net class is `SwinTransformerNet` (`swin_transformer.py`).
Below, the type strings name the builders/configs and `SwinTransformerNet` names the net class.

## Design decisions (settled with the user)

- **Two builders, two conditioning mechanisms, selected by a block-level flag**
  `conditioning: Literal["adaln", "cln"] = "adaln"`:
  - **Deterministic `SwinTransformer` → `"adaln"`**: keeps the existing native per-stage DiT
    6-param gated AdaLN. The AdaLN math is **byte-for-byte unchanged**; block/layer signatures
    only gain optional params (defaulting `None`) to thread the `Context` for cln, so the
    deterministic call paths and outputs are identical. Default flag value ⇒ no behavior change.
  - **`NoiseConditionedSwinTransformer` → `"cln"`**: reuses the SFNO's `ConditionalLayerNorm`
    class as the block norm, plus a learned gate (below). This is the user's explicit
    preference: reuse CLN for the noise-conditioned model, keep native AdaLN for the
    deterministic one.
- **Noise source: per-pixel i.i.d. Gaussian.** The Swin is not spectral, so the SFNO's
  isotropic/SHT noise does not apply. Reuse the gaussian branch of `NoiseConditionedModel`
  (`inverse_sht=None`).
- **Injection: per-block, per-pixel, zero-init `Conv2d`** — obtained *for free* by reusing
  `ConditionalLayerNorm`, which owns a zero-bias `Conv2d(embed_dim_noise, n_channels, 1)` per
  norm. Because CLN is inherently per-block, this gives the per-block stochastic capacity we
  want using the literal SFNO class (no custom AdaLN-noise math).
- **Learned gate on top of CLN.** CLN does normalize + scale/bias but does **not** gate.
  Add a LayerScale-style learned gate per residual branch: `gate = nn.Parameter(torch.zeros(dim))`,
  applied as `shortcut + gate * drop_path(branch)`. Zero-init ⇒ each CLN block starts as
  (near-)identity and ramps up, matching the AdaLN-Zero behavior the deterministic Swin
  already relies on.
- **Resampling across U-Net resolutions: one full-res field, strided subsample to the
  bottleneck** (Option 1 / subsample). Full-res noise feeds `layer1`/`layer4`, half-res
  (`noise[..., ::2, ::2]`) feeds `layer2`/`layer3`. Faithful to the SFNO's single-field
  design; preserves unit variance and independence; clean reuse of `NoiseConditionedModel`.
- **In CLN mode, all conditioning flows through CLN** (noise + labels + scalar), since CLN
  sums them into one scale/bias. For these configs `embed_dim_scalar=0` and labels are
  inactive, so noise is the only live source. The deterministic AdaLN scalar/label path is
  therefore not reused by the NC model — acceptable and simpler.
- **Layout.** `ConditionalLayerNorm` is channels-first `(B, C, H, W)`; Swin window attention
  is channels-last `(B, H, W, C)`. The CLN-mode block transposes around each norm call.

## Pattern to mirror / reuse

- `NoiseConditionedModel` (`fme/ace/registry/stochastic_sfno.py:49`) is a **generic** wrapper:
  `forward(x, labels=None)` samples `noise` `(B, embed_dim_noise, H, W)` (gaussian when
  `inverse_sht is None`) and calls `conditional_model(x, Context(..., noise=noise))`. Already
  exposes the `(x, labels)` interface the registry/stepper expect and re-samples noise each
  call → ensemble spread. Wrap the Swin net directly.
- `ConditionalLayerNorm` (`fme/core/models/conditional_sfno/layers.py:143`) — **reused as-is**
  as the NC block norm. Consumes the whole `Context`; its zero-init `Conv2d` paths
  (`layers.py:183-193, 281-307`) give the per-pixel additive noise scale/bias. Init already
  matches the zero-init convention we want.

## Implementation

### 1. `fme/core/models/swin_transformer/swin_layers.py` — CLN conditioning mode + learned gate

- `SwinTransformerBlock.__init__`: add `conditioning: Literal["adaln", "cln"] = "adaln"`,
  plus `context_config: ContextConfig | None` and `input_resolution`/`img_shape` for this
  stage.
  - `"adaln"` (default): build exactly as today (native per-stage AdaLN path untouched). The
    existing `conditioned: bool` arg keeps its meaning (set from `embed_dim_scalar/labels > 0`);
    `"cln"` mode ignores `conditioned` and always applies CLN.
  - `"cln"`: set `self.norm1 = ConditionalLayerNorm(dim, img_shape, context_config)` and
    `self.norm2 = ConditionalLayerNorm(dim, img_shape, context_config)`; build
    `self.gate1 = nn.Parameter(torch.zeros(dim))`, `self.gate2 = nn.Parameter(torch.zeros(dim))`.
    Do **not** build the native AdaLN projections.
- `SwinTransformerBlock.forward`: add an optional `context` so adaln stays byte-for-byte while
  cln threads it through:
  `forward(x, cond_params: CondParams | None = None, context: Context | None = None)`.
  - `"adaln"`: identical to today — `BasicLayer` still passes the precomputed `cond_params`
    6-tuple and the existing block math runs unchanged; the new `context` arg defaults to `None`
    and is ignored. (Note: the block does **not** derive its 6-tuple from a `Context` today —
    `BasicLayer` computes it and passes the tuple; that stays the case.)
  - `"cln"`: `BasicLayer` passes `context`; with `x` as `(B, H, W, C)`:
    ```
    shortcut = x
    h = self.norm1(x.permute(0,3,1,2), context).permute(0,2,3,1)   # CLN, channels-first
    h = window_attention(h)
    h = h + column_mixer(h)                                          # ColumnMixer folded in
    x = shortcut + self.gate1 * drop_path(h)                         # learned LayerScale gate
    y = self.norm2(x.permute(0,3,1,2), context).permute(0,2,3,1)
    x = x + self.gate2 * drop_path(mlp(y))
    ```
- `BasicLayer.__init__`: thread `conditioning`, `context_config`, `img_shape` to every block.
- `BasicLayer.forward`: add an optional `context` param
  (`forward(x, cond_scalar=None, cond_labels=None, context=None)`). `"adaln"` computes the
  per-stage `cond_params` from `cond_scalar`/`cond_labels` exactly as today and calls
  `blk(x, cond_params=cond_params)`; `"cln"` calls `blk(x, context=context)` so each block's
  CLN reads `context.noise`.

### 2. `fme/core/models/swin_transformer/swin_transformer.py` — wire noise through the U-Net

- `__init__`: read `self.conditioning` / `self.embed_dim_noise` from `context_config`; build
  the four `BasicLayer`s in the selected mode, passing each stage's resolution
  (`(Hp, Wp)` for layers 1/4, `(Hp//2, Wp//2)` for layers 2/3) so CLN sizing matches.
- `forward(x, context)`: when CLN mode and `embed_dim_noise > 0`, require `context.noise`
  `(B, embed_dim_noise, H, W)` at unpadded input res; pad it with the **same**
  `F.pad(..., (0, pad_w, 0, pad_h))` used for `x`; build `noise_half = noise_pad[..., ::2, ::2]`.
  Pass a stage-appropriate `Context` to each `BasicLayer` — e.g.
  `ctx_full = dataclasses.replace(context, noise=noise_pad)` and
  `ctx_half = dataclasses.replace(context, noise=noise_half)` — full-res to layers 1/4,
  half-res to layers 2/3. Both `Hp, Wp` are multiples of `2·window_size`, so `::2` is exact.

### 3. `fme/ace/registry/swin_transformer.py` — `NoiseConditionedSwinTransformer` builder

- Add `@ModuleSelector.register("NoiseConditionedSwinTransformer")` dataclass
  `NoiseConditionedSwinTransformerBuilder(ModuleConfig)` mirroring `NoiseConditionedSFNOBuilder`:
  Swin architecture fields (`embed_dim`, `depth_multiplier`, `num_heads`, `window_size`,
  `mlp_ratio`, `drop_path_rate`, `use_skip`, `mlp_layer`) plus `noise_embed_dim: int = 256`
  and `label_embed_dim: int = 0` (learned-embedding semantics; `0` ⇒ one-hot labels pass through).
  Unlike `SwinTransformerBuilder`, the NC builder does **not** expose
  `embed_dim_scalar`/`embed_dim_labels`: it hardwires `embed_dim_scalar=0` and derives
  `embed_dim_labels` from `label_embed_dim` via `effective_label_dim` (SFNO-style).
- `build`: compute `effective_label_dim` like the SFNO builder; build `ContextConfig(
  embed_dim_scalar=0, embed_dim_labels=effective_label_dim, embed_dim_noise=noise_embed_dim,
  embed_dim_pos=0)`; build `SwinTransformerNet(..., conditioning="cln", context_config=...)`;
  return `NoiseConditionedModel(net, img_shape=dataset_info.img_shape,
  embed_dim_noise=noise_embed_dim, embed_dim_pos=0, n_labels=n_labels,
  label_embed_dim=label_embed_dim, inverse_sht=None)` (gaussian). Import
  `NoiseConditionedModel` from `fme.ace.registry.stochastic_sfno`.
- Leave the deterministic `SwinTransformerBuilder` untouched (`conditioning="adaln"`,
  `embed_dim_noise=0`, wrapped in `_ContextWrappedModule`).

### 4. `configs/.../ace-train-config-4deg-AIMIP-nc-swin.yaml`

- Change `stepper.step.config.builder.type: SwinTransformer` → `NoiseConditionedSwinTransformer`
  and add `noise_embed_dim: 32` under `builder.config`. Keep all architecture fields and the
  entire `stepper_training` block (`EnsembleLoss`, `n_ensemble: 2`,
  `optimize_last_step_only: true`, weights, kwargs). `swin.yaml` is **not** modified.

### 5. Tests

- `fme/core/models/swin_transformer/test_swin_transformer.py`: forward+backward in CLN mode
  with `embed_dim_noise>0` (build `Context(noise=randn(n, embed_dim_noise, *img_shape))`,
  assert output shape and that all params — including `gate1/gate2` and CLN noise convs —
  receive gradients); a padded-`img_shape` case (exercises pad + subsample); two forwards
  with different noise differ; and an `"adaln"`-mode regression case that is identical to the
  pre-change deterministic block.
- `fme/ace/registry/test_swin_transformer.py` (create) — two complementary tests:
  (a) a mock-style test in the spirit of `fme/ace/registry/test_stochastic_sfno.py` asserting
  `NoiseConditionedSwinTransformerBuilder.build(...)` returns a `NoiseConditionedModel` and that
  noise reaches the wrapped net's `Context`; and (b) a **real-build** test (small `DatasetInfo`,
  no mock) asserting two forwards on identical input differ — a mocked inner model cannot show
  output divergence. Also assert the deterministic `SwinTransformer` builder path is unchanged.

## Out of scope / unchanged

- Deterministic `SwinTransformer` builder, its native AdaLN, and `swin.yaml`.
- `weather_2024`/`weather_1994` inference entries in `nc-swin.yaml` (`weight: 0.0`) — leave
  as-is for structural parity.
- No isotropic/SHT noise; no independent-per-resolution noise (documented escalation only).

## Reused utilities

- `NoiseConditionedModel` — `fme/ace/registry/stochastic_sfno.py:49`
- `ConditionalLayerNorm`, `Context`, `ContextConfig` — `fme/core/models/conditional_sfno/layers.py`
- `_ContextWrappedModule` (deterministic path) — `fme/ace/registry/swin_transformer.py:12`
  (an identical copy also lives at `fme/ace/registry/local_net.py:24`; the Swin builder uses the former)
- `"NoiseConditionedSwinTransformer"` must be added to `CONDITIONAL_BUILDERS`
  (`fme/core/registry/module.py:61`); `"SwinTransformer"` is already present.

## Verification

1. **Unit tests** (CPU):
   ```bash
   cd /Users/alexeyy/Git/ace
   conda run -n fme python -m pytest \
     fme/core/models/swin_transformer/test_swin_transformer.py \
     fme/ace/registry/test_swin_transformer.py -q
   ```
2. **Deterministic config still builds** (`conditioning="adaln"`, `embed_dim_noise=0`
   regression): parse `swin.yaml` via `dacite` strict + `prepare_config` into `TrainConfig`;
   assert builder type `SwinTransformer`, `loss.type == 'MSE'`, `n_ensemble == 1`.
3. **Noise-conditioned config builds and is stochastic**: parse `nc-swin.yaml`; assert
   builder type `NoiseConditionedSwinTransformer`, `loss.type == 'EnsembleLoss'`,
   `n_ensemble == 2`; build the module from a small `DatasetInfo` and confirm two forward
   passes on identical input differ.
4. **Lint/type**: `pre-commit run --files <changed files>` (ruff, ruff-format, mypy).

# Plan: Channel-conditioned variable masking for SFNO (FiLM via dedicated Context field)

> Deliverable file: this plan lives in `claude_condition_mask.md` at repo root.
> Branch off **`feature/var_masking_simple`**: `feature/sfno-channel-mask-conditioning`.
> This stacks on the input-dropout PR; merge order is var_masking_simple → this.

## Context

Two ways exist to tell the network which input variables are masked/absent:
1. **Real missing variables** via `data_mask` (genuinely-absent fields, also used
   for loss masking).
2. **Synthetic input dropout** via `SingleModuleStepConfig.input_dropout`
   (`VariableMaskingConfig`), added on `feature/var_masking_simple`: a per-sample
   presence mask sampled once per rollout in `TrainStepper._accumulate_loss`
   (`make_input_dropout_mask`), threaded through `StepArgs.input_dropout_mask`, and
   applied in `network_call` — it zeros dropped channels (`_apply_input_mask`) and is
   AND-combined with `data_mask` inside `_build_channel_mask_dict`.

Today the only way to *signal presence to the network* is
`include_channel_mask_inputs` (in `fme/core/step/single_module.py`): per-variable
presence indicators are broadcast to full `(batch, H, W)` maps and **concatenated**
onto the network input, doubling `n_in_channels`. That signal is inherently
per-sample, per-variable (whole variables are dropped) — a length-`n_channels`
**vector**, not a spatial field. Feeding a constant-over-space vector through the
spatial/spectral input path is wasteful (especially for SFNO, where a constant map
only excites the DC spectral bin) and forces the first layer to learn binary-channel
weights, which controls weight variance poorly.

Goal: add an alternative that injects the presence vector as **FiLM conditioning**
(scale/bias modulation) through the existing `ConditionalLayerNorm`, reusing a new
dedicated `Context` slot (kept separate from categorical `labels`). Scope: **SFNO
only** (`NoiseConditionedSFNO` builder). Keep the concat path intact for A/B.

**Critical:** the presence vector fed to FiLM must reflect **both** signals — real
`data_mask` AND synthetic `input_dropout_mask` — combined exactly as the concat path
combines them (`_build_channel_mask_dict`: present iff really present *and* not
synthetically dropped). The whole point of this revision is that FiLM conditioning
works on synthetically dropped variables, not just real missing ones.

All primitives this needs already exist on `feature/var_masking_simple`: `data_mask`
plumbing, `input_dropout` / `StepArgs.input_dropout_mask`, `_build_channel_mask_dict`
(with AND-combine), and the SFNO `Context` / `ConditionalLayerNorm` FiLM machinery.
No new input-dropout infrastructure is needed — this plan only adds the FiLM path
and routes the combined presence vector into it.

## Reference branches

This plan = **merge of two existing branches**, both off main. Use both as
reference; the work is to combine them, not build from scratch.

- **`feature/var_masking_conditional`** — already implements the FiLM conditional
  masking described below (the `Context.channel_mask` slot, `embed_dim_mask` in
  `ConditionalLayerNorm`, `condition_on_channel_mask` builder flag, `wants_channel_mask`
  on `Module`, `_build_channel_presence_vector` in `network_call`). It touches exactly
  this plan's files (`conditional_sfno/layers.py`, `registry/stochastic_sfno.py`,
  `registry/module.py`, `step/single_module.py`) and ships the `.pt` regression
  baselines and parallel test. **Limitation: conditions on `data_mask` only — it has
  no `input_dropout`.** This is the gap this revision closes.
- **`feature/var_masking_simple`** — implements the `input_dropout` infra
  (`VariableMaskingConfig`, `StepArgs.input_dropout_mask`, `make_input_dropout_mask`,
  stepper/multi_call/coupled plumbing) and the AND-combine in `_build_channel_mask_dict`,
  but signals presence via the **concat** path, not FiLM.

The deliverable: take `feature/var_masking_conditional`'s FiLM path and feed its
presence vector the combined (`data_mask` & `input_dropout_mask`) signal, pulling the
dropout machinery from `feature/var_masking_simple`. Concretely, the only net-new
work over `feature/var_masking_conditional` is extending
`_build_channel_presence_vector` to take + AND `input_dropout_mask`, and wiring
`args.input_dropout_mask` into the `network_call` presence build.

## Design (single source of truth = SFNO builder flag)

- New flag `condition_on_channel_mask: bool` on the **SFNO builder config**
  (conditioning is an architecture choice). When true, the builder sizes a new
  CLN conditioning input of dim `n_in_channels` and the wrapped `Module` advertises
  `wants_channel_mask=True`.
- This flag is independent of `ModuleSelector.conditional`, but note the current
  SFNO builder allocates label FiLM from `dataset_info.all_labels` alone. Therefore
  this plan's mask-only path is guaranteed to work for datasets with
  `dataset_info.all_labels == set()` and `conditional: false`. For labeled datasets,
  either set `conditional: true` and provide labels, or first redesign label FiLM
  allocation to depend on `ModuleSelector.conditional` (which would require a
  broader builder/wrapper interface change).
- The `Module` wrapper exposes `wants_channel_mask`; the Step reads it to decide
  whether to build + pass the per-sample presence vector. No duplicate Step flag,
  no `isinstance`.
- Mask conditioning is **independent of `labels` inside `Context`** and works with
  `n_labels == 0`. This is the whole reason for a dedicated `Context` field rather
  than the labels slot. Do not use `labels` as a carrier for mask-only conditioning.
- **Conditioning source = combined presence.** The presence vector is the AND of
  real `data_mask` and synthetic `input_dropout_mask`, in `in_packer.names` order —
  the same combination `_build_channel_mask_dict` uses for the concat indicators. So
  the FiLM and concat paths condition on an identical presence signal; only the
  delivery mechanism (scale/bias modulation vs. concatenated maps) differs. This
  makes the conditioning fire on synthetically dropped variables during training even
  when `data_mask` marks everything present.
- The FiLM flag (`condition_on_channel_mask`, architecture-side) and `input_dropout`
  (Step-side, training augmentation) are **independent**. Either can be on alone:
  FiLM with no dropout conditions only on real `data_mask`; dropout with concat
  (`include_channel_mask_inputs`) is the existing var_masking_simple behavior; FiLM
  with dropout is the new combination this plan targets.
- Opt-in configs that rely on incomplete inputs must still set
  `ModuleSelector.allow_missing_variables: true`; the architecture flag consumes
  `data_mask`/`input_dropout_mask` but does not change data-loading requirements.
- `ContextConfig.embed_dim_mask` must default to `0`, and
  `Context.channel_mask` must default to `None`, so existing `ContextConfig(...)`
  and `Context(...)` call sites on main remain valid.
- Mask FiLM linears are **zero-initialized** (mirroring `W_scale_labels`/`W_bias_labels`,
  layers.py:227-233): at init the mask adds nothing, so training starts from the
  unconditioned model and learns to use the mask gradually — directly the
  weight-variance control the review comment asked for.

## Changes

### 1. `fme/core/models/conditional_sfno/layers.py` — FiLM mechanism
- `ContextConfig`: add `embed_dim_mask: int = 0` as the final field.
- `Context`: add field `channel_mask: torch.Tensor | None = None`; update `__post_init__`
  (validate `ndim == 2` when not None, like `labels` at line 74-75), `asdict`,
  `from_dict`. Use `data.get("channel_mask")` in `from_dict` so older serialized
  contexts and SFNO checkpoint test inputs remain readable.
- `ConditionalLayerNorm.__init__`: store `self.embed_dim_mask =
  context_config.embed_dim_mask`; after the labels block (line 182), add
  `W_scale_mask` / `W_bias_mask = nn.Linear(embed_dim_mask, n_channels)` when
  `embed_dim_mask > 0`, else `None`.
- `reset_parameters` (after line 233): zero-init weight **and** bias of both mask
  linears (mirror labels).
- `forward` (labels block at lines 296-303): mirror the labels block —
  `scale = scale + W_scale_mask(context.channel_mask).unsqueeze(-1).unsqueeze(-1)`
  and same for bias; add a None-guard (raise if mask linears present but
  `context.channel_mask is None`, like line 265-268).

### 2. `fme/ace/registry/stochastic_sfno.py` — thread mask into Context
- `NoiseConditionedModel.forward`: add `channel_mask: torch.Tensor | None = None`;
  put it into the `Context(...)` built at line 166 (raw, no embedding).
- This wrapper is shared by SFNO, LocalNet, and NoiseConditionedSwinTransformer;
  the new arg must be optional and defaulted so those existing users do not change.
- `NoiseConditionedSFNOBuilder`: add config field
  `condition_on_channel_mask: bool = False`; in `build` set
  `embed_dim_mask = n_in_channels if self.condition_on_channel_mask else 0` on the
  `ContextConfig` (line 373-378). `n_in_channels` here is the un-doubled count.

### 3. `fme/core/registry/module.py` — forward mask through the wrapper
- `Module.__init__`: add `wants_channel_mask: bool = False`; store it; add a
  `wants_channel_mask` property; preserve it in `wrap_module` and `to`.
- `Module.__call__`: add `channel_mask: torch.Tensor | None = None`. Restructure so
  `channel_mask` is forwarded via kwargs **independent of the labels gating** (labels
  may be None when `n_labels == 0`), e.g. build a `kwargs` dict: add `labels` only on
  the conditional-label path, add `channel_mask` whenever not None, then
  `self._module(input, **kwargs)`.
- Keep the existing error when non-None `labels` are passed to a model without a
  `LabelEncoding`; mask-only conditioning should work because `labels` is None.
- `ModuleSelector.build`: set `wants_channel_mask` on the returned `Module` from the
  builder via `getattr(self.module_config, "condition_on_channel_mask", False)`
  (defaulted getattr keeps non-SFNO builders untouched — no protocol change).
- **Wiring through `wrap_module`:** `network_call` invokes the module via
  `self.module.wrap_module(wrapper)(input_tensor, labels=...)`. `wrap_module` returns a
  fresh `Module(wrapper(self._module), label_encoding)`, so it must propagate
  `wants_channel_mask` to that returned `Module` (as noted above), and the returned
  `Module.__call__` must forward `channel_mask` via the kwargs path. `wrapper` is either
  identity (`lambda x: x`, the default) or `torch.utils.checkpoint.checkpoint` (grad
  checkpointing, `fme/ace/stepper/single_module.py:1147`); both already forward `labels`
  as a kwarg, so `channel_mask` rides the identical `**kwargs` path with no extra work.

### 4. `fme/core/step/single_module.py` — build + pass the presence vector
- In `SingleModuleStepConfig.__post_init__`: fail fast if both
  `include_channel_mask_inputs` and the SFNO builder flag are set (two redundant
  presence-signal mechanisms). Validate here rather than only at runtime, matching
  the repo's config convention. The check can use `builder_wants_channel_mask =
  getattr(self.builder.module_config, "condition_on_channel_mask", False)`.
  **`input_dropout` is orthogonal and explicitly allowed with the FiLM flag** —
  that combination (synthetic dropout signalled via FiLM) is the target use case.
  Do not reject it.
- In `SingleModuleStep.__init__`: do **not** double `n_in_channels` for mask
  conditioning (only concat doubles). After build, set
  `self._wants_channel_mask = self.module.wants_channel_mask` as the runtime source
  of truth.
- Factor the per-sample presence computation out of `_build_channel_mask_dict` into
  `_build_channel_presence_vector(in_names, data_mask, input_dropout_mask, batch,
  device, dtype) -> Tensor[batch, n_channels]` (float, 1.0 present / 0.0 masked).
  Replicate `_build_channel_mask_dict`'s exact combine logic per name:
  - real presence: `data_mask[source]` via `extra_channel_source_field` source
    lookup (GMR extras inherit their source field), else all-ones;
  - synthetic presence: `input_dropout_mask[name]` keyed by the **packed channel
    name directly** (GMR extras independently maskable), else all-ones;
  - combined = `real & synthetic`, cast to `dtype`.
  **Column order is exactly `in_names` order**; build the tensor by stacking one
  `[batch]` column per name in iteration order. Have `_build_channel_mask_dict` call
  it, then expand each column with `presence[:, i].view(batch, 1, 1).expand(batch,
  *spatial)` keyed by `enumerate(in_names)` — single source of truth for the
  present/absent combination, shared by concat and FiLM paths. Correct because 1 name
  = 1 channel here (`n_in_channels = len(packed_in_names)`, single_module.py:289).
- **Channel-order alignment (critical):** in `network_call` call the presence helper
  with `self.in_packer.names` (the same names list the concat path packs with),
  `args.data_mask`, `args.input_dropout_mask`, and source
  `batch`/`device`/`dtype` from `input_tensor`. Because the network input is
  `self.in_packer.pack(...)` in that same name order, column `i` of the presence
  vector aligns with input channel `i`, and `embed_dim_mask == n_in_channels ==
  len(in_packer.names)`. So the FiLM `Linear(n_in_channels, n_channels)` sees the mask
  in the same channel order the network sees the data.
- **Dropout still zeros the input** independent of FiLM: the existing
  `_apply_input_mask(input_norm, args.input_dropout_mask)` call (added on
  var_masking_simple) stays. FiLM is an *additional* presence signal, not a
  replacement — exactly as concat indicators sit alongside the zeroed input. Build
  the presence vector from the un-zeroed mask args, not from the (already-zeroed)
  packed input.
- In `network_call`: when `self._wants_channel_mask`, build the presence vector
  (combined `data_mask` & `input_dropout_mask`) and pass `channel_mask=presence` to
  the module call (alongside `labels=args.labels`).

## Tests

- **CLN unit** (`fme/core/models/conditional_sfno/`): with `embed_dim_mask > 0`,
  compare two different non-None masks and assert the output is unchanged at init
  (zero-init identity); assert a missing `channel_mask` raises; assert output changes
  once mask linear weights are set / after a grad step; assert `Context` ndim guard
  raises on bad `channel_mask`.
- **Noise-conditioned SFNO wrapper** (`fme/ace/registry/test_stochastic_sfno.py`):
  call `NoiseConditionedModel.forward(..., channel_mask=mask)` and assert the wrapped
  module receives `Context.channel_mask` as the exact `[batch, n_channels]` tensor.
- **Step integration** (`fme/core/step/test_step.py`): SFNO step with
  `condition_on_channel_mask=True`: (a) network input channel count is **not** doubled;
  (b) the module receives the expected `[batch, n_channels]` `channel_mask`
  vector. If asserting output differences, manually set mask FiLM weights nonzero
  first, because the zero-init contract means masked/unmasked outputs should match at
  initialization; (c) constructing with both
  `include_channel_mask_inputs` and the builder flag raises; (d) unit-test
  `_build_channel_presence_vector` (incl. GMR source inheritance **and**
  `input_dropout_mask` AND-combine: a channel marked present by `data_mask` but
  dropped by `input_dropout_mask` yields a 0.0 column, and vice versa); (e)
  **dropout→FiLM end-to-end**: SFNO step with `condition_on_channel_mask=True` **and**
  `input_dropout` configured, training mode — pass a `StepArgs.input_dropout_mask`
  that drops a known channel, assert the module's `channel_mask` column for that
  channel is 0.0 while `data_mask`-present columns stay 1.0; assert `input_dropout`
  alongside the FiLM flag does **not** raise (orthogonality from `__post_init__`).
- **Module registry** (`fme/core/registry/test_module_registry.py`): `Module`
  forwards `channel_mask` with `n_labels == 0` and `labels=None`; `Module.to()` and
  `Module.wrap_module()` preserve `wants_channel_mask`.
- **Module config snapshot** (`fme/core/registry/testdata/`): adding
  `condition_on_channel_mask: false` to `NoiseConditionedSFNO_module_config.yaml`
  is required because `test_latest_module_backwards_compatibility` checks for new
  builder config keys. No state-dict update is required while the flag is false,
  because no mask FiLM parameters are allocated.
- **Spatial parallel**: presence vector is per-sample `[batch, n_channels]` (no
  spatial dim) → FiLM scale/bias are per-channel global and identical across tiles.
  `data_mask` is already per-sample and tile-consistent. The synthetic
  `input_dropout_mask` is **already `broadcast_spatial`'d upstream** in
  `make_input_dropout_mask` (var_masking_simple) — `torch.rand` advances independently
  per co-rank, so the spatial root's mask is broadcast there. The presence vector
  built in `network_call` consumes that already-consistent mask, so no additional
  `broadcast_spatial` is needed in the FiLM path. Add/extend a
  `@pytest.mark.parallel` SFNO test mirroring existing conditional-SFNO parallel
  coverage; cover the FiLM+dropout combo to prove `channel_mask` columns agree across
  co-ranks. Parallel SFNO tests live in
  `fme/core/distributed/parallel_tests/test_step.py`, **not** `fme/core/step/`.

## Backward compatibility
New `ContextConfig.embed_dim_mask` (default 0), builder field (default False), and
`Module.wants_channel_mask` (default False) mean existing checkpoints load unchanged
and allocate no new params when the feature is off. Existing training/inference YAMLs
do not need edits unless they opt in, but the latest-module registry test fixture YAML
does need the explicit default key described above.

## Verification
```
# targeted tests (conda env: fme; use python -m pytest)
python -m pytest fme/core/models/conditional_sfno -k "mask or conditional_layer_norm"
python -m pytest fme/core/step/test_step.py -k "channel_mask"
python -m pytest fme/core/registry/test_module_registry.py
python -m pytest fme/ace/registry/test_stochastic_sfno.py
# input-dropout infra inherited from var_masking_simple (sanity)
python -m pytest fme/core/test_var_masking.py
# spatial-parallel smoke
FME_FORCE_CPU=1 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 FME_DISTRIBUTED_W=1 \
  torchrun --nproc-per-node 2 -m pytest -m parallel fme/core/distributed/parallel_tests
pre-commit run --all-files
```

## Notes / open items for reviewer
- Flag lives on the SFNO builder (architecture-side), not the Step, unlike
  `include_channel_mask_inputs`. Alternative: keep a Step flag and pass a
  `channel_mask_dim` kwarg into `ModuleSelector.build` — rejected here because it
  changes the `build` signature across all builders for an SFNO-only feature.
- Mask values fed raw as 0/1 (matching one-hot labels). Centering (mask - 0.5) is a
  possible later tweak; not needed given zero-init.
- **Branch base / merge**: stacks on `feature/var_masking_simple` for the
  `input_dropout` infra and the AND-combine in `_build_channel_mask_dict`, and reuses
  `feature/var_masking_conditional`'s FiLM implementation wholesale (see *Reference
  branches* above). Most of this plan is already on `feature/var_masking_conditional`;
  the net-new work is the dropout combine. Merge `feature/var_masking_simple` first.
  Alternative — branching off main and re-porting both — rejected: duplicates code
  under review and conflicts on the same files (`single_module.py`, `args.py`,
  `step.py`, `layers.py`, `module.py`).
- This plan adds **no** new dropout infrastructure and little net-new FiLM code. It
  feeds the existing FiLM path the combined (`data_mask` & `input_dropout_mask`)
  presence vector via the refactored `_build_channel_presence_vector`. Coupled
  training already rejects `input_dropout` (var_masking_simple
  `CoupledTrainStepper.__init__`), so FiLM+dropout is uncoupled-training-only; FiLM
  alone (real `data_mask`, as on `feature/var_masking_conditional` today) still works
  in any path.
- **Regression baselines** (`sm_channel_mask_conditioned_{input,output}.pt`) exist on
  `feature/var_masking_conditional` for the data_mask-only FiLM path. Keep them valid
  (the dropout combine is off when `input_dropout=None`); add a separate baseline or
  in-test assertion for the FiLM+dropout combo rather than mutating the existing ones.

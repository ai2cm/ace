# VarMasking wandb projects — comparison

All six wandb projects (`VarMasking` … `VarMasking6`) are iterations of **one research
thread**: training a single ACE/SFNO emulator that is robust to *missing input
variables* by randomly dropping ("masking") input channels during training. The goal
is a model you can run at inference with an arbitrary subset of inputs available, and
(from V4 on) condition explicitly on the CO2 forcing.

All experiments live under `configs/experiments/2026-05-27-var-masking/`, all target the
**4° AIMIP** setup, all use a **noise-conditioned SFNO** (`NoiseConditionedSFNO`,
`noise_embed_dim: 32`, `noise_type: isotropic`) except where noted, and all keep the same
`global_mean_removal: {kind: shared, append_as_input: true}`. The projects differ mainly
in (a) the masking API/strategy, (b) how (or whether) the model is *told* which channels
are masked, (c) CO2 masking, and (d) coupled-stepper support.

> Mapping is in `wandb_to_git.yaml`. Branch lineage is `exp/alexey` (V1/V2),
> `exp/alexey2` (V3), `exp/alexey4..6` (V4/V5/V6). Chronological order is
> **V4 → V5 → V6**: V4 uses the older sampler API (`min_vars`/`max_vars`) with
> concatenated mask channels; V5 switches to the newer `max_masked_vars` sampler and
> FiLM conditioning; V6 is the targeted follow-up. (The anchor SHA is just an anchor —
> projects accumulated more same-branch runs after it; this matters most for V3.)
>
> **Per-experiment train configs are not committed** — they are emitted at submit time by
> `generate_masking_configs.py`. The committed YAML is only base templates + eval suites,
> so the *generator* is the authoritative source for what each run actually set.

---

## TL;DR evolution table

| | wandb | git | date | masking API | how model learns the mask | CO2 mask | coupled | distinguishing thing |
|---|---|---|---|---|---|---|---|---|
| **V1** | VarMasking | `ceaa7417` | Jun 03 | name/prefix-rate dict **+** uniform-count, two separate classes | appended indicator channels (`include_channel_mask_inputs`) | — | no | first cut: per-variable Bernoulli **and** uniform-count schemes |
| **V2** | VarMasking2 | `8525ae2a` | Jun 05 | hybrid `VariableMaskingConfig` (explicit groups **+** nested `uniform`), ensemble-aware, `can_mask`/`validate` | appended indicator channels | CO2 **present/absent** dim (`-co2-mask`/`-co2-nomask`), no rate override | no | broad expanded sweep (~46 runs) + 1 finetune; finer rates, GMR ref-field protection |
| **V3** | VarMasking3 | `8e52947b` | Jun 09 | **rewrite** to channel-index tensors; `kind`-tagged union `Uniform`\|`PerVariable` | appended indicator channels | — | **yes** (coupled stepper) | messy debug bucket (~63 runs, many failed/crashed): step1 vs step2, `-iid`, `-nonoise` ablations |
| **V4** | VarMasking4 | `54c489b3` | Jun 25* | union + `co2_rate`; `min_vars`/`max_vars` | appended indicator channels (`include_channel_mask_inputs`) | **yes — `co2_rate` override** | yes | first clean OFAT production matrix; CO2 always input; warmup/cooldown + eval suites |
| **V5** | VarMasking5 | `45db7372` | Jun 19* | union + `co2_rate`; `max_masked_vars` | **FiLM conditioning inside SFNO** (`condition_on_channel_mask`) | **yes — `co2_rate`** | yes | same matrix as V4 but FiLM conditioning + newer sampler API |
| **V6** | VarMasking6 | `17d4a8c5` | Jun 26 | union + `co2_rate` + **`same_mask_per_batch`** | **FiLM** (`condition_on_channel_mask`, V5-style) | yes | yes | targeted `same_mask_per_batch` ablation on 2 arms (~4 base runs); `data_mask` vs `input_dropout_mask` split |

---

## The two things that change the most

### 1. The masking API (`fme/core/var_masking.py`)

- **V1** — two unrelated dataclasses:
  - `VariableMaskingConfig`: a `{name → rate}` dict with `default_rate`. Keys ending in
    `_` are *prefix* keys (e.g. `air_temperature_` masks all 8 levels together with one
    draw); exact names get independent Bernoulli draws.
  - `UniformVariableMaskingConfig`: draw an integer `n ∈ [min_vars, max_vars]` per sample,
    mask `n` randomly-chosen eligible variables (`ignore_vars` excluded).
  - Output is a **dict `{name → [batch] bool}`** (True = present).
- **V2** — merges them: one `VariableMaskingConfig` with explicit per-group rates **and** a
  nested `UniformMaskingConfig`, plus `can_mask()` / `validate_variable_names()`. Adds
  **ensemble awareness** (`_repeat_ensemble_mask`, `_get_base_batch_size`) so all ensemble
  members of a base sample share a mask. Still name-keyed dicts.
- **V3** — **architectural rewrite**. Drops name-dicts for **per-sample channel-index
  tensors** `[batch, n_channels]`. Two small `kind`-discriminated dataclasses unified by a
  union type `VariableMaskingConfig = UniformMaskingConfig | PerVariableMaskingConfig`:
  - `UniformMaskingConfig` (`kind: uniform`, `min_vars`, `max_vars`): draws a count, ranks
    channels by random noise, masks the lowest-ranked.
  - `PerVariableMaskingConfig` (`kind: per_variable`, `rate`): independent Bernoulli per
    channel.
  - Each exposes `sample_mask(n_channels, batch_size, device, n_ensemble)`.
  - **This is the API all later projects (V4/V5/V6) build on.**
- **V4 / V5 / V6** — same union, plus a `co2_rate: float | None` field on both configs
  (see below). V5 renames the uniform bound to `max_masked_vars` in configs; V4 keeps
  `min_vars`/`max_vars`. V6 adds `same_mask_per_batch` (one shared mask for the whole batch
  instead of per-sample).

### 2. How the network is told which channels are masked

Masked channels are always **zeroed in normalized space** (≈ climatological mean in
physical space) via `_apply_input_mask`. What differs is whether the model is *informed*
of the mask:

- **V1–V4**: `include_channel_mask_inputs: true` — append a binary presence indicator
  channel per variable to the input (roughly **doubling input channels**).
  `_build_channel_mask_dict` builds those indicators.
- **V5**: moves conditioning **into the SFNO** as FiLM. `condition_on_channel_mask: true`
  on the builder; `fme/core/models/conditional_sfno/layers.py` gains `embed_dim_mask` and
  `W_scale_mask` / `W_bias_mask` linear layers that add to the block scale/bias from the
  channel-presence vector. The mask flows through `registry/module.py`
  (`wants_channel_mask` / `channel_mask` kwarg) rather than being concatenated. The two
  paths (`include_channel_mask_inputs` vs `condition_on_channel_mask`) are mutually
  exclusive.
- **V6**: keeps V5's FiLM path (`condition_on_channel_mask: true` in the generator). The
  new variable is `same_mask_per_batch` (one mask shared across the whole batch vs an
  independent mask per sample), tested on two arms.

---

## CO2 masking (new in V4/V5/V6)

`fme/core/var_masking.py` gains `CO2_NAME = "global_mean_co2"` and helpers
`_validate_co2_rate`, `_apply_co2_override` (V6 also `_validate_co2_in_names` +
`validate_names` methods on the configs). When `co2_rate` is set, the CO2 column is
**resampled independently** at that keep-rate, overriding the base uniform/per-variable
mask, so CO2 availability can be controlled separately from everything else. Requires
`global_mean_co2` to be an input channel (a `long_46year_constant_co2` dataset / forcing
input is added in these branches). The `co2_rate` *override* machinery is **V4+**; note
that CO2 already existed as a coarser **present/absent input dimension** in V2's generator
(`-co2-mask` / `-co2-nomask`).

## `data_mask` vs `input_dropout_mask` (clarified in V5/V6)

Early projects reused one `data_mask` channel for both loss-weighting and synthetic input
masking. V5/V6 split them in `fme/core/step/args.py`:

- `data_mask` — marks *genuinely absent* variables (preprocessing + loss masking).
- `input_dropout_mask` — *synthetic, training-only* corruption that only perturbs the
  network input, never the loss.

V5/V6 also add `make_input_dropout_mask()` / `has_input_dropout()` to the step interface
(`step.py`, `multi_call.py`).

## Coupled-stepper support

- V1, V2: **no** coupled support.
- V3 onward: `fme/coupled/stepper.py` integrates masking. V6 explicitly **raises** if
  `input_dropout` is set on a component step during coupled training (the dropout hook is
  only called by uncoupled training), preventing a silent no-op.

> The `evaluator.py` changes in V4/V6 are a `log_label` logging refactor, **not** masking —
> masking is training-only in every project.

---

## Per-project config matrix (what each experiment actually swept)

> Suffix `nc-sfno` = noise-conditioned (`noise_embed_dim: 32`); `sfno` = deterministic
> (`noise_embed_dim: 0`).

**V1 `ceaa7417`** — two schemes by filename:
- Bernoulli: `input_dropout: {default_rate: 0.2}` (or `0.0`); `noforcing` variant pins
  `land_fraction / ocean_fraction / sea_ice_fraction / DSWRFtoa / HGTsfc` to rate `0.0`.
- Uniform: `min_vars: min`, `max_vars: 15` (or `max`), same forcing vars in `ignore_vars`.
- `include_channel_mask_inputs: true`. No CO2.

**V2 `8525ae2a`** — broad expanded sweep (~46 runs): finer Bernoulli rates
(`0.05–0.40`), extra small uniform arms (`mask3`), explicit no-mask controls crossing
GMR on/off and CO2 present/absent, GMR reference field (`surface_temperature`) protected
from dropout. The CO2 dimension here is presence-based (`-co2-mask` / `-co2-nomask`), not
yet a `co2_rate` override. The committed `...-v2-finetune.yaml` (deterministic
`noise_embed_dim: 0`, no `input_dropout` block) is just the one finetune job, not the
whole project.

**V3 `8e52947b`** — a **messy debug/prototyping bucket** (~63 runs, many failed/crashed),
not a clean sweep, and it accumulated runs past the anchor SHA. Core grid:
`...nc-sfno-mask{0,5,40}-gmron-steps{1,2}`, uniform `kind: uniform, min_vars: 1,
max_vars: {5,40}`; `mask0` drops `input_dropout`; `steps1/steps2` = final-stage
`n_forward_steps`; `include_channel_mask_inputs: true`. Later same-branch ablations added
`mask10`, `-iid` (independent synthetic mask per ensemble member vs shared), and
`-nonoise` (`noise_embed_dim: 0`, isolate masking from noise conditioning).

**V4 `54c489b3`** — first clean OFAT production matrix. CO2 always an input; dose-response
`mask{0,5,10,20,30}` + Bernoulli `mask{0.11,0.22,0.33}`; `co2_rate` override sweep
`{0.4,0.8,0.9}`. Uses **appended indicator channels** (`include_channel_mask_inputs: true`)
and the older uniform shape (`min_vars`/`max_vars`). 150-epoch warmup/constant/cooldown,
pre-cooldown ckpt at epoch 142; adds `-cooldown` / `-bestinfcooldown` finetune variants
(drop `input_dropout`) and `-besttrain`/`-bestinf`/`-lastepoch` eval suites.

**V5 `45db7372`** — same scientific matrix and run/eval structure as V4, but **FiLM
conditioning** (`condition_on_channel_mask: true` in the builder, mutually exclusive with
`include_channel_mask_inputs`) and the newer uniform sampler (`max_masked_vars`). V4-vs-V5
isolates concat-channels vs presence-vector conditioning + the sampler API cleanup.

**V6 `17d4a8c5`** — narrow targeted follow-up: only `mask10-co2-0.8` and
`mask20-co2-default`, each with a `-same` twin (`same_mask_per_batch: true`) plus cooldown
variants (~4 base + 8 cooldown + 24 inference runs). Uniform `max_masked_vars: {10,20}`,
`co2_rate`, and keeps V5's `condition_on_channel_mask: true` (FiLM). Anchor commit itself
is just an evaluator-logging fix (`log_label`).

### Sweep dimensions, summarized
- **Mask strategy**: Bernoulli/per-variable rate vs uniform-count.
- **Mask amount**: rate `0.0–0.22` / count `5–40` (or `max`).
- **Forcing protection**: exclude forcings from masking (`ignore_vars` / per-var rate `0`).
- **CO2**: presence (`-co2-mask`/`-co2-nomask`, V2) → `co2_rate` override `0.4/0.8/0.9` (V4+).
- **Mask conditioning**: appended channels (V1–V4) → FiLM (V5, V6).
- **Noise conditioning**: on (`nc-sfno`) vs off (`sfno` / V3 `-nonoise`).
- **Forward steps** (V3): `steps1` vs `steps2`.
- **Ensemble mask sharing**: shared (default) vs `-iid` per-member (V3).
- **Batch-shared mask** (V6): `same_mask_per_batch`.
- **Cooldown/finetune** (V4–V6): finetune variants turn dropout off; `-besttrain`/
  `-bestinf`/`-lastepoch` eval suites.

---

## How to read the progression

1. **V1–V2**: prototype with name-keyed masks; figure out ensemble handling. V2 is the
   broad expanded sweep (rates, uniform counts, CO2 presence, GMR controls).
2. **V3**: rewrite to a clean channel-tensor union API + coupled support — the durable
   foundation. In practice a messy debug bucket (step-count, `-iid`, `-nonoise` ablations).
3. **V4**: first clean OFAT matrix — CO2 `co2_rate` override, warmup/cooldown, eval suites,
   still **appended-channel** conditioning.
4. **V5**: same matrix, swapped to **FiLM conditioning** + newer sampler API. The clean
   concat-vs-FiLM head-to-head.
5. **V6**: targeted follow-up on two V5 arms — `same_mask_per_batch` ablation; also where
   `input_dropout_mask` is cleanly separated from `data_mask` in the step args.

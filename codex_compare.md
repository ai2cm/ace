# W&B VarMasking Project Comparison

This compares the W&B projects listed in `wandb_to_git.yaml`:
`ai2cm/VarMasking`, `VarMasking2`, `VarMasking3`, `VarMasking4`,
`VarMasking5`, and `VarMasking6`.

Evidence used:

- `wandb_to_git.yaml` for the project to Beaker example to anchor-SHA mapping.
- Local git history and experiment scripts under
  `configs/experiments/2026-05-27-var-masking/`.
- Read-only W&B API inventory for `ai2cm/*` projects, queried on 2026-06-29.

Important caveat: the SHA in `wandb_to_git.yaml` is a useful anchor, but some
projects continued to accumulate same-branch runs after that SHA. This matters
most for `VarMasking3`, which is more of an iterative/debug project than a clean
single sweep.

Another chronology caveat: the W&B project numbers and anchor commit dates do
not perfectly order the experiment ideas. `VarMasking4` base W&B runs started
before `VarMasking5`, but the anchor commit in `wandb_to_git.yaml` for
`VarMasking4` is later than the `VarMasking5` anchor because it includes later
resubmission/eval work.

Config caveat: many per-run training YAMLs were generated at submit time by
`generate_masking_configs.py`. When the committed base templates and generated
run configs differ, the generator state and W&B run names are the better record
of what was actually submitted.

## Quick Map

| W&B project | Anchor SHA | Branch containing anchor | Example Beaker experiment | W&B inventory | Short description |
| --- | --- | --- | --- | --- | --- |
| `VarMasking` | `ceaa7417` | `exp/alexey` | https://beaker.org/ex/01KT76SYM2TSN754TS5SQ81DM6 | 18 training, all finished | First broad variable-masking submission. Bernoulli and uniform masking over `sfno` and `nc-sfno`, with channel-mask inputs concatenated. |
| `VarMasking2` | `8525ae2a` | `exp/alexey` | https://beaker.org/ex/01KTCZ0NV85ZZ3VV629JNJTAM8 | 46 training, all finished | Expanded/fixed V1-style sweep. Finer Bernoulli rates, additional uniform mask sizes, CO2 variants, protected GMR reference field, plus one fine-tune run. |
| `VarMasking3` | `8e52947b` | `exp/alexey2` | https://beaker.org/ex/01KTQERMR0P4FAV4PD4RX053Q7 | 63 training: 27 finished, 23 failed, 13 crashed | Iterative/simple-mask experiment bucket. Constant-LR runs testing mask count, 1-step vs 2-step training, later IID mask and no-noise ablations. Messy, not a clean final sweep. |
| `VarMasking4` | `54c489b` | `exp/alexey4` | https://beaker.org/ex/01KW09225ZQFSZD84AFH0RP82N | 27 base training + 54 cooldown training + 81 inference, all finished | First clean OFAT-style production matrix with CO2 always present, explicit CO2 masking rates, warmup/cooldown, eval suites. Uses concatenated channel-mask inputs. |
| `VarMasking5` | `45db7372` | `exp/alexey5` | https://beaker.org/ex/01KVGH4X8WSJX5QP208WGQ1MDM | 27 base training + 54 cooldown training + 81 inference, all finished | Same run matrix as V4, but with the newer mask sampler API, explicit CO2 override support, and FiLM-style channel-mask conditioning instead of concatenating mask channels. |
| `VarMasking6` | `17d4a8c5` | `exp/alexey6` | https://beaker.org/ex/01KW276NPW9SR1S0JTB335410Y | 4 base training + 8 cooldown training + 24 inference; 12 inference failed, then reran finished | Targeted V5-style follow-up testing per-sample masks vs `same_mask_per_batch` on two selected arms. The submit generator keeps V5-style FiLM mask conditioning, although the committed YAML snapshots at the anchor do not show that flag. Includes evaluator logging fix and rerun. |

## Shared Vocabulary

- `sfno` and `nc-sfno`: both are 4-degree AIMIP SFNO-family configs. The
  `nc-sfno` configs use the noise-conditioned SFNO builder.
- `gmron` / `gmroff`: global mean removal on/off. Later projects settle on
  `gmron`.
- `rpoff`: residual prediction disabled. All meaningful later runs use
  residual prediction off.
- `include_channel_mask_inputs`: older way to tell the network which variables
  are present, by concatenating per-channel presence maps to the spatial input.
- `condition_on_channel_mask`: newer way to tell the network which variables
  are present, by passing a per-sample presence vector into the SFNO builder for
  FiLM-style conditioning.
- `co2-default`: `global_mean_co2` is present and masked like the base mask
  sampler says.
- `co2-0.4`, `co2-0.8`, `co2-0.9`: override `global_mean_co2` with an
  independent masking rate of 0.4, 0.8, or 0.9.
- `data_mask`: real missing-data mask used by preprocessing/loss masking.
- `input_dropout_mask`: synthetic training-only mask used to corrupt model
  inputs without changing the loss target. This is split from `data_mask` in
  the V5/V6 implementation.
- `besttrain`, `bestinf`, `lastepoch`: evaluator jobs against
  `best_ckpt.tar`, `best_inference_ckpt.tar`, and final `ckpt.tar`.
- `cooldown`: re-runs the final short cooldown from the pre-cooldown checkpoint
  with masking disabled.
- `bestinfcooldown`: same cooldown idea, but initialized from
  `best_inference_ckpt.tar`.

## Implementation Evolution

This is the code-level progression behind the project-level differences:

1. `VarMasking`: early name-based masking configs, with separate Bernoulli-style
   per-variable rates and uniform-count masking paths.
2. `VarMasking2`: merged/cleaned early masking schema, added ensemble-aware mask
   sharing and validation, and protected the GMR reference field from dropout.
3. `VarMasking3`: moved toward the durable channel-index tensor masking API used
   by later projects, but W&B remained a mixed debugging bucket.
4. `VarMasking4`: added the clean production matrix and explicit
   `global_mean_co2` override rates while still using concatenated mask inputs.
5. `VarMasking5`: changed the model-conditioning path to FiLM-style
   channel-mask conditioning, cleaned up the uniform sampler API, and split
   synthetic `input_dropout_mask` from real `data_mask`.
6. `VarMasking6`: added `same_mask_per_batch` while keeping the V5-style
   FiLM mask-conditioning path in the submit generator.

## Project Notes

### `VarMasking`

This is the original broad sweep. The generator at the anchor commit produces
Bernoulli and "Jeremy/uniform" variable-masking configs for both `sfno` and
`nc-sfno`.

Design:

- Models: `sfno`, `nc-sfno`.
- Bernoulli mask rates: `0.00`, `0.20`, `0.40`.
- Bernoulli masking scope: `all` or `noforcing`.
- Uniform mask counts: `maskall`, plus `mask17` for all variables and `mask15`
  for no-forcing variants.
- GMR: on only in the submitted generator state (`gmron`).
- Residual prediction: off only (`rpoff`).
- Channel-mask signal: `include_channel_mask_inputs: true`, so mask indicators
  are concatenated to the network input.

W&B has 18 runs, all finished, all in group
`ace2-var-masking-2026-05-27`. The run names are a clean 2-model cross of:

- Bernoulli: `mask0.00-all`, `mask0.20-all`, `mask0.20-noforcing`,
  `mask0.40-all`, `mask0.40-noforcing`.
- Uniform: `maskall-all`, `maskall-noforcing`, `mask17-all`,
  `mask15-noforcing`.

How to remember it: V1 asked "does masking variables at all help, and does the
answer differ for SFNO vs noise-conditioned SFNO?" It was broad and early. It
did not yet have the later CO2-specific machinery or the cleaner OFAT design.

### `VarMasking2`

This is a corrected and expanded version of the first broad sweep. The anchor
commit is titled "Resubmitting fine tune", but the project contains the larger
V2 training sweep plus one fine-tune job.

Key changes from `VarMasking`:

- W&B project and run names get the `VarMasking2` / `-v2` suffix.
- Bernoulli rates expand to `0.05`, `0.10`, `0.20`, `0.30`, `0.40`.
- Uniform configs add smaller `mask3` arms.
- No-masking controls are more explicit, crossing GMR on/off and CO2 present vs
  absent.
- `input_dropout` config shape is updated to the nested schema:
  `per_variable` for Bernoulli and `uniform` for uniform masking.
- The shared global-mean-removal reference field (`surface_temperature`) is
  protected from dropout when GMR is enabled.
- CO2 variants are introduced, with `global_mean_co2` added to both `in_names`
  and `next_step_forcing_names` for CO2-enabled configs.
- A `pre_cooldown_checkpoint_epoch` is set, though V2 is not yet the full
  cooldown/eval pipeline used in V4/V5.

W&B has 46 runs, all finished:

- 45 regular training runs in group `ace2-var-masking-2026-06-04`.
- 1 fine-tune run in group `ace2-var-masking-finetune-2026-06-05`:
  `ace2-var-mask-sfno-mask0.20-noforcing-gmron-rpoff-bernoulli-v2-finetune`.
  That fine-tune config loads a checkpoint and no longer contains an
  `input_dropout` block, so the fine-tune itself is a no-dropout continuation
  despite the inherited mask-themed filename.

How to remember it: V2 is "V1, but fixed and much broader." It is the place to
look for the early full-grid comparison of rates, uniform counts, no-forcing
variants, CO2 presence/masking, and GMR controls.

### `VarMasking3`

This project is the least cleanly interpretable because it was used while the
masking implementation and config strategy were being simplified. The anchor
commit says "Updating configs", but W&B shows several waves of attempts on the
same project.

The cleaner intended design at the anchor SHA:

- Models: `sfno`, `nc-sfno`.
- Masking: uniform `mask40`, uniform `mask5`, and `mask0`.
- Steps: `n_forward_steps=1` and `n_forward_steps=2`.
- GMR: on.
- CO2: off in the anchor generator (`CO2_VALS = [False]`), although some
  earlier/later project runs include CO2-named attempts.
- Training: 150 epochs, constant LR, no warmup/cooldown.
- Residual prediction: off.
- Channel-mask signal: still `include_channel_mask_inputs: true`.

Later same-branch changes added more focused debugging dimensions:

- `mask10` replaces/augments the larger `mask40` arm in later generator state.
- `-iid` variants for `nc-sfno`, meaning independent synthetic masks per
  ensemble member instead of shared masks. These only apply when masking is
  active.
- `-nonoise` variants for `nc-sfno`, disabling noise conditioning
  (`noise_embed_dim=0`) to separate the masking effect from the noise embedding.
- Gradient clipping / seed changes were tried for IID runs.

W&B has 63 runs:

- 27 finished.
- 23 failed.
- 13 crashed.
- All are training jobs in group `ace2-var-masking-2026-06-04`.

The run names show the project's mixed nature:

- Early failed/crashed Bernoulli and CO2/GMR attempts, such as
  `mask0.00-gmroff-bernoulli-co2-v3` and `mask0.40-gmron-bernoulli-co2-v3`.
- Clean finished step-count runs, such as
  `nc-sfno-mask0-gmron-steps1-v3`,
  `nc-sfno-mask5-gmron-steps2-v3`, and `sfno-mask10-gmron-steps2-v3`.
- Later IID/no-noise runs, many with duplicate failed attempts before successful
  reruns.

How to remember it: V3 is a debugging/prototyping bucket for the simpler
variable-mask training path, step-count effects, IID mask sampling, and noise
conditioning. I would not use it as a clean headline comparison without first
filtering to the specific run family you care about.

### `VarMasking4`

This is the first clean production-style matrix. It moves away from the broad
V1/V2 grid and uses an OFAT-ish design centered on a noise-conditioned SFNO
baseline.

Actual matrix, based on code and W&B inventory:

- 27 base training runs:
  - 26 `nc-sfno` runs.
  - 1 `sfno` baseline: `sfno-mask0-uniform-co2-default`.
- For `nc-sfno`, the dose-response controls at `co2-default` are:
  - Uniform: `mask0`, `mask5`, `mask10`, `mask20`, `mask30`.
  - Bernoulli family checks: `mask0.11`, `mask0.22`, `mask0.33`.
- CO2 sweeps are run at six anchor masks:
  - `mask10`, `mask0.11`, `mask20`, `mask0.22`, `mask30`, `mask0.33`.
  - CO2 rates: `co2-0.4`, `co2-0.8`, `co2-0.9`.
  - Plus `co2-default` for the dose-response controls.

Training details:

- `global_mean_co2` is always an input channel and next-step forcing.
- GMR is always on.
- Residual prediction is always off.
- Training schedule is 150 epochs: 8 epoch linear warmup, constant LR, then
  8 epoch polynomial cooldown.
- A pre-cooldown checkpoint is saved at epoch 142.
- Channel-mask signal uses the old concat path:
  `include_channel_mask_inputs: true`.
- Uniform masking uses the older shape:
  `kind: uniform`, `min_vars: 0`, `max_vars: n`.

Follow-up jobs:

- 54 cooldown training jobs: 2 per base run.
  - `-cooldown` loads `training_checkpoints/pre_cooldown_ckpt.tar`.
  - `-bestinfcooldown` loads `training_checkpoints/best_inference_ckpt.tar`.
  - Both disable `input_dropout` during the cooldown.
- 81 inference jobs: 3 per base run.
  - `-besttrain`, `-bestinf`, `-lastepoch`.
  - Each evaluator job runs the generated eval suite entries under one W&B run.

W&B has 162 runs, all finished:

- 27 base training in group `ace2-var-masking-2026-06-15`.
- 54 cooldown training in group `ace2-var-masking-cooldown-2026-06-17`.
- 81 inference in group `ace2-var-masking-eval-2026-06-17`.

How to remember it: V4 is the clean OFAT matrix with old-style channel-mask
inputs. Use it when you want the "concat mask channels" baseline.

### `VarMasking5`

This repeats the V4 matrix, but with a materially different implementation of
how synthetic missingness is sampled and exposed to the model.

Same as `VarMasking4`:

- Same 27 base run names, except `-v5` instead of `-v4`.
- Same 26 `nc-sfno` arms plus 1 `sfno` baseline.
- Same CO2 sweep and mask-dose design.
- Same 150 epoch warmup/constant/cooldown schedule.
- Same 54 cooldown jobs and 81 inference jobs.
- Same W&B group structure.

Different from `VarMasking4`:

- Uniform config changes from `min_vars`/`max_vars` to `max_masked_vars`.
  The sampler draws the number of masked variables uniformly from
  `[0, max_masked_vars]`.
- CO2 masking is supported directly by the core masking configs via
  `co2_rate`, rather than being handled by generator-specific exclusions.
- Channel-mask information is no longer concatenated onto the spatial input.
  Instead, configs set:
  `builder.config.condition_on_channel_mask: true`.
- The step code builds a per-sample channel-presence vector and passes it to
  the module, so the SFNO builder can condition through its channel-mask path.
- `include_channel_mask_inputs` and `condition_on_channel_mask` are made
  mutually exclusive.
- Synthetic training dropout is split from real missing-data masking:
  `StepArgs` gains `input_dropout_mask`, while `data_mask` remains the signal
  for genuinely absent variables and loss masking. The training path samples an
  input-dropout mask through `make_input_dropout_mask()` / `has_input_dropout()`.
- Coupled training is guarded against silent no-ops: if a coupled component step
  has `input_dropout` configured, construction raises because the coupled
  training route does not call the input-dropout hook.

W&B has 162 runs, all finished:

- 27 base training in group `ace2-var-masking-2026-06-15`.
- 54 cooldown training in group `ace2-var-masking-cooldown-2026-06-17`.
- 81 inference in group `ace2-var-masking-eval-2026-06-17`.

How to remember it: V5 is the same scientific matrix as V4, but with the newer
conditioning path and cleaner CO2-aware mask sampler. The V4 vs V5 comparison
mostly isolates "concat mask channels" vs "FiLM/presence-vector conditioning",
plus the sampler API cleanup.

### `VarMasking6`

This is not another full matrix. It is a targeted follow-up to V5.

The base training arms are only four runs:

- `nc-sfno-mask10-uniform-co2-0.8-v6`
- `nc-sfno-mask10-uniform-co2-0.8-same-v6`
- `nc-sfno-mask20-uniform-co2-default-v6`
- `nc-sfno-mask20-uniform-co2-default-same-v6`

The key new variable is `same_mask_per_batch`:

- Without `-same`: `same_mask_per_batch: false`, so each sample gets an
  independently sampled synthetic variable mask.
- With `-same`: `same_mask_per_batch: true`, so one synthetic variable mask is
  sampled and repeated across the whole batch.

Other details:

- Uses the V5-style core masking API: `max_masked_vars`, optional `co2_rate`,
  the separated `input_dropout_mask` training path, and the coupled-training
  guard against unsupported input dropout.
- The V6 submit generator keeps the V5-style mask-conditioning path by setting
  `builder.config.condition_on_channel_mask: true`. One source caveat: the
  committed V6 train YAML snapshots at the anchor do not contain that flag, so
  the generator is the better record for submitted runs here.
- Uses the same warmup/constant/cooldown schedule as V4/V5.
- Uses the same cooldown/eval machinery, but only for the four selected base
  arms.
- The anchor commit `17d4a8c5` fixes evaluator-suite logging behavior by
  allowing `run_evaluator_from_config` to reuse a shared W&B logger and log
  suite entries under their own labels.

W&B has 36 runs:

- 4 base training, all finished.
- 8 cooldown training, all finished.
- 24 inference runs:
  - 12 failed first attempts.
  - 12 finished reruns with the same logical names.

How to remember it: V6 asks "for the promising V5-style arms, does sharing the
synthetic mask across the batch change behavior compared with per-sample masks?"
It is the targeted `same_mask_per_batch` ablation, not a broad model/mask/CO2
sweep.

## Timeline Of Design Changes

1. `VarMasking`: first broad sweep with concat mask inputs and early dropout
   config shape.
2. `VarMasking2`: broad sweep corrected/expanded; adds finer rates, CO2
   variants, and GMR reference-field protection.
3. `VarMasking3`: simplified constant-LR experiments and debugging of mask
   sampling, step count, IID ensemble masks, and noise conditioning.
4. `VarMasking4`: first clean OFAT matrix with CO2 always present, warmup and
   cooldown, generated cooldown jobs, and generated eval suites; still uses
   concat mask inputs.
5. `VarMasking5`: repeats V4's matrix with FiLM/presence-vector channel-mask
   conditioning and explicit CO2-aware mask sampler support.
6. `VarMasking6`: targeted V5 follow-up comparing per-sample masks with
   `same_mask_per_batch` on two selected arms.

## Practical Interpretation

- Use `VarMasking` / `VarMasking2` for historical broad-sweep context.
- Treat `VarMasking3` as exploratory/debugging data unless you filter to a
  specific sub-family of finished runs.
- Use `VarMasking4` vs `VarMasking5` when you want the cleanest comparison of
  channel-mask signaling mechanisms.
- Use `VarMasking5` as the cleaner full matrix if you trust the newer FiLM
  conditioning path.
- Use `VarMasking6` only for the targeted question of whether a batch-shared
  synthetic mask changes the selected V5 arms.

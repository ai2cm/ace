# ACE Downscaling Distillation (FastGen Spike)

Distills the ACE 100 km→25 km teacher into a 2–4-step student using
[FastGen](https://github.com/anuragajay/fastgen).  Three distillation
methods are provided for comparison:

| Config | Method | Target steps | Mode-coverage |
|---|---|---|---|
| `configs/sft_spike.py` | SFT (sanity check) | 18 (teacher) | n/a |
| `configs/fdistill_kl_spike.py` | f-distill, forward KL | 4 → 2 | mass-covering |
| `configs/scm_spike.py` | sCM | 2 | strong coverage |
| `configs/dmd2_baseline_spike.py` | DMD2, reverse KL | 4 | mode-seeking (baseline) |

Primary go/no-go metric:
`generation/histogram/prediction_frac_of_target/99.9999th-percentile`
within ~5 % of the teacher.

---

## Prerequisites

### 1. FastGen submodule

```bash
git submodule update --init FastGen
```

FastGen is not installed as a package; add it to `PYTHONPATH` at runtime:

```bash
export PYTHONPATH="$(git rev-parse --show-toplevel)/FastGen:$PYTHONPATH"
```

### 2. Conda environment

```bash
conda activate fme   # or fme-py312
pip install hydra-core boto3 torchvision   # FastGen extras not in fme env
```

### 3. Required environment variables

| Variable | Required | Description |
|---|---|---|
| `ACE_TEACHER_CKPT` | **yes** | Path to the pre-trained CONUS 100 km→25 km teacher `.ckpt` |
| `WANDB_API_KEY` | **yes** | W&B API key (or store in `./credentials/wandb_api.txt`) |
| `ACE_C_OUT` | no | Number of output channels (default `7`) |
| `ACE_H_FINE` | no | Fine-grid height in pixels (default `512`) |
| `ACE_W_FINE` | no | Fine-grid width in pixels (default `512`) |
| `ACE_STUDENT_STEPS` | no | Student inference steps (default per config; see table above) |
| `FASTGEN_OUTPUT_ROOT` | no | Root directory for checkpoints and the config snapshot (default: `./FASTGEN_OUTPUT`) |

```bash
export ACE_TEACHER_CKPT=/path/to/conus-100km-25km-teacher.ckpt
export WANDB_API_KEY=<your-key>
export FASTGEN_OUTPUT_ROOT=/path/to/experiment/output
```

### 4. W&B run configuration

Each run is tracked in W&B under three identifiers that **must be set per run**
via config overrides — the spike configs only set `group`; `project` and `name`
default to `"fastgen"` and `"debug"` if not overridden.

```bash
torchrun --nproc_per_node 8 \
    -m fme.downscaling.distillation.fastgen_train \
    --config fme/downscaling/distillation/configs/sft_spike.py \
    --teacher-checkpoint "$ACE_TEACHER_CKPT" \
    --data-yaml configs/baselines/downscaling/train-conus-100km-to-25km-augusta.yaml \
    - log_config.project=ace-distillation \
      log_config.group=ace_downscaling_sft_spike \
      log_config.name=sft-run1
```

| Field | Spike config default | Recommendation |
|---|---|---|
| `log_config.project` | `"fastgen"` | Set to `"ace-distillation"` (or your team project) |
| `log_config.group` | set per config (e.g. `"ace_downscaling_sft_spike"`) | Keep as-is to group runs by method |
| `log_config.name` | `"debug"` | Set to a unique run name, e.g. `"sft-run1"` or `"fdistill-lr2e-6"` |
| `log_config.wandb_mode` | `"online"` | Set to `"disabled"` to suppress W&B entirely |

Checkpoints and the `config.yaml` snapshot are written to:
`$FASTGEN_OUTPUT_ROOT/<project>/<group>/<name>/`

To disable W&B without changing the config file:

```bash
- log_config.wandb_mode=disabled
```

**What gets logged:**

| Metric | Cadence | W&B key pattern |
|---|---|---|
| Per-loss averages (averaged across DDP ranks) | every 500 iters | `train/<loss_name>` |
| Learning rate | every 500 iters | `optimizer/lr_<optimizer_name>` |
| Student sample images vs. real targets | every 500 iters | `train_media/student/generation`, `train_media/data/real` |
| Validation losses | every 1 000 iters | `val0/<loss_name>` |

---

## Step 1 — Dryrun smoke test

Verify environment wiring before committing GPU hours.  Loads the teacher,
builds one batch, runs one forward pass, and exits.

```bash
python -m fme.downscaling.distillation.fastgen_train \
    --config fme/downscaling/distillation/configs/sft_spike.py \
    --teacher-checkpoint "$ACE_TEACHER_CKPT" \
    --data-yaml configs/baselines/downscaling/train-conus-100km-to-25km-augusta.yaml \
    --dryrun
```

Expected output (shapes will match your teacher's output dimensions):

```
Dryrun OK — real: (2, 7, 512, 512), condition: (2, 7, 512, 512), x0_hat: (2, 7, 512, 512)
```

---

## Step 2 — SFT sanity run

SFT has no step reduction — it just fine-tunes the teacher on its own
trajectories.  A falling DSM loss and visually plausible 18-step samples
confirm the adapter wiring is correct before running the distillation methods.

**Single GPU:**

```bash
python -m fme.downscaling.distillation.fastgen_train \
    --config fme/downscaling/distillation/configs/sft_spike.py \
    --teacher-checkpoint "$ACE_TEACHER_CKPT" \
    --data-yaml configs/baselines/downscaling/train-conus-100km-to-25km-augusta.yaml
```

**Multi-GPU (DDP):**

```bash
torchrun --nproc_per_node 8 \
    -m fme.downscaling.distillation.fastgen_train \
    --config fme/downscaling/distillation/configs/sft_spike.py \
    --teacher-checkpoint "$ACE_TEACHER_CKPT" \
    --data-yaml configs/baselines/downscaling/train-conus-100km-to-25km-augusta.yaml
```

Gate: DSM loss decreases steadily over the first ~5 k iterations.

---

## Step 3 — Distillation variants

Run after the SFT sanity check passes.  The three methods can be submitted
in parallel; they are independent experiments.

**f-distill (forward KL — primary candidate):**

```bash
torchrun --nproc_per_node 8 \
    -m fme.downscaling.distillation.fastgen_train \
    --config fme/downscaling/distillation/configs/fdistill_kl_spike.py \
    --teacher-checkpoint "$ACE_TEACHER_CKPT" \
    --data-yaml configs/baselines/downscaling/train-conus-100km-to-25km-augusta.yaml
```

**sCM (score Consistency Model):**

```bash
torchrun --nproc_per_node 8 \
    -m fme.downscaling.distillation.fastgen_train \
    --config fme/downscaling/distillation/configs/scm_spike.py \
    --teacher-checkpoint "$ACE_TEACHER_CKPT" \
    --data-yaml configs/baselines/downscaling/train-conus-100km-to-25km-augusta.yaml
```

**DMD2 (reverse KL — expected to fail tail metric; serves as baseline):**

```bash
torchrun --nproc_per_node 8 \
    -m fme.downscaling.distillation.fastgen_train \
    --config fme/downscaling/distillation/configs/dmd2_baseline_spike.py \
    --teacher-checkpoint "$ACE_TEACHER_CKPT" \
    --data-yaml configs/baselines/downscaling/train-conus-100km-to-25km-augusta.yaml
```

### Config overrides

Append `- key=value` pairs after all other arguments to override FastGen
config fields at the command line:

```bash
python -m fme.downscaling.distillation.fastgen_train \
    --config fme/downscaling/distillation/configs/fdistill_kl_spike.py \
    --teacher-checkpoint "$ACE_TEACHER_CKPT" \
    --data-yaml configs/baselines/downscaling/train-conus-100km-to-25km-augusta.yaml \
    - model.net_optimizer.lr=1e-6 trainer.max_iter=100000
```

The leading `-` before the key=value pairs is required by FastGen's
`override_config_with_opts`.

---

## Step 4 — Export student checkpoint

After distillation training completes, export the student weights into ACE
checkpoint format so the existing `EvaluatorConfig` / `PatchPredictor`
pipeline can load them without modification:

```python
from fme.downscaling.models import CheckpointModelConfig
from fme.downscaling.distillation.student_checkpoint import save_student_checkpoint

teacher = CheckpointModelConfig(checkpoint_path="/path/to/teacher.ckpt").build()

# fastgen_model is the FastGen model object returned by Trainer.run().
# Access the bare denoiser via .net._ace_module (unwrapped from DDP).
save_student_checkpoint(
    student_module=fastgen_model.net._ace_module,
    teacher=teacher,
    path="student.ckpt",
    num_sampling_steps=4,   # overrides the step count saved in the config
)
```

The resulting `student.ckpt` is a drop-in replacement for the teacher
checkpoint: same grid, variable names, and normalisation stats; only the
denoiser weights differ.

---

## Step 5 — Evaluate with tail metrics

Load the student checkpoint with the existing evaluation pipeline and check
the 99.9999th-percentile tail metric:

```bash
python -m fme.downscaling.inference.inference \
    --checkpoint student.ckpt \
    --config configs/baselines/downscaling/eval-conus-25km-to-3km-augusta.yaml
```

Key metric to check in the evaluation output:
`generation/histogram/prediction_frac_of_target/99.9999th-percentile`

Target: within ~5 % of the teacher's value.  The f-distill (forward KL) and
sCM methods are expected to meet this bar; DMD2 (reverse KL) is not.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fastgen'`**
→ Set `PYTHONPATH` to include the `FastGen/` directory (see Prerequisites).

**`ModuleNotFoundError: No module named 'hydra'` / `boto3` / `torchvision`**
→ Install the missing FastGen extras: `pip install hydra-core boto3 torchvision`.

**`ValueError: Provide --teacher-checkpoint or set $ACE_TEACHER_CKPT`**
→ Export `ACE_TEACHER_CKPT` or pass `--teacher-checkpoint /path/to/ckpt`.

**Batch size warning in the logs**
→ If `batch_size_global` is not divisible by `per_gpu × world_size`, the
launcher logs a `CRITICAL` message and rounds down.  Adjust
`trainer.batch_size_global` via a config override or edit the spike config.

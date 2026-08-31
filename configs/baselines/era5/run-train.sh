#!/bin/bash
#
# Launcher for the 1-degree baseline cost-reduction screen (research repo
# investigation 2026-08-07-1deg-baseline-cost-reductions).
#
# Adopted by copying research/.claude/skills/launching-runs/run-train.reference.sh.
# The GUARDRAILS block below is verbatim from that reference; only the
# BASELINE-SPECIFIC gantry block and the run_training calls at the bottom are
# edited for this baseline.
#
# Usage (run FROM this directory, configs/baselines/era5/):
#   ./run-train.sh                 # launch every run_training call below
#   ./run-train.sh daily           # launch only the daily arm
#   ./run-train.sh fg16-sr0p125    # launch both bottleneck arms

set -euo pipefail

# === GUARDRAILS (copy verbatim from the reference; do not hand-edit) =========
WANDB_IDENTITY="mcgibbon"   # the wandb username every run must attribute to

SCRIPT_PATH=$(git rev-parse --show-prefix)   # repo-root-relative dir of this script
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

# WANDB attribution guard. The beaker job env does not carry WANDB_USERNAME, and
# the beaker account (jeremym) makes wandb fall back to the API-key service
# account, so an unset/null/jeremym value silently misattributes the run. Beaker
# specs are immutable, so a miss costs a full stop+relaunch+rewrite-every-record
# cycle — fail loud, before submit.
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  echo "       Run:  export WANDB_USERNAME=$WANDB_IDENTITY   before launching." >&2
  exit 1
fi

# cwd / path guard. An empty SCRIPT_PATH means the script was run from the repo
# root (or outside the configs dir): CONFIG_PATH would become "/<config>.yaml"
# and gantry would submit a doomed job even after local validate_config fails.
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: SCRIPT_PATH (git rev-parse --show-prefix) is empty." >&2
  echo "       Invoke run-train.sh FROM its own configs directory, not the repo root." >&2
  exit 1
fi

# Config-line filter. With no args every run_training call runs; with args, only
# calls whose config filename OR job name contains one of the substrings.
LAUNCH_FILTERS=("$@")
should_run() {  # should_run <config_filename> <job_name>
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${LAUNCH_FILTERS[@]}"; do
    [[ "$1" == *"$f"* || "$2" == *"$f"* ]] && return 0
  done
  return 1
}

# Post-launch attribution assertion. gantry submits asynchronously, so the wandb
# run may not exist at submit time; call this once the run has registered (or as
# a standalone follow-up check) to confirm wandb really recorded it under
# WANDB_IDENTITY before you write records / move on.
#   assert_wandb_attribution <wandb_run_id> [wandb_project]   # default ai2cm/ace
assert_wandb_attribution() {
  local run_id="$1" project="${2:-ai2cm/ace}"
  python - "$run_id" "$project" "$WANDB_IDENTITY" <<'PY'
import sys
import wandb
run_id, project, expected = sys.argv[1], sys.argv[2], sys.argv[3]
got = wandb.Api().run(f"{project}/{run_id}").user.username
assert got == expected, f"wandb run {run_id} attributed to {got!r}, expected {expected!r}"
print(f"OK: wandb run {run_id} attributed to {got}")
PY
}
# === END GUARDRAILS =========================================================

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local N_GPUS="${3:-1}"
  local CLUSTER="${4:-ai2/titan}"
  local PRIORITY="${5:-high}"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  should_run "$config_filename" "$job_name" || { echo "skip (filter): $job_name"; return 0; }

  # path guard: the resolved local config must exist before we pay for gantry.
  # (cwd is REPO_ROOT here, so CONFIG_PATH is the repo-relative path.)
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: $REPO_ROOT/$CONFIG_PATH" >&2
    echo "       Check the filename and that you launched from the configs dir." >&2
    exit 1
  fi

  echo "launching: $job_name  ($CONFIG_PATH)"

  # --- BASELINE-SPECIFIC: edit only the block below for this baseline ---------
  # Validate locally to fail fast on config bugs before paying for GPU spin-up.
  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional gantry flags from "# arg: ..." headers in the YAML.
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  # CM_PRIORITY is read by the beaker priority balancer, not by fme: high by
  # default, overridable per arm by a "# arg: --env CM_PRIORITY=low" config
  # header. gantry rejects a duplicated --env, so the default is dropped when
  # the config supplies its own. Only a value right after --env (or the fused
  # --env=NAME=value form) counts, so a lookalike name (MY_CM_PRIORITY=…), a
  # quoted header, or the string appearing in some other flag's value keeps the
  # default instead of silently leaving the job with no usable label.
  local cm_priority_args=(--env CM_PRIORITY=high) arg prev=""
  for arg in "${extra_args[@]}"; do
    if [[ ("$prev" == --env && "$arg" == CM_PRIORITY=*) || "$arg" == --env=CM_PRIORITY=* ]]; then
      cm_priority_args=()
      break
    fi
    prev="$arg"
  done

  # Target per Jeremy 2026-08-07: ai2/ace workspace, high priority, titan only,
  # 4 GPUs (1-degree runs), CM_PRIORITY=high. --shared-memory 400GiB matches the
  # control runs (4s0rnth6, glvk7uxz) this screen compares against.
  # Per-arm overrides of cluster/priority go through run_training's optional
  # 4th/5th arguments (e.g. the GAN arm targets jupiter at urgent).
  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'ACE2S-ERA5 1-degree baseline cost-reduction screen' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority "$PRIORITY" \
    --min-runtime 4h \
    --cluster "$CLUSTER" \
    "${cm_priority_args[@]}" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=ace2s-era5-cost-reductions \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
  # --- END BASELINE-SPECIFIC --------------------------------------------------
}

# Launch targets: the two spectral-bottleneck arms of the cost-reduction screen.
# Their controls already exist and are NOT relaunched here:
#   6h control    wandb 4s0rnth6  (main's ace-train-config-1-step-pretrain.yaml)
#   daily control wandb glvk7uxz  (ace-train-config-1-step-pretrain-daily.yaml)
run_training "ace-train-config-1-step-pretrain-fg16-sr0p125.yaml" \
  "ace2s-era5-fg16-sr0p125-1-step-pre-training-rs0" 4

run_training "ace-train-config-1-step-pretrain-daily-fg16-sr0p125.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-1-step-pre-training-rs0" 4

# 3-step BPTT fine-tunes of the two completed control cells (added 2026-08-10).
# The screen's standard fine-tune recipe (mostly-1-step, last-step-only,
# detached) moved small-scale precipitation power only marginally and both of
# its control fine-tunes died; these replace it with a fixed 3-step rollout,
# the loss on all 3 steps, and BPTT (use_gradient_accumulation: false).
# The two bottleneck cells' fine-tunes wait on their pretrains finishing.
# The pretrain arms above are LIVE -- launch these alone:
#   ./run-train.sh ft3-bptt
run_training "ace-train-config-ft3-bptt.yaml" \
  "ace2s-era5-ft3-bptt-multi-step-fine-tuning-rs0" 8

run_training "ace-train-config-ft3-bptt-daily.yaml" \
  "ace2s-era5-daily-ft3-bptt-multi-step-fine-tuning-rs0" 8

# 3-step BPTT fine-tune of the daily+bottleneck cell (added 2026-08-17), donor
# r95iprjt — also the headline run of the paper-recipe investigation.
# The arms above are done or live -- launch this alone:
#   ./run-train.sh ft3-bptt-daily-fg16
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-ft3-bptt-multi-step-fine-tuning-rs0" 8

# Deterministic analog of the paper recipe (added 2026-08-17): same pipeline
# with noise and the ensemble loss removed (MSE + ACE2-paper variable weights).
# Its 3-step BPTT fine-tune is wired below (donor: this pretrain's checkpoint).
# The arms above are done or live -- launch this alone:
#   ./run-train.sh det
run_training "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-det.yaml" \
  "ace2-era5-daily-fg16-sr0p125-deterministic-1-step-pre-training-rs0" 4

# 3-step BPTT fine-tune of the deterministic pretrain (added 2026-08-20), donor
# xskvz4nx via dataset 01M0DA7TWW34ET5HHCTGJEQHHS. The arms above are done or
# live -- launch this alone:
#   ./run-train.sh ft3-bptt-daily-fg16-sr0p125-det
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-det.yaml" \
  "ace2-era5-daily-fg16-sr0p125-deterministic-ft3-bptt-multi-step-fine-tuning-rs0" 8

# Deterministic 2-step BPTT from-scratch pretrain (added 2026-08-20): trains
# over 2 forward steps with the graph kept across both (no gradient
# accumulation / no detaching), matching the ACE2 paper's training behavior.
# 4 GPUs: the stochastic 2-step OOMed at 4 with n_ensemble 2; deterministic
# n_ensemble 1 halves activation memory, matching the 1-step stochastic
# footprint that fits. Launch alone:
#   ./run-train.sh 2-step-bptt-pretrain-daily-fg16-sr0p125-det
run_training "ace-train-config-2-step-bptt-pretrain-daily-fg16-sr0p125-det.yaml" \
  "ace2-era5-daily-fg16-sr0p125-deterministic-2-step-bptt-pre-training-rs0" 4

# CRPS/ES 50/50 split ablation (added 2026-08-17): paper-recipe gating ablation
# comparing 50/50 vs 90/10 CRPS/energy-score weights. Pretrain launches first;
# fine-tune waits on the pretrain checkpoint.
# The arms above are done or live -- launch the pretrain alone:
#   ./run-train.sh crps50
run_training "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-crps50.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-crps50-1-step-pre-training-rs0" 4

# Fine-tune — launch separately after the pretrain is done and the checkpoint
# dataset ID is filled into the config's "# arg: --dataset" header:
#   ./run-train.sh ft3-bptt-daily-fg16-sr0p125-crps50
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-crps50.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-crps50-ft3-bptt-multi-step-fine-tuning-rs0" 8

# GAN discriminator arm (added 2026-08-18): the stochastic pretrain plus a
# step-conditional discriminator (research repo investigation
# 2026-08-18-gan-loss-small-scale-precip-spectra; ace PR #1446).
# Target per Jeremy 2026-08-18: jupiter at urgent priority, 8 GPUs,
# CM_PRIORITY stays high. The arms above are done or live -- launch this
# alone:
#   ./run-train.sh gan
run_training "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-gan.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-gan-1-step-pre-training-rs0" 8 ai2/jupiter urgent

# Diurnal-mean decoder stage (added 2026-08-20): a small secondary decoder
# (embed 128, filter groups 4, deterministic) trained on the FROZEN paper
# backbone (c8hp09jm best_inference_ckpt) to predict the daily-mean *_mean
# surface fields, on the 2026-08-13 store. Launch alone:
#   ./run-train.sh decoder-daily-mean
run_training "ace-train-config-decoder-daily-mean-fg16-sr0p125.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-mean-decoder-training-rs0-v2" 4

# GAN discriminator 3-step BPTT fine-tune (added 2026-08-24): introduces
# the discriminator at fine-tune time from the r95iprjt pretrain checkpoint
# (no GAN during pretrain). The pretrained generator narrows the real/fake
# gap so the R1 penalty starts manageable. Same discriminator config as
# the pretrain GAN arm. Jupiter/urgent/8 GPUs. Launch alone:
#   ./run-train.sh ft3-bptt-daily-fg16-sr0p125-gan
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-gan.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-gan-ft3-bptt-multi-step-fine-tuning-rs0" 4 ai2/titan urgent

# No-R1 variant of the GAN fine-tune: tests whether D overshoot is avoided
# by the pretrained generator alone, without R1 regularization.
#   ./run-train.sh gan-no-r1
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-gan-no-r1.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-gan-no-r1-ft3-bptt-multi-step-fine-tuning-rs0" 4 ai2/titan urgent

# PatchDiscriminator variant: CNN D with spectral normalization focused on 8
# surface fields (no R1, no input conditioning). Spectral norm constrains D's
# Lipschitz constant directly — avoids the R1 freeze/overshoot problem.
#   ./run-train.sh gan-patch
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-gan-patch.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-gan-patch-ft3-bptt-multi-step-fine-tuning-rs0" 4 ai2/titan urgent

# PatchDiscriminator 1-step pretrains (added 2026-08-27): high-capacity CNN D
# (hidden_dim 512) with spectral norm, D lr 1e-4 matching G. Prior PatchD
# run (h9v70lqi, hidden_dim 64, D lr 1e-5) froze D at chance for 15 epochs.
# Two arms: surface-only (8 fields) and all in/out fields.
#   ./run-train.sh gan-patch-sfc
run_training "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-gan-patch-sfc.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-gan-patch-sfc-1-step-pre-training-rs0" 8 ai2/jupiter urgent

#   ./run-train.sh gan-patch-all
run_training "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-gan-patch-all.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-gan-patch-all-1-step-pre-training-rs0" 8 ai2/jupiter urgent

# 3-step BPTT fine-tune of the all-D GAN pretrain (added 2026-08-31), donor
# sufug8tp via dataset 01M11T34XP492Y999NCQKBA6P9. Batch size 4 (D overhead on
# top of the 3-step BPTT graph; the SFNO-D fine-tune OOMed at batch 8). Titan,
# 4 GPUs. The arms above are done or live -- launch this alone:
#   ./run-train.sh ft3-bptt-daily-fg16-sr0p125-gan-patch-all
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-gan-patch-all.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-gan-patch-all-ft3-bptt-multi-step-fine-tuning-rs0" 4

# Patch energy score arm (reworked 2026-08-26): the stochastic pretrain with
# EnsembleLoss 0.9 patch-ES (3x3) / 0.1 spectral-ES, no CRPS -- the patch ES
# takes CRPS's place (research repo investigation
# 2026-08-25-patch-vs-spectral-energy-score; ace PR #1459). The first cut
# (0.9 CRPS / 0.1 patch-ES, wandb v0d7iiz4) was canceled at epoch 21 per
# Jeremy. Titan, 4 GPUs, matching donor r95iprjt -- jupiter kills this family
# with NCCL unhandled cuda errors at every epoch-boundary validation
# (80 GB H100s). Launch alone:
#   ./run-train.sh pes90-es10-1-step
run_training "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-pes90-es10.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-pes90-es10-1-step-pre-training-rs0" 4 ai2/titan

# 0.8 patch-ES / 0.2 spectral-power-CRPS pretrain arm (per Jeremy 2026-08-26):
# same fork with the 0.1 spectral ES swapped for a 0.2-weight per-degree
# log-power CRPS (ported from bake-off commit 1ce0c1d40). Jupiter per Jeremy
# (note: earlier jupiter pretrains died at epoch-boundary validation with
# NCCL unhandled cuda errors). Launch alone:
#   ./run-train.sh pes80-spcrps20
run_training "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-pes80-spcrps20.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-pes80-spcrps20-1-step-pre-training-rs0" 4 ai2/jupiter

# CRPS control for the arm above (per Jeremy 2026-08-27): 0.8 CRPS /
# 0.2 spectral-power-CRPS at the same embed_dim 384, isolating patch-ES vs
# CRPS at matched spectral-power-CRPS weight and width. Jupiter to match its
# sibling. Launch alone:
#   ./run-train.sh crps80-spcrps20
run_training "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-crps80-spcrps20.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-crps80-spcrps20-1-step-pre-training-rs0" 4 ai2/jupiter

# Fine-tune — launch separately after the pes90-es10 pretrain is done and its
# final job's result dataset is filled into the config's "# arg: --dataset"
# header. All FTs per Jeremy 2026-08-26: titan, 3 steps (jupiter kills this
# family at epoch-boundary validation; 2-step jupiter cut abandoned).
#   ./run-train.sh pes90-es10-ft3
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-pes90-es10.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-pes90-es10-ft3-bptt-multi-step-fine-tuning-rs0" 8 ai2/titan

# Own-pretrain BPTT fine-tunes for the 384-wide arms (pes80-spcrps20 and its
# CRPS control crps80-spcrps20), part of the patch-vs-spectral ES
# investigation. Titan, 8 GPUs (4 GPUs OOM on 3-step BPTT). Launch alone:
#   ./run-train.sh pes80-spcrps20-ft
#   ./run-train.sh crps80-spcrps20-ft
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-pes80-spcrps20-ft.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-pes80-spcrps20-ft3-bptt-multi-step-fine-tuning-rs0" 8 ai2/titan
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-crps80-spcrps20-ft.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-crps80-spcrps20-ft3-bptt-multi-step-fine-tuning-rs0" 8 ai2/titan

# Cross-donor patch-ES fine-tunes: patch-ES-led losses fine-tuning the
# CRPS-led 90/10 pretrain (donor r95iprjt, same donor as c8hp09jm) -- fast
# viability reads on the patch ES that don't wait for its own pretrain.
# Two arms per Jeremy 2026-08-26: 0.9 patch-ES / 0.1 spectral-ES, and 100%
# patch-ES. Titan, 8 GPUs, 3 steps (exact c8hp09jm compute match). Launch:
#   ./run-train.sh pes90-es10-from-es
#   ./run-train.sh pes100-from-es
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-pes90-es10-from-es.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-pes90-es10-from-es-ft3-bptt-multi-step-fine-tuning-rs0" 8 ai2/titan
run_training "ace-train-config-ft3-bptt-daily-fg16-sr0p125-pes100-from-es.yaml" \
  "ace2s-era5-daily-fg16-sr0p125-pes100-from-es-ft3-bptt-multi-step-fine-tuning-rs0" 8 ai2/titan

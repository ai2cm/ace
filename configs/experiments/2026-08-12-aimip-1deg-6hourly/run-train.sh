#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)

# wandb runs on a service-account key, so this is the only thing attributing them to a
# human. It differs from the beaker username, so it cannot be derived -- fail before submit.
WANDB_IDENTITY="bhenn1983"
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       Unset WANDB_USERNAME, or export WANDB_USERNAME=$WANDB_IDENTITY, before launching." >&2
  exit 1
fi

# Cluster selection. batch_size is global and fme splits it across data-parallel ranks, so it
# is held fixed and the rank count varies with per-GPU memory: the batch of 8 is 2 samples per
# GPU on titan and 1 on jupiter. Cluster and N_GPUS move together and must never be set
# independently.
#
# Running a seed on the other cluster does NOT make it a different experiment. DistributedSampler
# builds one seeded global permutation and hands rank r the stride indices[r::R], so with a local
# batch of 8/R the union at each step is exactly indices[8k:8k+8] for any R -- the sequence of
# global batches is identical at 4 and 8 ranks (verified against torch directly). Only the
# gradient reduction order differs, which is the same float nondeterminism as re-running on the
# same hardware. rs0 trained on titan; rs1-rs3 may go wherever there is room.
CLUSTER=${CLUSTER:-titan}
case "$CLUSTER" in
  titan)   BEAKER_CLUSTER="ai2/titan";   N_GPUS=4 ;;
  jupiter) BEAKER_CLUSTER="ai2/jupiter"; N_GPUS=8 ;;
  *)
    echo "ERROR: CLUSTER must be 'titan' or 'jupiter', got '$CLUSTER'." >&2
    exit 1
    ;;
esac

# Preemption guarantee. --preemptible/--not-preemptible are deprecated and, since the
# 2026-08-28 scheduler change, priority no longer buys protection either; --min-runtime is the
# lever that remains. rs0's stage 2 took 81h wall across 35 preempted attempts without it.
MIN_RUNTIME=8h

# Priority orders contention within our own budget -- `high` rather than the `urgent` rs0 was
# submitted with, to stay polite to the rest of the group. Set once: it feeds both the gantry
# flag and the CM_PRIORITY env var the job reads, which must not drift apart.
PRIORITY=high

# cwd guard: an empty SCRIPT_PATH means this was run from the repo root, which would make
# CONFIG_PATH absolute and submit a doomed job.
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run this script from configs/experiments/2026-08-12-aimip-1deg-6hourly, not the repo root." >&2
  exit 1
fi

cd "$REPO_ROOT"  # so the config path is valid no matter where this is run from

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local override="${3:-}"          # e.g. "seed=1"; empty for none
  local donor="${4:-}"             # this chain's own previous-stage result dataset id
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  # The stage 2 and 3 configs carry rs0's donor dataset id baked into their `# arg:` header,
  # so a new seed's chain MUST override it. Getting this wrong is silent: the job runs, and
  # initializes seed N from seed 0's weights. Guard both directions.
  local mounts_weights=0
  [[ "${extra_args[*]}" == *:/weights* ]] && mounts_weights=1

  if [[ "$donor" == FILL-* ]]; then
    echo "ERROR: $job_name has an unfilled donor placeholder ('$donor'). Fill in the result" >&2
    echo "       dataset id of this seed's preceding stage before launching." >&2
    exit 1
  fi

  if [[ -n "$donor" ]]; then
    if (( ! mounts_weights )); then
      echo "ERROR: donor '$donor' given but $config_filename mounts no /weights dataset." >&2
      exit 1
    fi
    local rebuilt=()
    local a
    for a in "${extra_args[@]}"; do
      [[ "$a" == *:/weights ]] && a="${donor}:/weights"
      rebuilt+=("$a")
    done
    extra_args=("${rebuilt[@]}")
  elif (( mounts_weights )) && [[ "$override" != "seed=0" && -n "$override" ]]; then
    echo "ERROR: $config_filename mounts rs0's donor checkpoint, but this chain is" >&2
    echo "       '$override' with no donor supplied. Pass the result dataset id of THIS" >&2
    echo "       seed's preceding stage as the 4th argument to run_training." >&2
    exit 1
  fi

  # fme divides the config's global batch size among data-parallel ranks and raises if it
  # does not divide evenly -- but only once the job is running. Check before submit.
  python - "$CONFIG_PATH" "$N_GPUS" <<'BATCHCHECK'
import sys, yaml
config_path, n_ranks = sys.argv[1], int(sys.argv[2])
config = yaml.safe_load(open(config_path))
loaders = (("train_loader", config.get("train_loader")),
           ("validation.loader", config.get("validation", {}).get("loader")))
for name, loader in loaders:
    size = (loader or {}).get("batch_size")
    if size is not None and size % n_ranks:
        sys.exit(f"ERROR: {name}.batch_size={size} is not divisible by {n_ranks} ranks.")
BATCHCHECK

  local train_cmd=(torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH")
  [[ -n "$override" ]] && train_cmd+=(--override $override)

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description "ACE2.2 seed ensemble (AIMIP-like baseline, 1°/6-hourly) on $CLUSTER" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority "$PRIORITY" \
    --min-runtime "$MIN_RUNTIME" \
    --timeout 0 \
    --no-logs \
    --cluster "$BEAKER_CLUSTER" \
    --env CM_PRIORITY="$PRIORITY" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
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
    -- "${train_cmd[@]}"
}

# 6-hourly retrain of the 1deg/daily v2 ERA5-only no-residual no-CO2 baseline submitted to
# AIMIP. The daily-step original emits daily snapshots, which cannot be evaluated against
# the monthly-/daily-AVERAGED AIMIP ERA5 data; a 6h step resolves the diurnal cycle so its
# output averages into comparable means. See README.md for the full change list.
#
# Trained in three stages: 1-step pretraining, a multi-step fine-tune initialized from its
# checkpoint, then a pressure-level fine-tune adding the AIMIP plev diagnostics. Each stage's
# config already carries our run's donor dataset id; to reproduce from scratch, substitute the
# id your own preceding stage produced.

base_name="train-1deg-6hourly-v2-era5-only-no-residual-no-co2"

# =============================== the rs0 chain (COMPLETE) ===============================
# Submitted to workspace ai2/climate-titan at --priority urgent with no --min-runtime; the
# flags above have since moved to ai2/ace, high, and 8h. Those are scheduling settings only
# and do not affect the trained result, so rs0 remains an exchangeable member of the ensemble.
#
#   stage 1  experiment 01KZYJ4HT4ZMZH296KBNWMPCQF  launched 2026-08-13 at commit b4a688d85
#            wandb g94277n6, 40 epochs, 44.5h on 4 GPUs
#            result dataset 01KZYJ4HTBWED5VG3VFTRYKDRC   <- baked into the stage 2 config
#
#   stage 2  experiment 01M06W0WN4WY1HJBWFXJNJEXC7  completed 2026-08-24, 40 epochs,
#            81h wall across 35 preempted attempts. wandb 78crdqjr. best_val_loss 0.16528,
#            best_inference_error 0.026124 -- unchanged from epoch 8, so best_inference_ckpt
#            is the epoch-8 checkpoint.
#            result dataset 01M0RFP2DKAGABV89KRPMXX5C3   <- baked into the stage 3 config
#
#   stage 3  wandb train-...-plev-ft-rs0, completed 2026-08-24 20:35Z
#
# Re-running rs0 would produce a different checkpoint (nondeterministic reductions) and
# invalidate every published ACE2.2 number, so no rs0 line is live in this script.

# ============================== the seed ensemble (rs1-rs3) ==============================
# ACE2.1 was selected best-of-4 seeds; ACE2.2 is a single seed. That asymmetry is a confound
# in every ACE2.2-vs-ACE2.1 comparison, worth ~12% in expectation on mean-climate error
# (E[max of 4 standard normals] = 1.03 against a measured seed spread of ~12%). Three more
# seeds bring ACE2.2 to n=4, matching the older model's protocol, and cut the uncertainty on
# its mean from a single draw to SE ~ sd/2.
#
# What this does NOT buy: significance on the training-span trend question. The gap is ~10
# points against a seed sd of 7.8, so even n=8 reaches only ~39% power. Judge that on the
# point estimate and the +4K response, not on a test.
#
# Selection discipline: report the 4-seed MEAN when comparing against P1 or ACE2.1, and the
# selected best only as the submission. Select on validation inference error -- never on
# trend, which the intercomparison scores as E2.
#
# Stages 1 and 2 run for every seed; STAGE 3 RUNS FOR THE SELECTED SEED ONLY. Stage-1
# metrics do not predict final quality (the P0 probe), so seeds cannot be culled before
# stage 2. But the comparison is scored on stage-2 checkpoints -- stage 3 only fits the
# pressure-level secondary decoder with the trunk frozen, and every field the seed
# comparison uses already exists after stage 2 (this is why the findings report scores
# long36-ace22-stage2 and long36-p1-rs*, not their stage-3 successors). Stage 3 is needed
# only for the model that is actually submitted, whose E1-E5 scoring reads the plev
# diagnostics. So: STAGE=3 SEEDS="<winner>", not the default.
#
# Evaluating a seed means a standalone long36 backfill of its stage-2 checkpoint, not the
# inline long_36year entry. The inline one fires on a fixed epoch schedule (29/34/39) and
# the selected checkpoint is not on it -- rs0's stage-2 best_inference_error was flat from
# epoch 8. Matching the selected epoch would mean running the rollout every epoch, ~6 GPU-h
# x 40 = ~240 GPU-h against 165 for the training itself. The inline entry is an in-flight
# drift signal, not the evaluation.
#
# Usage:  STAGE=1 ./run-train.sh                  # all three seeds
#         STAGE=2 SEEDS="1 2" ./run-train.sh
#         STAGE=3 SEEDS="2" ./run-train.sh        # winner only
#         CLUSTER=jupiter STAGE=1 ./run-train.sh

SEEDS=${SEEDS:-"1 2 3"}
STAGE=${STAGE:-}

# Result dataset id of each seed's own stage-1 run. Each experiment ran a single job (no
# restarts), so the job's result dataset is the experiment's; a restart would have issued a
# new id, and the id below would then point at the dead job's short checkpoints.
#   rs1  experiment 01M1J6WF8B2FDS0GCFMXVNEA87  job 01M1J6WFDXMS3FREJ8Z0PCEJ9V
#   rs2  experiment 01M1J6WW2Y7EQB8KWZGACRC2JC  job 01M1J6WW7C7N6SN2732ABZ2104
#   rs3  experiment 01M1J6X90YXZ7EFFG01BD8CWEB  job 01M1J6X94M9ZNC8J41SSJ2CKJ7
# Stage 1 launched 2026-09-02 23:23-23:58Z, 4 GPUs each on titan.
declare -A STAGE1_DONOR=(
  [1]=01M1J6WF8NQM6AV14WW90MF2T9
  [2]=01M1J6WW372RKP423AYD0A9JYQ
  [3]=01M1J6X918YK0W7ZP9WA422MT9
)

# Stage 2 launched 2026-09-04 06:05-06:29Z on jupiter (8 GPUs, high, 8h) at commit e10a9be:
#   rs1  experiment 01M1NGC06NPTEZJS1H2KSE8KV2
#   rs2  experiment 01M1NGNEMV2A1ETM4ZGAZ1X2T4
#   rs3  experiment 01M1NHPDTMZ0N8K9W7XEWPRYP5
# Job and result-dataset ids are deliberately NOT recorded here yet: these jobs auto-resume,
# and a preemption after the 8h min-runtime issues a new job with a new result dataset. Take
# both from the job that exited 0, after completion.
#
# Result dataset id of each seed's own stage-2 run, filled after STAGE=2 completes. Take the
# id from the JOB that exited 0, not from the experiment: a preempted-and-resumed experiment
# commits the dead job's dataset too, with checkpoints short of the final epoch, and mounting
# it fails silently. This bit the P1 chains.
declare -A STAGE2_DONOR=(
  [1]=FILL-AFTER-STAGE-2
  [2]=FILL-AFTER-STAGE-2
  [3]=FILL-AFTER-STAGE-2
)

case "$STAGE" in
  1)
    for SEED in $SEEDS; do
      run_training "$base_name.yaml" "$base_name-rs${SEED}" "seed=${SEED}"
    done
    ;;
  2)
    for SEED in $SEEDS; do
      run_training "$base_name-multi-step-ft.yaml" "$base_name-multi-step-ft-rs${SEED}" \
        "seed=${SEED}" "${STAGE1_DONOR[$SEED]}"
    done
    ;;
  3)
    for SEED in $SEEDS; do
      run_training "$base_name-plev-ft.yaml" "$base_name-plev-ft-rs${SEED}" \
        "seed=${SEED}" "${STAGE2_DONOR[$SEED]}"
    done
    ;;
  *)
    echo "Usage: STAGE=<1|2|3> [SEEDS=\"1 2 3\"] $0" >&2
    echo "  1  1-step pretraining         (no donor)" >&2
    echo "  2  multi-step fine-tuning     (needs STAGE1_DONOR filled)" >&2
    echo "  3  pressure-level fine-tuning (needs STAGE2_DONOR filled)" >&2
    exit 1
    ;;
esac

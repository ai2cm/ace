#!/bin/bash
# 1979-2014 rollout metrics for models that never logged `long_36year`, so they can be
# compared against the P1/P2 chains without touching the AIMIP 2015-2024 holdout.
#
#   ACE2.2 stage 2   -- its own training logged `long_46year`, which spans the holdout.
#   ACE2.1 RS0-RS3   -- their 1979-2014 runs used ONE initial condition; this uses the
#                       same eight as the inline entry, so their seed spread is not
#                       inflated by single-member internal variability.
#
# Each job is ~45 min on 8 GPUs (~6 GPU-h), from the inline cost at epochs 30/35/40.
#
# SMOKE=1 runs 2400 of the 52566 steps (~4.6%). That is long enough past startup to
# measure steady-state throughput, so the job's wall time x ~22 estimates the full run --
# which matters because the single-GPU cost is not known: the inline entry does this
# rollout in ~44 min across 8 GPUs, but how 8 ICs batched onto one GPU compares is
# untested. Inference progress logs are suppressed (ace issue #1461), so timing the smoke
# is the only way to size the real runs. Use it once before the real
# batch: the ACE2.1 checkpoints were trained at ace ref 70c966ed5 against the 2024 ERA5
# store, and this evaluates them against the 2026 store.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)

WANDB_IDENTITY="bhenn1983"
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  exit 1
fi
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run this from the config directory, not the repo root." >&2
  exit 1
fi

# One GPU. Multi-GPU ACE inference is not a supported entrypoint: torchrun is documented
# for fme.ace.train and fme.downscaling.inference but not for fme.ace.evaluator, and no
# parallel test covers it. The plumbing looks present (fme/ace/evaluator.py enters
# Distributed.context(), and the loader shards ICs across ranks at
# fme/ace/data_loading/inference.py:292-297), so N_GPUS=8 is available to try -- but it is
# untested, so it is not the default.
#
# The inline entry runs this same rollout in ~45 min across 8 GPUs, or ~6 GPU-hours, so a
# single GPU takes several hours. Hence the 8h min-runtime.
N_GPUS=${N_GPUS:-1}
if [[ "$N_GPUS" -eq 1 ]]; then MIN_RUNTIME=8h; else MIN_RUNTIME=4h; fi

# PREEMPTIBLE=1 gives up the min-runtime guarantee (--min-runtime 0 means "preemptible at
# any time") and so does not consume allocated slots. Fine for a short smoke run; for the
# real rollouts a preemption restarts from zero, since this config has no --segments.
# Preemptible work takes no allocated slots, so it does not compete with the group's
# guaranteed capacity and goes in at urgent -- at high it just sits behind allocated work
# on a busy cluster. Both the gantry flag and CM_PRIORITY follow this one variable.
PRIORITY=high
if [[ -n "$PREEMPTIBLE" ]]; then
  MIN_RUNTIME=0
  PRIORITY=urgent
fi

# Optional substring filter on the run label: `bash run-long36-backfill.sh ace21-rs0`.
SELECT="${1:-}"
CLUSTER=${CLUSTER:-titan}
case "$CLUSTER" in
  titan)   BEAKER_CLUSTER="ai2/titan";   GPU="B200";      PRODUCT_LIMIT=500 ;;
  jupiter) BEAKER_CLUSTER="ai2/jupiter"; GPU="H100 80GB"; PRODUCT_LIMIT=100 ;;
  ceres)   BEAKER_CLUSTER="ai2/ceres";   GPU="H100 80GB"; PRODUCT_LIMIT=100 ;;
  *) echo "ERROR: CLUSTER must be 'titan', 'jupiter' or 'ceres', got '$CLUSTER'." >&2; exit 1 ;;
esac

cd "$REPO_ROOT"
# Two configs. ACE2.2 and the variant chains read the 2026 ERA5 store; the ACE2.1
# checkpoints require the 2024 build, whose ak coefficients they were trained against
# (fme checks the vertical coordinate against the checkpoint). The ak difference is tiny,
# so metrics from the two stores stay comparable.
CONFIG_2026="$SCRIPT_PATH/long36-evaluator-config.yaml"
CONFIG_2024="$SCRIPT_PATH/long36-ace21-evaluator-config.yaml"
python -m fme.ace.validate_config --config_type evaluator "$CONFIG_2026"
python -m fme.ace.validate_config --config_type evaluator "$CONFIG_2024"

# Memory. Per-window cost scales as (initial conditions per rank) x forward_steps_in_memory,
# and that product has an OOM ceiling that depends on the GPU:
#   H100 80GB  (ceres, jupiter): ~100
#   B200 ~180GB (titan):         ~500, evidenced by
#     configs/experiments/2026-06-16-ace2s-land-feedback-inference/frameworkA-era5.yaml
#     running 5 ICs x forward_steps_in_memory 100 on titan.
# The config's forward_steps_in_memory=40 is sized for the inline run, which puts 1 IC on
# each of 8 ranks (1 x 40, fine). On one GPU all 8 ICs share it, so 8 x 40 = 320 -- fine on
# B200, an OOM on H100. Derive the largest safe value; FSIM overrides it.
N_ICS=8
ICS_PER_RANK=$(( N_ICS / N_GPUS ))
DERIVED_FSIM=$(( PRODUCT_LIMIT / ICS_PER_RANK ))
[[ $DERIVED_FSIM -gt 40 ]] && DERIVED_FSIM=40   # never exceed the config's own value
FSIM=${FSIM:-$DERIVED_FSIM}
echo "  ${CLUSTER} (${GPU}): ${N_ICS} ICs / ${N_GPUS} rank(s) = ${ICS_PER_RANK} per rank"
echo "  -> forward_steps_in_memory=${FSIM}, product $(( ICS_PER_RANK * FSIM )) of limit ${PRODUCT_LIMIT}"

# `--override` takes nargs="*" and is NOT append-style (fme/core/cli.py:115-120), so a
# second --override replaces the first. Collect the pairs and emit one flag.
SUFFIX=""
DOTLIST=()
DOTLIST+=("forward_steps_in_memory=$FSIM")
if [[ -n "$SMOKE" ]]; then
  SUFFIX="-smoke"
  DOTLIST+=("n_forward_steps=2400")
fi
OVERRIDE=()
[[ ${#DOTLIST[@]} -gt 0 ]] && OVERRIDE=(--override "${DOTLIST[@]}")

run_eval () {
  local label="$1" checkpoint_dataset="$2"
  local job_name="long36-${label}${SUFFIX}"

  if [[ -n "$SELECT" && "$label" != *"$SELECT"* ]]; then
    return 0
  fi
  local CONFIG="$CONFIG_2026"
  [[ "$label" == ace21-* ]] && CONFIG="$CONFIG_2024"
  echo "  submitting $job_name  ($(basename "$CONFIG"))"

  gantry run \
    --allow-dirty \
    --name "$job_name" \
    --task-name "$job_name" \
    --description '1979-2014 rollout metrics, holdout-clean comparison against the ACE2.2 variant campaign' \
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
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=ace22-variants \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${checkpoint_dataset}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus "$N_GPUS" \
    --shared-memory 200GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.evaluator "$CONFIG" "${OVERRIDE[@]}"
}

# ACE2.2 stage 2 (multi-step FT), the counterpart to the P1/P2 stage-2 chains.
run_eval "ace22-stage2" "01M0RFP2DKAGABV89KRPMXX5C3"

# ACE2.1's four training seeds. Ids from
# ACE2.1-ERA5-AIMIP/scripts/run-ace-evaluator-seed-selection-single.sh.
run_eval "ace21-rs0" "01K9B1MR70QWN90KNY7NM22K5M"
run_eval "ace21-rs1" "01K9B1MT4QY1ZEZPPS53G2SXPK"
run_eval "ace21-rs2" "01K9B1MVP3VS3NEABHT0W151AX"
run_eval "ace21-rs3" "01K9B1MXD6V26S8BQH5CKY514C"

# The variant chains' own stage-2 checkpoints. The inline `long_36year` entry evaluates
# whatever weights the chain holds at that epoch, which is a different object from the
# backfills above: those evaluate `best_inference_ckpt.tar`, a selected checkpoint. Epoch
# snapshots also carry occasional large excursions -- P2's epoch-35 evaluation reads 0.044
# against 0.027/0.028 either side of it, and P1-rs0's epoch 40 is similarly elevated -- so a
# mean over inline evaluations is not comparable to a single backfill number. Running the
# chains through the same path makes all six models the same object.
run_eval "p1-rs0" "01M19ZTFCEAKE8ZP8SGXP8KBZN"
run_eval "p1-rs1" "01M19ZTFGBA97R0Z2RX3VHSEFY"
run_eval "p1-rs2" "01M19ZTGKSNXXT7GPFGKDGT654"
run_eval "p2-rs0" "01M1A95R5J7J54GANKMG4GFJB8"

# Stage-3 checkpoints, for the two models whose near-surface fields exist only after the
# plev fine-tune. P1 and ACE2.2 need no stage-3 equivalent: their near-surface fields are
# prognostic, so the stage-2 backfills above already carry them, and the stage-3 trunk is
# frozen. ACE2.1's plev-FT is decoder seed 0 on the RS3 trunk, from
# ACE2.1-ERA5-AIMIP/scripts/run-ace-evaluator-seed-selection-single.sh; the `ace21-` prefix
# routes it to the 2024 ERA5 config.
run_eval "ace21-plevft" "01KAKXY0EK24K7BZK2N8SPJ5SJ"

# P2 stage 3 (experiment 01M1EXEGPJ4PVSNPYA1WM1XGJP, job 01M1EXEGT6MSMGT306E0EJYVG1).
# Still running at time of writing -- CONFIRM exit 0 and a committed dataset before
# launching, since a restart would issue a new id.
run_eval "p2-stage3" "01M1EXEGPV085WQVT9EMY87PQ7"

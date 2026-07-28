#!/bin/bash
#
# ERA5-CASE response evals for the seven finished combined SHiELD+ + ERA5
# trainings. The trainings' eight inline evaluations all read c96-SHiELD stores
# with labels: [shield-plus], so the investigation's own question -- does joint
# training carry the forcing responses to the ERA5 case? -- was never measured.
# These jobs make that measurement offline from the epoch-240 EMA checkpoints.
#
# Per (arm, label mode), three jobs:
#   - p0k / p4k : ERA5-forced 4-yr rollouts, control and +4 K ocean-masked SST
#                 (fme.ace.inference). Response = p4k - p0k; time-mean maps are
#                 written so the land/ocean warming ratio can be measured.
#   - trend46yr : ERA5-forced 46-yr rollout + its constant-CO2 twin, eval-only
#                 fme.ace.train (max_epochs 0 + evaluate_before_training).
#
# Label modes: the two unconditional arms run `nolabel`; the labelled anchor B
# runs `era5`; the four label_embed_dim-2 arms run `era5` AND `withheld`
# (labels: [], the all-zeros unconditional state that label dropout trains --
# the mode dropout exists to create, and the one a shortcut-free model should be
# able to use). 33 jobs total.
#
# research: tasks/2026-07-08-combined-shield-era5-training.md
#           tasks/2026-07-08-masked-label-conditioning.md
#           investigations/2026-07-16-combined-shield-era5-training.md
#
# Usage (run FROM this configs directory):
#   ./run-era5-response-evals.sh                     # all 33 jobs
#   ./run-era5-response-evals.sh drop50-era5         # canary: one arm+mode
#   ./run-era5-response-evals.sh p4k trend46yr       # substring OR filter

set -euo pipefail

# === GUARDRAILS (from research run-train.reference.sh) ======================
WANDB_IDENTITY="mcgibbon"
SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  exit 1
fi
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run this script from its own configs directory, not the repo root." >&2
  exit 1
fi
LAUNCH_FILTERS=("$@")
should_run() {
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${LAUNCH_FILTERS[@]}"; do
    [[ "$1" == *"$f"* ]] && return 0
  done
  return 1
}
# === END GUARDRAILS ==========================================================

JOB_GROUP="combined-shield-era5-era5-response-evals"
EMA="ema_ckpt_0240.tar"

# arm -> beaker result dataset of its epoch-240 training run
declare -A CKPT=(
  [nolabel-pooled]="01KYBPVHY2R0EV1BQT4ED7NSFN"   # A  wandb d3oopmrt
  [nolabel-shieldstats]="01KXS8002HPFZFTKQB8K4KYK4Z"   # G  wandb 605kg08o
  [label-pooled]="01KY8KS3K7RVYW7XD9MQP13AEF"   # B  wandb tklu8mif
  [drop0]="01KYBJN91V9K819KKAMMW6W3V9"   # B' wandb j2ox5jvv
  [drop10]="01KYGHJQSB36E7TBETW0N1D2QR"   # C  wandb bntku16k
  [drop50]="01KYGHJE0WQGWA1R0NDDZM89Q7"   # D  wandb tfort5ji
  [drop90]="01KYAMGX32R4JQ6ZB8ZCKEQZDG"   # E  wandb v9irj2he
)
# arm -> label modes to evaluate (space separated)
declare -A MODES=(
  [nolabel-pooled]="nolabel"
  [nolabel-shieldstats]="nolabel"
  [label-pooled]="era5"
  [drop0]="era5 withheld"
  [drop10]="era5 withheld"
  [drop50]="era5 withheld"
  [drop90]="era5 withheld"
)
ARMS=(nolabel-pooled nolabel-shieldstats label-pooled drop0 drop10 drop50 drop90)

cd "$REPO_ROOT"

for mode in nolabel era5 withheld; do
  for pert in p0k p4k; do
    python -m fme.ace.validate_config --config_type inference \
      "$SCRIPT_PATH/ace-inference-era5-$pert-$mode.yaml"
  done
  python -m fme.ace.validate_config --config_type train \
    "$SCRIPT_PATH/eval-trend-era5-46yr-$mode.yaml"
done

gantry_common=(
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")"
  --workspace ai2/ace
  --priority high
  --not-preemptible
  --cluster ai2/jupiter
  --cluster ai2/titan
  --env WANDB_USERNAME="$WANDB_USERNAME"
  --env WANDB_JOB_TYPE=inference
  --env WANDB_RUN_GROUP="$JOB_GROUP"
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa
  --dataset-secret google-credentials:/tmp/google_application_credentials.json
  --gpus 1
  --shared-memory 50GiB
  --allow-dirty
  --weka climate-default:/climate-default
  --budget ai2/atec-climate
  --system-python
  --install "pip install --no-deps ."
)

launch () {   # launch <job_name> <ckpt_dataset> <entrypoint...>
  local JOB_NAME=$1 CKPT_DATASET=$2; shift 2
  should_run "$JOB_NAME" || { echo "skip (filter): $JOB_NAME"; return 0; }
  echo "launching: $JOB_NAME"
  gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description 'ERA5-case response evals, combined SHiELD+ERA5 arms' \
    --env WANDB_NAME="$JOB_NAME" \
    "${gantry_common[@]}" \
    --dataset "$CKPT_DATASET":"training_checkpoints/$EMA":/ckpt.tar \
    -- "$@"
}

for arm in "${ARMS[@]}"; do
  ds="${CKPT[$arm]}"
  for mode in ${MODES[$arm]}; do
    launch "era5resp-$arm-$mode-p0k" "$ds" \
      python -I -m fme.ace.inference "$SCRIPT_PATH/ace-inference-era5-p0k-$mode.yaml"
    launch "era5resp-$arm-$mode-p4k" "$ds" \
      python -I -m fme.ace.inference "$SCRIPT_PATH/ace-inference-era5-p4k-$mode.yaml"
    launch "era5resp-$arm-$mode-trend46yr" "$ds" \
      torchrun --nproc_per_node 1 -m fme.ace.train "$SCRIPT_PATH/eval-trend-era5-46yr-$mode.yaml"
  done
done

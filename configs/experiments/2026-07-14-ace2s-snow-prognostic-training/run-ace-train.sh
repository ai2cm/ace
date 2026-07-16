#!/bin/bash
# One launcher for the ACE2S snow-prognostic training runs.
# Two models (ERA5, CM4-piControl), each trained from scratch in two stages
# (1-step pretrain -> multi-step finetune). Both stages are part of the recipe:
# stage 2 warm-starts from the stage-1 checkpoint, so it is launched as a second
# invocation once the pretrain jobs finish and their dataset IDs are known.
#
#   ERA5:  era5-snow-prognostic   (base recipe: feature/add-ACE2S-ERA5-baseline)
#   CM4:   cm4-snow-prognostic    (base recipe: exp/ace2s-cm4-piControl-train)
#
# Snow water equivalent (surface_snow_amount) is added as a prognostic variable and
# snow-covered area (surface_snow_area_fraction) as a diagnostic; the moisture-budget
# corrector's clip_frozen_precipitation is enabled (see the pretrain configs).
#
# Usage:
#   ./run-ace-train.sh                    # submit both pretrain jobs
#   ./run-ace-train.sh era5               # optional substring filter on the job name
# After the pretrain jobs finish, paste their beaker dataset IDs into
# PRETRAIN_DATASETS_* below and uncomment the finetune block, then re-run.
#
# Per-config `# arg:` header lines (e.g. the stats dataset mount) are extracted and
# appended to the gantry command, mirroring the baseline training scripts.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)   # this dir, relative to the repo root
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}   # W&B handle (differs from the Beaker username)
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4
JOB_GROUP="ace2s-snow-prognostic"
DATA_TAG="1deg-6h"                            # data resolution + timestep for this experiment dir;
                                              # a new data regime gets its own dated dir (e.g. 4deg-daily)
SELECT="${1:-}"                               # optional substring filter on job name

cd "$REPO_ROOT"  # so the config path resolves regardless of where this is run from

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"
  shift 2

  if [ -n "$SELECT" ] && [[ "$job_name" != *"$SELECT"* ]]; then
    return 0
  fi

  local ckpt_dataset=""
  local override_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --ckpt) ckpt_dataset="$2"; shift 2 ;;
      *) override_args+=("$1"); shift ;;
    esac
  done

  local ckpt_arg=()
  if [[ -n "$ckpt_dataset" ]]; then
    ckpt_arg=(--dataset "$ckpt_dataset:/weights")
  fi

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional gantry args from the config header (lines starting "# arg: ").
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description "ACE2S snow-prognostic training: $job_name" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority high \
    --preemptible \
    --cluster ai2/titan \
    --weka climate-default:/climate-default \
    "${extra_args[@]}" \
    "${ckpt_arg[@]}" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH" \
    ${override_args:+--override "${override_args[@]}"}
}

# ---- Stage 1: 1-step pretrain (one seed each) --------------------------------------
run_training "era5-snow-prognostic-1-step-pretrain.yaml" "ace2s-snowprog-era5-${DATA_TAG}-pretrain" "seed=0"
run_training "cm4-snow-prognostic-1-step-pretrain.yaml"  "ace2s-snowprog-cm4-${DATA_TAG}-pretrain"  "seed=0"

# ---- Stage 2: multi-step finetune (warm-start from the pretrain checkpoints) --------
# Fill in each pretrain job's beaker result dataset ID, then uncomment the block and
# re-run. The finetune configs are variable-agnostic (they load in_names/out_names/
# corrector from the mounted pretrain checkpoint at /weights).
PRETRAIN_DATASET_ERA5=""   # e.g. 01K...
PRETRAIN_DATASET_CM4=""

# run_training "era5-multi-step-finetuning.yaml" "ace2s-snowprog-era5-${DATA_TAG}-finetune" --ckpt "$PRETRAIN_DATASET_ERA5" "seed=0"
# run_training "cm4-multi-step-finetuning.yaml"  "ace2s-snowprog-cm4-${DATA_TAG}-finetune"  --ckpt "$PRETRAIN_DATASET_CM4"  "seed=0"

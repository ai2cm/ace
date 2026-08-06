#!/bin/bash
#
# ERA5-case +4 K response of the ERA5-ONLY trainings that are closest in recipe
# to the combined SHiELD+ + ERA5 arms.
#
# Why this wave exists. The combined-training report compared its arms' ERA5-case
# +4 K response against one ERA5-only emulator, `shared-t-append-clipped`
# (wandb ww3nugpp). That model is NOT close to the joint recipe: 60 epochs
# against 240, filter_num_groups 1 against 16 (and no spectral_ratio),
# hand-tuned per-variable loss weights, and `remove_global_mean: false` in its
# network normalization. Review asked for the closest available ERA5-only run
# instead.
#
# The closest is `train-4deg-daily-v2-era5-only-no-residual-resume400k`
# (wandb znnaox7t). Against the joint arms it matches on: NoiseConditionedSFNO,
# embed_dim 512, 8 layers, filter_num_groups 16, spectral_ratio 0.125,
# clip_latent_global_means, normalize_big_skip, isotropic noise (noise_embed_dim
# 32), shared global-mean removal appended as input, the dry-air/moisture
# correctors, EnsembleLoss with crps_weight 0.9 + energy_score_weight 0.1,
# n_ensemble 2, unweighted per-variable loss, lr 1e-4, EMA decay 0.999,
# validate_using_ema, seed 0, and residual_prediction: false. It differs in
# exactly three things, two of them unavoidable for an ERA5-only control:
#   1. normalization statistics are ERA5-only, not the pooled SHiELD+ERA5 file;
#   2. single-step objective throughout, where the joint arms switch to the
#      stochastic multi-step distribution at epoch 121;
#   3. ocean.interpolate is false (its training default), true on the joint arms.
# The stepper config travels in the checkpoint, so (3) is applied exactly as the
# reference trained -- donor-consistent by construction, and stated as a caveat
# rather than silently matched to the joint arms.
#
# Its matched residual control, `train-4deg-daily-v2-era5-only-resume400k`
# (wandb oshj5u79), is run too: residual_prediction is the one recipe axis that
# is known to couple global-mean temperature to the sea-surface forcing, so it
# bounds how much of the reference's response is that axis rather than the data.
#
# Two checkpoints per reference, mirroring the two waves the report already
# carries:
#   - best_inference_ckpt.tar : the trainer's own selection, as the joint arms
#     are read. NOTE the criterion differs -- these trainings' inline inference
#     evaluations are ERA5 rollouts, so selection is on ERA5 skill, where the
#     joint arms select on the weight-1.0 slab-ocean CO2-response evaluations.
#   - ema_ckpt_0240.tar : epoch-matched to the report's epoch-240 wave. Both
#     references trained to 259 epochs with EMA checkpoints every 5, so epoch 240
#     exists for both.
#
# The rollout configs are the 2026-07-29 wave's `nolabel` files, reused
# unmodified (`../2026-07-29-combined-shield-era5-best-inference-evals/`), so
# store, initial condition, 1460 daily steps and the constant ocean-masked 4 K
# perturbation are byte-identical to what every joint arm ran. Only the mounted
# checkpoint differs.
#
# research: tasks/2026-07-08-combined-shield-era5-training.md
#           investigations/2026-07-16-combined-shield-era5-training.md
#           report review: ai2cm/reports#60
#
# Usage (run FROM this configs directory):
#   ./run-era5-only-reference-evals.sh                # all 8 jobs
#   ./run-era5-only-reference-evals.sh nores-bi       # one reference+checkpoint
#   ./run-era5-only-reference-evals.sh p4k            # substring OR filter

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

JOB_GROUP="era5-only-reference-plus4k"

# The rollout configs live in the wave this one references, not beside this
# script, so that they are provably the same files the joint arms ran.
CONFIG_DIR="configs/experiments/2026-07-29-combined-shield-era5-best-inference-evals"

# reference -> beaker result dataset of its training run
declare -A CKPT_DATASET=(
  [nores]="01KWEJ1A9X2P3HQEGHGZYAMC4F"   # wandb znnaox7t, no residual  (closest)
  [res]="01KWEA26DSV8TVT6CYT9QZ0Y8H"     # wandb oshj5u79, residual     (its control)
)
# checkpoint tag -> file inside training_checkpoints/
declare -A CKPT_FILE=(
  [bi]="best_inference_ckpt.tar"
  [ep240]="ema_ckpt_0240.tar"
)

cd "$REPO_ROOT"

for pert in p0k p4k; do
  python -m fme.ace.validate_config --config_type inference \
    "$CONFIG_DIR/ace-inference-era5-$pert-nolabel.yaml"
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

launch () {   # launch <job_name> <ckpt_dataset> <ckpt_file> <entrypoint...>
  local JOB_NAME=$1 CKPT_DATASET=$2 CKPT=$3; shift 3
  should_run "$JOB_NAME" || { echo "skip (filter): $JOB_NAME"; return 0; }
  echo "launching: $JOB_NAME"
  gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description 'ERA5-only reference +4K response, closest recipe to the combined SHiELD+ERA5 arms' \
    --env WANDB_NAME="$JOB_NAME" \
    "${gantry_common[@]}" \
    --dataset "$CKPT_DATASET":"training_checkpoints/$CKPT":/ckpt.tar \
    -- "$@"
}

for ref in nores res; do
  for tag in bi ep240; do
    for pert in p0k p4k; do
      launch "era5onlyref-$ref-$tag-$pert" "${CKPT_DATASET[$ref]}" "${CKPT_FILE[$tag]}" \
        python -I -m fme.ace.inference "$CONFIG_DIR/ace-inference-era5-$pert-nolabel.yaml"
    done
  done
done

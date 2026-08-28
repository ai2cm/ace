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

# Cluster selection. The global batch size is a property of the config -- fme splits it
# across data-parallel ranks -- so it is held fixed across clusters for training
# consistency, and the rank count is what varies with per-GPU memory: the batch of 8 is
# 2 samples per GPU on titan and 1 on jupiter. Cluster and N_GPUS therefore move together
# and must never be set independently.
CLUSTER=${CLUSTER:-titan}
case "$CLUSTER" in
  titan)   BEAKER_CLUSTER="ai2/titan";   N_GPUS=4 ;;
  jupiter) BEAKER_CLUSTER="ai2/jupiter"; N_GPUS=8 ;;
  *)
    echo "ERROR: CLUSTER must be 'titan' or 'jupiter', got '$CLUSTER'." >&2
    exit 1
    ;;
esac

# Preemption guarantee. --preemptible/--not-preemptible are deprecated, and since the
# 2026-08-28 scheduler change priority no longer buys protection either; --min-runtime is
# the lever that remains.
MIN_RUNTIME=8h

# Priority orders contention within our own budget -- `high` rather than `urgent` so these
# stay polite to the rest of the group. Set once: it feeds both the gantry flag and the
# CM_PRIORITY env var the job reads, which must not drift apart.
PRIORITY=high

# cwd guard: an empty SCRIPT_PATH means this was run from the repo root, which would make
# CONFIG_PATH absolute and submit a doomed job.
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run this script from configs/experiments/2026-08-28-ace22-variants, not the repo root." >&2
  exit 1
fi

cd "$REPO_ROOT"  # so the config path is valid no matter where this is run from

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local override="${3:-}"          # e.g. "seed=1"; empty for none
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Per-stage beaker args (the donor checkpoint mount) are declared in the config itself
  # as `# arg:` lines, keeping the config and the dataset it needs in one place.
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  # Fail before submit on an unfilled donor placeholder rather than after scheduling.
  if [[ "${extra_args[*]}" == *FILL-AFTER-PREVIOUS-STAGE-SELECTION* ]]; then
    echo "ERROR: $config_filename still has the donor placeholder in its '# arg:' header." >&2
    echo "       Fill in the previous stage's result dataset id before launching." >&2
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
    --description "ACE2.2 variant campaign (harmonized split; near-surface diagnostics) on $CLUSTER" \
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
    --env WANDB_RUN_GROUP=ace22-variants \
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

# Two variants of the 6-hourly ACE2.2 recipe, both on the ACE2.1-harmonized train/val
# split (train 1979-2008, validate 2009-2014, out-of-sample checkpoint selection):
#
#   P1  harmonized split only                  -- 3 seeds; tests whether the shorter,
#                                                 cooler training span costs forced-response
#                                                 skill, and supplies the seed noise floor
#                                                 every other comparison needs.
#   P2  P1 + near-surface fields as stage-3     -- 1 seed; tests whether prognostic
#       secondary diagnostics, a la ACE2.1         near-surface fields were material.
#
# Stage 1 of each is independent. Stages 2-3 run only for the selected stage-1 seed, and
# their `# arg:` headers must be filled with the donor result dataset id first.

p1="train-1deg-6hourly-v2-harmonized-split"
p2="train-1deg-6hourly-v2-harmonized-split-ns-diag"

# --- P1 stage 1: three seeds ---
for SEED in 0 1 2; do
  run_training "$p1.yaml" "ace22-p1-harmonized-1step-rs${SEED}" "seed=${SEED}"
done

# --- P2 stage 1: one seed ---
run_training "$p2.yaml" "ace22-p2-nsdiag-1step-rs0" "seed=0"

# --- Stages 2 and 3 (run after stage-1 seed selection) ---
# Fill the `# arg:` donor id in each config first; the guard above will refuse otherwise.
# Select the P1 seed by best_inference_error, which now scores the out-of-sample
# 2009-2014 window, then:
#
# run_training "$p1-multi-step-ft.yaml" "ace22-p1-harmonized-multistep-rs<N>"
# run_training "$p1-plev-ft.yaml"       "ace22-p1-harmonized-plev-rs<N>"
# run_training "$p2-multi-step-ft.yaml" "ace22-p2-nsdiag-multistep-rs0"
# run_training "$p2-plev-ft.yaml"       "ace22-p2-nsdiag-plev-rs0"

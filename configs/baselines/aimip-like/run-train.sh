#!/bin/bash
#
# Reference ACE training launcher. Canonical source:
#   research/.claude/skills/launching-runs/run-train.reference.sh
#
# Baseline branches ADOPT this by copying it to
# configs/<baseline>/run-train.sh and editing ONLY:
#   (a) the BASELINE-SPECIFIC gantry block inside run_training(), and
#   (b) the run_training() calls at the bottom.
# Keep the GUARDRAILS block verbatim so it does not drift between baselines —
# that is the whole point of having a canonical reference. `git grep` for the
# block markers detects drift. The guardrails are mcgibbon-specific (wandb
# attribution) and deliberately live in research/, NOT in the ace repo.
#
# Usage (run FROM the configs directory that contains this script):
#   ./run-train.sh                  # launch every run_training call below
#   ./run-train.sh no-residual      # launch only calls whose config filename
#                                   # or job name contains "no-residual"
#   ./run-train.sh seed1 seed2      # multiple substrings = OR
#
# The config filter lets you add a new arm to an existing baseline's script
# and launch only that arm, without commenting out / relaunching the runs that
# are already live.

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
  # (Swap fme.ace -> fme.diffusion etc. as the baseline requires.)
  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional gantry flags from "# arg: ..." headers in the YAML.
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  # Optional checkpoint mount (arg 4): "<beaker-result-ds>:/prior-results".
  # The ssrl1-eval configs are checkpoint-agnostic (they reference
  # /prior-results/... in the yaml) and set the mount HERE rather than via a
  # yaml "# arg: --dataset" header, so the two configs stay byte-identical and
  # differ only in which checkpoint this launcher mounts. Older configs that
  # carry their own "# arg: --dataset ...:/prior-results" pass no arg 4.
  local dataset_mount="${4:-}"
  local dataset_args=()
  [[ -n "$dataset_mount" ]] && dataset_args+=(--dataset "$dataset_mount")

  gantry run \
    --name "$job_name" \
    --description 'Run ACE training (AIMIP-like baseline)' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
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
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    "${dataset_args[@]}" \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
  # --- END BASELINE-SPECIFIC --------------------------------------------------
}

# =============================================================================
# v2 ERA5-only residual baseline (canonical).
#
# The selected 4°/daily residual recipe (filter_num_groups 16 × spectral_ratio
# 0.125 "ws64", embed_dim 512; residual-recipe-selection, run j8r0z322), with
# one change from v1: the train_loader splits each ERA5 production stream at the
# canonical in-window stitch boundaries (1986-04-01, 1993-08-01, 2000-01-01,
# 2010-01-01) so no training sample's 1-step residual target straddles a
# stream-to-stream discontinuity. See
# research/knowledge/era5-stitching-discontinuities.md. All weights-affecting
# settings other than the training data window are identical to v1.
# =============================================================================

# --- v2 baseline, seed 0 (1 GPU; Jupiter+Titan, high) ---
# Already launched and running as wandb oshj5u79 / beaker 01KVTHCVW0DB3F4Q8CNTV8ZRB7
# (relaunched 2026-06-23 @ 6e8cf916f). Commented out so this script does not
# relaunch it; uncomment to launch a fresh v2 seed.
# run_training "train-4deg-daily-v2-era5-only.yaml" "train-4deg-daily-v2-era5-only-rs0" 1

# =============================================================================
# v2 ERA5-only NO-RESIDUAL ablation (matched control).
#
# Identical to the v2 baseline above EXCEPT residual_prediction: false — the
# network predicts the full next-state field instead of the delta. Everything
# else (recipe, stitched train window, loss, residual-scaled loss
# normalization, eval methods) is held fixed, isolating the contribution of the
# residual-prediction forward step. Mirrors Jeremy's established matched-control
# methodology (the 2026-06-05 residual-prediction-configuration controls flip
# exactly this one line). Note: the stitched train window was motivated by
# residual targets, so its effect may be smaller here; it is kept fixed to keep
# this a single-variable ablation.
# =============================================================================

# --- v2 no-residual ablation, seed 0 (1 GPU; Jupiter+Titan, high) ---
# Already launched as wandb znnaox7t / beaker 01KVWYW6KKM2K8PY6MWXJVNQ6R; commented out so this script does not relaunch it.
# run_training "train-4deg-daily-v2-era5-only-no-residual.yaml" "train-4deg-daily-v2-era5-only-no-residual-rs0" 1

# =============================================================================
# eswhiten cross with residual-prediction (v2 default weights, NO fd-CRPS).
#
# Energy-score per-sample spectral whitening (energy_score_whitening:
# per_sample, EnergyScoreLoss option from the 2026-06-19 spectral-whitening A/B)
# added to the v2 default loss (crps 0.9 / energy 0.1, no fd-CRPS term). These
# two complete the {residual, no-residual} x {eswhiten, no-eswhiten} 2x2: the
# no-eswhiten cells are the v2 baseline (oshj5u79) and the v2 no-residual
# control (znnaox7t). Isolates the eswhiten effect (the real signal from the
# fd-CRPS weighting investigation: ~-58% h500 spectral bias with no T0 1-step
# cost) within the v2 window, crossed with residual prediction on/off. Each is
# its base config + a single energy_score_whitening: per_sample line.
# =============================================================================

# --- v2 eswhiten (with residual), seed 0 (1 GPU; Jupiter+Titan, high) ---
# Already launched as wandb nfknkl0j / beaker 01KW2NYBYXKQEX7M389NEDW5P7; commented out so this script does not relaunch it.
# run_training "train-4deg-daily-v2-era5-only-eswhiten.yaml" "train-4deg-daily-v2-era5-only-eswhiten-rs0" 1

# --- v2 no-residual eswhiten, seed 0 (1 GPU; Jupiter+Titan, high) ---
# Already launched as wandb 53em1oyx / beaker 01KW2NZ0Z3MJ5QYQKX3QGRWYMR; commented out so this script does not relaunch it.
# run_training "train-4deg-daily-v2-era5-only-no-residual-eswhiten.yaml" "train-4deg-daily-v2-era5-only-no-residual-eswhiten-rs0" 1

# =============================================================================
# Small-scale spectral-calibration arms (research task
# 2026-07-02-small-scale-calibration-four-arms). All are single-knob
# perturbations of the v2 residual baseline (oshj5u79):
#
#   1. eswhiten-gamma0p5 — PARTIAL energy-score whitening (exponent 0.5):
#      full whitening (nfknkl0j) blew up the long rollout with residual ON
#      (stratospheric at0-led, onset ~ep 60-80); gamma=0.5 halves the log-boost
#      of noise-dominated high-l modes. 120 epochs.
#   2. spower-crps — CRPS of the per-degree log spectral power (weight 0.1),
#      a phase-free high-SNR gradient toward correct per-scale amplitude.
#      120 epochs.
#   3. noise-shaping — stochastic spectral output perturbation with learnable
#      per-(channel, degree) amplitude (init 1e-3), a direct parameterization
#      of per-scale stochastic variance. 120 epochs.
#   4. nens4 — n_ensemble 2->4 mechanism test (does more ensemble members'
#      higher-SNR dispersion estimate steepen early small-scale spectral
#      convergence?), truncated to 30 epochs.
# =============================================================================

# --- arm 1: partial whitening gamma=0.5, residual ON, seed 0 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-eswhiten-gamma0p5.yaml" "train-4deg-daily-v2-era5-only-eswhiten-gamma0p5-rs0" 1

# --- arm 2: spectral-power CRPS term, seed 0 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-spower-crps.yaml" "train-4deg-daily-v2-era5-only-spower-crps-rs0" 1

# --- arm 3: learned spectral noise shaping, seed 0 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-noise-shaping.yaml" "train-4deg-daily-v2-era5-only-noise-shaping-rs0" 1

# --- arm 4: n_ensemble 4 truncated SNR test, seed 0 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-nens4.yaml" "train-4deg-daily-v2-era5-only-nens4-rs0" 1

# =============================================================================
# Ensemble-size fine-tuning sweep (2026-07-07): does a larger TRAINING ensemble
# relax the 1-step SSR under-dispersion, and does it converge?
#
# All arms resume the v2 baseline (oshj5u79) epoch-120 checkpoint (mounted at
# /prior-results via each config's `# arg: --dataset ...:/prior-results`) and
# fine-tune 30 epochs (120->150) at constant LR 1e-4, resume_wandb: false (each
# is a fresh wandb run continuing the same weights). Sole change across arms is
# n_ensemble (2 = fine-tuning control, 4, 8, 16); inference: [] since the metric
# (train/val 1-step SSR bias) is logged every epoch from the train/val loops.
# Peak activation ~ batch_size x n_ensemble (baseline 8x2); the 8/16-member arms
# may OOM at batch 8 -> relaunch that arm at a lower batch (comparison is
# per-epoch). Launch with:  ./run-train.sh ftens
# =============================================================================

# --- ftens2: n_ensemble 2 fine-tuning control, seed 0 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-ftens2.yaml" "train-4deg-daily-v2-era5-only-ftens2-rs0" 1

# --- ftens4: n_ensemble 4, seed 0 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-ftens4.yaml" "train-4deg-daily-v2-era5-only-ftens4-rs0" 1

# --- ftens8: n_ensemble 8, seed 0 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-ftens8.yaml" "train-4deg-daily-v2-era5-only-ftens8-rs0" 1

# --- ftens16: n_ensemble 16, seed 0 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-ftens16.yaml" "train-4deg-daily-v2-era5-only-ftens16-rs0" 1

# =============================================================================
# Frozen-eval controls (metric-effect isolation, 2026-07-07): load the v2
# baseline (oshj5u79) epoch-120 weights via parameter_init as a FRESH run
# (epoch 0) and, with evaluate_before_training + max_epochs 0, run ONE
# validation and exit — NO training. Gives val 1-step SSR bias at each N on the
# SAME epoch-120 weights. The frozen val-SSR curve over N is the pure
# evaluation/metric effect; overlaid on the trained ftens arms it separates the
# training effect from the metric. Launch with:  ./run-train.sh ftens-eval
# =============================================================================

# --- frozen eval, n_ensemble 2 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-ftens-eval-n2.yaml" "train-4deg-daily-v2-era5-only-ftens-eval-n2-rs0" 1

# --- frozen eval, n_ensemble 4 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-ftens-eval-n4.yaml" "train-4deg-daily-v2-era5-only-ftens-eval-n4-rs0" 1

# --- frozen eval, n_ensemble 8 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-ftens-eval-n8.yaml" "train-4deg-daily-v2-era5-only-ftens-eval-n8-rs0" 1

# --- frozen eval, n_ensemble 16 (1 GPU) ---
run_training "train-4deg-daily-v2-era5-only-ftens-eval-n16.yaml" "train-4deg-daily-v2-era5-only-ftens-eval-n16-rs0" 1

# --- ftens16 RERUN (expandable_segments allocator fix after OOM at ep134) ---
run_training "train-4deg-daily-v2-era5-only-ftens16-rerun.yaml" "train-4deg-daily-v2-era5-only-ftens16-rerun-rs0" 1

# =============================================================================
# SSR-bias-L1 frozen eval (2026-07-09): show the CRPS-consistent L1 metric
# (ssr_bias_l1) reproduces the L2 val/ensemble/ssr_bias 1-step miscalibration
# signal. Two checkpoints, ONE checkpoint-agnostic config each (byte-identical;
# they differ only in the --dataset checkpoint mounted here at /prior-results).
# n_ensemble=16, n_forward_steps=1, evaluate_before_training + max_epochs 0 ->
# one validation over the train_subset + val loaders, then exit. Both configs
# log ssr_bias AND ssr_bias_l1 on each loader. Launch with:  ./run-train.sh ssrl1
# =============================================================================

# --- baseline v2 (oshj5u79) epoch-120 weights (result ds 01KVYC47QN4PM0MXD4FYPFVWV6) ---
run_training "train-4deg-daily-v2-era5-only-ssrl1-eval-baseline.yaml" \
  "train-4deg-daily-v2-era5-only-ssrl1-eval-baseline-rs0" 1 \
  "01KVYC47QN4PM0MXD4FYPFVWV6:/prior-results"

# --- 2-member finetuned (fautk4uz) epoch-150 weights (result ds 01KWZD5SRC95H55TG8BV55TASA) ---
# Matched-epoch control for ftens16: same 120->150 fine-tune, constant LR 1e-4,
# 2 members per training ensemble. Isolates ensemble size (2 vs 16) from the
# extra 30 epochs the baseline-vs-ftens16 contrast otherwise confounds.
run_training "train-4deg-daily-v2-era5-only-ssrl1-eval-ftens2.yaml" \
  "train-4deg-daily-v2-era5-only-ssrl1-eval-ftens2-rs0" 1 \
  "01KWZD5SRC95H55TG8BV55TASA:/prior-results"

# --- 16-member finetuned (a3uqkhyz) epoch-150 weights (result ds 01KWZN0PF2QPPYF00KWT3J2FJW) ---
run_training "train-4deg-daily-v2-era5-only-ssrl1-eval-ftens16.yaml" \
  "train-4deg-daily-v2-era5-only-ssrl1-eval-ftens16-rs0" 1 \
  "01KWZN0PF2QPPYF00KWT3J2FJW:/prior-results"

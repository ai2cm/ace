#!/bin/bash
# 200 yr piControl and 130 yr 1pctCO2 coupled free rollouts at 1deg from a coupled fine-tune's selected checkpoint
# (best_inference_ckpt), the 1deg twin of ../2026-10-09-4deg-hybrid-ppad/launch-4deg-ppad-coupled-ckpt-evals.sh.
#   ARM=alt CKPT_DS=<dataset holding training_checkpoints/best_inference_ckpt.tar> ./launch-1deg-coupled-ckpt-evals.sh
# A results dataset of a RUNNING job is not mountable: fetch its best_inference_ckpt.tar and upload it as a new dataset.
# ~1.7 s per coupled step on one GPU (inline inference timing): piC ~7 h, 1pct ~4.5 h; no resume, so the min runtime
# covers the whole rollout (8h is the Beaker cap).
set -euo pipefail
ARM="${ARM:?arm name, e.g. alt or alt-replay}"
CKPT_DS="${CKPT_DS:?dataset}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
TAG_SUFFIX="${TAG_SUFFIX:-}"   # e.g. -ep4 when the checkpoint is from a run still training
SCENARIOS="${SCENARIOS:-piC 1pct}"
PRIORITY="${PRIORITY:-high}"
CLUSTERS="${CLUSTERS:-ai2/ceres ai2/jupiter ai2/titan}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for S in $SCENARIOS; do
  total=$((total+1))
  case "$S" in
    piC)  CFG=coupled-evaluator-config-piC-ic0151-200yr-1deg-singleckpt.yaml;  TAG=piC-ic0151-200yr;  MIN_RUNTIME=8h ;;   # Beaker caps min runtime at 8h
    1pct) CFG=coupled-evaluator-config-1pct-ic0001-130yr-1deg-singleckpt.yaml; TAG=1pct-ic0001-130yr; MIN_RUNTIME=8h ;;
    *) echo "unknown scenario $S"; exit 1 ;;
  esac
  JOB="1deg-randco2rs1-fmean-${ARM}${TAG_SUFFIX}-eval-${TAG}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "1deg rs1 random-CO2 ACE2S + ocean_ppad_fmean, ${ARM} coupled FT selected checkpoint: ${TAG} coupled free rollout" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
    $(for C in $CLUSTERS; do printf -- "--cluster %s " "$C"; done) \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=ace1deg-randco2-centennial \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1) || true
  echo "$out" | grep -qm1 "beaker.org/ex/" && { ok=$((ok+1)); echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')"; } || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "coupled checkpoint evals launched: $ok/$total"

#!/bin/bash
# 1deg UFS + ERA5 coupled fine-tune stages (configs in ./coupled from make_configs.py), resources as the 1deg CM4 coupled
# fine-tunes (8 GPUs, one sample per GPU, 840 GiB shared memory). Troy 2026-11-05.
#   ARM=ace2s6h  STAGE=fto ./launch-1deg-ufs-era5-coupled-ft.sh                                  # stage 1: FTO
#   ARM=ace2s6h  STAGE=fto-then-alt OCEAN_DS=<lifted FTO ocean> ./launch-1deg-ufs-era5-coupled-ft.sh   # stage 2: alternating
#   ARM=ace2s512 ...                                                                              # the summer-2026 ACE2S arm
#   OCEAN=ohc ARM=ace2s6h ...                                                                      # corrector-kept UFS ocean (jobs samudra-ufs1deg-era5-ohc-*)
#   OCEAN=ohc ARM=ace2s6h STAGE=jfto ...                     # CM4 1deg two-stage template (James's recipe), stage 1 (Troy 2026-11-10)
#   OCEAN=ohc ARM=ace2s6h STAGE=jfto-then-alt CKPT_DS=<jfto results> ...   # stage 2 from the stage-1 coupled checkpoint (/ckpt.tar)
#   EXISTING_RESULTS_DS=<results dataset> ...  continues a previous submission (the -resume config)
set -euo pipefail
OCEAN="${OCEAN:-noohc}"             # noohc = corrector-free UFS ocean (paper-2 line); ohc = scaled-temperature OHC corrector kept (2026-11-09 cooling test)
ARM="${ARM:?ace2s6h or ace2s512}"
STAGE="${STAGE:?fto, fto-then-alt, fto-mse, fto-mse-then-alt[-mseocean|-mse4], jfto or jfto-then-alt}"
case "$ARM" in
  ace2s6h)  ATMOS_DS_DEFAULT=01M49ERVM1DVWV12G5DRDXM0PC; ATMOS_CKPT_DEFAULT=training_checkpoints/ema_ckpt_0002.tar ;;        # 1deg-6h-ace2s-ft12-detached-rs0 (wandb cerls68n), the ACE2S paper's sixhourly checkpoint
  ace2s512) ATMOS_DS_DEFAULT=01KSVC6YS7C18SGYV4VPZYZ232; ATMOS_CKPT_DEFAULT=training_checkpoints/best_inference_ckpt.tar ;;  # ace2s-era5-multi-step-fine-tuning-no-var-weighting-rs0 (wandb u9ek57d9), the summer-2026 coupled runs' atmosphere
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
ATMOS_DS="${ATMOS_DS:-$ATMOS_DS_DEFAULT}"
ATMOS_CKPT="${ATMOS_CKPT:-$ATMOS_CKPT_DEFAULT}"
case "$OCEAN" in
  noohc) OCEAN_DS_DEFAULT=01M6H7NBRG4W31DHT2MKD4WBWA; OTAG="" ;;      # samudra-ufs1deg-resid-noohc-ppad-sstdiag-fmean-ufsft results (wandb s5j0xr6o)
  ohc)   OCEAN_DS_DEFAULT=01M6HWDXRVCZC1W5HFD5WPM19Z; OTAG="ohc-" ;;  # samudra-ufs1deg-resid-ohc-ppad-sstdiag-fmean-ufsft results (wandb 75s8184h)
  ohc-uhm1p7) OCEAN_DS_DEFAULT=01M6HWDXRVCZC1W5HFD5WPM19Z; OTAG="ohc-uhm1p7-" ;;  # same ocean, corrector constant -1.7 W/m2 during coupled training
  *) echo "unknown OCEAN $OCEAN"; exit 1 ;;
esac
OCEAN_DS="${OCEAN_DS:-$OCEAN_DS_DEFAULT}"                           # stage 2 passes the lifted FTO ocean instead
OCEAN_CKPT="${OCEAN_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_STATS_DS="${OCEAN_STATS_DS:-01M6F9VRGMY0NYX19R37PRTW92}"     # 1deg UFS 2026-10-06 stats + CM4 placeholder entries (coupled stages use UFS ocean + ERA5 atmosphere stats, Troy 2026-10-20)
ATMOS_STATS_DS="${ATMOS_STATS_DS:-01KPC6VHH4H3D800KV4VX83KK9}"     # andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019 (both atmospheres' own stats)
FRAC_DS="${FRAC_DS:?surface-fraction store dataset (era5-1deg-6h-ufs-coupled-fractions-1994-2023)}"
FRAC_ZARR=2026-11-05-era5-1deg-6h-ufs-coupled-fractions-1994-2023.zarr
N_GPUS="${N_GPUS:-8}"
PRIORITY="${PRIORITY:-high}"
MIN_RUNTIME="${MIN_RUNTIME:-6h}"   # guaranteed time before Beaker may preempt (gantry --min-runtime; 1deg template)
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
case "$STAGE" in
  fto) JOB="samudra-ufs1deg-era5-${OTAG}${ARM}-fto" ;;
  fto-then-alt) JOB="samudra-ufs1deg-era5-${OTAG}${ARM}-fto-then-ftalt" ;;
  fto-mse) JOB="samudra-ufs1deg-era5-${OTAG}${ARM}-fto-mse" ;;
  fto-mse-then-alt) JOB="samudra-ufs1deg-era5-${OTAG}${ARM}-fto-mse-then-ftalt" ;;   # CRPS alternating stage from the MSE-loss FTO ocean
  fto-mse-then-alt-mseocean) JOB="samudra-ufs1deg-era5-${OTAG}${ARM}-fto-mse-then-ftalt-mseocean" ;;   # ... with the ocean loss kept on MSE
  fto-mse-then-alt-mse4) JOB="samudra-ufs1deg-era5-${OTAG}${ARM}-fto-mse-then-ftalt-mse4" ;;           # ... ocean MSE on every step of 4-step windows
  jfto) JOB="samudra-ufs1deg-era5-${OTAG}${ARM}-jfto" ;;                        # CM4 1deg two-stage template (James's recipe), stage 1
  jfto-then-alt) JOB="samudra-ufs1deg-era5-${OTAG}${ARM}-jfto-then-ftalt"; : "${CKPT_DS:?stage-2 needs CKPT_DS=<jfto results dataset>}" ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
JOB="${JOB}${TAG:+-$TAG}"
CFG="${SCRIPT_PATH}/coupled/ufs1deg-${OCEAN}-${ARM}-${STAGE}${EXISTING_RESULTS_DS:+-resume}.yaml"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "1deg UFS + ERA5 coupled FT, stage ${STAGE}, ocean ${OCEAN}, atmosphere ${ARM} (${ATMOS_DS}:${ATMOS_CKPT}), ocean ${OCEAN_DS}:${OCEAN_CKPT}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  $(for C in $CLUSTERS; do printf -- "--cluster %s " "$C"; done) \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-ufs1deg-era5-coupled-ft \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${OCEAN_STATS_DS}:/ocean_stats" \
  --dataset "${ATMOS_STATS_DS}:/atmos_stats" \
  --dataset "${OCEAN_DS}:${OCEAN_CKPT}:/ocean_ckpt.tar" \
  --dataset "${ATMOS_DS}:${ATMOS_CKPT}:/atmos_ckpt.tar" \
  --dataset "${FRAC_DS}:/era5frac/${FRAC_ZARR}" \
  ${CKPT_DS:+--dataset "${CKPT_DS}:${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}:/ckpt.tar"} \
  ${EXISTING_RESULTS_DS:+--dataset "${EXISTING_RESULTS_DS}:/existing-results"} \
  --gpus "$N_GPUS" --shared-memory 840GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node="${N_GPUS}" -m fme.coupled.train "${CFG}" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }

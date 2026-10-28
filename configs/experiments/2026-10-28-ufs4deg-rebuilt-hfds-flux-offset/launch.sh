#!/bin/bash
# "Throw everything at the corrector" test at 4deg: the pretrain-vs-scratch pretrained arm (CM4 pretrain -> UFS fine-tune ->
# alternating coupled FT with ERA5; diagnosed SST, heat-content corrector with zero unaccounted heating) retrained on the
# rebuilt-flux UFS store (hfds/hfds_total_area = the FV3-component rebuild, not MOM6 hfds), then run 30 years with and without
# a -2.7 W/m2 inference-only flux offset. Training configs are the pretrain-vs-scratch ones, unchanged; only the store and
# statistics differ. Job names carry "rbh" (rebuilt hfds).
#   STAGE=ftcm4 ./launch.sh                                                   # UFS fine-tune from the CM4 pretrain (1 GPU, 60 ep)
#   STAGE=coupled OCEAN_CKPT_DS=<ufs results ds> ./launch.sh                  # alternating coupled FT with ERA5 (2 GPUs, 12 ep)
#   STAGE=eval EVAL=30yr|30yr-offset|hindcast CKPT_DS=<coupled results ds> [ARM=rbh] ./launch.sh
#   ARM=pvs UFS_DS=01M4E83P5GMXQEN2JN8SZFB5BN STAGE=eval EVAL=30yr-offset CKPT_DS=01M60XHGS7M66WX9H487MY5FNN ./launch.sh
#     (the offset alone on the existing MOM6-flux pretrained arm)
set -euo pipefail
STAGE="${STAGE:?ftcm4 | coupled | eval}"
ARM="${ARM:-rbh}"
MIN_RUNTIME="${MIN_RUNTIME:-3h}"   # team template: 4deg 3h (evals 1h)
PRIORITY="${PRIORITY:-high}"
UFS_DS="${UFS_DS:-01M5ZPJE5XF3F84CAMZRKCSKK1}"                  # rebuilt-flux twin of ufs-replay-ocean-4deg-19level-5day-2026-10-06-cm4vars-uh
UFS_STATS_DS="${UFS_STATS_DS:-01M5ZPWQV6F3KRP8KNV5H424YS}"      # its statistics (rebuilt hfds entries)
CM4_CKPT_DS="${CM4_CKPT_DS:-01M5YWBS8G629G550W6F9P9VCP}"        # CM4 4deg pretrain of the recipe (samudra-ppad4deg-resid-ohc-ppad-sstdiag-fmean)
ERA5_DS="${ERA5_DS:-01M4W630AM1HQRYKJ2AJ0JMNN1}"                # era5-4deg-8layer-daily-ufs-coupled-1994-2023-v2
ERA5_ZARR=2026-10-14-era5-4deg-8layer-daily-ufs-coupled-1994-2023.zarr
ATMOS_CKPT_DS="${ATMOS_CKPT_DS:-01M39XBD8K0KB3GAQ79ETVN470}"    # 4deg-daily-ace2s-ft3-detached-rs0 (Jeremy, wandb 1r7oyu58)
CKPT_FILE=training_checkpoints/best_inference_ckpt.tar
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
PVS_PATH=configs/experiments/2026-10-26-ufs4deg-pretrain-vs-scratch
COUPLING_PATH=configs/experiments/2026-10-14-ufs4deg-era5-coupling
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
UFS_MOUNT="${UFS_DS}:/ufs4deg/2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr"
case "$STAGE" in
  ftcm4)
    JOB=samudra-ufs4deg-rbh-ftcm4; DESC="UFS 4deg fine-tune from the CM4 pretrain on the rebuilt-flux store (diagnosed SST, corrector)"
    GPUS=1; SHM=200GiB; GROUP=samudra-ufs4deg-rebuilt-hfds; TYPE=training
    MOUNTS=(--dataset "${UFS_STATS_DS}:/ocean_stats" --dataset "$UFS_MOUNT" --dataset "${CM4_CKPT_DS}:${CKPT_FILE}:/ckpt.tar")
    CMD=(torchrun --nproc_per_node=1 -m fme.ace.train "${PVS_PATH}/ufs-ftcm4.yaml") ;;
  coupled)
    JOB=samudra-ufs4deg-era5-rbh-ftcm4-coupledft-spb; DESC="alternating coupled FT with the ERA5 4deg ACE2S of the rebuilt-flux UFS ocean"
    GPUS=2; SHM=200GiB; GROUP=samudra-ufs4deg-era5-coupled-ft; TYPE=training
    MOUNTS=(--dataset "${UFS_STATS_DS}:/ocean_stats" --dataset "$UFS_MOUNT" --dataset "${ERA5_DS}:/era5/${ERA5_ZARR}"
            --dataset "${OCEAN_CKPT_DS:?UFS ocean results dataset}:${CKPT_FILE}:/ocean_ckpt.tar" --dataset "${ATMOS_CKPT_DS}:${CKPT_FILE}:/atmos_ckpt.tar")
    CMD=(torchrun --nproc_per_node=2 -m fme.coupled.train "${PVS_PATH}/coupled/ftcm4-era5-coupledft-spb.yaml") ;;
  eval)
    EVAL="${EVAL:?30yr | 30yr-offset | hindcast}"; MIN_RUNTIME="${MIN_RUNTIME_EVAL:-1h}"
    case "$EVAL" in
      30yr) CFG="${SCRIPT_PATH}/coupled-evaluator-30yr-1994-deep.yaml"; NAME="${ARM}-ftcm4-30yr-1994-deep" ;;
      30yr-offset) CFG="${SCRIPT_PATH}/coupled-evaluator-30yr-1994-deep-offset2p7.yaml"; NAME="${ARM}-ftcm4-30yr-1994-deep-offset2p7" ;;
      hindcast) CFG="${COUPLING_PATH}/coupled-evaluator-config-ufs4deg-era5-hindcast-2002-2011-singleckpt.yaml"; NAME="${ARM}-ftcm4-hindcast-2002-2011" ;;
      *) echo "EVAL must be 30yr, 30yr-offset or hindcast"; exit 1 ;;
    esac
    JOB="samudra-ufs4deg-era5-coupled-${NAME}"; DESC="coupled evaluator ${NAME} (rebuilt-flux / flux-offset test)"
    GPUS=1; SHM=100GiB; GROUP=samudra-ufs4deg-era5-coupled; TYPE=inference
    MOUNTS=(--dataset "$UFS_MOUNT" --dataset "${ERA5_DS}:/era5/${ERA5_ZARR}" --dataset "${CKPT_DS:?coupled FT results dataset}:${CKPT_FILE}:/ckpt.tar")
    CMD=(python -I -m fme.coupled.evaluator "$CFG") ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
out=$(gantry run --name "$JOB" --task-name "$JOB" --description "$DESC" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" --env WANDB_JOB_TYPE="$TYPE" --env WANDB_RUN_GROUP="$GROUP" \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  "${MOUNTS[@]}" \
  --gpus "$GPUS" --shared-memory "$SHM" --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- "${CMD[@]}" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }

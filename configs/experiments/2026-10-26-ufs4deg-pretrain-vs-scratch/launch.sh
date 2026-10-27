#!/bin/bash
# Pretrain-vs-scratch test at 4deg (see make_configs.py). One launcher, one STAGE per call; every stage reuses the program's
# datasets. Job names carry "pvs" (pretrain vs scratch).
#   STAGE=cm4 ./launch.sh                                    # CM4 4deg pretrain of the recipe (1 GPU, 75 ep)
#   STAGE=scratch ./launch.sh                                # UFS from scratch (1 GPU, 200 ep)
#   STAGE=ftcm4 CKPT_DS=<cm4 results ds> ./launch.sh         # UFS fine-tune from the CM4 pretrain (1 GPU, 60 ep)
#   STAGE=coupled ARM=scratch|ftcm4 OCEAN_CKPT_DS=<ufs results ds> ./launch.sh      # alternating coupled FT with ERA5 (2 GPUs, 12 ep)
#   STAGE=eval ARM=scratch|ftcm4 EVAL=30yr|hindcast CKPT_DS=<coupled results ds> ./launch.sh   # 30-yr rollout / 120 hindcasts (1 GPU)
set -euo pipefail
STAGE="${STAGE:?cm4 | scratch | ftcm4 | coupled | eval}"
MIN_RUNTIME="${MIN_RUNTIME:-3h}"   # team template: 4deg 3h (evals 1h via EVAL)
PRIORITY="${PRIORITY:-high}"
CM4_STATS_DS="${CM4_STATS_DS:-01KY8D816E67NXHP2EB29FHBJM}"      # elynn/2026-07-22-cm4-1pctco2-4deg-coupled-stats (ocean/), pretraining only
UFS_STATS_DS="${UFS_STATS_DS:-01M5D5GG2FB3MWWDD5GGSH8QR2}"      # UFS 4deg 2026-10-06 stats (+ uh + CM4 placeholder entries), every UFS stage
UFS_DS="${UFS_DS:-01M4E83P5GMXQEN2JN8SZFB5BN}"                  # ufs-replay-ocean-4deg-19level-5day-2026-10-06-cm4vars-uh
#   rebuilt-flux twins (hfds/hfds_total_area = the corrector's FV3-component rebuild instead of MOM6 hfds, 2026-10-27):
#   UFS_DS=01M5ZPJE5XF3F84CAMZRKCSKK1 UFS_STATS_DS=01M5ZPWQV6F3KRP8KNV5H424YS  (both arms of a test must use the same pair)
ERA5_DS="${ERA5_DS:-01M4W630AM1HQRYKJ2AJ0JMNN1}"                # era5-4deg-8layer-daily-ufs-coupled-1994-2023-v2
ERA5_ZARR=2026-10-14-era5-4deg-8layer-daily-ufs-coupled-1994-2023.zarr
ATMOS_CKPT_DS="${ATMOS_CKPT_DS:-01M39XBD8K0KB3GAQ79ETVN470}"    # 4deg-daily-ace2s-ft3-detached-rs0 (Jeremy, wandb 1r7oyu58)
CKPT_FILE=training_checkpoints/best_inference_ckpt.tar
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
COUPLING_PATH=configs/experiments/2026-10-14-ufs4deg-era5-coupling
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
UFS_MOUNT="${UFS_DS}:/ufs4deg/2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr"
case "$STAGE" in
  cm4)
    JOB=samudra-ppad4deg-resid-ohc-ppad-sstdiag-fmean; DESC="CM4 4deg ppad pretrain of the paper-1 hybrid recipe (pretrain-vs-scratch test)"
    GPUS=1; SHM=200GiB; GROUP=samudra-ppad-4deg; TYPE=training
    MOUNTS=(--dataset "${CM4_STATS_DS}:ocean:/ocean_stats")
    CMD=(torchrun --nproc_per_node=1 -m fme.ace.train "${SCRIPT_PATH}/cm4-pretrain.yaml") ;;
  scratch)
    JOB=samudra-ufs4deg-pvs-scratch; DESC="UFS 4deg from scratch, paper-1 hybrid recipe (pretrain-vs-scratch test)"
    GPUS=1; SHM=200GiB; GROUP=samudra-ufs4deg-pretrain-vs-scratch; TYPE=training
    MOUNTS=(--dataset "${UFS_STATS_DS}:/ocean_stats" --dataset "$UFS_MOUNT")
    CMD=(torchrun --nproc_per_node=1 -m fme.ace.train "${SCRIPT_PATH}/ufs-scratch.yaml") ;;
  ftcm4)
    JOB=samudra-ufs4deg-pvs-ftcm4; DESC="UFS 4deg fine-tune from the CM4 pretrain, paper-1 hybrid recipe (pretrain-vs-scratch test)"
    GPUS=1; SHM=200GiB; GROUP=samudra-ufs4deg-pretrain-vs-scratch; TYPE=training
    MOUNTS=(--dataset "${UFS_STATS_DS}:/ocean_stats" --dataset "$UFS_MOUNT" --dataset "${CKPT_DS:?CM4 pretrain results dataset}:${CKPT_FILE}:/ckpt.tar")
    CMD=(torchrun --nproc_per_node=1 -m fme.ace.train "${SCRIPT_PATH}/ufs-ftcm4.yaml") ;;
  coupled)
    ARM="${ARM:?scratch | ftcm4}"; JOB="samudra-ufs4deg-era5-pvs-${ARM}-coupledft-spb"; DESC="alternating coupled FT with the ERA5 4deg ACE2S of the ${ARM} UFS ocean (pretrain-vs-scratch test)"
    GPUS=2; SHM=200GiB; GROUP=samudra-ufs4deg-era5-coupled-ft; TYPE=training
    MOUNTS=(--dataset "${UFS_STATS_DS}:/ocean_stats" --dataset "$UFS_MOUNT" --dataset "${ERA5_DS}:/era5/${ERA5_ZARR}"
            --dataset "${OCEAN_CKPT_DS:?UFS ocean results dataset}:${CKPT_FILE}:/ocean_ckpt.tar" --dataset "${ATMOS_CKPT_DS}:${CKPT_FILE}:/atmos_ckpt.tar")
    CMD=(torchrun --nproc_per_node=2 -m fme.coupled.train "${SCRIPT_PATH}/coupled/${ARM}-era5-coupledft-spb.yaml") ;;
  eval)
    ARM="${ARM:?scratch | ftcm4}"; EVAL="${EVAL:?30yr | hindcast}"; MIN_RUNTIME="${MIN_RUNTIME_EVAL:-1h}"
    case "$EVAL" in
      30yr) CFG=coupled-evaluator-config-ufs4deg-era5-30yr-1994-singleckpt.yaml; NAME="pvs-${ARM}-30yr-1994" ;;
      hindcast) CFG=coupled-evaluator-config-ufs4deg-era5-hindcast-2002-2011-singleckpt.yaml; NAME="pvs-${ARM}-hindcast-2002-2011" ;;
      *) echo "EVAL must be 30yr or hindcast"; exit 1 ;;
    esac
    JOB="samudra-ufs4deg-era5-coupled-${NAME}"; DESC="coupled evaluator ${NAME} (pretrain-vs-scratch test)"
    GPUS=1; SHM=100GiB; GROUP=samudra-ufs4deg-era5-coupled; TYPE=inference
    MOUNTS=(--dataset "$UFS_MOUNT" --dataset "${ERA5_DS}:/era5/${ERA5_ZARR}" --dataset "${CKPT_DS:?coupled FT results dataset}:${CKPT_FILE}:/ckpt.tar")
    CMD=(python -I -m fme.coupled.evaluator "${COUPLING_PATH}/${CFG}") ;;
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

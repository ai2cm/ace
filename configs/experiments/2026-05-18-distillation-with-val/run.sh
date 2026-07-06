#!/bin/bash
# Distillation training with BestStudentCheckpointCallback enabled.
#
# Saves best_student.ckpt (ACE format, by validation CRPS) into /results so
# it is captured as a Beaker dataset artifact alongside the raw .pth files.
#
# Usage: ./run.sh <method> [--suffix <variant>] [--moe-teacher] \
#            [--expert <0|1>] [--student-steps <N>]
#   method:      dmd2 | fdistill | scm
#   --suffix:    optional training-variant tag appended to JOB_NAME, e.g.
#                "1step" → ace-downscaling-distillation-fdistill-with-val-1step.
#                Useful when running multiple variants in parallel so each gets
#                a distinct wandb run name.
#   --moe-teacher: use the bundled multivariate MoE checkpoint
#                (01KTCHVDHY0SATWH9E0AW2PDS6 / bundled_moe_multivariate.ckpt)
#                as the teacher instead of the default single-model checkpoint.
#                Note: DMD2 + MoE teacher requires additional discriminator
#                wiring via the primary expert; fdistill/scm are fully supported.
#   --expert:    per-expert distillation (requires --moe-teacher). Distil a
#                single expert in-domain over its own sigma range, no dispatch:
#                  0 = low-noise  Student-Lo, sigma [0.005, 200], val=lo_renoise
#                  1 = high-noise Student-Hi, sigma [200, 2000], val=hi_cascade
#                Expert 1 validates end-to-end through a frozen Lo student, so it
#                requires --frozen-lo <dataset> (train Lo first). The sigma=200
#                boundary is taken from the Hi student's own sigma_min.
#   --frozen-lo: Beaker dataset id holding the trained Student-Lo checkpoint,
#                mounted at /frozen_lo for expert-1 hi_cascade validation.
#   --frozen-lo-path: filename of the Lo checkpoint within --frozen-lo
#                (default best_student.ckpt).
#   --student-steps: override ACE_STUDENT_STEPS (student denoising steps).
#                Lo: try 1 vs 2; Hi: 1.
#   --gan-r1:    R1 discriminator regularization weight (ACE_GAN_R1_REG_WEIGHT;
#                fdistill only). Off by default; the standard fix for the
#                disc-winning PRMSL spectral collapse (see MOE_DISTILLATION_STATUS.md).
#   --gan-weight: generator GAN loss weight (ACE_GAN_LOSS_WEIGHT_GEN; default 1e-3).
#                Lower to let forward-KL carry more of the signal.
#   --lr-decay-steps: linearly decay all three LRs to ~5% over N iters and cap
#                max_iter at N (ACE_LR_DECAY_STEPS). 0/unset = constant LR.
#   --disc-feature-depth: encoder level the GAN discriminator taps, as an offset
#                toward finer resolution from the deepest/bottleneck level
#                (ACE_DISC_FEATURE_DEPTH; default 0 = bottleneck/coarsest). Raise
#                to move the policed spectral band finer (candidate fix for the
#                coarse-PRMSL GAN damage). Resolved (res, channels) print at launch.

set -e

METHOD="${1:?Usage: $0 <dmd2|fdistill|scm> [--suffix <variant>] [--moe-teacher] [--expert <0|1>] [--student-steps <N>]}"
shift

SUFFIX=""
MOE_TEACHER=false
EXPERT=""
STUDENT_STEPS=""
GAN_R1=""
GAN_WEIGHT=""
LR_DECAY_STEPS=""
DISC_FEATURE_DEPTH=""
FROZEN_LO_DATASET=""
FROZEN_LO_PATH="best_student.ckpt"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suffix)
            SUFFIX="${2:?--suffix requires a value}"
            shift 2
            ;;
        --moe-teacher)
            MOE_TEACHER=true
            shift
            ;;
        --expert)
            EXPERT="${2:?--expert requires 0 or 1}"
            shift 2
            ;;
        --frozen-lo)
            FROZEN_LO_DATASET="${2:?--frozen-lo requires a Beaker dataset id}"
            shift 2
            ;;
        --frozen-lo-path)
            FROZEN_LO_PATH="${2:?--frozen-lo-path requires a value}"
            shift 2
            ;;
        --student-steps)
            STUDENT_STEPS="${2:?--student-steps requires a value}"
            shift 2
            ;;
        --gan-r1)
            GAN_R1="${2:?--gan-r1 requires a value}"
            shift 2
            ;;
        --gan-weight)
            GAN_WEIGHT="${2:?--gan-weight requires a value}"
            shift 2
            ;;
        --lr-decay-steps)
            LR_DECAY_STEPS="${2:?--lr-decay-steps requires a value}"
            shift 2
            ;;
        --disc-feature-depth)
            DISC_FEATURE_DEPTH="${2:?--disc-feature-depth requires a value}"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$EXPERT" && "$MOE_TEACHER" != "true" ]]; then
    echo "--expert requires --moe-teacher" >&2
    exit 1
fi

case "$METHOD" in
    dmd2)
        CONFIG="fme/downscaling/distillation/configs/dmd2_baseline_spike.py"
        DESCRIPTION="ACE downscaling DMD2 distillation with val CRPS checkpoint"
        ;;
    fdistill)
        CONFIG="fme/downscaling/distillation/configs/fdistill_kl_spike.py"
        DESCRIPTION="ACE downscaling f-distill (forward-KL) with val CRPS checkpoint"
        ;;
    scm)
        CONFIG="fme/downscaling/distillation/configs/scm_spike.py"
        DESCRIPTION="ACE downscaling sCM distillation with val CRPS checkpoint"
        ;;
    *)
        echo "Unknown method: $METHOD. Choose from: dmd2, fdistill, scm" >&2
        exit 1
        ;;
esac

JOB_NAME="ace-downscaling-distillation-${METHOD}-with-val${SUFFIX:+-${SUFFIX}}"

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
# Relative path from repo root to this script's directory — stable regardless
# of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR#$REPO_ROOT/}"

cd $REPO_ROOT

NGPU=4
IMAGE="$(cat $REPO_ROOT/latest_distillation_image.txt)"

if [[ "$MOE_TEACHER" == "true" ]]; then
    # Bundled multivariate MoE teacher: precip + winds + pressure.
    TEACHER_DATASET=01KTCHVDHY0SATWH9E0AW2PDS6
    TEACHER_CKPT_FLAG="--teacher-moe-checkpoint /checkpoints/bundled_moe_multivariate.ckpt"
    JOB_NAME="${JOB_NAME}-moe-teacher"
    VAL_DATASET=/climate-default/2026-06-09-distillation-teacher-moe-multivar-val-dataset/conus_multivar_val_2023.zarr
    # MoE teacher training parameters: 4 output variables (u10/v10/PRMSL/
    # PRATEsfc), sigma ~ loguniform on [0.005, 2000], generated with 18
    # diffusion steps.  The spike configs read the ACE_* env vars.
    #
    # The MoE teacher has two experts split at sigma=200: a low-noise expert
    # [0.005, 200] and a high-noise expert [200, 2000].  ACE_SIGMA_MAX must be
    # the full schedule maximum (2000), not the expert boundary (200) -- the
    # student's inference sigma grid spans the full range, so its first
    # generation step starts near sigma=2000 and must be trained there.
    TEACHER_NUM_STEPS=18
    TEACHER_ENV_FLAGS=(
        --env ACE_C_OUT=4
        --env ACE_NOISE_DIST=loguniform
        --env ACE_SIGMA_MIN=0.005
        --env ACE_SIGMA_MAX=2000.0
    )

    # Per-expert distillation: restrict the teacher to one expert over its own
    # sigma range (no dispatch) and pick the matching validation mode.  The
    # boundaries match the bundled MoE teacher (expert 0 [0.005, 200], expert 1
    # [200, 2000]); ACE_SIGMA_MIN/MAX must equal the chosen expert's range so
    # training noise is sampled in-domain.
    if [[ -n "$EXPERT" ]]; then
        case "$EXPERT" in
            0) E_SIGMA_MIN=0.005; E_SIGMA_MAX=200.0;  E_VAL_MODE=lo_renoise ;;
            1) E_SIGMA_MIN=200.0; E_SIGMA_MAX=2000.0; E_VAL_MODE=hi_cascade
               if [[ -z "$FROZEN_LO_DATASET" ]]; then
                   echo "expert 1 (Student-Hi) uses hi_cascade validation, which "\
                        "requires a frozen Lo student: pass --frozen-lo "\
                        "<beaker-dataset-id> (the dataset holding the trained "\
                        "Student-Lo best_student.ckpt). Train Lo (expert 0) "\
                        "first." >&2
                   exit 1
               fi ;;
            *) echo "--expert must be 0 or 1, got $EXPERT" >&2; exit 1 ;;
        esac
        TEACHER_ENV_FLAGS=(
            --env ACE_C_OUT=4
            --env ACE_NOISE_DIST=loguniform
            --env ACE_SIGMA_MIN=$E_SIGMA_MIN
            --env ACE_SIGMA_MAX=$E_SIGMA_MAX
            --env ACE_EXPERT_INDEX=$EXPERT
            --env ACE_VAL_MODE=$E_VAL_MODE
        )
        if [[ "$EXPERT" == "1" ]]; then
            # hi_cascade cascades the trained Hi student through the frozen Lo
            # mounted at /frozen_lo (see the --dataset flag below).  The segment
            # boundary (sigma=200) is taken from the Hi student's own sigma_min,
            # so ACE_FROZEN_LO_SIGMA_MIN only sets the low end of the Lo segment.
            TEACHER_ENV_FLAGS+=(
                --env ACE_FROZEN_LO_CKPT=/frozen_lo/$FROZEN_LO_PATH
                --env ACE_FROZEN_LO_STEPS=2
                --env ACE_FROZEN_LO_SIGMA_MIN=0.005
            )
        fi
        JOB_NAME="${JOB_NAME}-expert${EXPERT}"
    fi
else
    # Default single-model teacher, trained with sigma ~ lognormal(-1.2, 1.8)
    # on [0.002, 150] (the spike config defaults).
    TEACHER_DATASET=01KNM6H3JB1ZNS76HX17AAZRF7:checkpoints
    TEACHER_CKPT_FLAG="--teacher-checkpoint /checkpoints/best_histogram_tail.ckpt"
    VAL_DATASET=/climate-default/2026-04-29-distillation-teacher-val-dataset/conus_val_2023.zarr
    TEACHER_NUM_STEPS=15
    TEACHER_ENV_FLAGS=()
fi

# Optional override of the student denoising-step count (read by the spike
# config as ACE_STUDENT_STEPS; drives both training and validation sampling).
if [[ -n "$STUDENT_STEPS" ]]; then
    TEACHER_ENV_FLAGS+=(--env ACE_STUDENT_STEPS=$STUDENT_STEPS)
fi

# Optional GAN-stabilizer knobs (fdistill): R1 reg, generator GAN weight, LR
# decay.  Read by fdistill_kl_spike.py as ACE_GAN_R1_REG_WEIGHT /
# ACE_GAN_LOSS_WEIGHT_GEN / ACE_LR_DECAY_STEPS.
if [[ -n "$GAN_R1" ]]; then
    TEACHER_ENV_FLAGS+=(--env ACE_GAN_R1_REG_WEIGHT=$GAN_R1)
fi
if [[ -n "$GAN_WEIGHT" ]]; then
    TEACHER_ENV_FLAGS+=(--env ACE_GAN_LOSS_WEIGHT_GEN=$GAN_WEIGHT)
fi
if [[ -n "$LR_DECAY_STEPS" ]]; then
    TEACHER_ENV_FLAGS+=(--env ACE_LR_DECAY_STEPS=$LR_DECAY_STEPS)
fi

# Optional discriminator tap depth (read by fastgen_train as
# ACE_DISC_FEATURE_DEPTH): offset toward finer resolution from the bottleneck.
if [[ -n "$DISC_FEATURE_DEPTH" ]]; then
    TEACHER_ENV_FLAGS+=(--env ACE_DISC_FEATURE_DEPTH=$DISC_FEATURE_DEPTH)
fi

# Frozen Lo student for hi_cascade validation (expert 1): mount the dataset
# holding its best_student.ckpt at /frozen_lo.
EXTRA_DATASET_FLAGS=()
if [[ -n "$FROZEN_LO_DATASET" ]]; then
    EXTRA_DATASET_FLAGS+=(--dataset $FROZEN_LO_DATASET:/frozen_lo)
fi

gantry run \
    --name $JOB_NAME \
    --description "$DESCRIPTION" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_JOB_TYPE=distillation \
    --env FASTGEN_OUTPUT_ROOT=/results \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    "${TEACHER_ENV_FLAGS[@]}" \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $TEACHER_DATASET:/checkpoints \
    "${EXTRA_DATASET_FLAGS[@]}" \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 100GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc-per-node $NGPU -m fme.downscaling.distillation.fastgen_train \
        --config $CONFIG \
        $TEACHER_CKPT_FLAG \
        --teacher-num-steps $TEACHER_NUM_STEPS \
        --data-yaml $SCRIPT_PATH/data-config.yaml \
        --val-dataset $VAL_DATASET \
        --val-data-yaml $SCRIPT_PATH/val-data-config.yaml \
        - log_config.name=$JOB_NAME

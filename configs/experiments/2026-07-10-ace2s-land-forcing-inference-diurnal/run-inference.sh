#!/bin/bash
# ACE2S land-forcing short-lead evaluation — DIURNAL-SAMPLING variant of
# 2026-07-09-ace2s-land-forcing-inference. Identical checkpoints/variables/settings; the only
# change is the IC sampling: the interval is NOT a multiple of 4 steps, so the 300 ICs' init hours
# rotate uniformly through 00/06/12/18 UTC (75 each). This makes the first predicted step sample
# all local solar times, removing the fixed-UTC terminator/low-sun bias in albedo & SW fluxes that
# contaminated the 07-09 run and making ERA5 vs CM4 directly comparable. Writes to a separate GCS
# prefix so the 07-09 outputs are preserved for comparison.
# Each run is a single-GPU `fme.ace.evaluator` job that rolls the model 40 steps (~10 days) from
# 100 ICs, computes near-surface skill vs the target, and writes paired prediction+target fields to
# a GCS zarr. Two dataset configs (era5.yaml, cm4.yaml) are reused across checkpoints, each split
# into 3 interleaved 100-IC chunks (300 total/checkpoint); checkpoint mounted at /ckpt.tar,
# per-run experiment_dir + IC `first` set via --override.
#
# Usage:
#   ./run-inference.sh                 # submit every run (6 checkpoints x 3 chunks = 18 jobs)
#   ./run-inference.sh era5            # only runs whose config/job name matches this substring
#   ./run-inference.sh -c0             # e.g. only chunk 0 across all checkpoints
# Recommended: submit one chunk of one checkpoint first, confirm the GCS zarr has the derived +
# component vars and GPU memory sits ~68%, then submit the rest.

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"          # this dir relative to the repo root
WANDB_USERNAME="bhenn1983"
JOB_GROUP="ace2s-land-forcing-eval-diurnal"
GCS_PREFIX="gs://vcm-ml-intermediate/2026-07-10-ace2s-land-forcing-inference-diurnal"
SELECT="${1:-}"                                   # optional substring filter; empty => all

# --- Checkpoint beaker dataset IDs (best_inference_ckpt.tar) -------------------------------------
# Controls: the deployed ACE2S checkpoints (no extra land forcing), reused as baseline.
ERA5_CONTROL=01KSVC6YS7C18SGYV4VPZYZ232
CM4_CONTROL_RS0=01KTYXNSJX90Y5E2CQ6SV8K37D
# CM4_CONTROL_RS1=01KTWGH2VEZ4DNXXF1H5FTJK1S   # optional second control seed

# Control PRETRAIN checkpoints (best_ckpt.tar) — the exact no-forcing 1-step pretrains the deployed
# controls were multi-step-finetuned FROM (traced via each deployed control's finetune-job /weights
# warmstart mount). This is the *fair* baseline for the 1-step-pretrain treatments: same lineage +
# same checkpoint-selection criterion, WITHOUT the multi-step-finetune confound the deployed
# controls carry.
#   ERA5: warmstart of ace2s-era5-multi-step-fine-tuning-no-var-weighting-rs0 (deployed control
#         01KSVC...). Residual epoch gap vs treatments (ep 6-14) — mostly-converged.
#   CM4:  warmstart of ace2s-cm4-picontrol-multi-step-finetuning-rs0 (deployed control 01KTYX...);
#         ran to ep 11, closely matching the CM4 treatments' ep 9-10.
ERA5_CONTROL_PT=01KSJKJKXRHA2Q0QCW91B1P8EG
CM4_CONTROL_PT=01KTPTS6C23P8SWB9RBFWB09BE

# Treatments: the *1-step pretrain* land-forcing checkpoints (NOT the multi-step finetuned ones —
# long rollouts are meteorologically confounded by the leaked land forcing). Fill in the four
# 1-step-pretrain beaker dataset IDs before submitting the treatment runs.
# Most-recent *committed* result dataset per pretrain experiment (best_ckpt.tar lives under
# training_checkpoints/). Result-dataset IDs are ULIDs, so they sort in retry order.
ERA5_SNOW=01KX47AYFXZR5GBP5236ZQ2H8G   # newest retry (captures the final epoch); confirm it has
                                       # committed before launching (prior committed: 01KX1S6C2G5E4856CSV70QN544).
ERA5_SOIL=01KWZDSCZ69NPBR39H1JS946RF
CM4_SNOW=01KX017K0Z48ZW91769NF7MQXH
CM4_SOIL=01KX1RVPA4YQYZK9NK76ZEPWMJ

# 4th arg = checkpoint file within the beaker dataset. Treatments use best_ckpt.tar (lowest
# *validation* loss — the short-horizon criterion, and what multi-step finetuning warm-started
# from); the pretrain jobs' best_inference_ckpt.tar is selected on a 7300-step rollout, i.e. the
# confounded long-rollout regime, so it is not used here. Controls default to best_inference_ckpt
# (their as-deployed checkpoint).
# Each checkpoint is run as 3 interleaved IC chunks of 100 ICs (memory-safe on the B200; 300
# total). The chunks tile each dataset's holdout at the dense stride baked into the config's
# `interval`; chunk c just shifts `first` by one stride. DIURNAL: strides are odd (not mult. of 4)
# so init hours rotate through 00/06/12/18 UTC. ERA5 dense stride 63 steps -> firsts
# 84738/84801/84864 (interval 189); CM4 stride 193 steps -> firsts 233600/233793/233986 (interval
# 579). Outputs go to <job>-c{0,1,2}; concatenate the 3 zarrs per checkpoint on the sample axis.
submit() {
    local config="$1" ckpt="$2" job="$3" ckpt_file="${4:-training_checkpoints/best_ckpt.tar}"
    if [ -z "$ckpt" ]; then
        echo "SKIP $job: checkpoint ID not set" >&2
        return 0
    fi
    local config_path="${CONFIG_DIR}/${config}"
    local firsts
    case "$config" in
        era5.yaml) firsts=(84738 84801 84864) ;;
        cm4.yaml)  firsts=(233600 233793 233986) ;;
        *) echo "unknown config $config" >&2; return 1 ;;
    esac
    python -m fme.ace.validate_config --config_type evaluator "$config_path"
    local c
    for c in 0 1 2; do
        local cjob="${job}-c${c}"
        if [ -n "$SELECT" ] && [[ "$config" != *"$SELECT"* ]] && [[ "$cjob" != *"$SELECT"* ]]; then
            continue
        fi
        gantry run \
            --name "$cjob" \
            --task-name "$cjob" \
            --description "ACE2S land-forcing eval: ${cjob}" \
            --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
            --workspace ai2/ace \
            --priority high \
            --preemptible \
            --cluster ai2/titan \
            --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            --env WANDB_USERNAME="$WANDB_USERNAME" \
            --env WANDB_NAME="$cjob" \
            --env WANDB_JOB_TYPE=inference \
            --env WANDB_RUN_GROUP="$JOB_GROUP" \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            --dataset "$ckpt:${ckpt_file}:/ckpt.tar" \
            --gpus 1 \
            --shared-memory 400GiB \
            --weka climate-default:/climate-default \
            --budget ai2/atec-climate \
            --system-python \
            --install "pip install --no-deps ." \
            -- python -I -m fme.ace.evaluator "$config_path" \
                 --override "experiment_dir=${GCS_PREFIX}/${cjob}" "loader.start_indices.first=${firsts[$c]}"
    done
}

cd "$REPO_ROOT"  # so the config paths resolve regardless of where this is run from

# config       checkpoint         job name                 [checkpoint file]
# Treatments (1-step pretrain) use the default best_ckpt.tar; controls use their as-deployed
# best_inference_ckpt.tar.
submit era5.yaml "$ERA5_CONTROL"    lf-eval-era5-control-diurnal     training_checkpoints/best_inference_ckpt.tar
submit era5.yaml "$ERA5_SNOW"       lf-eval-era5-snow-diurnal
submit era5.yaml "$ERA5_SOIL"       lf-eval-era5-soil-diurnal
submit cm4.yaml  "$CM4_CONTROL_RS0" lf-eval-cm4-control-rs0-diurnal  training_checkpoints/best_inference_ckpt.tar
submit cm4.yaml  "$CM4_SNOW"        lf-eval-cm4-snow-diurnal
submit cm4.yaml  "$CM4_SOIL"        lf-eval-cm4-soil-diurnal

# Control PRETRAIN baseline (best_ckpt.tar, matching the treatments' selection). Run these to get a
# recipe- and criterion-matched control that removes the multi-step-finetune confound. Launch just
# these six with:  ./run-inference.sh control-pt
submit era5.yaml "$ERA5_CONTROL_PT"  lf-eval-era5-control-pt-diurnal
submit cm4.yaml  "$CM4_CONTROL_PT"    lf-eval-cm4-control-pt-diurnal

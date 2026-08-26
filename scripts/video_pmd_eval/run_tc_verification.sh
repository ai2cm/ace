#!/bin/bash
# Full TC track verification pipeline for a 4-region-tiled stage-2 model
# (default: st-singlestage-flat), matching the methodology already run for
# st-flat/st-ou in crps_eval_results_stage2_st-flat-st-ou/tc_verification/:
#   1. Combine the 4 region zarrs into one global store (ensemble member 0).
#   2. Bootstrap TempestExtremes via micromamba (conda-forge).
#   3. Run detect_tc_tracks.py (SLP-only, 3hr cadence -- model output has no
#      upper-air temperature or HGTsfc, so no warm-core/topo filtering).
#   4. Rectify against the existing known_tracks_2023_filtered.csv (mounted
#      as a Beaker dataset -- generated for the st-flat/st-ou run, reused
#      as-is here since it's model-independent reference/truth tracks).
#   5. Compute the summary table (tc_summary.py) + comparison plots.
#
# Known-tracks dataset: 01KZT2P2PS0QZC5J69PQGE6XK8
#
# Run:  bash scripts/video_pmd_eval/run_tc_verification.sh [model]
#   model: any label in combine_regions_to_zarr.py's PATCHED_MODELS
#     (default: st-singlestage-flat).
#
# Timeout is 8h, not the 2h that's plenty for most models: any model
# sharing hiro's checkpoint/pipeline (hiro, cascade-infill-then-sr) is
# produced by fme.downscaling.inference's ZarrWriter, which writes zarr v3
# shards -- combine_regions_to_zarr.py's re-chunking .to_zarr() write on
# that source is measured at only ~9.5 GB/h (vs. finishing well within 2h
# for plainly-chunked video-PMD sources), so the ~60GB combined store alone
# can take ~6h+. A cascade-infill-then-sr run hit exactly this and was
# killed by the old 2h timeout with zero progress logged after "Writing
# to ..." -- not stuck/hung, just genuinely that slow.
set -e

MODEL="${1:-st-singlestage-flat}"
JOB_NAME="tc-verification-${MODEL}-$(date +%s)"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/neptune"
KNOWN_TRACKS_DATASET="01KZT2P2PS0QZC5J69PQGE6XK8"
# "hiro" is not on weka -- its zarr lives in its own Beaker result dataset,
# mounted at /hiro_result only when requested (same pattern as
# run_crps_eval.sh).
HIRO_RESULT_DATASET="01M08MXPM2EA4TX8QCN9WD1HPN"
CPUS=8
MEMORY="96GiB"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

# Base64-embed the repo's scripts/ tree pieces we need so the job doesn't
# depend on a git checkout inside the container -- same pattern as
# run_crps_eval.sh / run_hiro_vs_video_pmd_compare.sh.
COMBINE_B64=$(base64 < scripts/video_pmd_eval/combine_regions_to_zarr.py | tr -d '\n')
SUMMARY_B64=$(base64 < scripts/video_pmd_eval/tc_summary.py | tr -d '\n')
DETECT_B64=$(base64 < scripts/tropical_cyclones/detect_tc_tracks.py | tr -d '\n')
RECTIFY_B64=$(base64 < scripts/tropical_cyclones/rectify_tc_tracks.py | tr -d '\n')

COMBINED_ZARR="/climate-default/2026-06-25-temporal-diffusion/inference/tc_combined/${MODEL}-ens0-global-combined.zarr"

MOUNT_ARGS=(
    --mount "src=weka,ref=climate-default,dst=/climate-default"
    --mount "src=beaker,ref=$KNOWN_TRACKS_DATASET,dst=/known_tracks"
)
if [ "$MODEL" = "hiro" ]; then
    MOUNT_ARGS+=(--mount "src=beaker,ref=$HIRO_RESULT_DATASET,dst=/hiro_result")
fi

set +e
CREATE_OUTPUT=$(beaker session create \
    --bare --detach \
    --cluster "$CLUSTER" \
    --priority urgent \
    --budget ai2/atec-climate \
    --workspace "$WORKSPACE" \
    --image "beaker://$DEPS_ONLY_IMAGE" \
    "${MOUNT_ARGS[@]}" \
    --cpus "$CPUS" \
    --memory "$MEMORY" \
    --gpus 0 \
    --timeout 8h \
    --name "$JOB_NAME" \
    --result /results \
    -- bash -c "
set -e
echo '=== writing scripts ==='
echo $COMBINE_B64 | base64 -d > /tmp/combine.py
echo $SUMMARY_B64 | base64 -d > /tmp/tc_summary.py
echo $DETECT_B64 | base64 -d > /tmp/detect_tc_tracks.py
echo $RECTIFY_B64 | base64 -d > /tmp/rectify_tc_tracks.py

echo '=== bootstrapping micromamba + tempest-extremes ==='
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba
export MAMBA_ROOT_PREFIX=/tmp/micromamba
/tmp/bin/micromamba create -y -n tempest -c conda-forge tempest-extremes
DETECT_EXE=\$(/tmp/bin/micromamba run -n tempest which DetectNodes)
STITCH_EXE=\$(/tmp/bin/micromamba run -n tempest which StitchNodes)
echo \"DetectNodes at \$DETECT_EXE, StitchNodes at \$STITCH_EXE\"

echo '=== installing dask (needed for chunked/streamed zarr write) ==='
python3 -m pip install --quiet dask

echo '=== combining 4 regions into one global zarr ==='
python3 /tmp/combine.py --model $MODEL --ensemble-member 0 \
    --out-zarr '$COMBINED_ZARR'

echo '=== running TC detection (SLP-only, 3hr) ==='
mkdir -p /results/tc_out
python3 /tmp/detect_tc_tracks.py '$COMBINED_ZARR' /results/tc_out \
    --no-warm-core --timefilter 3hr --chunk-size 128 --workers 6 \
    --u-var eastward_wind_at_ten_meters --v-var northward_wind_at_ten_meters \
    --detect-exe \"\$DETECT_EXE\" --stitch-exe \"\$STITCH_EXE\" --no-write-sel-args
cp /results/tc_out/tracks.csv /results/detect_raw_tracks.csv

echo '=== rectifying against known tracks ==='
python3 /tmp/rectify_tc_tracks.py /known_tracks/known_tracks_2023_filtered.csv \
    /results/detect_raw_tracks.csv /results/tc_rectified
cp /results/tc_rectified/rectified_tracks.csv /results/rectified_tracks.csv

echo '=== computing summary ==='
python3 /tmp/tc_summary.py --label $MODEL \
    --known-csv /known_tracks/known_tracks_2023_filtered.csv \
    --raw-csv /results/detect_raw_tracks.csv \
    --rectified-csv /results/rectified_tracks.csv \
    --outdir /results

echo TC_VERIFICATION_DONE
" 2>&1)
set -e

echo "$CREATE_OUTPUT"
SESSION_ID=$(echo "$CREATE_OUTPUT" | grep -oE '01[A-Z0-9]{24}' | head -1)
if [ -z "$SESSION_ID" ]; then
    echo "Error: could not parse session ID from beaker output" >&2
    exit 1
fi
echo "Session ID: $SESSION_ID"
echo "Follow logs with: beaker session logs -f $SESSION_ID"

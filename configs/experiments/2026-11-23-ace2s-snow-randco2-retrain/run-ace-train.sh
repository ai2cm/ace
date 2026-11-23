#!/bin/bash
# Stage 1: 1-step pre-training of the control and masked-naive arms: the
# pic-1pct-randco2 arms on the corrected random-CO2 data, seeds 0 and 1, and the
# pic-1pct arms of the 2026-10-06 round, seed 1 (one config per arm; the seed is
# passed as an override).
#
#   ./run-ace-train.sh                                        # all six runs
#   ./run-ace-train.sh cm4-masked-naive-pic-1pct-randco2-rs0  # a subset
#
# Dry run (validation and both inline rollouts before the first iterations; cancel
# once training iterations have logged):
#   OVERRIDE="evaluate_before_training=true" ./run-ace-train.sh cm4-masked-naive-pic-1pct-randco2-rs0
# Job names then get a -dry suffix (set JOB_SUFFIX for another). DRY_RUN=1 prints
# the gantry commands instead of submitting them.

source "$(dirname "$0")/launch-common.sh"

ALL_RUNS=(cm4-control-pic-1pct-randco2-rs0 cm4-control-pic-1pct-randco2-rs1
          cm4-masked-naive-pic-1pct-randco2-rs0 cm4-masked-naive-pic-1pct-randco2-rs1
          cm4-control-pic-1pct-rs1 cm4-masked-naive-pic-1pct-rs1)

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
  RUNS=("${ALL_RUNS[@]}")
fi
for run in "${RUNS[@]}"; do
  if [[ ! " ${ALL_RUNS[*]} " =~ " $run " ]]; then
    echo "unknown run $run; choose from ${ALL_RUNS[*]}" >&2
    exit 1
  fi
done

suffix="${JOB_SUFFIX:-}"
override_args=()
if [[ -n "${OVERRIDE:-}" ]]; then
  suffix="${JOB_SUFFIX--dry}"
  override_args=($OVERRIDE)
fi

for run in "${RUNS[@]}"; do
  arm="${run%-rs*}"
  seed="${run##*-rs}"
  run_training "$arm-1-step-pretrain-daily.yaml" \
    "ace2s-snowfix-${arm}-daily-1-step-pretrain-rs${seed}${suffix}" "seed=$seed" "${override_args[@]}"
done

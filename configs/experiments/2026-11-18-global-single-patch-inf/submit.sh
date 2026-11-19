#!/usr/bin/env bash
# Submit one evaluator job per (model, method) through the Beaker proxy.
#
#   ./submit.sh                       # dry run, all six jobs
#   ./submit.sh --submit              # submit all six
#   ./submit.sh --submit MODEL METHOD # submit one, e.g. no_tmp2m_acf single_pass
#
# Run from the repository root's checkout of the pushed branch; jobs run the
# pushed commit. Job IDs are appended to experiments.txt next to this script.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REL=configs/experiments/2026-11-18-global-single-patch-inf
GANTRY=/usr/local/bin/gantry
cd "$HERE/../../.."

MODE=${1:-}
DRY=(--dry-run)
[[ $MODE == --submit ]] && DRY=()
ONLY_MODEL=${2:-}
ONLY_METHOD=${3:-}

declare -A HIGH=(
    [no_tmp2m_acf]=01M5JP35DFARTGRGBPAPXS6RK1
    [tmp2m_acf_matched]=01M71BAY8SFQZW5VH6832K1C9P
)
declare -A LOW=(
    [no_tmp2m_acf]=01M5JP3AHKQ5BE48EKH2SN1ZE2
    [tmp2m_acf_matched]=01M72ASTXQX1CFVE5ZQTR5MYH8
)
# The single pass needs the most GPU memory: B200 (titan). Never use ai2/holmes.
declare -A CLUSTERS=(
    [single_pass]="ai2/titan"
    [tiles]="ai2/titan ai2/jupiter"
    [multidiffusion]="ai2/titan ai2/jupiter"
)

for model in no_tmp2m_acf tmp2m_acf_matched; do
    for method in single_pass tiles multidiffusion; do
        [[ -n $ONLY_MODEL && $model != "$ONLY_MODEL" ]] && continue
        [[ -n $ONLY_METHOD && $method != "$ONLY_METHOD" ]] && continue
        config="$REL/eval-${model//_/-}-${method//_/-}.yaml"
        [[ -f $config ]] || { echo "missing $config" >&2; exit 1; }
        cluster_args=()
        for c in ${CLUSTERS[$method]}; do cluster_args+=(--cluster "$c"); done
        out=$(mktemp)
        $GANTRY run "${DRY[@]}" \
            --name "gspi-${model//_/-}-${method//_/-}" \
            --description "global-single-patch-inf: $method over lon 0-360, lat -65..65, 4 snapshots in 2023, $model" \
            "${cluster_args[@]}" \
            --gpus 1 \
            --shared-memory 64GiB \
            --priority high \
            --min-runtime 8h \
            --task-timeout 12h \
            --beaker-image oliverwm/fme-deps-only-54045d546 \
            --no-python \
            --install "pip install --no-deps ." \
            --weka climate-default:/climate-default \
            --dataset "${HIGH[$model]}:checkpoints:/checkpoints_high_sigma" \
            --dataset "${LOW[$model]}:checkpoints:/checkpoints_low_sigma" \
            -- bash "$REL/run_in_job.sh" "$config" \
            /climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling \
            /climate-default/2026-02-03-downscaling-data-X-SHiELD-AMIP-pressure-data \
            2>&1 | tee "$out"
        if [[ $MODE == --submit ]]; then
            id=$(grep -oE '^ +id: +[0-9A-Z]{26}' "$out" | awk '{print $2}' | head -1)
            [[ -n $id ]] || { rm -f "$out"; echo "submission failed for $model $method" >&2; exit 1; }
            echo "$model $method $id $(date -u +%FT%TZ)" >> "$HERE/experiments.txt"
        fi
        rm -f "$out"
    done
done

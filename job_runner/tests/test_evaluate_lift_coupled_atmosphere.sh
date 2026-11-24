#!/bin/bash
# Tests the fme module and job command evaluate.sh --dry-run picks, with and without
# --lift-coupled-atmosphere, on coupled and atmosphere-only experiment dirs. Runs the
# real script in a throwaway git repo; beaker and python are stubs, so nothing is launched.
# Usage: bash job_runner/tests/test_evaluate_lift_coupled_atmosphere.sh
# JOB_RUNNER_SRC=<dir> tests another copy of job_runner/.

set -u

JOB_RUNNER_SRC="${JOB_RUNNER_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

HEADER="group|tag|experiment_id|status|checkpoint|priority|min_runtime|override|results_dataset|workspace|cluster|ocean_results_dataset|atmos_results_dataset|shared_mem|n_gpus"
LIFT_PREFIX="lift-coupled-atmosphere"
LIFT_OVERRIDE="checkpoint_path=/tmp/lifted_atmosphere_ckpt.tar"
FAILURES=0

fail() {
    echo "FAIL: $*"
    FAILURES=$((FAILURES + 1))
}

pass() {
    echo "ok: $*"
}

mkdir -p "$WORK/bin"
cat >"$WORK/bin/beaker" <<'EOF'
#!/bin/bash
if [[ "$1 $2" == "account whoami" ]]; then
    echo '[{"name": "tester"}]'
fi
EOF
cat >"$WORK/bin/python" <<EOF
#!/bin/bash
echo "\$*" >>"$WORK/python.log"
EOF
chmod +x "$WORK/bin/"*
export PATH="$WORK/bin:$PATH"

REPO="$WORK/repo"
git init -q -b main "$REPO"
git -C "$REPO" config user.email test@example.com
git -C "$REPO" config user.name test
cp -r "$JOB_RUNNER_SRC" "$REPO/job_runner"
echo image >"$REPO/latest_deps_only_image.txt"
mkdir -p "$REPO/eval_configs"
echo "{}" >"$REPO/eval_configs/evaluator-config-randco2.yaml"
for DIR in exper/coupled/cfg exper/atmos/cfg; do
    mkdir -p "$REPO/$DIR"
    echo "$HEADER
G1|rs0|01TRAINEXPER|run_randco2|best_inference_ckpt|high|||01RESULTSDS||||||1
G1|rs0|01TRAINEXPER|run_inf_randco2|best_inference_ckpt|high|||01RESULTSDS||||||1" >"$REPO/$DIR/experiments.txt"
done
git -C "$REPO" add -A
git -C "$REPO" commit -qm init

# Args: NAME EXPERIMENT_DIR VALIDATE_MODULE EVALUATOR_MODULE LIFT(true|false) [FLAGS...]
check_case() {
    local NAME="$1" EXP="$2" VALIDATE="$3" EVALUATOR="$4" LIFT="$5"
    shift 5
    rm -f "$WORK/python.log"
    if ! (cd "$REPO" && bash job_runner/evaluate.sh "$EXP" . --config-dir eval_configs --dry-run "$@") \
        >"$WORK/out" 2>&1; then
        fail "$NAME: exit"
        cat "$WORK/out"
        return
    fi
    local COMMAND
    COMMAND=$(grep -F "Job command:" "$WORK/out")
    local OK=true
    [[ "$(grep -c . "$WORK/python.log" 2>/dev/null)" == 1 ]] || OK=false
    grep -qF -- "-m $VALIDATE --config_type evaluator eval_configs/evaluator-config-randco2.yaml" \
        "$WORK/python.log" || OK=false
    [[ "$(grep -c "Job command:" "$WORK/out")" == 1 ]] || OK=false
    [[ "$COMMAND" == *" $EVALUATOR eval_configs/evaluator-config-randco2.yaml"* ]] || OK=false
    if [[ "$LIFT" == "true" ]]; then
        [[ "$COMMAND" == *"$LIFT_PREFIX"* ]] && [[ "$COMMAND" == *"$LIFT_OVERRIDE"* ]] || OK=false
        grep -qF -- "$LIFT_OVERRIDE" "$WORK/python.log" || OK=false
    else
        [[ "$COMMAND" != *"$LIFT_PREFIX"* ]] && [[ "$COMMAND" != *"$LIFT_OVERRIDE"* ]] || OK=false
    fi
    if [[ "$OK" == "true" ]]; then
        pass "$NAME"
    else
        fail "$NAME"
        cat "$WORK/out" "$WORK/python.log"
    fi
}

check_case "coupled dir, no flag: fme.coupled.*" exper/coupled/cfg \
    fme.coupled.validate_config fme.coupled.evaluator false
check_case "atmosphere dir, no flag: fme.ace.*" exper/atmos/cfg \
    fme.ace.validate_config fme.ace.evaluator false
check_case "atmosphere dir, lift: fme.ace.* after lift" exper/atmos/cfg \
    fme.ace.validate_config fme.ace.evaluator true --lift-coupled-atmosphere
check_case "coupled dir, lift: fme.ace.* after lift" exper/coupled/cfg \
    fme.ace.validate_config fme.ace.evaluator true --lift-coupled-atmosphere

echo
if [[ $FAILURES -eq 0 ]]; then
    echo "all passed"
else
    echo "$FAILURES failed"
    exit 1
fi

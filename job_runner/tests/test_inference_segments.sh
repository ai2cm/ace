#!/bin/bash
# Tests the experiments.txt `segments` column: inference.sh passes `--segments N`
# to the inference command only (never to validate_config) and keeps gantry's
# auto-resume for segmented rows; a blank or absent column changes nothing; bad
# values and segments on evaluate.sh rows stop the submission. Runs the real
# scripts in a throwaway git repo; beaker, gantry and python are stubs, so nothing
# is launched.
# Usage: bash job_runner/tests/test_inference_segments.sh
# JOB_RUNNER_SRC=<dir> tests another copy of job_runner/.

set -u

JOB_RUNNER_SRC="${JOB_RUNNER_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

OLD_HEADER="group|tag|experiment_id|status|checkpoint|priority|min_runtime|override|results_dataset|workspace|cluster|ocean_results_dataset|atmos_results_dataset|shared_mem|n_gpus"
NEW_HEADER="${OLD_HEADER}|segments"
CONFIG=inf_configs/inference-config-piC.yaml
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
cat >"$WORK/bin/gantry" <<EOF
#!/bin/bash
printf '%s\n' "\$@" >>"$WORK/gantry.log"
EOF
chmod +x "$WORK/bin/"*
export PATH="$WORK/bin:$PATH"

REPO="$WORK/repo"
git init -q -b main "$REPO"
git -C "$REPO" config user.email test@example.com
git -C "$REPO" config user.name test
cp -r "$JOB_RUNNER_SRC" "$REPO/job_runner"
echo image >"$REPO/latest_deps_only_image.txt"
mkdir -p "$REPO/inf_configs"
echo "{}" >"$REPO/$CONFIG"
git -C "$REPO" add -A
git -C "$REPO" commit -qm init

# Write <experiment_dir>/experiments.txt with HEADER and one run_inf_piC row whose
# fields after `override` are the given trailing fields.
# Args: EXPERIMENT_DIR HEADER TRAILING_FIELDS
write_experiments() {
    mkdir -p "$REPO/$1"
    printf '%s\nG1|rs0|01TRAINEXPER|run_inf_piC|best_inference_ckpt|high||seed=0|01RESULTSDS%s\n' \
        "$2" "$3" >"$REPO/$1/experiments.txt"
}

# Args: SCRIPT EXPERIMENT_DIR [FLAGS...]; output in $WORK/out
run_script() {
    local SCRIPT="$1" EXP="$2"
    shift 2
    rm -f "$WORK/python.log" "$WORK/gantry.log"
    (cd "$REPO" && bash "job_runner/$SCRIPT" "$EXP" . --config-dir inf_configs "$@") \
        >"$WORK/out" 2>&1
}

# Args: NAME EXPERIMENT_DIR MODULE EXPECTED_SEGMENTS_ARGS(e.g. "--segments 4" or "")
check_dry_run() {
    local NAME="$1" EXP="$2" MODULE="$3" SEGS="$4"
    if ! run_script inference.sh "$EXP" --dry-run; then
        fail "$NAME: exit"
        cat "$WORK/out"
        return
    fi
    local COMMAND EXPECTED OK=true
    COMMAND=$(grep -F "Job command:" "$WORK/out")
    EXPECTED="Job command: python -I -m $MODULE $CONFIG ${SEGS:+$SEGS }--override seed=0"
    [[ "$COMMAND" == *"$EXPECTED" ]] || OK=false
    grep -qF -- "--config_type inference $CONFIG --override seed=0" "$WORK/python.log" || OK=false
    ! grep -qF -- "--segments" "$WORK/python.log" || OK=false
    if [[ -n "$SEGS" ]]; then
        grep -qF "Auto-resume: on" "$WORK/out" || OK=false
    else
        grep -qF "Auto-resume: off" "$WORK/out" || OK=false
        [[ "$COMMAND" != *"--segments"* ]] || OK=false
    fi
    if [[ "$OK" == "true" ]]; then
        pass "$NAME"
    else
        fail "$NAME (expected '...$EXPECTED')"
        cat "$WORK/out" "$WORK/python.log"
    fi
}

# Args: NAME EXPECTED_SEGMENTS_ARGS EXPECT_NO_AUTO_RESUME(true|false)
check_submit() {
    local NAME="$1" SEGS="$2" NO_AUTO_RESUME="$3"
    if ! run_script inference.sh exper/coupled/submit; then
        fail "$NAME: exit"
        cat "$WORK/out"
        return
    fi
    local OK=true GANTRY
    GANTRY=$(tr '\n' ' ' <"$WORK/gantry.log")
    [[ "$GANTRY" == *"-- python -I -m fme.coupled.inference $CONFIG ${SEGS:+$SEGS }--override seed=0 " ]] || OK=false
    if [[ "$NO_AUTO_RESUME" == "true" ]]; then
        [[ "$GANTRY" == *"--no-auto-resume"* ]] || OK=false
    else
        [[ "$GANTRY" != *"--no-auto-resume"* ]] || OK=false
    fi
    if [[ "$OK" == "true" ]]; then
        pass "$NAME"
    else
        fail "$NAME"
        cat "$WORK/out" "$WORK/gantry.log"
    fi
}

# Args: NAME SCRIPT EXPERIMENT_DIR EXPECTED_ERROR_SUBSTRING
check_rejected() {
    local NAME="$1" SCRIPT="$2" EXP="$3" MESSAGE="$4"
    if run_script "$SCRIPT" "$EXP" --dry-run; then
        fail "$NAME: exited 0"
        cat "$WORK/out"
    elif grep -qF -- "$MESSAGE" "$WORK/out"; then
        pass "$NAME"
    else
        fail "$NAME: missing '$MESSAGE'"
        cat "$WORK/out"
    fi
}

write_experiments exper/coupled/old "$OLD_HEADER" "||||||1"
write_experiments exper/coupled/blank "$NEW_HEADER" "||||||1|"
write_experiments exper/coupled/seg "$NEW_HEADER" "||||||1|4"
write_experiments exper/atmos/seg "$NEW_HEADER" "||||||2|3"
write_experiments exper/coupled/zero "$NEW_HEADER" "||||||1|0"
write_experiments exper/coupled/word "$NEW_HEADER" "||||||1|four"
write_experiments exper/coupled/gpus "$NEW_HEADER" "||||||2|4"
mkdir -p "$REPO/exper/coupled/eval"
printf '%s\nG1|rs0|01TRAINEXPER|run_piC|best_inference_ckpt|high|||01RESULTSDS||||||1|4\n' \
    "$NEW_HEADER" >"$REPO/exper/coupled/eval/experiments.txt"
echo "{}" >"$REPO/inf_configs/evaluator-config-piC.yaml"

check_dry_run "file without segments column: no flag" exper/coupled/old fme.coupled.inference ""
check_dry_run "blank segments: no flag" exper/coupled/blank fme.coupled.inference ""
check_dry_run "coupled segments=4" exper/coupled/seg fme.coupled.inference "--segments 4"
if ! grep -qF "Segments: 4" "$WORK/out"; then
    fail "coupled segments=4: dry-run summary lacks 'Segments: 4'"
fi
check_rejected "segments=0 rejected" inference.sh exper/coupled/zero \
    "segments must be a positive integer or blank, got '0'"
check_rejected "segments=four rejected" inference.sh exper/coupled/word \
    "segments must be a positive integer or blank, got 'four'"
check_rejected "coupled n_gpus=2 with segments rejected" inference.sh exper/coupled/gpus \
    "n_gpus is 2"
check_rejected "evaluate.sh row with segments rejected" evaluate.sh exper/coupled/eval \
    "has no"

# Atmosphere inference on 2 GPUs: torchrun prefix instead of python -I -m.
if run_script inference.sh exper/atmos/seg --dry-run &&
    grep -qF "Job command: torchrun --nproc_per_node 2 -m fme.ace.inference $CONFIG --segments 3 --override seed=0" "$WORK/out" &&
    ! grep -qF -- "--segments" "$WORK/python.log"; then
    pass "atmosphere segments=3 on 2 GPUs"
else
    fail "atmosphere segments=3 on 2 GPUs"
    cat "$WORK/out"
fi

mkdir -p "$REPO/exper/coupled/submit"
cp "$REPO/exper/coupled/old/experiments.txt" "$REPO/exper/coupled/submit/experiments.txt"
check_submit "submit without segments: --no-auto-resume" "" true
cp "$REPO/exper/coupled/seg/experiments.txt" "$REPO/exper/coupled/submit/experiments.txt"
check_submit "submit segments=4: auto-resume kept" "--segments 4" false

echo
if [[ $FAILURES -eq 0 ]]; then
    echo "all passed"
else
    echo "$FAILURES failed"
    exit 1
fi

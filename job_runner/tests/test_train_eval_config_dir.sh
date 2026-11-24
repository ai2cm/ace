#!/bin/bash
# Tests {uncoupled,coupled}_{train,finetune}.sh experiments.txt rows with and without
# --eval-config-dir. Runs the real scripts in a throwaway git repo with a bare
# origin; gantry, beaker and python are stubs, so nothing is launched.
# Usage: bash job_runner/tests/test_train_eval_config_dir.sh
# JOB_RUNNER_SRC=<dir> tests another copy of job_runner/.

set -u

JOB_RUNNER_SRC="${JOB_RUNNER_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

EXPER_ID=01TESTTRAININGEXPERIMENT
HEADER="group|tag|experiment_id|status|checkpoint|priority|min_runtime|override|results_dataset|workspace|cluster|ocean_results_dataset|atmos_results_dataset|shared_mem|n_gpus|segments"
FAILURES=0

fail() {
    echo "FAIL: $*"
    FAILURES=$((FAILURES + 1))
}

pass() {
    echo "ok: $*"
}

mkdir -p "$WORK/bin"
cat >"$WORK/bin/gantry" <<EOF
#!/bin/bash
echo "Experiment submitted, see progress at https://beaker.org/ex/${EXPER_ID}"
EOF
cat >"$WORK/bin/beaker" <<'EOF'
#!/bin/bash
if [[ "$1 $2" == "account whoami" ]]; then
    echo '[{"name": "tester"}]'
fi
EOF
cat >"$WORK/bin/python" <<'EOF'
#!/bin/bash
exit 0
EOF
chmod +x "$WORK/bin/"*
export PATH="$WORK/bin:$PATH"

# Fresh repo + bare origin per case; prints the repo path
new_repo() {
    local NAME="$1"
    local REPO="$WORK/$NAME"
    git init -q --bare "$WORK/$NAME-origin.git"
    git init -q -b main "$REPO"
    git -C "$REPO" config user.email test@example.com
    git -C "$REPO" config user.name test
    git -C "$REPO" remote add origin "$WORK/$NAME-origin.git"

    cp -r "$JOB_RUNNER_SRC" "$REPO/job_runner"
    cat >"$REPO/job_runner/create_coupled_train_config.sh" <<'EOF'
#!/bin/bash
cp "$3" "$4"
EOF
    local CREATE_FINETUNE_CONFIG
    for CREATE_FINETUNE_CONFIG in create_finetune_config.sh create_coupled_finetune_config.sh; do
        cat >"$REPO/job_runner/$CREATE_FINETUNE_CONFIG" <<'EOF'
#!/bin/bash
cp "$2" "$3"
EOF
    done
    echo image >"$REPO/latest_deps_only_image.txt"

    local DIR="$REPO/exper/coupled/cfg"
    mkdir -p "$DIR" "$REPO/eval_configs" "$REPO/empty_eval_configs"
    echo "{}" >"$DIR/train-config.yaml"
    echo "{}" >"$DIR/train-config-template.yaml"
    echo "{}" >"$DIR/finetune-config-template.yaml"
    echo "group|tag|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override_args|min_runtime
G1|rs0|train|normal||8|100GiB||||
G1|rs1|skip|normal||8|100GiB||||
G1|rs2|train|normal||8|100GiB||||" >"$DIR/training.txt"
    echo "group|tag|ocean_project|ocean_wandb_id|ocean_ckpt_type|atmos_project|atmos_wandb_id|atmos_ckpt_type|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override|min_runtime|results_dataset_ocean|results_dataset_atmos
C1|rs0|p|w|best_ckpt|p|w|best_ckpt|train|normal||8|100GiB|||||01OCEANDS|01ATMOSDS" >"$DIR/pretraining.txt"
    echo "group|tag|wandb_project|wandb_id|ckpt_type|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override|results_dataset|min_runtime
F1|ft0|p|w|best_ckpt|train|normal||8|100GiB||||01RESULTSDS|
F1|ft1|p|w|best_ckpt|skip|normal||8|100GiB||||01RESULTSDS|
F1|ft2|p|w|best_ckpt|train|normal||8|100GiB||||01RESULTSDS|" >"$DIR/finetuning.txt"
    echo "{}" >"$REPO/eval_configs/evaluator-config-piC-ic0151.yaml"
    echo "{}" >"$REPO/eval_configs/evaluator-config-1pct.yaml"
    echo "{}" >"$REPO/eval_configs/inference-config-ignored.yaml"
    echo "{}" >"$REPO/empty_eval_configs/inference-config-x.yaml"

    git -C "$REPO" add -A
    git -C "$REPO" commit -qm init
    git -C "$REPO" push -q origin main 2>/dev/null
    echo "$REPO"
}

# Args: REPO SCRIPT [ARGS...]; output in $WORK/out, exit status returned
run_script() {
    local REPO="$1" SCRIPT="$2"
    shift 2
    (cd "$REPO" && bash "job_runner/$SCRIPT" exper/coupled cfg --coupled_stats 01STATS "$@") \
        >"$WORK/out" 2>&1
}

check_file() {
    local NAME="$1" FILE="$2" EXPECTED="$3"
    if [[ "$(cat "$FILE")" == "$EXPECTED" ]] && [[ "$(tail -c1 "$FILE" | od -An -c)" == *'\n'* ]]; then
        pass "$NAME"
    else
        fail "$NAME"
        diff <(echo "$EXPECTED") "$FILE"
    fi
}

check_pushed() {
    local NAME="$1" REPO="$2"
    if [[ "$(git -C "$REPO" rev-parse HEAD)" == "$(git -C "$REPO" rev-parse origin/main)" ]] &&
        [[ -z "$(git -C "$REPO" status --porcelain)" ]]; then
        pass "$NAME: committed and pushed"
    else
        fail "$NAME: experiments.txt not committed and pushed"
    fi
}

TRAINING_ROW_TAIL="${EXPER_ID}|training|best_inference_ckpt|normal|--min-runtime 8h"
WAIT_ROW_TAIL="|best_inference_ckpt|high|--min-runtime 8h||||||||1|"

# Without --eval-config-dir: one training row per submitted run, as before the flag existed
REPO=$(new_repo uncoupled-plain)
run_script "$REPO" uncoupled_train.sh || { fail "uncoupled_train.sh exit"; cat "$WORK/out"; }
check_file "uncoupled_train.sh without flag" "$REPO/exper/coupled/cfg/experiments.txt" "$HEADER

G1|rs0|${TRAINING_ROW_TAIL}

G1|rs2|${TRAINING_ROW_TAIL}"
check_pushed "uncoupled_train.sh without flag" "$REPO"

REPO=$(new_repo coupled-plain)
run_script "$REPO" coupled_train.sh || { fail "coupled_train.sh exit"; cat "$WORK/out"; }
check_file "coupled_train.sh without flag" "$REPO/exper/coupled/cfg/experiments.txt" "$HEADER

C1|rs0|${TRAINING_ROW_TAIL}"

# With --eval-config-dir: wait_<tag> rows per run (glob order), no training row
REPO=$(new_repo uncoupled-eval)
run_script "$REPO" uncoupled_train.sh --eval-config-dir eval_configs || { fail "uncoupled_train.sh --eval-config-dir exit"; cat "$WORK/out"; }
check_file "uncoupled_train.sh --eval-config-dir" "$REPO/exper/coupled/cfg/experiments.txt" "$HEADER

G1|rs0|${EXPER_ID}|wait_1pct${WAIT_ROW_TAIL}
G1|rs0|${EXPER_ID}|wait_piC-ic0151${WAIT_ROW_TAIL}

G1|rs2|${EXPER_ID}|wait_1pct${WAIT_ROW_TAIL}
G1|rs2|${EXPER_ID}|wait_piC-ic0151${WAIT_ROW_TAIL}"
check_pushed "uncoupled_train.sh --eval-config-dir" "$REPO"

REPO=$(new_repo coupled-eval)
run_script "$REPO" coupled_train.sh --eval-config-dir "$REPO/eval_configs" ||
    { fail "coupled_train.sh --eval-config-dir exit"; cat "$WORK/out"; }
check_file "coupled_train.sh --eval-config-dir (absolute)" "$REPO/exper/coupled/cfg/experiments.txt" "$HEADER

C1|rs0|${EXPER_ID}|wait_1pct${WAIT_ROW_TAIL}
C1|rs0|${EXPER_ID}|wait_piC-ic0151${WAIT_ROW_TAIL}"

for SCRIPT in uncoupled_finetune.sh coupled_finetune.sh; do
    REPO=$(new_repo "${SCRIPT%.sh}-plain")
    run_script "$REPO" "$SCRIPT" || { fail "$SCRIPT exit"; cat "$WORK/out"; }
    check_file "$SCRIPT without flag" "$REPO/exper/coupled/cfg/experiments.txt" "$HEADER

F1|ft0|${TRAINING_ROW_TAIL}

F1|ft2|${TRAINING_ROW_TAIL}"
    check_pushed "$SCRIPT without flag" "$REPO"

    REPO=$(new_repo "${SCRIPT%.sh}-eval")
    run_script "$REPO" "$SCRIPT" --eval-config-dir eval_configs || { fail "$SCRIPT --eval-config-dir exit"; cat "$WORK/out"; }
    check_file "$SCRIPT --eval-config-dir" "$REPO/exper/coupled/cfg/experiments.txt" "$HEADER

F1|ft0|${EXPER_ID}|wait_1pct${WAIT_ROW_TAIL}
F1|ft0|${EXPER_ID}|wait_piC-ic0151${WAIT_ROW_TAIL}

F1|ft2|${EXPER_ID}|wait_1pct${WAIT_ROW_TAIL}
F1|ft2|${EXPER_ID}|wait_piC-ic0151${WAIT_ROW_TAIL}"
    check_pushed "$SCRIPT --eval-config-dir" "$REPO"
done

# Dry run: rows printed, nothing written or committed
for CASE in "uncoupled_train.sh:G1|rs2:4" "coupled_train.sh:C1|rs0:2" "uncoupled_finetune.sh:F1|ft2:4" \
    "coupled_finetune.sh:F1|ft2:4"; do
    SCRIPT="${CASE%%:*}"
    N_ROWS="${CASE##*:}"
    ROW_KEY="${CASE#*:}"
    ROW_KEY="${ROW_KEY%:*}"
    REPO=$(new_repo "${SCRIPT%.sh}-dry-run")
    run_script "$REPO" "$SCRIPT" --dry-run --eval-config-dir eval_configs || { fail "$SCRIPT dry run exit"; cat "$WORK/out"; }
    if [[ ! -e "$REPO/exper/coupled/cfg/experiments.txt" ]] &&
        grep -qxF "      ${ROW_KEY}|01234567890abcdef|wait_piC-ic0151${WAIT_ROW_TAIL}" "$WORK/out" &&
        grep -qF "Append ${N_ROWS} wait_<tag> entries" "$WORK/out"; then
        pass "$SCRIPT dry run prints wait_ rows, writes nothing"
    else
        fail "$SCRIPT dry run"
        cat "$WORK/out"
    fi
done

# Bad --eval-config-dir: refused before any submission
for CASE in "missing:no_such_dir:is not a directory" \
    "empty:empty_eval_configs:holds no evaluator-config-<tag>.yaml" \
    "outside:/tmp:must be inside the repository root" \
    "no value::requires a path argument"; do
    IFS=: read -r NAME DIR MESSAGE <<<"$CASE"
    REPO=$(new_repo "bad-${NAME// /-}")
    if [[ -n "$DIR" ]]; then
        ARGS=(--eval-config-dir "$DIR")
    else
        ARGS=(--eval-config-dir)
    fi
    if ! run_script "$REPO" uncoupled_train.sh "${ARGS[@]}" &&
        grep -qF -- "$MESSAGE" "$WORK/out" &&
        ! grep -qF "Launching" "$WORK/out" &&
        [[ ! -e "$REPO/exper/coupled/cfg/experiments.txt" ]]; then
        pass "--eval-config-dir $NAME refused"
    else
        fail "--eval-config-dir $NAME"
        cat "$WORK/out"
    fi
done

echo
if [[ $FAILURES -eq 0 ]]; then
    echo "all passed"
else
    echo "$FAILURES failed"
    exit 1
fi

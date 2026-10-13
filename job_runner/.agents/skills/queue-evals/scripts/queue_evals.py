#!/usr/bin/env python3
"""Replace finished `training` rows of experiments.txt with evaluate/inference rows.

  queue_evals.py mark <dir> [--accept <expected_name> ...]
  queue_evals.py status <dir>
  queue_evals.py apply <dir> --row run_<tag>[:<min_runtime>] [--row ...]
      [--priority P] [--keep-training] [--config-dir PATH]

<dir> is searched recursively for experiments.txt. A row is finished when its
experiment's final beaker job (jobs[-1]) exited 0. `apply` checks that each
row's config (evaluator-config-<tag>.yaml, or inference-config-<tag>.yaml for
run_inf_<tag>) exists where evaluate.sh / inference.sh will look for it.

`mark` looks up each run_* / running_* row's launched job by the name evaluate.sh /
inference.sh give it, `<group>-{eval,inf}_<ckpt>-<tag>[-<row tag>]`, in the
workspace lib.sh resolves for the row's cluster. Beaker appends `-<4 hex>`
when the name is taken, so suffixed names are matched too, but only an exact
name that exited 0 is marked ran_ unless passed to --accept. Rows whose job
is pending or running become running_*, and rows whose every match failed
become failed_*; running_* rows are rechecked on the next `mark`.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

# Column positions, set by load_columns from EXPERIMENTS_COLUMNS in lib.sh
TAG = EXPER_ID = STATUS = CKPT = PRIORITY = MIN_RUNTIME = WORKSPACE = CLUSTER = -1


def repo_root(path):
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return pathlib.Path(out.stdout.strip())


def load_columns(root):
    global TAG, EXPER_ID, STATUS, CKPT, PRIORITY, MIN_RUNTIME, WORKSPACE, CLUSTER
    out = subprocess.run(
        ["bash", "-c", "source job_runner/lib.sh; input_header EXPERIMENTS_COLUMNS"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    index = {name: i for i, name in enumerate(out.stdout.strip().split("|"))}
    TAG, EXPER_ID, STATUS, CKPT, PRIORITY, MIN_RUNTIME, WORKSPACE, CLUSTER = (
        index[name]
        for name in (
            "tag",
            "experiment_id",
            "status",
            "checkpoint",
            "priority",
            "min_runtime",
            "workspace",
            "cluster",
        )
    )


def fields(line):
    f = line.split("|")
    return f if len(f) > STATUS and not line.lstrip().startswith("#") else None


def job_state(experiment_id):
    out = subprocess.run(
        ["beaker", "experiment", "get", experiment_id, "--format", "json"],
        capture_output=True,
        text=True,
    )
    if out.returncode:
        return "error", out.stderr.strip().splitlines()[-1:]
    s = json.loads(out.stdout)[0]["jobs"][-1]["status"]
    if s.get("exited"):
        return "exited", s.get("exitCode")
    return ("running" if s.get("started") else "pending"), None


def training_states(root):
    cache = {}
    for path in sorted(pathlib.Path(root).rglob("experiments.txt")):
        for line in path.read_text().splitlines():
            f = fields(line)
            if f and f[STATUS] == "training":
                if f[EXPER_ID] not in cache:
                    cache[f[EXPER_ID]] = job_state(f[EXPER_ID])
                yield path, f, *cache[f[EXPER_ID]]


def config_name(status):
    if status.startswith("run_inf_"):
        return f"inference-config-{status[len('run_inf_'):]}.yaml"
    if status.startswith("run_"):
        return f"evaluator-config-{status[len('run_'):]}.yaml"
    sys.exit(f"status must start with run_ or run_inf_: {status}")


def field(f, i):
    return f[i] if len(f) > i else ""


def launch_workspace(root, cluster, workspace, cache={}):
    key = (cluster, workspace)
    if key not in cache:
        script = (
            'source job_runner/lib.sh; build_cluster_args "$1" "$2"; '
            'for i in "${!CLUSTER_ARGS[@]}"; do '
            "[[ ${CLUSTER_ARGS[$i]} == --workspace ]] && "
            'echo "${CLUSTER_ARGS[$((i+1))]}"; done; true'
        )
        out = subprocess.run(
            ["bash", "-c", script, "_", cluster, workspace],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        cache[key] = out.stdout.split()[0]
    return cache[key]


LAUNCHABLE = ("running_", "run_")
NEW_PREFIX = {"ran": "ran_", "in-flight": "running_", "failed": "failed_"}


def launch_tag(status):
    for prefix in LAUNCHABLE:
        if status.startswith(prefix):
            return status[len(prefix) :]
    return None


def launched_name(f):
    tag = launch_tag(f[STATUS])
    kind, tag = ("inf", tag[len("inf_") :]) if tag.startswith("inf_") else ("eval", tag)
    name = f"{f[0]}-{kind}_{f[CKPT]}-{tag}"
    return f"{name}-{f[TAG]}" if f[TAG] else name


def launched(workspace, name):
    out = subprocess.run(
        [
            "beaker",
            "workspace",
            "experiments",
            workspace,
            "--text",
            name,
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pattern = re.compile(re.escape(name) + r"(-[0-9a-f]{4})?")
    found = []
    for e in json.loads(out.stdout or "[]"):
        if pattern.fullmatch(e["name"]):
            s = e["jobs"][-1]["status"] if e.get("jobs") else {}
            done = s.get("exited") or s.get("finalized") or s.get("canceled")
            state = "exited" if done else ("running" if s.get("started") else "pending")
            found.append((e["name"], e["id"], state, s.get("exitCode")))
    return found


def classify(name, found):
    ok = [m for m in found if m[2] == "exited" and m[3] == 0]
    if any(m[0] == name for m in ok):
        return "ran"
    if ok:
        return "suffixed-success"
    if any(m[2] != "exited" for m in found):
        return "in-flight"
    return "failed" if found else "not-launched"


def cmd_mark(args):
    root = repo_root(args.dir)
    for path in sorted(pathlib.Path(args.dir).rglob("experiments.txt")):
        lines = path.read_text().splitlines()
        changed = False
        for i, line in enumerate(lines):
            f = fields(line)
            if not (f and launch_tag(f[STATUS]) is not None):
                continue
            name = launched_name(f)
            found = launched(
                launch_workspace(root, field(f, CLUSTER), field(f, WORKSPACE)), name
            )
            kind = classify(name, found)
            if kind == "suffixed-success" and name in args.accept:
                kind = "ran"
            new = NEW_PREFIX.get(kind, "") + launch_tag(f[STATUS])
            if kind in NEW_PREFIX and new != f[STATUS]:
                f[STATUS] = new
                lines[i] = "|".join(f)
                changed = True
                kind += f" -> {new}"
            matches = ",".join(f"{n}:{x}:{s}:{c}" for n, x, s, c in found)
            print(f"{path}|{name}|{kind}|{matches}")
        if changed:
            path.write_text("\n".join(lines) + "\n")


def cmd_status(args):
    for path, f, state, code in training_states(args.dir):
        print(f"{path}|{f[0]}|{f[EXPER_ID]}|{state}|{code}")


def cmd_apply(args):
    rows = [(r.split(":", 1) + [None])[:2] for r in args.row]
    finished: dict[pathlib.Path, set[str]] = {}
    for path, f, state, code in training_states(args.dir):
        if state == "exited" and code == 0:
            finished.setdefault(path, set()).add(f[EXPER_ID])
    for path, ids in finished.items():
        config_dir = (
            repo_root(args.dir) / args.config_dir if args.config_dir else path.parent
        )
        for status, _ in rows:
            if not (config_dir / config_name(status)).is_file():
                print(
                    f"warning: {config_dir / config_name(status)} not found",
                    file=sys.stderr,
                )
        out = []
        for line in path.read_text().splitlines():
            f = fields(line)
            if not (f and f[EXPER_ID] in ids and f[STATUS] == "training"):
                out.append(line)
                continue
            if args.keep_training:
                out.append(line)
            for status, runtime in rows:
                g = f + [""] * (MIN_RUNTIME + 1 - len(f))
                g[STATUS] = status
                if args.priority:
                    g[PRIORITY] = args.priority
                if runtime:
                    g[MIN_RUNTIME] = f"--min-runtime {runtime}"
                out.append("|".join(g))
        comment = f"# --config-dir {args.config_dir}" if args.config_dir else None
        if comment and comment not in out:
            out[1:1] = ["", comment]
        path.write_text("\n".join(out) + "\n")
        print(f"{path}: {sorted(ids)}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(required=True)
    m = sub.add_parser("mark")
    m.add_argument("dir")
    m.add_argument(
        "--accept",
        action="append",
        default=[],
        help="expected name whose suffixed success counts as ran",
    )
    m.set_defaults(func=cmd_mark)
    s = sub.add_parser("status")
    s.add_argument("dir")
    s.set_defaults(func=cmd_status)
    a = sub.add_parser("apply")
    a.add_argument("dir")
    a.add_argument("--row", action="append", required=True, help="STATUS[:MIN_RUNTIME]")
    a.add_argument("--priority", help="default: copied from the training row")
    a.add_argument("--keep-training", action="store_true")
    a.add_argument("--config-dir", help="repo-root-relative, as passed to evaluate.sh")
    a.set_defaults(func=cmd_apply)
    args = p.parse_args()
    load_columns(repo_root(args.dir))
    args.func(args)


if __name__ == "__main__":
    main()

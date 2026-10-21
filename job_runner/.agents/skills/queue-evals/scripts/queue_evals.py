#!/usr/bin/env python3
"""Replace finished `training` rows of experiments.txt with evaluate/inference rows.

  queue_evals.py mark <dir> [--accept <expected_name> ...]
  queue_evals.py status <dir> [FILTERS]
  queue_evals.py settle <dir> [FILTERS]
  queue_evals.py apply <dir> --row run_<tag>[:<min_runtime>] [--row ...]
      [--priority P] [--keep-training] [--config-dir PATH|auto] [--settle-rest]
      [FILTERS]

  FILTERS: [--only <experiment_id> ...] [--exclude <experiment_id> ...]
      [--group-regex RE]

<dir> is searched recursively for experiments.txt. A `training` row's state is
its experiment's final beaker job (jobs[-1]): pending, running, canceled, or
exited with an exit code. A row is finished when that job exited 0. `settle`
sets each final `training` row to trained (exit 0), canceled, or failed.

`apply` checks that each row's config (evaluator-config-<tag>.yaml, or
inference-config-<tag>.yaml for run_inf_<tag>) exists where evaluate.sh /
inference.sh will look for it. A --row without :<min_runtime> takes the
config's `# --min-runtime <t>` comment, else the training row's value.
`--config-dir auto` uses, per experiments.txt, its own directory when that
holds every requested config, else the one configs/experiments/*_eval/*/ dir
whose requested configs' file_pattern set equals that of the experiment's
train-config.yaml + finetune-config.yaml; any other outcome skips the file.
When the configs are not in the experiments.txt directory, a
`# --config-dir <dir>` note for readers goes above the first queued row; the
script never reads these notes back.
`--settle-rest` then settles every remaining `training` row, filters ignored.

`mark` looks up each run_* / running_* row's launched job by the name evaluate.sh /
inference.sh give it, `<group>-{eval,inf}_<ckpt>-<tag>[-<row tag>]`, in the
workspace lib.sh resolves for the row's cluster. Beaker appends `-<4 hex>`
when the name is taken, so suffixed names are matched too, but only an exact
name that exited 0 is marked ran_ unless passed to --accept. Rows whose job
is pending or running become running_*, and rows whose every match failed
become failed_*; running_* rows are rechecked on the next `mark`.
"""

import argparse
import functools
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


@functools.cache
def job_state(experiment_id):
    if not experiment_id:
        return "error", ["empty experiment_id"]
    out = subprocess.run(
        ["beaker", "experiment", "get", experiment_id, "--format", "json"],
        capture_output=True,
        text=True,
    )
    if out.returncode:
        return "error", out.stderr.strip().splitlines()[-1:]
    s = json.loads(out.stdout)[0]["jobs"][-1]["status"]
    if s.get("exited") and s.get("exitCode") == 0:
        return "exited", 0
    if s.get("canceled"):
        return "canceled", s.get("exitCode")
    if s.get("exited"):
        return "exited", s.get("exitCode")
    return ("running" if s.get("started") else "pending"), None


def settled_status(state, code):
    if state == "exited":
        return "trained" if code == 0 else "failed"
    return "canceled" if state == "canceled" else None


def selected(f, args):
    return args is None or (
        (not args.only or f[EXPER_ID] in args.only)
        and f[EXPER_ID] not in args.exclude
        and (not args.group_regex or re.search(args.group_regex, f[0]))
    )


def training_states(root, args=None):
    for path in sorted(pathlib.Path(root).rglob("experiments.txt")):
        for line in path.read_text().splitlines():
            f = fields(line)
            if f and f[STATUS] == "training" and selected(f, args):
                yield path, f, *job_state(f[EXPER_ID])


def row_key(f):
    return f[0], f[EXPER_ID]


def config_name(status):
    if status.startswith("run_inf_"):
        return f"inference-config-{status[len('run_inf_') :]}.yaml"
    if status.startswith("run_"):
        return f"evaluator-config-{status[len('run_') :]}.yaml"
    sys.exit(f"status must start with run_ or run_inf_: {status}")


def file_patterns(config):
    text = config.read_text() if config.is_file() else ""
    return {m.strip("'\"") for m in re.findall(r"file_pattern:\s*(\S+)", text)}


def auto_config_dir(root, exper_dir, statuses):
    if all((exper_dir / config_name(s)).is_file() for s in statuses):
        return None, None
    train = set().union(
        *(
            file_patterns(exper_dir / n)
            for n in ("train-config.yaml", "finetune-config.yaml")
        )
    )
    if not train:
        return None, "no file_pattern in train-config.yaml or finetune-config.yaml"
    matches = [
        d.relative_to(root)
        for d in sorted(root.glob("configs/experiments/*_eval/*/"))
        if all((d / config_name(s)).is_file() for s in statuses)
        and set().union(*(file_patterns(d / config_name(s)) for s in statuses)) == train
    ]
    if len(matches) != 1:
        return None, f"{len(matches)} eval dirs match its file_patterns: {matches}"
    return str(matches[0]), None


def runtime_hint(config):
    text = config.read_text() if config.is_file() else ""
    m = re.search(r"#\s*--min-runtime\s+(\S+)", text)
    return m.group(1) if m else None


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
    for path, f, state, code in training_states(args.dir, args):
        print(f"{path}|{f[0]}|{f[EXPER_ID]}|{state}|{code}")


def settle(root, args=None, skip=()):
    new_status: dict[pathlib.Path, dict[tuple, str]] = {}
    for path, f, state, code in training_states(root, args):
        new = settled_status(state, code)
        if new and path not in skip:
            new_status.setdefault(path, {})[row_key(f)] = new
    for path, by_key in new_status.items():
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            f = fields(line)
            if f and f[STATUS] == "training" and row_key(f) in by_key:
                f[STATUS] = by_key[row_key(f)]
                lines[i] = "|".join(f)
                print(f"{path}|{f[0]}|{f[EXPER_ID]}|{f[STATUS]}")
        path.write_text("\n".join(lines) + "\n")


def cmd_settle(args):
    settle(args.dir, args)


def cmd_apply(args):
    root = repo_root(args.dir)
    rows = [(r.split(":", 1) + [None])[:2] for r in args.row]
    statuses = [status for status, _ in rows]
    finished: dict[pathlib.Path, set[tuple]] = {}
    for path, f, state, code in training_states(args.dir, args):
        if state == "exited" and code == 0:
            finished.setdefault(path, set()).add(row_key(f))
    skipped = set()
    for path, keys in finished.items():
        config_dir_arg = args.config_dir
        if config_dir_arg == "auto":
            config_dir_arg, err = auto_config_dir(root, path.parent, statuses)
            if err:
                print(f"error: {path}: {err}; skipped", file=sys.stderr)
                skipped.add(path)
                continue
        config_dir = root / config_dir_arg if config_dir_arg else path.parent
        lines = path.read_text().splitlines()
        shared = config_dir.resolve() != path.parent.resolve()
        comment = f"# --config-dir {config_dir_arg}" if shared else None
        runtimes = {}
        for status, runtime in rows:
            config = config_dir / config_name(status)
            if not config.is_file():
                print(f"warning: {config} not found", file=sys.stderr)
            runtimes[status] = runtime or runtime_hint(config)
            if not runtimes[status]:
                print(
                    f"warning: {config}: no --min-runtime hint; "
                    "min_runtime copied from the training row",
                    file=sys.stderr,
                )
        out = []
        for line in lines:
            f = fields(line)
            if not (f and row_key(f) in keys and f[STATUS] == "training"):
                out.append(line)
                continue
            if comment:
                if [x for x in out if x.strip()][-1:] != [comment]:
                    out += ["", comment] if out and out[-1].strip() else [comment]
                comment = None
            if args.keep_training:
                out.append(line)
            for status in statuses:
                g = f + [""] * (MIN_RUNTIME + 1 - len(f))
                g[STATUS] = status
                if args.priority:
                    g[PRIORITY] = args.priority
                if runtimes[status]:
                    g[MIN_RUNTIME] = f"--min-runtime {runtimes[status]}"
                out.append("|".join(g))
        path.write_text("\n".join(out) + "\n")
        print(f"{path}: {config_dir_arg or path.parent}: {sorted(k[1] for k in keys)}")
    if args.settle_rest:
        settle(args.dir, skip=skipped)


def add_filters(parser):
    parser.add_argument("--only", action="append", default=[], metavar="EXPERIMENT_ID")
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="EXPERIMENT_ID"
    )
    parser.add_argument("--group-regex", help="re.search on the group column")


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
    add_filters(s)
    s.set_defaults(func=cmd_status)
    t = sub.add_parser("settle")
    t.add_argument("dir")
    add_filters(t)
    t.set_defaults(func=cmd_settle)
    a = sub.add_parser("apply")
    a.add_argument("dir")
    a.add_argument("--row", action="append", required=True, help="STATUS[:MIN_RUNTIME]")
    a.add_argument("--priority", help="default: copied from the training row")
    a.add_argument("--keep-training", action="store_true")
    a.add_argument(
        "--config-dir",
        help="repo-root-relative, as passed to evaluate.sh, or `auto`",
    )
    a.add_argument(
        "--settle-rest",
        action="store_true",
        help="then settle every remaining training row, filters ignored",
    )
    add_filters(a)
    a.set_defaults(func=cmd_apply)
    args = p.parse_args()
    load_columns(repo_root(args.dir))
    args.func(args)


if __name__ == "__main__":
    main()

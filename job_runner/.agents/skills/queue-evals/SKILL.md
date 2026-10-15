---
name: queue-evals
description: Mark `run_*` rows of experiments.txt files as `ran_*`, `running_*` or `failed_*` from their beaker job, then check beaker for every `training` row in the experiments.txt files under a job_runner experiment directory (nested dirs included), replace each finished row (final job exit 0) with `run_<tag>` / `run_inf_<tag>` rows for evaluate.sh / inference.sh, and optionally settle the rest as `trained` / `canceled` / `failed`. Edits stay uncommitted; nothing is launched. Use when asked to queue evaluation or inference for trainings that have finished.
---

# queue-evals

Script: `scripts/queue_evals.py` next to this file. Run it from the repo root
on the branch that owns the experiment directory (`../../../AGENTS.md`,
"Check the branch").

## Row semantics

`experiments.txt` columns are `EXPERIMENTS_COLUMNS` in `../../../lib.sh`,
which the script also reads. The `status` column decides which script acts on
a row:

| status | acted on by | config read |
| --- | --- | --- |
| `training` | nothing (written by the train scripts) | — |
| `trained` / `canceled` / `failed` | nothing (written by `settle`) | — |
| `run_<tag>` | `evaluate.sh` | `evaluator-config-<tag>.yaml` |
| `run_inf_<tag>` | `inference.sh` | `inference-config-<tag>.yaml` |
| anything else (`ran_`, `running_`, `failed_`, `resumed`, …) | nothing | — |

Neither script rewrites the status after launching, so launched rows are
re-prefixed in step 1 (`running_` / `ran_` / `failed_`) to keep a rerun from
relaunching them. Switching a `failed_` row back to `run_` relaunches it.
Configs are read from `<experiment_dir>/<config_subdir>/` unless
`--config-dir` is passed (`../../../README.md`, "Shared Config Directory").
Blank `priority` / `min_runtime` fall back to the defaults in `evaluate.sh`.

## Workflow

1. **Mark launched rows.** `python3 <skill>/scripts/queue_evals.py mark <dir>`
   finds each `run_*` or `running_*` row's job by the name the launcher gives it
   (`<group>-eval_<ckpt>-<tag>-<row tag>`, `-inf_` for inference), in the
   workspace `lib.sh` resolves from the row's `workspace`/`cluster` fields. It
   prints `path|expected_name|class|matches` and re-prefixes the status:

   | class | meaning | action |
   | --- | --- | --- |
   | `ran` | exact name, final job exit 0 | marked `ran_` |
   | `suffixed-success` | only `<name>-<4 hex>` exited 0 | report, ask |
   | `in-flight` | a match is pending or running | marked `running_` |
   | `failed` | every match exited ≠ 0, or was canceled | marked `failed_` |
   | `not-launched` | no match | none |

   Beaker appends `-<4 hex>` when a name is already taken, usually by an
   earlier failed attempt of the same row, so a suffixed success is most
   likely this row's retry. It could also be a different launch that
   reused the name. Show the user every `suffixed-success` row with its
   matches, and ask which ones count as ran. Then rerun with
   `--accept <expected_name>` for each one they confirm. `running_` rows are
   rechecked on every `mark`, so a later run moves them to `ran_` or
   `failed_`.
2. `python3 <skill>/scripts/queue_evals.py status <dir> [FILTERS]` prints
   `path|group|experiment_id|state|exit_code` for each `training` row.
   Only `jobs[-1]` counts (`../../../AGENTS.md`, "Reading results"):

   | state | `jobs[-1].status` |
   | --- | --- |
   | `exited`, code 0 | finished |
   | `canceled` | `canceled` set (beaker or by hand, before or after start) |
   | `exited`, code ≠ 0 | crashed |
   | `running` / `pending` | started / not started |

   FILTERS select rows in `status`, `settle` and `apply`:
   `--only <experiment_id>` and `--exclude <experiment_id>` (repeatable),
   `--group-regex <re>` (`re.search` on `group`).
3. From the request, settle the statuses (`run_<tag>` and/or `run_inf_<tag>`),
   `priority`, the config(s), which finished rows to queue, and whether to
   settle the rest. When in doubt about the tags or the configs, ask the
   user.
4. Apply, once per set of experiments.txt files that share a config dir:
   ```
   python3 <skill>/scripts/queue_evals.py apply <dir> \
     --row run_<tag_a> --row run_<tag_b>[:<min_runtime>] \
     [--priority high] [--config-dir <repo-relative dir>|auto] \
     [--keep-training] [--settle-rest] [FILTERS]
   ```
   Each finished, selected `training` row becomes one row per `--row`. The
   other fields are copied, then `--priority` overrides theirs, and
   `min_runtime` comes from, in order: the `:<min_runtime>` suffix, the
   config's `# --min-runtime <t>` comment, the training row.
   - `--config-dir auto` uses, per experiments.txt, its own directory when
     that holds every requested config. Otherwise it uses the one
     `configs/experiments/*_eval/*/` dir holding every requested config
     whose `file_pattern` set equals that of the experiment's
     `train-config.yaml` + `finetune-config.yaml`. Zero or several matches
     skip the file with an error: ask the user which configs to use, then
     pass the dir explicitly.
   - When the configs are not in the experiments.txt directory, a
     `# --config-dir <dir>` line goes directly above the first queued row,
     unless the nearest line above already says the same. It is a note for
     whoever launches: one file may hold several, each read as covering the
     rows below it, and a file with local configs may have none. Nothing
     parses it; the launch still needs `--config-dir` on the command line.
   - A warning means a config the launcher will look for is missing, or has
     no `--min-runtime` hint.
   - `--settle-rest` then runs `settle` on every remaining `training` row
     under `<dir>`, ignoring FILTERS, except in skipped files.
5. `python3 <skill>/scripts/queue_evals.py settle <dir> [FILTERS]` sets each
   `training` row whose final job is done to `trained` (exit 0), `canceled`
   or `failed`, and prints `path|group|experiment_id|new status`. Pending and
   running rows stay `training`. Run it only when asked to settle rows
   without queueing them.
6. Report the diff plus, per experiments.txt: rows marked `ran_`, `running_` and
   `failed_`, ids queued, rows settled, ids still running/pending, and failed
   ids (exit ≠ 0) left untouched. Do not commit and do not launch, unless asked
   to. `evaluate.sh <experiment_dir> <config_subdir> --dry-run [--config-dir
   <dir>]` previews the launch.

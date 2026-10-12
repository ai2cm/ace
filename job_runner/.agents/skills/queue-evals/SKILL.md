---
name: queue-evals
description: Mark `run_*` rows of experiments.txt files as `ran_*`, `running_*` or `failed_*` from their beaker job, then check beaker for every `training` row in the experiments.txt files under a job_runner experiment directory (nested dirs included), and replace each finished row (final job exit 0) with `run_<tag>` / `run_inf_<tag>` rows for evaluate.sh / inference.sh. Edits stay uncommitted; nothing is launched. Use when asked to queue evaluation or inference for trainings that have finished.
---

# queue-evals

Script: `scripts/queue_evals.py` next to this file. Run it from the repo root
on the branch that owns the experiment directory (`../../../AGENTS.md`,
"Check the branch").

## Row semantics

`experiments.txt` columns are in `../../../README.md`. The `status` field
(field 4) decides which script acts on a row:

| status | acted on by | config read |
| --- | --- | --- |
| `training` | nothing (written by the train scripts) | — |
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
2. `python3 <skill>/scripts/queue_evals.py status <dir>` prints
   `path|group|experiment_id|state|exit_code` for each `training` row.
   Only `jobs[-1]` counts (`../../../AGENTS.md`, "Reading results").
3. From the request, settle the statuses (`run_<tag>` and/or `run_inf_<tag>`),
   per-status `min_runtime`, `priority`, and the config dir. Ask when the
   request leaves the tags or the config dir open.
4. Apply, once per set of experiments.txt files that share a config dir:
   ```
   python3 <skill>/scripts/queue_evals.py apply <dir> \
     --row run_<tag_a>:1h --row run_<tag_b>:2h \
     [--priority high] [--config-dir <repo-relative dir>] [--keep-training]
   ```
   Each finished `training` row becomes one row per `--row`. The other fields
   are copied, then `--priority` and the `:<min_runtime>` suffix override
   theirs. `--config-dir` also adds a `# --config-dir <dir>` line under the
   header. A warning means the config file the launcher will look for is
   missing.
5. Report the diff plus, per experiments.txt: rows marked `ran_`,
   `running_` and `failed_`, ids
   queued, ids still running/pending, and failed ids (exit ≠ 0), which stay
   untouched. Do not
   commit and do not launch. `evaluate.sh <experiment_dir> <config_subdir>
   --dry-run [--config-dir <dir>]` previews the launch.

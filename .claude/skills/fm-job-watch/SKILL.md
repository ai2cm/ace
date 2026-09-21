---
name: fm-job-watch
description: One tick of the FM (2026-06-26) Beaker job watcher. Fills in missing norm-ablation jobs whose dependencies have landed, retries failed ones (seed bump on non-finite loss), and reports exhausted jobs. Run every 20 minutes from an in-session cron; also runnable by hand.
---

# FM job watcher tick

Runs `configs/experiments/2026-06-26-fm/watch_fm_jobs.py` once. The script
does the work; this skill is the fixed way to invoke it and to act on its
output.

## Run

From the repository root of the worktree on branch `exp/alexeyfm-2`
(`/Users/alexeyy/orca/workspaces/ace/exp-alexeyfm-2`):

```bash
cd /Users/alexeyy/orca/workspaces/ace/exp-alexeyfm-2/configs/experiments/2026-06-26-fm && \
PATH=/Users/alexeyy/mamba/envs/fme/bin:$HOME/.local/bin:$PATH \
PYTHONPATH=/Users/alexeyy/orca/workspaces/ace/exp-alexeyfm-2 \
/Users/alexeyy/mamba/envs/fme/bin/python watch_fm_jobs.py
```

Use a 20 minute Bash timeout (`timeout: 1200000`); a tick with many
submissions runs several minutes. Never run two ticks at once.

## What one tick does

1. `git pull --ff-only origin exp/alexeyfm`, list `ai2/ace`, refresh
   `wandb_to_beaker_map.json`, commit and push if it changed.
2. For every expected job (nc-sfno and nc-swin-v2 x cells x stages; nc-swin-v2.1
   is excluded via `EXCLUDED_ARCHS`) that is missing
   or failed and whose dependencies succeeded: generate configs, commit, push,
   submit with `--skip-if-in-beaker` on `ai2/jupiter ai2/titan` at priority
   `normal`. Failures are retried up to 5 times; a swin training or fine-tune
   with a non-finite loss is resubmitted at seed + 1 via `seed_overrides.json`.
3. Prints a summary. State is `job_watch_state.json` (gitignored).

## After the run

- If the output contains any `NOTIFY:` line, call `PushNotification` with
  those lines verbatim (one notification per tick) and repeat them to the
  user. These are jobs that used up their retries.
- If the script exited non-zero, report the last 30 lines of output to the
  user. Do not retry within the same tick.
- Otherwise reply with the `=== FM job watcher summary ===` block only.

## Re-arming after a session restart

The cron lives in the Claude session and dies with it. To re-arm:

```
CronCreate every 20 minutes: /fm-job-watch
```

To stop: `CronDelete` on that cron. To clear a job's retry count or its
exhausted mark, edit `job_watch_state.json` by hand.

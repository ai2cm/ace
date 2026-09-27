---
name: fm-job-watch
description: One tick of the FM (2026-06-26) Beaker job watcher. Fills in missing norm-ablation and paper-replication jobs whose dependencies have landed, retries failed ones (seed bump on non-finite loss), reports exhausted jobs, and says when everything is done. Run every 30 minutes from an in-session cron; also runnable by hand.
---

# FM job watcher tick

Runs `configs/experiments/2026-06-26-fm/watch_fm_jobs.py` once. The script
does the work; this skill is the fixed way to invoke it and to act on its
output.

## Run

From the current working directory's checkout, which must be on branch
`exp/alexeyfm` (the script pulls from and pushes to `origin/exp/alexeyfm`).
Run it as a background Bash command: a tick that submits many jobs takes
longer than the foreground tool timeout, and the script holds a lock file
(`job_watch.lock`) so a tick that starts while one is still running exits
at once with "tick already running".

```bash
cd "$(git rev-parse --show-toplevel)/configs/experiments/2026-06-26-fm" && \
PATH=/Users/alexeyy/mamba/envs/fme/bin:$HOME/.local/bin:$PATH \
PYTHONPATH="$(git rev-parse --show-toplevel)" \
/Users/alexeyy/mamba/envs/fme/bin/python watch_fm_jobs.py
```

Use `run_in_background: true` and read the output file when the task
notification arrives. Never start a second tick by hand while one is
running.

## What one tick does

1. `git pull --ff-only origin exp/alexeyfm`, list `ai2/ace`, refresh
   `wandb_to_beaker_map.json`, commit and push if it changed.
2. For every expected job (nc-sfno, nc-swin-v2 and nc-swin-v2.1 x cells x
   stages: train, eval, sst, q0, finetune, ft-sst, paper) that is missing or
   failed and whose dependencies succeeded: generate configs, commit, push,
   submit with `--skip-if-in-beaker` on `ai2/jupiter ai2/titan` at priority
   `normal`. Failures are retried up to 5 times; a swin training or fine-tune
   with a non-finite loss is resubmitted at seed + 1 via `seed_overrides.json`.
   A job the user canceled is left alone for good.
3. Prints a summary. State is `job_watch_state.json` (gitignored).

## After the run

- If the output contains any `NOTIFY:` line, call `PushNotification` with
  those lines verbatim (one notification per tick) and repeat them to the
  user. These are jobs that used up their retries.
- If the output contains a `DONE:` line, every expected job other than the
  1000-year slab runs has succeeded or been canceled. Call
  `PushNotification` with "FM jobs complete" plus the `DONE:` line and the
  `canceled:` lines, repeat them to the user, and delete the watcher cron
  (`CronList`, then `CronDelete` on the `/fm-job-watch` entry).
- If the script exited non-zero, report the last 30 lines of output to the
  user. Do not retry within the same tick.
- Otherwise reply with the `=== FM job watcher summary ===` block only.

## Re-arming after a session restart

The cron lives in the Claude session and dies with it. To re-arm:

```
CronCreate every 30 minutes: /fm-job-watch
```

To stop: `CronDelete` on that cron. To clear a job's retry count or its
exhausted mark, edit `job_watch_state.json` by hand.

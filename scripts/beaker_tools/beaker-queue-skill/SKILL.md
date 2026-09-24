---
name: beaker-queue
description: Answer "what is running / queued / ahead of my job on Beaker cluster X" for a budget's allocated jobs. Use whenever the user asks about a Beaker cluster's queue, slot quota, why a job has been queued for hours, whether a running job can be preempted, or who is using the budget's slots (default budget ai2/atec-climate).
---

# Beaker queue

Run the bundled script and relay its output. Do not reconstruct the queue by hand from
`beaker job list`. The script is `scripts/beaker_queue.py` under this skill's base directory.

```
python3 <skill base directory>/scripts/beaker_queue.py <cluster> [--budget ai2/<name>] [--include-unallocated] [--json]
```

Examples: `ai2/jupiter` (default budget `ai2/atec-climate`); `ai2/titan --budget ai2/atec-olmoearth`;
add `--include-unallocated` for two more tables covering the budget's backfill jobs.

## How Beaker schedules

Verified against the scheduler source (allenai/beaker, `scheduling/internal/sortpolicy.go`,
`preemption.go`, `resources.go`, commit ae24142 of 2026-09-24), which implements each cluster's
`schedulerPolicy` as written.

- Each budget has two numbers per cluster (`beaker allocation list <cluster> --org ai2`): a
  runtime quota percentage, whose share of the cluster's slots is its **target**, and a
  `slotLimit`, the **burst cap** it may reach when slots are free. A budget can run over target
  but never over the cap. Budgets do not compete with each other for those slots.
- Jobs submitted with a minimum runtime are **allocated**. They compete only with the same
  budget's other allocated jobs for its quota. Jobs with no minimum runtime are **unallocated**
  backfill: a separate, age-ordered, preemptible queue that never explains why an allocated job
  waits.
- The allocated queue sorts by allocation balance at the workspace-group level, then at the
  workspace level, then priority, then age. Balance is (quota − usage) / quota over a decaying
  multi-day window. A balance criterion is an entity boundary: priority and age only ever order
  jobs within one workspace. When every waiting job of the budget is in one workspace the balance
  keys tie and the order is priority, then age; the script checks this and says which case applies.
- A running allocated job is protected only for its minimum runtime. After that it is
  interruptible: a queued allocated job of higher priority in the same workspace can preempt it,
  as can one whose workspace is further under its target when the victim's workspace is over
  target. Ranking ahead only by age never licenses preemption. The script prints each running job's
  minimum runtime left; blank means it has elapsed and the job is interruptible now.
- Allocated means minimum runtime > 0, for running and queued jobs alike (`Allocated()` in
  `msg/job.go`; the same test in the `cluster usage` SQL). `beaker job list --cluster X` returns
  jobs that *requested* X, and a job naming several clusters may run on another one, so the script
  keeps running and scheduled jobs only when their node is on the cluster and shows, for queued
  jobs, the other clusters they could land on.
- A job with `status.scheduled` set but no `status.started` has a node and is starting (image
  pull, setup); it is no longer waiting on the queue.

## Reporting

The script prints markdown: a header with the budget's slots in use, its target and burst cap, and
its usage over the scheduler's lookback window against target (undecayed, so approximate), then
two tables: the
budget's allocated jobs running on the cluster with their minimum runtime left, and its allocated
jobs scheduled or queued in scheduling order. With `--include-unallocated` the same two tables
follow for the budget's backfill jobs. The logged-in user's jobs are marked `*`. Relay all the
tables as they are. Add at most a short paragraph answering the user's actual
question (is the budget at quota, what is ahead of their job, can their running job be preempted);
do not restate the tables in prose. Times are already local. If the script fails because `beaker`
is not logged in or the cluster name is wrong, say so and stop.

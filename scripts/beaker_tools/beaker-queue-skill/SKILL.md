---
name: beaker-queue
description: Answer "what is running / queued / ahead of my job on Beaker cluster X" for a budget's allocated jobs. Use whenever the user asks about a Beaker cluster's queue, slot quota, why a job has been queued for hours, whether a running job can be preempted, or who is using the budget's slots (default budget ai2/atec-climate).
---

# Beaker queue

Run the bundled script and relay its output. Do not reconstruct the queue by hand from
`beaker job list`. The script is `scripts/beaker_queue.py` under this skill's base directory.

```
uv run --script <skill base directory>/scripts/beaker_queue.py <cluster> [--budget ai2/<name>] [--include-unallocated] [--authors USER ...] [--json]
```

`uv` installs the script's one dependency, beaker-py, which it uses to read recent allocated usage.

Examples: `ai2/jupiter` (default budget `ai2/atec-climate`); `ai2/titan --budget ai2/atec-olmoearth`;
add `--include-unallocated` for two more tables covering the budget's backfill jobs.

## How Beaker schedules

Verified against the scheduler source (allenai/beaker, `scheduling/internal/sortpolicy.go`,
`preemption.go`, `resources.go`, commit ae24142 of 2026-09-24), which implements each cluster's
`schedulerPolicy` as written.

- Each budget has two numbers per cluster (`beaker allocation list <cluster> --org ai2`): a
  runtime quota percentage, whose share of the cluster's slots is its **target allocation**, and
  a `slotLimit`, the **slot limit** on the allocated slots it may have in use at once. Allocated
  slots in use can exceed the target allocation but never the slot limit. The price of exceeding
  it is the **balance score**, as the Beaker UI shows it: recent allocated usage minus target
  allocation, as a percentage of the target allocation, with usage decay-weighted over a
  multi-day window; positive means over target. A group with a positive balance score ranks
  behind every group with a lower one, from any budget, and its jobs past their minimum runtime
  can be preempted by them.
- Jobs submitted with a minimum runtime are **allocated**. They compete only with the same
  budget's other allocated jobs for its quota. Jobs with no minimum runtime are **unallocated**
  backfill: a separate, age-ordered, preemptible queue that never explains why an allocated job
  waits.
- The allocated queue sorts by balance score at the workspace-group level across the whole
  cluster (lowest first), then at the workspace level, then priority, then age. A balance
  criterion is an entity boundary: priority and age only ever order jobs within one group. Within
  a budget whose waiting jobs share one workspace the balance keys tie and the order is priority,
  then age; the script checks this and says which case applies. The header's balance score is
  computed as the Beaker UI computes it, from the `ListClusterUsage` RPC; like the UI's, it leaves
  out the slot-hours running jobs are guaranteed but have not used yet, so the scheduler's can be
  higher. A preemption message (`canceledFor` on the job) quotes the scheduler's exact figure.
- A running allocated job is protected only for its minimum runtime. After that it is
  interruptible: a queued allocated job of higher priority in the same group can preempt it, and so
  can a job of any group with a lower balance score, from any budget, while the victim's group has
  a positive balance score.
  Ranking ahead only by age never licenses preemption. The script prints each running job's
  minimum runtime left; blank means it has elapsed and the job is interruptible now.
- Allocated means minimum runtime > 0, for running and queued jobs alike (`Allocated()` in
  `msg/job.go`; the same test in the `cluster usage` SQL). `beaker job list --cluster X` returns
  jobs that *requested* X, and a job naming several clusters may run on another one, so the script
  keeps running and scheduled jobs only when their node is on the cluster and shows, for queued
  jobs, the other clusters they could land on.
- A job with `status.scheduled` set but no `status.started` has a node and is starting (image
  pull, setup); it is no longer waiting on the queue.
- Each queued job's `FailedScheduling` events (`beaker job events <job>`) give the scheduler's own
  reason, counted in nodes across every cluster the job can use: nodes without enough free slots,
  cordoned nodes, nodes excluded by an allocation slot limit, and nodes waiting for a job higher in
  the queue to be scheduled, named by job ID. That job can be from any budget, so the budget's own
  tables cannot show it.

## Reporting

The script prints markdown: a header with the budget's allocated slots in use and free under its
slot limit, its target allocation, and its balance score, then two tables: the
budget's allocated jobs running on the cluster with their minimum runtime left, and its allocated
jobs scheduled or queued in scheduling order. With `--include-unallocated` the same two tables
follow for the budget's backfill jobs. Jobs are named by experiment, as the Beaker UI lists
them. Under a queued table, a list gives each job that Beaker says must be scheduled first on
this cluster, from any budget, and the queued jobs it is ahead of. The logged-in user's jobs are
marked `*`. Relay all the tables and that list as they are. Add at most a short paragraph answering the user's actual
question (is the budget at its slot limit, what is ahead of their job, can their running job be
preempted); do not restate the tables in prose. Use the Beaker UI's terms (balance score, target
allocation, allocated slots in use or free, slot limit), and say which one a comparison is about:
allocated slots in use above the target allocation is not the same as a positive balance score.
Times are already local. When the header says
`beaker job list --cluster` is failing, queued jobs come only from the users it names;
say so if the answer depends on the queue being complete, and rerun with `--authors`
for anyone missing (the script remembers every user it has seen). If the script fails because `beaker`
is not logged in or the cluster name is wrong, say so and stop.

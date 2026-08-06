# Beaker priority balancer

Beaker does not enforce our urgent-priority allocation, so queueing urgent jobs
can quietly put the team over it. This script keeps us within the allocation
without anyone having to hand-manage priorities.

Our allocation is **72 GPU slots on `ai2/jupiter` and 32 on `ai2/titan`**. A
Beaker slot is one GPU; a CPU-only job counts as one slot.

## Opting a job in

Set `CM_PRIORITY` on the job to `low`, `normal`, `high` or `urgent`:

```bash
gantry run --env CM_PRIORITY=high ...
```

The balancer only modifies jobs that set it. Jobs that don't are left alone
entirely, though their urgent slots still count against the allocation.

`CM_PRIORITY` is both a ranking and a resting place. It decides who gets a
scarce urgent slot first, and what a job falls back to when it doesn't get one.
A job labelled `urgent` rests at `high`, since resting at urgent would defeat
the point.

Two things stop a job being managed:

- **It targets more than one cluster.** Beaker offers no way to pin a queued
  job to a cluster, so a job eligible for both jupiter and titan could consume
  either allocation and there is no way to know which in advance. Pin your job
  to one cluster if you want it balanced. The script logs a warning for
  labelled jobs it has to skip for this reason.
- **It targets a cluster with no allocation** (`ai2/saturn`, `ai2/ceres`).

## What it does

Each pass recomputes the whole allocation from scratch and applies the
difference:

1. Read every unfinished job in the workspace.
2. Subtract the slots held at urgent by jobs it cannot modify. Those are a
   fixed charge; only what is left is available.
3. Walk the managed jobs in `CM_PRIORITY` order, granting urgent while the
   job's cluster has room.
4. Set every managed job to urgent if granted, or to its resting priority if
   not.

Recomputing rather than adjusting incrementally is what makes step 3 work: a
`high` job is always considered before any `normal` job, so it never has to
wait for slots a lesser job already took. If `normal` jobs are holding urgent
slots that a `high` job needs, they are dropped to make room.

Within one `CM_PRIORITY` level, urgent is granted to running jobs that already
hold it first — so a queued job always gives up its slot before a running one —
ordered so the job that has held a GPU longest is the first to lose it. Queued
jobs follow, longest-waiting first. The two orderings use different clocks:
time on GPU for running jobs, and time since entering the queue for queued
ones. The queue clock resets when a job is preempted and requeued, so a job
that has been bounced does not keep seniority it no longer has.

A job too large for the remaining slots is skipped and smaller ones behind it
still get a chance. Nothing is reserved for it, because a level that could not
fit has already taken everything it was entitled to.

## What it will not do

**It never stops or preempts a job.** The only call it makes is
`UpdateJobSourcePriority`. Demoting a running job does not stop it — the job
keeps running and merely becomes preemptible again, which is the honest
consequence of being outside the allocation. Whether it is then preempted is
the cluster scheduler's decision, not ours.

**It never promotes a running job.** Beaker rejects this outright:

```
BeakerPermissionsError: cannot increase priority for running job "01K..."
```

So a job that starts at `high` stays there for life, even if urgent slots free
up later. Such a job is also excluded from the allocation walk — granting it
urgent on paper would displace jobs that *can* be promoted, for no benefit. If
it is preempted and returns to the queue it becomes eligible again, at its
proper rank.

The practical consequence: **submit at the priority you want, and let the
balancer take it away** rather than hoping to be raised later.

## Accounting

A job that has landed counts only against the cluster it is on. A *queued*
urgent job counts at full weight against every cluster it could land on.
Nothing outranks urgent and nothing can preempt it, so a queued urgent job is a
committed future occupant, not a maybe.

## Running it

Needs `beaker-py` and a Beaker token in the environment:

```bash
pip install beaker-py
python balance.py --dry-run          # report what a pass would change
python balance.py                    # apply one pass
python balance.py --interval 180     # keep running, one pass every 3 minutes
```

Useful flags: `--workspace` (default `ai2/ace`), `--limit CLUSTER=SLOTS`
(repeatable, overrides the allocation), `-v` for debug logging.

A pass is stateless and converges, so it is safe to run from cron:

```cron
*/3 * * * * /path/to/python /path/to/balance.py >> /var/log/beaker_balancer.log 2>&1
```

It runs as whoever's token is in the environment and modifies teammates' jobs
too. A job it lacks permission to change is logged and skipped rather than
failing the pass, so watch the log the first time it runs against jobs it does
not own.

## Tests

```bash
python -m pytest
```

`test_balance.py` covers the decision logic, including randomised populations
asserting the invariants that matter: a pass never pushes usage above the
allocation, a second pass is always a no-op (so cron cannot flap jobs), and
only labelled single-cluster jobs are ever touched — never raising a running
one. `test_beaker_io.py` drives the Beaker-facing layer against a fake client
built from real protobuf messages.

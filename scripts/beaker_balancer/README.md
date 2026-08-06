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

Things that stop a job being managed, each reported in the log as an aggregate
count per pass:

- **It targets more than one cluster.** Beaker offers no way to pin a queued
  job to a cluster, so a job eligible for both jupiter and titan could consume
  either allocation and there is no way to know which in advance. Pin your job
  to one cluster if you want it balanced. Most `ai2/ace` jobs today target
  `titan+jupiter`, so expect this to be the common case during adoption.
- **It targets a cluster with no allocation** (`ai2/saturn`, `ai2/ceres`).
- **It is part of a replica group.** The allocation walk would grant urgent to
  some ranks and not others, which for a synchronised-start group wastes the
  slots it did grant. It is also unconfirmed whether `UpdateJobSourcePriority`
  acts per job or per source; if per source, one call would move every replica.
- **It has landed on a node the script cannot resolve to a cluster.** Its slots
  are then charged to every budget as a precaution, which is inconsistent with
  managing it against one.

### What may surprise you

- **A job labelled `low` can still be promoted to `urgent`.** `CM_PRIORITY` is
  a ranking, not a ceiling: when the cluster is quiet, any labelled job is
  eligible. If you label a job `low` to be a good citizen, it may still be made
  unpreemptible and charged to the team allocation. It is simply the first to
  be dropped when the allocation gets tight.
- **A demotion is permanent for the life of that run.** Beaker will not raise
  the priority of a job it has already placed, so a job the balancer drops
  during a busy spell stays down even after the spell passes. It becomes
  eligible again only if it is preempted and returns to the queue.
- **Demotion deliberately targets the job with the most work at risk.** Within
  a `CM_PRIORITY` level, the job that has held its GPUs longest is the first to
  lose urgent, on the grounds that it has already had its turn.

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
(repeatable; merges into the default allocation rather than replacing it), `-v`
for debug logging. Cluster names in `--limit` are checked against Beaker at
startup, since a typo would otherwise manage nothing and look exactly like a
quiet cluster.

A pass is stateless and converges, so it is safe to run from cron. Note that
cron does not read your shell profile, so `BEAKER_TOKEN` has to be supplied
explicitly:

```cron
BEAKER_TOKEN=...
*/3 * * * * /path/to/python /path/to/balance.py >> /var/log/beaker_balancer.log 2>&1
```

There is no lockfile, so pick an interval comfortably longer than a pass (a pass
is a few seconds against the current workspace).

It runs as whoever's token is in the environment and modifies teammates' jobs
too. A job it lacks permission to change is logged and skipped rather than
failing the pass, so watch the log the first time it runs against jobs it does
not own. Demotions are attempted before promotions, so a pass that dies partway
through is never left over allocation.

## Tests

```bash
python -m pytest
```

Requires `beaker-py`, which is not a core `fme` dependency; both test modules
skip themselves when it is absent so a plain `pytest` at the repo root still
collects cleanly.

`test_balance.py` covers the decision logic, including randomised populations
asserting the invariants that matter: a pass never pushes usage above the
allocation, *any prefix* of a pass stays within it, a second pass is always a
no-op (so cron cannot flap jobs), and only labelled, manageable jobs are ever
touched — never raising a placed one. Those tests compute usage with an oracle
written independently of the production accounting, so a bug in the accounting
cannot hide behind itself; a further test asserts the random populations
actually reach the states that matter, so the suite cannot quietly go vacuous.

`test_beaker_io.py` drives the Beaker-facing layer against a fake client built
from real protobuf messages, covering environment-variable and
placement-constraint parsing, org-qualified cluster resolution, the
scheduled-but-not-started case, slot accounting, dry-run, action ordering, and
fail-soft behaviour.

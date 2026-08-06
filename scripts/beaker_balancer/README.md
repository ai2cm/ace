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
- **It is an interactive session.** There is a person attached to it, and
  taking their priority away mid-session is not the script's call.
- **It is at `immediate` priority.** Beaker requires a human-supplied reason for
  `immediate`, so it represents a deliberate decision the balancer should not
  quietly undo. `immediate` is not an accepted `CM_PRIORITY` value either: it is
  not something the script may hand out.
- **It has landed on a node the script cannot resolve to a cluster.** Its slots
  are then charged to every budget as a precaution, which is inconsistent with
  managing it against one.

Sessions and `immediate` jobs still count against the allocation — see
[Accounting](#accounting).

### What may surprise you

- **A job labelled `low` can still be promoted to `urgent`.** `CM_PRIORITY` is
  a ranking, not a ceiling: when the cluster is quiet, any labelled job is
  eligible. If you label a job `low` to be a good citizen, it may still be made
  unpreemptible and charged to the team allocation. It is simply the first to
  be dropped when the allocation gets tight.
- **A queued job below its `CM_PRIORITY` is raised to it, grant or no grant.**
  The resting priority is where the label says the job belongs, so the balancer
  moves it there in both directions. Submitting at `low` with `CM_PRIORITY=high`
  therefore gets you `high`, not `low`, once a pass has seen the job. This costs
  the allocation nothing — only `urgent` is budgeted — but it does change how the
  job is scheduled against everyone else's.
- **A demotion is permanent for the life of that run.** Beaker will not raise
  the priority of a job it has already placed, so a job the balancer drops
  during a busy spell stays down even after the spell passes. It becomes
  eligible again only if it is preempted and returns to the queue.
- **Demotion deliberately targets the job with the most work at risk.** Within
  a `CM_PRIORITY` level, the job that has held its GPUs longest is the first to
  lose urgent, on the grounds that it has already had its turn.

## Replica groups

Beaker runs a multi-node job as one job per rank. The balancer decides for the
whole **replica group** at once and grants urgent all-or-nothing. A lone job is
simply a group of one.

A half-granted group is the worst available outcome: it cannot start until every
rank is placed, so it waits at the priority of its lowest rank while the granted
ranks hold slots the allocation has already spent. A four-rank group needing 80
slots against 72 free therefore gets nothing, rather than stranding 64 slots on
a job that still cannot run.

A group is managed only if *every* rank is: one unlabelled rank, one session, or
one at `immediate` makes the whole group untouchable. The same applies if the
ranks disagree about their `CM_PRIORITY`, or if the group comes back short —
only unfinished jobs in one workspace are read, so granting the ranks that
happen to be visible would half-grant the real group. Skipped groups are counted
in the log.

Splits *below* urgent are left alone deliberately. They cost the allocation
nothing, and raising a queued rank whose siblings are stuck at a lower priority
is what gets the group placed.

The grouping is keyed on the replica group rather than the workload, since one
workload can hold independent tasks that have no reason to move together.

Deciding for the group is only half of it: the ranks are still raised one call
at a time, so a refusal partway through would split the group anyway. A refused
rank therefore abandons the rest of the group's grant instead of spending more
slots on a job that cannot start. Ranks already raised keep urgent — the next
pass re-decides from what is really there and either finishes the group or takes
it back, which is cheaper than a rollback that can fail in its own turn. In
practice permission is per owner and every rank shares one, so the first rank
fails and nothing is spent.

## What it does

Each pass recomputes the whole allocation from scratch and applies the
difference:

1. Read every unfinished job in the workspace and group it with its ranks.
2. Subtract the slots held at urgent or `immediate` by jobs it cannot modify.
   Those are a fixed charge; only what is left is available.
3. Walk the managed groups in `CM_PRIORITY` order, granting urgent while the
   group's cluster has room for all of it.
4. Set every managed group to urgent if granted, or to its resting priority if
   not.

Recomputing rather than adjusting incrementally is what makes step 3 work: a
`high` group is always considered before any `normal` one, so it never has to
wait for slots a lesser one already took. If `normal` jobs are holding urgent
slots that a `high` job needs, they are dropped to make room.

Within one `CM_PRIORITY` level, urgent is granted to running groups that already
hold it first — so a queued group always gives up its slot before a running one
— ordered so the group that has held its GPUs longest is the first to lose it.
Queued groups follow, longest-waiting first. The two orderings use different
clocks: time on GPU for running groups, and time since entering the queue for
queued ones. A group's queue clock is that of its *latest* rank, because it
cannot start until the last one is placed; since the clock resets when a rank is
preempted and requeued, a group that has been bounced does not keep seniority it
no longer has.

A group too large for the remaining slots is skipped and smaller ones behind it
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
up later. A group with a placed rank below urgent is also excluded from the
allocation walk — granting it urgent on paper would displace groups that *can*
be promoted, for no benefit, and it could not be brought wholly to urgent
anyway. If it is preempted and returns to the queue it becomes eligible again,
at its proper rank.

The practical consequence: **submit at the priority you want, and let the
balancer take it away** rather than hoping to be raised later.

**It never touches an interactive session or an `immediate` job**, in either
direction.

## Accounting

A job that has landed counts only against the cluster it is on. A *queued*
urgent job counts at full weight against every cluster it could land on.
Nothing outranks urgent and nothing can preempt it, so a queued urgent job is a
committed future occupant, not a maybe.

`immediate` counts the same as urgent: it outranks urgent, so the slot is just
as occupied. Interactive sessions count too.

Sessions and `immediate` jobs are counted but never reclaimable, so a pass can
be unable to get within the allocation no matter what it does. Each pass logs
how many slots per cluster are held that way, so that shows up as a fact rather
than as a balancer that appears not to be working.

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
not own.

Such a refusal is why demotions are attempted before promotions, and why a
refused demotion also *defers* the grant it was paying for. Ordering alone makes
any prefix of a pass safe, but a refusal does not end the pass — it skips one
call and carries on — so what lands is an arbitrary subset. Granting urgent
after the demotion paying for it was refused would end the pass over the
allocation. The deferral is per cluster and lasts one pass: the next one
recomputes from the state that really exists. A group is deferred whole, since
it is granted whole.

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
no-op (so cron cannot flap jobs), a replica group is never left holding urgent
on only some of its ranks, sessions and `immediate` jobs are never touched, and
only labelled, manageable jobs are ever modified — never raising a placed one.
Those tests compute usage with an oracle
written independently of the production accounting, so a bug in the accounting
cannot hide behind itself; a further test asserts the random populations
actually reach the states that matter, so the suite cannot quietly go vacuous.

Those invariants are one-sided — never *above* the allocation — and a balancer
that granted nothing would satisfy every one of them, so one more asserts the
other half: any candidate group left below urgent was left there because it did
not fit in what remained, not because the walk overlooked it.

Those invariants are all properties of `decide`, which assumes every call it
plans succeeds. The ones that only exist once calls can fail belong to
`run_pass` and are tested there.

`test_beaker_io.py` drives the Beaker-facing layer against a fake client built
from real protobuf messages, covering environment-variable and
placement-constraint parsing, org-qualified cluster resolution, the
scheduled-but-not-started case, slot accounting, dry-run, action ordering,
fail-soft behaviour, the per-pass reports, and the entrypoint's one-shot and
`--interval` modes. Its randomised passes make an arbitrary subset of the calls
fail and assert the pass still never ends above the allocation — the property
that the ordering alone does not buy.

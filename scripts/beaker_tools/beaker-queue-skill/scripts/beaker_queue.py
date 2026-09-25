#!/usr/bin/env python3
"""Where a budget's allocated jobs stand on a Beaker cluster.

Each budget holds an allocation on a cluster (`beaker allocation list <cluster> --org
ai2`): a runtime quota, whose share of the cluster's slots is its target, and a slot
limit, the cap. Jobs submitted with a minimum runtime are allocated jobs; jobs without
one are unallocated backfill in a separate, age-ordered, preemptible queue.

Ordering and preemption follow the cluster's scheduler policy (`beaker cluster get`,
field schedulerPolicy), which the scheduler implements as written (allenai/beaker,
scheduling/internal/sortpolicy.go and preemption.go). On the ai2 clusters the
allocated partition sorts by allocation balance at the workspace-group level across the
whole cluster, then at the workspace level, then priority, then age. Balance is
(quota - usage) / quota over a decaying multi-day window (usageWindow lookback and decay
half-life), and a balance criterion is an entity boundary: priority and age only ever
order jobs within one group. So budgets do interact: a group over its target ranks
behind every group under target, whatever their budgets. A running allocated job is
protected for its minimum runtime and then becomes interruptible; a queued allocated
job that ranks ahead of it on balance (when the victim's group is over target, including
a job from another budget) or on priority within the group may preempt it; one that
ranks ahead only by age may not.

`beaker job list --cluster X` returns jobs that *requested* X, and a job that named
several clusters may be running on another one. Running and scheduled jobs are
therefore kept only when their node belongs to the cluster (`beaker cluster nodes`);
queued jobs are kept whenever the cluster is among their constraints, and the other
clusters they could land on are shown. The allocated GPU total is cross-checked against
`beaker cluster usage`, whose allocated flag is likewise min_runtime > 0.

Usage:
  beaker_queue.py CLUSTER [--budget ai2/NAME] [--include-unallocated] [--json]

  beaker_queue.py ai2/jupiter                       # ai2/atec-climate
  beaker_queue.py ai2/titan --budget ai2/atec-olmoearth

The header gives the budget's target (its runtime quota percentage of the cluster's
slots), its slot limit (the cap), and its standing against target over the scheduler's
lookback window, from `beaker report gpu-usage` in one-day bins weighted by the policy's
decay half-life. That approximates the scheduler's balance (it ignores the forgiveness
horizon and quota changes); a preemption message quotes the exact figure.

Output (markdown): a header with the budget's standing and use, then the budget's
allocated jobs running on the cluster with their minimum runtime left (blank =
interruptible), and its allocated jobs scheduled or queued, in scheduling order.
--include-unallocated adds the same two tables for the budget's unallocated jobs. Jobs
of the logged-in user are marked with *. Times are local.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HOUR_NS = 3.6e12
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "beaker-queue")
REPORT_TIMEOUT_S = 25
PRIORITY_RANK = {
    "immediate": -1,
    "urgent": 0,
    "high": 1,
    "normal": 2,
    "low": 3,
    "preemptible": 4,
}


class BeakerError(RuntimeError):
    pass


def beaker(*args, attempts=2, timeout=None):
    """Run a beaker CLI command and parse its JSON, retrying once on a transient
    failure; `timeout` (seconds) bounds each attempt."""
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                ["beaker", *args, "--format", "json"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            error = f"no answer within {timeout} s"
        else:
            if result.returncode == 0:
                return json.loads(result.stdout)
            error = result.stderr.strip()[:300]
        transient = "deadline" in error.lower() or "within" in error
        if attempt + 1 < attempts and transient:
            continue
        raise BeakerError(f"beaker {' '.join(args)} failed: {error}")
    raise BeakerError(f"beaker {' '.join(args)} failed")


def first(d):
    return d[0] if isinstance(d, list) else d


def parse_time(ts):
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def local(ts):
    return parse_time(ts).astimezone().strftime("%b %d %H:%M") if ts else "?"


def hours(ns):
    return f"{ns / HOUR_NS:.1f} h"


def hours_left(ns):
    return hours(ns) if ns > 0 else ""


def whoami():
    return first(beaker("account", "whoami")).get("name", "")


def budget_record(name):
    d = first(beaker("budget", "get", name))
    return d["id"], d.get("fullName", name)


def slot_limit(cluster, budget_full_name):
    org = budget_full_name.split("/")[0]
    for a in beaker("allocation", "list", cluster, "--org", org):
        if (a.get("budget") or {}).get("fullName") == budget_full_name:
            return a.get("slotLimit"), a.get("runtimeQuotaPercent")
    return None, None


def cluster_record(cluster):
    return first(beaker("cluster", "get", cluster))


def scheduler_policy(record):
    policy = record.get("schedulerPolicy") or {}
    keys = []
    for part in (policy.get("sort") or {}).get("partitions") or []:
        if "allocated" not in (part.get("match") or {}):
            continue
        for crit in part.get("criteria") or []:
            for key in crit.get("keys") or []:
                ((name, cfg),) = key.items()
                level = (cfg or {}).get("level", "")
                level = level.replace("ALLOCATION_LEVEL_", "").lower()
                keys.append(f"{name}({level})" if level else name)
    return keys, bool(policy.get("interruptibleAfterMinRuntime"))


def lookback_hours(record):
    window = (record.get("schedulerPolicy") or {}).get("usageWindow") or {}
    seconds = float(str(window.get("lookback") or "0s").rstrip("s"))
    return seconds / 3600 if seconds > 0 else None


def total_slots(record):
    return ((record.get("clusterOccupancy") or {}).get("slotCounts") or {}).get("total")


def half_life_hours(record):
    window = (record.get("schedulerPolicy") or {}).get("usageWindow") or {}
    seconds = float(str(window.get("usageDecayHalfLife") or "0s").rstrip("s"))
    return seconds / 3600 if seconds > 0 else None


def allocated_gpu_hours_between(cluster, budget_full_name, start, end):
    org = budget_full_name.split("/")[0]
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    rows = beaker(
        "report",
        "gpu-usage",
        "--organizations",
        org,
        "--budgets",
        budget_full_name,
        "--clusters",
        cluster,
        "--start",
        start.strftime(fmt),
        "--end",
        end.strftime(fmt),
        "--allocated",
        "true",
        "--group-by",
        "budget",
        timeout=REPORT_TIMEOUT_S,
    )
    return sum(float(r.get("gpuHours") or 0) for r in rows)


def cached_day_hours(cluster, budget_full_name, day_start):
    """Allocated GPU-hours of one completed UTC day, fetched once and kept on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{cluster}_{budget_full_name}_{day_start:%Y-%m-%d}.json".replace("/", "-")
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["gpu_hours"]
    hours = allocated_gpu_hours_between(
        cluster, budget_full_name, day_start, day_start + datetime.timedelta(days=1)
    )
    with open(path, "w") as f:
        json.dump({"gpu_hours": hours}, f)
    return hours


def usage_bins(cluster, bname, now, hours_back, pool):
    """(age of bin centre in hours, bin length in hours, GPU-hours) over the lookback:
    today's partial UTC day fetched live, completed days from the cache."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    n_days = int(hours_back // 24)
    today = pool.submit(allocated_gpu_hours_between, cluster, bname, midnight, now)
    days = [midnight - datetime.timedelta(days=d) for d in range(1, n_days + 1)]
    past = [pool.submit(cached_day_hours, cluster, bname, day) for day in days]
    elapsed = (now - midnight).total_seconds() / 3600
    bins = [(elapsed / 2, elapsed, today.result())]
    for d, future in enumerate(past, start=1):
        bins.append((elapsed + (d - 0.5) * 24, 24.0, future.result()))
    return bins


def decayed_standing(bins, target_slots, half_life):
    """Usage and target over the bins, each weighted by the decay half-life."""
    used = target = 0.0
    for age, length, hours in bins:
        weight = 0.5 ** (age / half_life) if half_life else 1.0
        used += weight * hours
        target += weight * target_slots * length
    return used, target


def target_clause(cluster, bname, record, limit, pct, now, pool):
    """The header's standing sentence and the over-target fraction (None if unknown)."""
    slots = total_slots(record)
    hours_back = lookback_hours(record)
    if not slots or pct is None or not hours_back:
        return "", None
    target_slots = pct / 100 * slots
    half_life = half_life_hours(record)
    try:
        bins = usage_bins(cluster, bname, now, hours_back, pool)
        used, target = decayed_standing(bins, target_slots, half_life)
    except BeakerError:
        return (
            f" Target {target_slots:.0f} slots ({pct}% of {slots}), cap {limit}; "
            "standing against target unavailable (usage report too slow)."
        ), None
    over = (used - target) / target if target else 0.0
    word = f"{abs(over) * 100:.0f}% {'over' if over > 0 else 'under'} target"
    text = (
        f" Target {target_slots:.0f} slots ({pct}% of {slots}), cap {limit}; "
        f"decay-weighted allocated use over the {hours_back / 24:.0f}-day lookback "
        f"is {word}."
    )
    return text, over


def cluster_node_ids(cluster):
    return {n["id"] for n in beaker("cluster", "nodes", cluster)}


def usage_slots(cluster, budget_full_name):
    used = {"allocated": 0, "unallocated": 0}
    for row in beaker("cluster", "usage", cluster):
        a = row.get("assigned")
        if a and a.get("budgetReference") == budget_full_name:
            key = "allocated" if a.get("allocated") else "unallocated"
            used[key] += int(row.get("slotSeconds") or 0)
    return used


def job_gpus(job, spec):
    resources = spec.get("resources") or {}
    return resources.get("gpuCount") or (job.get("requests") or {}).get("gpuCount") or 0


def active_jobs(cluster, budget_id, now, node_ids):
    rows, workspace_names = [], {}
    for j in beaker("job", "list", "--cluster", cluster, "--kind", "execution"):
        st = j.get("status") or {}
        if j.get("kind") != "execution" or j.get("budget") != budget_id:
            continue
        if st.get("finalized") or st.get("exitCode") is not None or st.get("canceled"):
            continue
        spec = (j.get("execution") or {}).get("spec") or {}
        ctx = spec.get("context") or {}
        min_runtime = ctx.get("minRuntime") or 0
        placed = st.get("started") or st.get("scheduled")
        if placed and j.get("node") not in node_ids:
            continue
        ws = j.get("workspace")
        if ws not in workspace_names:
            workspace_names[ws] = first(beaker("workspace", "get", ws)).get(
                "fullName", ws
            )
        if st.get("started"):
            state, when = "running", st["started"]
            elapsed_ns = (now - parse_time(when)).total_seconds() * 1e9
            min_left_ns = min_runtime - elapsed_ns
        elif st.get("scheduled"):
            state, when, min_left_ns = "scheduled", st["scheduled"], None
        else:
            state, when, min_left_ns = "queued", st.get("created"), None
        clusters = (spec.get("constraints") or {}).get("cluster") or []
        rows.append(
            dict(
                state=state,
                allocated=min_runtime > 0,
                author=(j.get("author") or {}).get("name", "?"),
                workspace=workspace_names[ws],
                name=j.get("name") or j.get("id"),
                id=j.get("id"),
                gpus=job_gpus(j, spec),
                priority=ctx.get("priority") or "?",
                min_runtime_ns=min_runtime,
                min_left_ns=min_left_ns,
                time=when,
                created=st.get("created"),
                other_clusters=[c for c in clusters if c != cluster],
            )
        )
    return rows


def mark(r, me):
    return "*" if r["author"] == me else ""


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


def gpu_total(rows):
    return sum(r["gpus"] for r in rows)


def queue_order(r):
    not_scheduled = r["state"] != "scheduled"
    return (not_scheduled, PRIORITY_RANK.get(r["priority"], 9), r["created"] or "")


def preemptible_by(r, over, interruptible):
    """Who can take a running allocated job now: nobody while its minimum runtime holds,
    a higher-priority job of this budget once it has elapsed, and any under-target
    group's job as well while this budget is over target."""
    if not interruptible or r["min_left_ns"] > 0:
        return ""
    if over is None:
        return "higher priority; any under-target group if over target"
    if over > 0:
        return "any under-target group"
    return "higher priority in budget"


def running_rows(jobs, me, allocated, over=None, interruptible=True):
    rows = [r for r in jobs if r["state"] == "running" and r["allocated"] == allocated]
    rows.sort(key=lambda r: r["min_left_ns"])
    cells = []
    for r in rows:
        runtime = (
            (
                hours(r["min_runtime_ns"]),
                hours_left(r["min_left_ns"]),
                preemptible_by(r, over, interruptible),
            )
            if allocated
            else ()
        )
        cells.append(
            (mark(r, me), r["author"], r["gpus"], r["priority"])
            + runtime
            + (local(r["time"]), r["name"][:60])
        )
    return cells, rows


def waiting_rows(jobs, me, allocated):
    rows = [r for r in jobs if r["state"] != "running" and r["allocated"] == allocated]
    rows.sort(key=queue_order)
    cells = []
    for r in rows:
        runtime = (hours(r["min_runtime_ns"]),) if allocated else ()
        cells.append(
            (mark(r, me), r["state"], r["author"], r["gpus"], r["priority"])
            + runtime
            + (local(r["created"]), ", ".join(r["other_clusters"]), r["name"][:60])
        )
    return cells, rows


def running_header(allocated):
    runtime = ("min runtime", "min runtime left", "preemptible by") if allocated else ()
    return ["", "user", "GPUs", "priority", *runtime, "started", "name"]


def waiting_header(allocated):
    runtime = ("min runtime",) if allocated else ()
    return [
        "",
        "state",
        "user",
        "GPUs",
        "priority",
        *runtime,
        "created",
        "also eligible for",
        "name",
    ]


def print_header(bname, cluster, now, clause, allocated_gpus, usage, me):
    when = now.astimezone().strftime("%b %d %H:%M %Z")
    line = (
        f"**{bname} on {cluster}** at {when}: {allocated_gpus} slots held by "
        f"allocated jobs, {usage['unallocated']} by unallocated jobs.{clause}"
        f" Jobs of {me} are marked *."
    )
    if usage["allocated"] != allocated_gpus:
        line += (
            f" (`beaker cluster usage` reports {usage['allocated']} allocated slots; "
            "the difference is jobs assigned but not yet started or just finished.)"
        )
    print(line)


def print_allocated(jobs, me, interruptible, sort_keys, over):
    cells, rows = running_rows(
        jobs, me, allocated=True, over=over, interruptible=interruptible
    )
    note = (
        "'preemptible by' is blank while the minimum runtime protects the job; after "
        "that a higher-priority job of this budget can take it, and any under-target "
        "group's job as well while this budget is over target."
    )
    print(
        f"\n**Allocated jobs running** ({len(rows)} jobs, {gpu_total(rows)} GPUs). "
        f"{note}\n"
    )
    table(running_header(True), cells)

    cells, rows = waiting_rows(jobs, me, allocated=True)
    workspaces = sorted({r["workspace"] for r in rows})
    order = " > ".join(sort_keys) if sort_keys else "unknown"
    print(
        f"\n**Allocated jobs scheduled or queued** ({len(rows)} jobs, "
        f"{gpu_total(rows)} GPUs). Scheduler sort for allocated jobs: {order}.",
        end=" ",
    )
    if len(workspaces) <= 1:
        print(
            "All waiting jobs share one workspace, so the balance keys tie and the "
            "order below (scheduled first, then priority, then age) is the "
            "scheduling order."
        )
    else:
        print(
            f"Waiting jobs span {len(workspaces)} workspaces "
            f"({', '.join(workspaces)}), so workspace allocation balance decides "
            "before priority and the order below (priority, then age) holds only "
            "within a workspace."
        )
    print(
        "A scheduled job has a node and is starting. A job also eligible for other "
        "clusters may start there instead.\n"
    )
    table(waiting_header(True), cells)


def print_unallocated(jobs, me):
    cells, rows = running_rows(jobs, me, allocated=False)
    print(
        f"\n**Unallocated jobs running** ({len(rows)} jobs, {gpu_total(rows)} GPUs; "
        "backfill, preemptible at any time)\n"
    )
    table(running_header(False), cells)
    cells, rows = waiting_rows(jobs, me, allocated=False)
    print(
        f"\n**Unallocated jobs scheduled or queued** ({len(rows)} jobs, "
        f"{gpu_total(rows)} GPUs; a separate queue, sorted by age)\n"
    )
    table(waiting_header(False), cells)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("cluster")
    p.add_argument("--budget", default="ai2/atec-climate")
    p.add_argument(
        "--include-unallocated",
        action="store_true",
        help="add the same two tables for the budget's unallocated (backfill) jobs",
    )
    p.add_argument("--json", action="store_true", help="emit the job rows as JSON")
    a = p.parse_args()

    now = datetime.datetime.now(datetime.timezone.utc)
    with ThreadPoolExecutor(max_workers=12) as pool:
        me_f = pool.submit(whoami)
        budget_f = pool.submit(budget_record, a.budget)
        nodes_f = pool.submit(cluster_node_ids, a.cluster)
        record_f = pool.submit(cluster_record, a.cluster)
        bid, bname = budget_f.result()
        jobs_f = pool.submit(active_jobs, a.cluster, bid, now, nodes_f.result())
        limit_f = pool.submit(slot_limit, a.cluster, bname)
        usage_f = pool.submit(usage_slots, a.cluster, bname)
        record = record_f.result()
        limit, pct = limit_f.result()
        clause_f = pool.submit(
            target_clause, a.cluster, bname, record, limit, pct, now, pool
        )
        me, jobs, usage = me_f.result(), jobs_f.result(), usage_f.result()
        clause, over = clause_f.result()
    if a.json:
        print(json.dumps(dict(user=me, jobs=jobs), indent=1))
        return
    sort_keys, interruptible = scheduler_policy(record)
    allocated_gpus = gpu_total(
        [r for r in jobs if r["allocated"] and r["state"] != "queued"]
    )
    print_header(bname, a.cluster, now, clause, allocated_gpus, usage, me)
    print_allocated(jobs, me, interruptible, sort_keys, over)
    if a.include_unallocated:
        print_unallocated(jobs, me)


if __name__ == "__main__":
    try:
        main()
    except BeakerError as error:
        sys.exit(str(error))

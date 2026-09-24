#!/usr/bin/env python3
"""Where a budget's allocated jobs stand on a Beaker cluster.

Each budget holds a fixed allocation of slots on a cluster (`beaker allocation list
<cluster> --org ai2`, field slotLimit). Jobs submitted with a minimum runtime are
allocated jobs and compete only with the same budget's other allocated jobs for that
quota. Jobs with no minimum runtime are unallocated: they backfill idle slots anywhere
on the cluster in a separate, age-ordered queue and are preemptible at any time.

Ordering and preemption follow the cluster's scheduler policy (`beaker cluster get`,
field schedulerPolicy), which the scheduler implements as written (allenai/beaker,
scheduling/internal/sortpolicy.go and preemption.go). On the ai2 clusters the
allocated partition sorts by allocation balance at the workspace-group level, then at
the workspace level, then priority, then age. Balance is (quota - usage) / quota over
a decaying multi-day window, and a balance criterion is an entity boundary: priority
and age only ever order jobs within one workspace. A budget whose jobs all sit in one
workspace therefore orders by priority, then age. A running allocated job is protected
for its minimum runtime and then becomes interruptible; a queued allocated job that
ranks ahead of it on priority (or on balance, when the victim's workspace is over its
target) may preempt it, one that ranks ahead only by age may not.

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
slots), its slot limit (the burst cap), and its allocated GPU-hours over the scheduler's
lookback window against the target, from `beaker report gpu-usage`. That standing is
undecayed, so it approximates the scheduler's balance rather than reproducing it.

Output (markdown): a header with the budget's standing and use, then the budget's
allocated jobs running on the cluster with their minimum runtime left (blank =
interruptible), and its allocated jobs scheduled or queued, in scheduling order.
--include-unallocated adds the same two tables for the budget's unallocated jobs. Jobs
of the logged-in user are marked with *. Times are local.
"""

import argparse
import datetime
import json
import subprocess
import sys

HOUR_NS = 3.6e12
PRIORITY_RANK = {
    "immediate": -1,
    "urgent": 0,
    "high": 1,
    "normal": 2,
    "low": 3,
    "preemptible": 4,
}


def beaker(*args):
    result = subprocess.run(
        ["beaker", *args, "--format", "json"], capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(f"beaker {' '.join(args)} failed: {result.stderr.strip()[:300]}")
    return json.loads(result.stdout)


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


def windowed_allocated_gpu_hours(cluster, budget_full_name, hours_back):
    org = budget_full_name.split("/")[0]
    rows = beaker(
        "report",
        "gpu-usage",
        "--organizations",
        org,
        "--budgets",
        budget_full_name,
        "--clusters",
        cluster,
        "--since",
        f"{int(hours_back)}h",
        "--allocated",
        "true",
        "--group-by",
        "budget",
    )
    return sum(float(r.get("gpuHours") or 0) for r in rows)


def target_clause(cluster, bname, record, limit, pct):
    slots = total_slots(record)
    hours_back = lookback_hours(record)
    if not slots or pct is None or not hours_back:
        return ""
    target_slots = pct / 100 * slots
    used = windowed_allocated_gpu_hours(cluster, bname, hours_back)
    target_hours = target_slots * hours_back
    over = (used - target_hours) / target_hours * 100 if target_hours else 0
    standing = f"{abs(over):.0f}% {'over' if over > 0 else 'under'} target"
    return (
        f" Target {target_slots:.0f} slots ({pct}% of {slots}), burst cap {limit}; "
        f"over the {hours_back / 24:.0f}-day lookback the budget used {used:.0f} of "
        f"{target_hours:.0f} target GPU-h, {standing} (undecayed; the scheduler "
        "weights recent use more)."
    )


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
    for j in beaker("job", "list", "--cluster", cluster):
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


def running_rows(jobs, me, allocated):
    rows = [r for r in jobs if r["state"] == "running" and r["allocated"] == allocated]
    rows.sort(key=lambda r: r["min_left_ns"])
    cells = []
    for r in rows:
        runtime = (
            (hours(r["min_runtime_ns"]), hours_left(r["min_left_ns"]))
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
    runtime = ("min runtime", "min runtime left") if allocated else ()
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


def print_header(bname, cluster, now, record, limit, pct, allocated_gpus, usage, me):
    when = now.astimezone().strftime("%b %d %H:%M %Z")
    line = (
        f"**{bname} on {cluster}** at {when}: {allocated_gpus} slots held by "
        f"allocated jobs, {usage['unallocated']} by unallocated jobs."
        f"{target_clause(cluster, bname, record, limit, pct)}"
        f" Jobs of {me} are marked *."
    )
    if usage["allocated"] != allocated_gpus:
        line += (
            f" (`beaker cluster usage` reports {usage['allocated']} allocated slots; "
            "the difference is jobs assigned but not yet started or just finished.)"
        )
    print(line)


def print_allocated(jobs, me, interruptible, sort_keys):
    cells, rows = running_rows(jobs, me, allocated=True)
    if interruptible:
        note = (
            "A blank 'min runtime left' means the job is past its minimum runtime and "
            "interruptible: a queued allocated job of higher priority in the same "
            "workspace can preempt it (one that is merely older cannot)."
        )
    else:
        note = (
            "This cluster does not make jobs interruptible after their minimum runtime"
        )
        note += "."
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
    me = whoami()
    bid, bname = budget_record(a.budget)
    jobs = active_jobs(a.cluster, bid, now, cluster_node_ids(a.cluster))
    if a.json:
        print(json.dumps(dict(user=me, jobs=jobs), indent=1))
        return
    limit, pct = slot_limit(a.cluster, bname)
    record = cluster_record(a.cluster)
    sort_keys, interruptible = scheduler_policy(record)
    usage = usage_slots(a.cluster, bname)
    allocated_gpus = gpu_total(
        [r for r in jobs if r["allocated"] and r["state"] != "queued"]
    )
    print_header(bname, a.cluster, now, record, limit, pct, allocated_gpus, usage, me)
    print_allocated(jobs, me, interruptible, sort_keys)
    if a.include_unallocated:
        print_unallocated(jobs, me)


if __name__ == "__main__":
    main()

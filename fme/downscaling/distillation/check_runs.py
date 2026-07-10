# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Inspect distillation runs on Weights & Biases and generate run reports.

Helper for the distillation experiment workflow (see ``experiments/WORKFLOW.md``).
Pure wandb — does NOT import fastgen, so it runs in the plain ``fme`` conda env.

Usage:
    # List the most recent runs in the project (id | state | step | name):
    conda run -n fme python -m fme.downscaling.distillation.check_runs --list

    # Compare validation/training metrics across one or more runs:
    conda run -n fme python -m fme.downscaling.distillation.check_runs a1b2c3 d4e5f6

    # Generate a pre-filled per-run report (Phase 4 of the workflow):
    conda run -n fme python -m fme.downscaling.distillation.check_runs \
        --report <wandb_id> --beaker <ULID> --out experiments/reports/

    # Print a LOG.md registry row for a run:
    conda run -n fme python -m fme.downscaling.distillation.check_runs \
        --registry-row <wandb_id> --beaker <ULID>

    # Compare a distilled eval bundle against its teacher (Phase 4, eval):
    conda run -n fme python -m fme.downscaling.distillation.check_runs \
        --compare-eval <teacher_id> <distilled_id> --project andrep-downscaling \
        --out experiments/reports/

The report generator auto-discovers the run's output variables, so it handles the
single-variable PRATEsfc student and the 4-variable MoE student without change.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Iterable, Sequence

import wandb

DEFAULT_PROJECT = "ai2cm/fastgen"

# GitHub repo the runs are launched from — used to turn a wandb-recorded commit
# sha into a browsable link in the generated reports / registry rows.
GITHUB_REPO = "ai2cm/ace"

# GAN binary-cross-entropy reference points: a discriminator at chance sits at
# ln2 (generator) / 2*ln2 (discriminator).  Used to detect a GAN that never
# engaged vs one that collapsed.
LN2 = 0.6931
TWO_LN2 = 1.3863

DEFAULT_KEYS = [
    "train/total_loss",
    "train/f_distill_loss",
    "train/gan_loss_gen",
    "val/crps_PRMSL",
    "val/crps_PRATEsfc",
    "val/crps_eastward_wind_at_ten_meters",
    "val/crps_northward_wind_at_ten_meters",
    "val/spec_mae_mean",
    "val/spec_mae_hi_PRATEsfc",
    "val/tail_99.99_PRATEsfc",
]

# Training-run loss terms reported in the loss-domination table, in display order.
LOSS_TERMS = [
    "train/f_distill_loss",
    "train/spectral_loss_weighted",
    "train/gan_loss_gen",
    "train/fake_score_loss",
    "train/total_loss",
]


# --------------------------------------------------------------------------- #
# Pure helpers (no wandb) — unit-tested in test_check_runs.py.
# --------------------------------------------------------------------------- #
def discover_variables(keys: Iterable[str]) -> list[str]:
    """Discover output variable names from a run's metric keys.

    Prefers the per-band spectral keys (``val/spec_mae_hi_<VAR>``); falls back to
    the tail keys (``val/tail_<pct>_<VAR>``) and per-variable CRPS.  Returns a
    sorted, de-duplicated list excluding the ``mean`` aggregate.
    """
    patterns = [
        re.compile(r"val/spec_mae_hi_(.+)$"),
        re.compile(r"val/tail_[\d.]+_(.+)$"),
        re.compile(r"val/crps_(.+)$"),
        re.compile(r"metrics/crps/(.+)$"),  # eval-bundle runs
        re.compile(r"power_spectrum/mean_abs_norm_bias/(.+)$"),  # eval-bundle runs
    ]
    found: set[str] = set()
    for key in keys:
        for pat in patterns:
            m = pat.match(key)
            if m:
                var = m.group(1)
                if var not in ("mean", "best"):
                    found.add(var)
    return sorted(found)


def present_keys(requested: Iterable[str], available: Iterable[str]) -> list[str]:
    """Intersect ``requested`` with ``available``, preserving requested order.

    ``wandb``'s ``run.history(keys=...)`` returns an EMPTY frame if *any* requested
    key was never logged by the run, so a single stale key (e.g. a loss term added
    after the run) silently blanks every metric.  Filtering to the keys actually
    present (the run's summary keys) before querying avoids that.
    """
    available_set = set(available)
    return [k for k in requested if k in available_set]


def discover_tail_percentiles(keys: Iterable[str]) -> list[str]:
    """Return the tail percentiles present (e.g. ``["99.99", "99.9999"]``)."""
    pat = re.compile(r"val/tail_([\d.]+)_")
    pcts = {m.group(1) for k in keys if (m := pat.match(k))}
    return sorted(pcts, key=float)


def series_summary(values: Sequence[float], objective: str = "min") -> dict | None:
    """Summarize a metric trajectory as first / best@frac / last.

    ``objective='min'`` treats the minimum as best (errors); ``objective='target'``
    treats the value closest to 1.0 as best (tail ratios).  Returns ``None`` for an
    empty series.
    """
    vals = [float(v) for v in values]
    if not vals:
        return None
    if objective == "target":
        best_idx = min(range(len(vals)), key=lambda i: abs(vals[i] - 1.0))
    else:
        best_idx = min(range(len(vals)), key=lambda i: vals[i])
    frac = best_idx / max(len(vals) - 1, 1)
    return {
        "first": vals[0],
        "last": vals[-1],
        "best": vals[best_idx],
        "best_frac": frac,
        "n": len(vals),
    }


def fmt_fbl(summary: dict | None) -> str:
    """Format a series_summary as ``first → best@frac → last``."""
    if summary is None:
        return "—"
    return (
        f"{summary['first']:.3g} → {summary['best']:.3g}"
        f"@{summary['best_frac']:.0%} → {summary['last']:.3g}"
    )


def assess_gan_health(
    gen_first: float | None,
    gen_last: float | None,
    disc_first: float | None,
    disc_last: float | None,
) -> tuple[str, str]:
    """Classify GAN training behavior. Returns (verdict, rationale)."""
    if gen_last is None or disc_last is None:
        return "unknown", "gan losses not logged"
    if abs(gen_last - LN2) < 0.05 and abs(disc_last - TWO_LN2) < 0.15:
        return (
            "not-engaged",
            f"gen≈ln2 ({gen_last:.3g}), disc≈2·ln2 ({disc_last:.3g}) — "
            "discriminator stuck at chance",
        )
    disc_collapsed = disc_last < 0.25 or (
        disc_first is not None and disc_last < 0.5 * disc_first
    )
    gen_rising = gen_first is not None and gen_last > 1.15 * gen_first
    if disc_collapsed and gen_rising:
        return (
            "disc-winning collapse",
            f"disc fell {disc_first:.3g}→{disc_last:.3g} while gen rose "
            f"{gen_first:.3g}→{gen_last:.3g}",
        )
    return "healthy", f"gen {gen_last:.3g}, disc {disc_last:.3g} (both engaged)"


def assess_trajectory(summary: dict | None) -> str:
    """Describe first→best→last movement of an aggregate error metric."""
    if summary is None:
        return "—"
    first, best, last = summary["first"], summary["best"], summary["last"]
    improve = (first - best) / abs(first) if first else 0.0
    drift = (last - best) / abs(best) if best else 0.0
    if drift > 0.2:
        label = "degrading late (checkpoint-selection trap)"
    elif improve > 0.1:
        label = "improving"
    else:
        label = "flat"
    return (
        f"{label}: improved {improve:.0%} first→best, then drifted "
        f"{drift:+.0%} best→last"
    )


def loss_domination_rows(
    last_values: dict[str, float], gan_weight: float
) -> tuple[list[tuple[str, str, str]], str]:
    """Build the loss-domination table rows + a one-line dominance note.

    Rows are (term, last-value, note).  Ratios are taken against
    ``f_distill_loss``; the GAN generator term is scaled by ``gan_weight`` to get
    its actual optimized contribution.
    """
    ref = last_values.get("train/f_distill_loss")
    rows: list[tuple[str, str, str]] = []
    flags: list[str] = []
    for term in LOSS_TERMS:
        val = last_values.get(term)
        if val is None:
            continue
        note = ""
        contrib = val
        if term == "train/gan_loss_gen":
            contrib = gan_weight * val
            note = f"×weight {gan_weight:g} = {contrib:.3g}"
        if ref and term in ("train/spectral_loss_weighted", "train/gan_loss_gen"):
            ratio = contrib / ref
            note = (note + "; " if note else "") + f"{ratio:.2g}× f_distill"
            if ratio > 1.0:
                flags.append(f"{term.split('/')[-1]} exceeds f_distill_loss")
        rows.append((term, f"{val:.4g}", note))
    if not flags:
        note = "f_distill_loss dominates the generator objective (expected)"
    else:
        note = "; ".join(flags)
    return rows, note


def registry_row(fields: dict) -> str:
    """Format a LOG.md run-registry table row from a field dict.

    When ``commit_url`` is present the commit renders as a markdown link so the
    LOG.md registry, like the per-run reports, points straight at GitHub.
    """
    commit = fields.get("commit", "?")
    commit_url = fields.get("commit_url")
    commit_cell = (
        f"[`{commit}`]({commit_url})" if commit_url and commit != "?" else f"`{commit}`"
    )
    cols = [
        f"`{fields.get('wandb', '?')}`",
        fields.get("date", "?"),
        fields.get("name", "?"),
        f"`{fields.get('beaker', '—')}`",
        commit_cell,
        fields.get("knobs", "?"),
        fields.get("state", "?"),
        fields.get("verdict", "⏳"),
        fields.get("report", "—"),
    ]
    return "| " + " | ".join(cols) + " |"


# --------------------------------------------------------------------------- #
# wandb I/O.
# --------------------------------------------------------------------------- #
def list_runs(project: str, limit: int) -> None:
    api = wandb.Api()
    runs = list(api.runs(project, order="-created_at")[:limit])
    print(f"{'id':12s} {'state':9s} {'step':>7s}  name")
    for r in runs:
        step = r.summary.get("_step")
        print(f"{r.id:12s} {r.state:9s} {str(step):>7s}  {r.name}")


def compare_runs(project: str, run_ids: list[str], keys: list[str]) -> None:
    api = wandb.Api()
    for rid in run_ids:
        r = api.run(f"{project}/{rid}")
        runtime_min = round(r.summary.get("_runtime", 0) / 60, 1)
        print("=" * 76)
        print(
            f"{r.name}\n  {rid} | {r.state} | step {r.summary.get('_step')} "
            f"| runtime {runtime_min} min"
        )
        run_keys = present_keys(keys, r.summary.keys())
        hist = r.history(keys=run_keys, samples=4000, pandas=True) if run_keys else None
        for key in keys:
            if hist is not None and key in hist.columns:
                summ = series_summary(hist[key].dropna().tolist())
                if summ:
                    print(f"  {key:42s} {fmt_fbl(summ)}  n={summ['n']}")


def _run_metadata(run) -> dict:
    """Pull git commit + launch args from the run's wandb-metadata.json."""
    meta: dict = {"commit": "?", "args": []}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            f = run.file("wandb-metadata.json").download(root=tmp, replace=True)
            data = json.load(open(f.name))
        meta["commit"] = (data.get("git", {}) or {}).get("commit", "?")
        meta["args"] = data.get("args", []) or []
    except Exception:
        pass
    return meta


def _history_lists(run, keys: list[str]) -> dict[str, list[float]]:
    """Return {key: [non-nan values in step order]} for the requested keys.

    Only keys the run actually logged are queried — see ``present_keys`` for why a
    stale key would otherwise blank the whole history frame.
    """
    keys = present_keys(keys, run.summary.keys())
    if not keys:
        return {}
    hist = run.history(keys=keys, samples=100000, pandas=True)
    out: dict[str, list[float]] = {}
    for key in keys:
        if key in hist.columns:
            out[key] = hist[key].dropna().tolist()
    return out


def _short_commit(commit: str) -> str:
    return commit[:7] if commit and commit != "?" else "?"


def _github_commit_url(commit: str) -> str | None:
    """GitHub URL for a full commit sha, or None if the commit is unknown."""
    if not commit or commit == "?":
        return None
    return f"https://github.com/{GITHUB_REPO}/commit/{commit}"


def _commit_cell(commit: str) -> str:
    """Render a commit as a short sha followed by its GitHub URL for report tables.

    Falls back to the bare short sha when the commit is unknown.
    """
    short = f"`{_short_commit(commit)}`"
    url = _github_commit_url(commit)
    return f"{short} — {url}" if url else short


def _suffix_from_name(name: str) -> str:
    return re.sub(r"^ace-downscaling-distillation-\w+-with-val-?", "", name) or name


def _method_from_args(args: list[str]) -> str:
    for a in args:
        if "fdistill" in a:
            return "fdistill"
        if "dmd2" in a:
            return "dmd2"
        if "scm_spike" in a:
            return "scm"
    return "?"


# --------------------------------------------------------------------------- #
# Report generation.
# --------------------------------------------------------------------------- #
def build_run_report(project: str, rid: str, beaker: str, gan_weight: float) -> str:
    api = wandb.Api()
    run = api.run(f"{project}/{rid}")
    meta = _run_metadata(run)
    summary_keys = list(run.summary.keys())
    variables = discover_variables(summary_keys)
    pcts = discover_tail_percentiles(summary_keys) or ["99.99", "99.9999"]

    # Assemble the metric keys we need from the discovered variables.
    train_keys = [
        "train/total_loss",
        "train/f_distill_loss",
        "train/spectral_loss",
        "train/spectral_loss_weighted",
        "train/gan_loss_gen",
        "train/gan_loss_disc",
        "train/fake_score_loss",
    ]
    agg_keys = ["val/crps_mean", "val/spec_mae_mean"]
    var_keys: list[str] = []
    for v in variables:
        var_keys += [f"val/spec_mae_{b}_{v}" for b in ("lo", "mid", "hi")]
        var_keys += [f"val/tail_{p}_{v}" for p in pcts]
    hist = _history_lists(run, train_keys + agg_keys + var_keys)

    def last(key: str) -> float | None:
        vals = hist.get(key)
        return vals[-1] if vals else None

    def first(key: str) -> float | None:
        vals = hist.get(key)
        return vals[0] if vals else None

    # --- Training behavior ---
    gan_verdict, gan_note = assess_gan_health(
        first("train/gan_loss_gen"),
        last("train/gan_loss_gen"),
        first("train/gan_loss_disc"),
        last("train/gan_loss_disc"),
    )
    last_losses: dict[str, float] = {}
    for k in LOSS_TERMS:
        lv = last(k)
        if lv is not None:
            last_losses[k] = lv
    loss_rows, loss_note = loss_domination_rows(last_losses, gan_weight)
    crps_traj = assess_trajectory(series_summary(hist.get("val/crps_mean", [])))
    spec_traj = assess_trajectory(series_summary(hist.get("val/spec_mae_mean", [])))

    # --- Config bits ---
    method = _method_from_args(meta["args"])
    suffix = _suffix_from_name(run.name)
    spec_raw, spec_w = last("train/spectral_loss"), last("train/spectral_loss_weighted")
    spectral_weight = (spec_w / spec_raw) if (spec_raw and spec_w) else None

    # --- Render ---
    L: list[str] = []
    L.append(f"# Run report — `{run.name}`")
    L.append("")
    L.append("_Hypothesis: TODO — what this run tests and against which baseline._")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| Experiment name | `{run.name}` |")
    L.append(f"| wandb run | `{rid}` — {run.url} |")
    bk = f"`{beaker}` — https://beaker.org/ex/{beaker}" if beaker else "`TODO`"
    L.append(f"| Beaker experiment | {bk} |")
    L.append(f"| Commit | {_commit_cell(meta['commit'])} |")
    L.append(f"| State / last step | `{run.state}` @ `{run.summary.get('_step')}` |")
    L.append("")
    L.append("## Config")
    L.append("")
    L.append(f"- **Method:** {method}")
    if spectral_weight is not None:
        L.append(f"- **Spectral weight (derived):** {spectral_weight:.2g}")
    L.append(f"- **GAN generator weight (assumed for ratios):** {gan_weight:g}")
    L.append(f"- **Suffix:** `{suffix}`")
    L.append("")
    L.append("## 1 · Training behavior")
    L.append("")
    L.append(f"- **GAN health:** `{gan_verdict}` — {gan_note}")
    L.append("- **Loss domination:**")
    L.append("")
    L.append("  | term | last | note |")
    L.append("  |---|---|---|")
    for term, val, note in loss_rows:
        L.append(f"  | `{term}` | {val} | {note} |")
    L.append("")
    L.append(f"  _{loss_note}._")
    L.append("")
    L.append(f"- **`val/crps_mean`:** {crps_traj}")
    L.append(f"- **`val/spec_mae_mean`:** {spec_traj}")
    L.append("")
    L.append("## 2 · Tail behavior")
    L.append("")
    L.append("Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under.")
    L.append("")
    header = "| variable | " + " | ".join(f"tail_{p}" for p in pcts) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(pcts) + 1))
    worst_tail = _render_var_rows(
        L, variables, [f"val/tail_{p}" for p in pcts], hist, "target"
    )
    L.append("")
    L.append(f"Worst tail: **{worst_tail}**.")
    L.append("")
    L.append("## 3 · Power spectrum")
    L.append("")
    L.append("`spec_mae` per band (relative error; lower better).")
    L.append("")
    L.append("| variable | lo | mid | hi |")
    L.append("|---|---|---|---|")
    worst_spec = _render_var_rows(
        L, variables, ["val/spec_mae_lo", "val/spec_mae_mid", "val/spec_mae_hi"], hist
    )
    L.append("")
    L.append(
        f"Worst spectrum: **{worst_spec}** · PSD figures: `val/power_spectrum/<VAR>`"
        f" in wandb."
    )
    L.append("")
    L.append("## Verdict  <!-- HUMAN: fill this in -->")
    L.append("")
    L.append("- **Outcome vs baseline:** TODO")
    L.append("- **Recommended checkpoint:** TODO (mid-training if late drift)")
    L.append("- **Next action:** TODO")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- ⚠️ _Prepend here if a later run invalidates this one._")
    L.append("")
    return "\n".join(L)


def _render_var_rows(
    lines: list[str],
    variables: list[str],
    key_prefixes: list[str],
    hist: dict[str, list[float]],
    objective: str = "min",
) -> str:
    """Append one table row per variable; return the worst variable label.

    "Worst" ranks by the last value of the final column (hi band / finest tail),
    using distance-from-1.0 for tail ratios and raw magnitude for errors.
    """
    worst_var, worst_score = "—", -1.0
    for v in variables:
        cells = []
        last_final = None
        for prefix in key_prefixes:
            summ = series_summary(hist.get(f"{prefix}_{v}", []), objective)
            cells.append(fmt_fbl(summ))
            if summ is not None:
                last_final = summ["last"]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
        if last_final is not None:
            score = abs(last_final - 1.0) if objective == "target" else abs(last_final)
            if score > worst_score:
                worst_score, worst_var = score, v
    return worst_var


def write_report(
    text: str, out: str | None, run_name: str, rid: str, date: str
) -> None:
    if out is None:
        print(text)
        return
    if os.path.isdir(out) or out.endswith("/"):
        os.makedirs(out, exist_ok=True)
        suffix = _suffix_from_name(run_name)
        path = os.path.join(out, f"{date}-{suffix}-{rid}.md")
    else:
        path = out
    with open(path, "w") as fh:
        fh.write(text)
    print(f"wrote {path}")


def compare_eval(project: str, teacher: str, distilled: str) -> str:
    api = wandb.Api()
    t = api.run(f"{project}/{teacher}")
    d = api.run(f"{project}/{distilled}")
    t_commit = _run_metadata(t)["commit"]
    d_commit = _run_metadata(d)["commit"]
    variables = discover_variables(list(t.summary.keys()))
    # eval percentile keys look like histogram/prediction_frac_of_target/<p>th-...
    pat = re.compile(r"histogram/prediction_frac_of_target/([\d.]+)th-percentile/")
    pcts = sorted(
        {m.group(1) for k in t.summary.keys() if (m := pat.match(k))}, key=float
    )
    tail_pct = pcts[-1] if pcts else "99.9999"

    def cell(run, key: str) -> str:
        val = run.summary.get(key)
        return f"{val:.4g}" if isinstance(val, int | float) else "—"

    L: list[str] = []
    L.append("# Eval comparison — distilled vs teacher")
    L.append("")
    L.append("_Hypothesis: TODO._")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("| role | wandb run | commit |")
    L.append("|---|---|---|")
    L.append(f"| Teacher | `{teacher}` — {t.url} | {_commit_cell(t_commit)} |")
    L.append(f"| Distilled | `{distilled}` — {d.url} | {_commit_cell(d_commit)} |")
    L.append("")
    L.append("## CRPS  (`metrics/crps/<VAR>` — lower better)")
    L.append("")
    L.append("| variable | teacher | distilled | Δ (dist−teach) |")
    L.append("|---|---|---|---|")
    for v in variables:
        key = f"metrics/crps/{v}"
        tv, dv = t.summary.get(key), d.summary.get(key)
        delta = (
            f"{dv - tv:+.3g}"
            if isinstance(tv, int | float) and isinstance(dv, int | float)
            else "—"
        )
        L.append(f"| {v} | {cell(t, key)} | {cell(d, key)} | {delta} |")
    L.append("")
    L.append(
        f"## Tail ratio  (`prediction_frac_of_target` @ {tail_pct}th pct — ~1.0 ideal)"
    )
    L.append("")
    L.append("| variable | teacher | distilled |")
    L.append("|---|---|---|")
    for v in variables:
        key = f"histogram/prediction_frac_of_target/{tail_pct}th-percentile/{v}"
        L.append(f"| {v} | {cell(t, key)} | {cell(d, key)} |")
    L.append("")
    L.append("## Power spectrum bias  (`power_spectrum/mean_abs_norm_bias/<VAR>`)")
    L.append("")
    L.append("| variable | teacher | distilled |")
    L.append("|---|---|---|")
    for v in variables:
        key = f"power_spectrum/mean_abs_norm_bias/{v}"
        L.append(f"| {v} | {cell(t, key)} | {cell(d, key)} |")
    L.append("")
    L.append("## Verdict  <!-- HUMAN: fill this in -->")
    L.append("")
    L.append("- **Does the distilled student hold up vs teacher?** TODO")
    L.append("- **Regressions to watch:** TODO")
    L.append("")
    return "\n".join(L)


def do_registry_row(project: str, rid: str, beaker: str) -> None:
    api = wandb.Api()
    run = api.run(f"{project}/{rid}")
    meta = _run_metadata(run)
    print(
        registry_row(
            {
                "wandb": rid,
                "date": str(run.created_at)[:10],
                "name": run.name,
                "beaker": beaker or "—",
                "commit": _short_commit(meta["commit"]),
                "commit_url": _github_commit_url(meta["commit"]),
                "knobs": _method_from_args(meta["args"]),
                "state": f"{run.state}@{run.summary.get('_step')}",
                "verdict": "⏳",
                "report": "—",
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_ids", nargs="*", help="W&B run ids to compare (e.g. syz25njv r9lerxok)"
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="entity/project")
    parser.add_argument(
        "--list", action="store_true", help="list recent runs instead of comparing"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="number of runs to list (with --list)"
    )
    parser.add_argument(
        "--keys", nargs="+", default=DEFAULT_KEYS, help="metric keys to report"
    )
    parser.add_argument("--report", metavar="RUN_ID", help="generate a per-run report")
    parser.add_argument(
        "--beaker", default="", help="Beaker experiment ULID for report"
    )
    parser.add_argument(
        "--gan-weight",
        type=float,
        default=1e-3,
        help="generator GAN weight, for loss-domination ratios (default 1e-3)",
    )
    parser.add_argument(
        "--registry-row", metavar="RUN_ID", help="print a LOG.md registry row"
    )
    parser.add_argument(
        "--compare-eval",
        nargs=2,
        metavar=("TEACHER", "DISTILLED"),
        help="compare two eval-bundle runs (use --project andrep-downscaling)",
    )
    parser.add_argument("--out", help="output dir or file for reports (default stdout)")
    args = parser.parse_args()

    if args.registry_row:
        do_registry_row(args.project, args.registry_row, args.beaker)
        return
    if args.report:
        api = wandb.Api()
        run = api.run(f"{args.project}/{args.report}")
        text = build_run_report(args.project, args.report, args.beaker, args.gan_weight)
        write_report(text, args.out, run.name, args.report, str(run.created_at)[:10])
        return
    if args.compare_eval:
        teacher, distilled = args.compare_eval
        text = compare_eval(args.project, teacher, distilled)
        if args.out:
            api = wandb.Api()
            d = api.run(f"{args.project}/{distilled}")
            write_report(
                text,
                args.out,
                f"moe-eval-distilled-vs-teacher-{distilled}",
                distilled,
                str(d.created_at)[:10],
            )
        else:
            print(text)
        return

    if args.list or not args.run_ids:
        list_runs(args.project, args.limit)
        if not args.run_ids:
            return
    if args.run_ids:
        compare_runs(args.project, args.run_ids, args.keys)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure helpers in check_runs.py (no wandb access)."""

from fme.downscaling.distillation.check_runs import (
    assess_gan_health,
    assess_trajectory,
    discover_tail_percentiles,
    discover_variables,
    fmt_fbl,
    loss_domination_rows,
    registry_row,
    series_summary,
)


def test_discover_variables_single_and_multivar():
    single = ["val/spec_mae_hi_PRATEsfc", "val/tail_99.99_PRATEsfc", "_step"]
    assert discover_variables(single) == ["PRATEsfc"]

    multivar = [
        "val/spec_mae_hi_PRMSL",
        "val/spec_mae_hi_PRATEsfc",
        "val/spec_mae_hi_eastward_wind_at_ten_meters",
        "val/spec_mae_mean",  # aggregate must be excluded
        "val/crps_best",  # selection metric must be excluded
    ]
    assert discover_variables(multivar) == [
        "PRATEsfc",
        "PRMSL",
        "eastward_wind_at_ten_meters",
    ]


def test_discover_variables_eval_keys():
    keys = [
        "metrics/crps/PRATEsfc",
        "power_spectrum/mean_abs_norm_bias/PRMSL",
    ]
    assert discover_variables(keys) == ["PRATEsfc", "PRMSL"]


def test_discover_tail_percentiles():
    keys = ["val/tail_99.99_PRATEsfc", "val/tail_99.9999_PRATEsfc", "val/crps_mean"]
    assert discover_tail_percentiles(keys) == ["99.99", "99.9999"]


def test_series_summary_min_objective():
    # min is at index 2 of 5 -> frac 0.5
    summ = series_summary([1.0, 0.8, 0.3, 0.5, 0.9], objective="min")
    assert summ is not None
    assert summ["first"] == 1.0
    assert summ["last"] == 0.9
    assert summ["best"] == 0.3
    assert summ["best_frac"] == 0.5
    assert summ["n"] == 5


def test_series_summary_target_objective_picks_closest_to_one():
    # tail ratio: value closest to 1.0 is 1.05 at index 1
    summ = series_summary([0.5, 1.05, 3.0], objective="target")
    assert summ is not None
    assert summ["best"] == 1.05
    assert summ["best_frac"] == 0.5


def test_series_summary_empty():
    assert series_summary([]) is None
    assert fmt_fbl(None) == "—"


def test_fmt_fbl():
    summ = series_summary([1.0, 0.3, 0.9])
    assert fmt_fbl(summ) == "1 → 0.3@50% → 0.9"


def test_assess_gan_health_not_engaged():
    verdict, _ = assess_gan_health(0.693, 0.694, 1.386, 1.387)
    assert verdict == "not-engaged"


def test_assess_gan_health_disc_winning_collapse():
    # disc falls to near zero while gen rises
    verdict, _ = assess_gan_health(0.7, 1.5, 1.4, 0.1)
    assert verdict == "disc-winning collapse"


def test_assess_gan_health_healthy():
    verdict, _ = assess_gan_health(0.9, 1.05, 1.2, 1.15)
    assert verdict == "healthy"


def test_assess_gan_health_unknown_when_missing():
    verdict, _ = assess_gan_health(None, None, None, None)
    assert verdict == "unknown"


def test_assess_trajectory_flags_late_drift():
    # improves 1.0 -> 0.1 then drifts back to 0.5 (huge best->last drift)
    text = assess_trajectory(series_summary([1.0, 0.1, 0.5]))
    assert "degrading late" in text


def test_assess_trajectory_improving():
    text = assess_trajectory(series_summary([1.0, 0.6, 0.55]))
    assert text.startswith("improving")


def test_loss_domination_flags_dominant_term():
    last_values = {
        "train/f_distill_loss": 0.1,
        "train/spectral_loss_weighted": 0.3,  # 3x f_distill -> flagged
        "train/gan_loss_gen": 1.0,
        "train/total_loss": 1.4,
    }
    rows, note = loss_domination_rows(last_values, gan_weight=1e-3)
    assert any("spectral_loss_weighted" in r[0] for r in rows)
    assert "spectral_loss_weighted exceeds f_distill_loss" in note


def test_loss_domination_expected_when_f_distill_dominates():
    last_values = {
        "train/f_distill_loss": 0.1,
        "train/spectral_loss_weighted": 0.004,
        "train/gan_loss_gen": 1.0,  # x1e-3 = 0.001 << f_distill
    }
    _, note = loss_domination_rows(last_values, gan_weight=1e-3)
    assert "f_distill_loss dominates" in note


def test_registry_row_formatting():
    row = registry_row(
        {
            "wandb": "abc123",
            "date": "2026-07-09",
            "name": "run-name",
            "beaker": "01ABC",
            "commit": "e29f797",
            "knobs": "fdistill",
            "state": "running@100",
            "verdict": "⏳",
            "report": "—",
        }
    )
    assert row.startswith("| `abc123` | 2026-07-09 | run-name | `01ABC` |")
    assert row.endswith("| ⏳ | — |")

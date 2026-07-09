# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for the spectral-aware f-distill method subclass.

These require FastGen, which is an optional dependency; they are skipped where
it is not installed.
"""

import pytest

pytest.importorskip("fastgen")

from fastgen.methods.distribution_matching.f_distill import FdistillModel  # noqa: E402

from fme.downscaling.distillation.spectral_method import AceFdistillModel  # noqa: E402


def test_is_fdistill_subclass():
    assert issubclass(AceFdistillModel, FdistillModel)


def test_overrides_student_update_step():
    # The override is what carries the spectral term; if upstream renames the
    # method this guard flags the drift.
    assert (
        AceFdistillModel._student_update_step is not FdistillModel._student_update_step
    )


def test_exposes_set_spectral_loss():
    assert callable(AceFdistillModel.set_spectral_loss)

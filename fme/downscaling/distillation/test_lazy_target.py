# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
from fme.downscaling.distillation._lazy_target import lazy_target_name


class FdistillModel:  # stand-in mirroring FastGen's class-object _target_
    pass


def test_class_object_target_uses_class_name():
    # LazyCall stores the class object itself; this is the case that regressed
    # (AttributeError: type object has no attribute 'endswith').
    assert lazy_target_name(FdistillModel) == "FdistillModel"
    assert lazy_target_name(FdistillModel).endswith("FdistillModel")


def test_string_target_passes_through():
    dotted = "fastgen.methods.distribution_matching.f_distill.FdistillModel"
    assert lazy_target_name(dotted) == dotted
    assert lazy_target_name(dotted).endswith("FdistillModel")


def test_none_target_is_stringified_not_crashing():
    assert lazy_target_name(None) == "None"

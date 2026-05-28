# Forked from torch-harmonics (BSD-3-Clause)
# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from copy import deepcopy


def lru_cache(maxsize=20, typed=False, copy=False):
    """LRU cache decorator with optional deep copying of cached results."""

    def decorator(f):
        cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(f)

        def wrapper(*args, **kwargs):
            res = cached_func(*args, **kwargs)
            if copy:
                return deepcopy(res)
            else:
                return res

        return wrapper

    return decorator

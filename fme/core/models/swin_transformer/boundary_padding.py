# Adapted from
# https://github.com/NCAR/miles-credit/blob/main/credit/boundary_padding.py
# CREDIT (Community Research Earth Digital Intelligence Twin) is an open
# software platform by the NSF NCAR MILES group for training and deploying
# AI atmospheric prediction models.
# Reference: Schreck et al., npj Clim Atmos Sci 8, 239 (2025).
#   https://doi.org/10.1038/s41612-025-01125-6
#
# Copyright 2025 NSF NCAR MILES group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Literal


@dataclasses.dataclass
class TensorPaddingConfig:
    """Boundary padding of a global lat-lon grid, validated at construction time.

    Args:
        activate: Whether to apply ``pad_lat``/``pad_lon``.
        mode: ``"earth"`` crosses the poles (rows reverse and longitude
            rotates by 180 degrees); ``"mirror"`` reflects about the edge row.
            Longitude always wraps circularly.
        pad_lat: ``[north, south]`` rows of padding.
        pad_lon: ``[west, east]`` columns of padding.
    """

    activate: bool = False
    mode: Literal["mirror", "earth"] = "earth"
    pad_lat: list[int] = dataclasses.field(default_factory=lambda: [40, 40])
    pad_lon: list[int] = dataclasses.field(default_factory=lambda: [40, 40])

    def __post_init__(self):
        if self.mode not in ("mirror", "earth"):
            raise ValueError(f"Unknown padding mode {self.mode!r}")
        for name, sizes in (("pad_lat", self.pad_lat), ("pad_lon", self.pad_lon)):
            if len(sizes) != 2 or any(size < 0 for size in sizes):
                raise ValueError(
                    f"{name} must be two non-negative sizes, got {list(sizes)}"
                )

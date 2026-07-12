"""
Standalone reproduction of ai2cm/ace#579.

Isolates the exact pattern from fme/ace/stepper/single_module.py (lines 561-562,
569, 693-695) and fme/coupled/stepper.py (lines 297-333, 343, 358) without
pulling in ace's full torch/xarray dependency stack. Mirrors the real
annotations (`list[str]`) so mypy sees the same casting boilerplate.
"""

from __future__ import annotations


class SingleModuleStepperConfig:
    # base config values -- order is real ML channel order, must stay list[str]
    input_names: list[str] = ["a", "b", "c"]
    output_names: list[str] = ["b", "d"]

    @property
    def input_only_names(self) -> list[str]:
        # single_module.py:562 -- unordered set difference, cast back to list
        # only to satisfy the list[str] return annotation
        return list(set(self.input_names) - set(self.output_names))

    @property
    def all_names(self) -> list[str]:
        # single_module.py:695 -- dedup via set, cast back to list
        return list(set(self.input_names + self.output_names))


def build_coupled_forcing_names(
    ocean_input_only: list[str], atmosphere_output: list[str]
) -> list[str]:
    # coupled/stepper.py:296-299 -- caller immediately re-casts the list back
    # to a set to do more set algebra, then casts the result back to a list
    return list(set(ocean_input_only).difference(atmosphere_output))


if __name__ == "__main__":
    cfg = SingleModuleStepperConfig()
    print("input_only_names:", cfg.input_only_names)
    print("all_names:", cfg.all_names)
    print(
        "coupled forcing names:",
        build_coupled_forcing_names(cfg.input_only_names, ["d"]),
    )
    for _ in range(5):
        # same logical inputs, list() has no ordering guarantee over a set
        print("re-run input_only_names:", cfg.input_only_names)

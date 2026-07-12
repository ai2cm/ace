"""
Proposed-fix version of repro_579.py: retype the genuinely-unordered values as
set[str] and drop the redundant list()/set() casting. Base config values
(input_names, output_names) stay list[str] since their order is real ML
channel order.
"""

from __future__ import annotations


class SingleModuleStepperConfig:
    input_names: list[str] = ["a", "b", "c"]
    output_names: list[str] = ["b", "d"]

    @property
    def input_only_names(self) -> set[str]:
        return set(self.input_names) - set(self.output_names)

    @property
    def all_names(self) -> set[str]:
        return set(self.input_names) | set(self.output_names)


def build_coupled_forcing_names(
    ocean_input_only: set[str], atmosphere_output: list[str]
) -> set[str]:
    return ocean_input_only.difference(atmosphere_output)


if __name__ == "__main__":
    cfg = SingleModuleStepperConfig()
    print("input_only_names:", cfg.input_only_names)
    print("all_names:", cfg.all_names)
    print(
        "coupled forcing names:",
        build_coupled_forcing_names(cfg.input_only_names, ["d"]),
    )

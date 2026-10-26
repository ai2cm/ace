"""Print a pip constraints file pinning each dependency to its declared minimum.

Every ``>=`` lower bound in pyproject.toml's ``[project]`` dependencies (and
the requested optional-dependency groups) is converted into an ``==`` pin, so
that CI can verify the lower bounds we advertise actually work. Dependencies
without a lower bound are left to the resolver.

Usage: python minimum_constraints.py [EXTRA ...] > constraints.txt
"""

import sys
import tomllib

from packaging.requirements import Requirement


def _minimum_pins(requirements: list[str]) -> list[str]:
    pins = []
    for line in requirements:
        requirement = Requirement(line)
        for specifier in requirement.specifier:
            if specifier.operator == ">=":
                pins.append(f"{requirement.name}=={specifier.version}")
    return pins


def main(extras: list[str]):
    with open("pyproject.toml", "rb") as f:
        project = tomllib.load(f)["project"]
    requirements = list(project["dependencies"])
    for extra in extras:
        requirements.extend(project["optional-dependencies"][extra])
    for pin in _minimum_pins(requirements):
        print(pin)


if __name__ == "__main__":
    main(sys.argv[1:])

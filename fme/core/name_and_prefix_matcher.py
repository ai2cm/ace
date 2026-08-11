import dataclasses
import re
from collections.abc import Iterable


class NameAndPrefixMatcher:
    """Match variable names against a list of names and prefixes.

    The matching convention is:
        - A bare name (e.g. ``thetao``) matches the 2D variable ``thetao`` and
          all of its 3D levels ``thetao_<level>``.
        - A trailing-underscore prefix (e.g. ``thetao_``) matches all
          ``thetao_<level>``.
        - An explicit ``name_<level>`` (e.g. ``thetao_0``) matches exactly.
    """

    def __init__(self, names_and_prefixes: list[str] | None = None):
        self._regex = self._build_regex(names_and_prefixes)

    def _build_regex(self, names_and_prefixes: list[str] | None) -> str | None:
        if names_and_prefixes:
            regex = []
            for name in names_and_prefixes:
                if name.endswith("_"):
                    regex.append(rf"^{name}\d+$")
                elif not re.match(r".+_\d+$", name):
                    regex.append(f"^{name}$")
                    regex.append(rf"^{name}_\d+$")
                else:
                    regex.append(rf"^{name}$")
            return r"|".join(regex)
        return None

    def match(self, name: str) -> bool:
        """Return whether ``name`` matches any configured name or prefix."""
        if self._regex is None:
            return False
        return bool(re.match(self._regex, name))


@dataclasses.dataclass(frozen=True)
class NameAndPrefixSelection:
    """A name-and-prefix selection that keeps its entries reportable.

    ``NameAndPrefixMatcher`` has no per-entry reporting; keeping the entry
    tuple alongside the matcher enables per-entry validation
    (``unmatched_entries``) without adding state to the matcher itself.

    Parameters:
        entries: Names and prefixes following the ``NameAndPrefixMatcher``
            matching convention.
    """

    entries: tuple[str, ...]

    @property
    def matcher(self) -> NameAndPrefixMatcher:
        """A matcher over all of the selection's entries."""
        return NameAndPrefixMatcher(list(self.entries))

    def matched(self, names: Iterable[str]) -> list[str]:
        """Names (sorted) that match any entry."""
        matcher = self.matcher
        return sorted(name for name in names if matcher.match(name))

    def unmatched_entries(self, names: Iterable[str]) -> list[str]:
        """Entries that match none of ``names``."""
        names = list(names)
        return [
            entry
            for entry in self.entries
            if not any(NameAndPrefixMatcher([entry]).match(name) for name in names)
        ]

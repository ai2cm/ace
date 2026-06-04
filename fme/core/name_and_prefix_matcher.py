import re


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

    def matches(self, name: str) -> bool:
        """Return whether ``name`` matches any configured name or prefix."""
        return bool(self._regex and re.match(self._regex, name))

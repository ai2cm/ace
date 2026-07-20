from collections.abc import Iterable
from fnmatch import fnmatchcase


def match_names(patterns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    """Resolve fnmatch wildcard patterns against a set of candidate names.

    Each pattern is matched case-sensitively (``fnmatchcase``) against every
    candidate, so a pattern with no wildcard characters behaves as an exact
    name match.  Every pattern must match at least one candidate or a
    ``ValueError`` is raised; this turns typos and stale config names — which
    would otherwise silently select nothing — into errors.

    Args:
        patterns: fnmatch patterns (e.g. ``["specific_total_water_*"]``).
        candidates: the names to match against.

    Returns:
        The matched candidate names, de-duplicated and in candidate order.
    """
    candidate_list = list(candidates)
    matched: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        pattern_matches = [c for c in candidate_list if fnmatchcase(c, pattern)]
        if not pattern_matches:
            raise ValueError(
                f"pattern '{pattern}' matched none of the available names: "
                f"{candidate_list}"
            )
        for name in pattern_matches:
            if name not in seen:
                seen.add(name)
                matched.append(name)
    return matched

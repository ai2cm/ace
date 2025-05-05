from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import OmegaConf


def update_dict_with_dotlist(
    base: Mapping[str, Any], dotlist: Sequence[str] | None = None
) -> Mapping[str, Any]:
    """Update a dictionary with a dotlist of key-value pairs.

    Args:
        base: The dictionary to update.
        dotlist: A list of key-value pairs with dots in the keys indicating nesting.

    Returns:
        The updated dictionary.

    Note:
        Uses omegaconf.from_dotlist to parse the dotlist.

    Example:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> dotlist = ["b.d=3", "e=4"]
        >>> update_dict_with_dotlist(base, dotlist)
        {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
    """
    if dotlist:
        override = OmegaConf.from_dotlist(dotlist)
        merged = OmegaConf.merge(base, override)
        return OmegaConf.to_container(merged, resolve=True)
    else:
        return base

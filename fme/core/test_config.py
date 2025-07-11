from fme.core.config import update_dict_with_dotlist


def test_update_dict_with_dotlist():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    dotlist = ["a=4", "b.c=0", "b.f.g=5"]
    expected = {"a": 4, "b": {"c": 0, "d": 3, "f": {"g": 5}}}
    merged = update_dict_with_dotlist(base, dotlist)
    assert merged == expected
    assert isinstance(merged, dict)


def test_update_dict_with_dotlist_bool():
    base = {"a": 1}
    dotlist = ["b=true"]
    expected = {"a": 1, "b": True}
    assert update_dict_with_dotlist(base, dotlist) == expected


def test_update_dict_with_none_dotlist():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    dotlist = None
    assert update_dict_with_dotlist(base, dotlist) == base


def test_update_dict_with_empty_dotlist():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    dotlist: list[str] = []
    assert update_dict_with_dotlist(base, dotlist) == base

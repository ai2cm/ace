from fme.core.name_and_prefix_matcher import NameAndPrefixMatcher


def test_empty_matcher_matches_nothing():
    matcher = NameAndPrefixMatcher()
    assert not matcher.matches("thetao")
    assert not matcher.matches("thetao_0")

    matcher = NameAndPrefixMatcher([])
    assert not matcher.matches("thetao")
    assert not matcher.matches("thetao_0")


def test_bare_name_matches_2d_and_levels():
    matcher = NameAndPrefixMatcher(["thetao"])
    # bare name matches the 2D variable
    assert matcher.matches("thetao")
    # and all of its 3D levels
    assert matcher.matches("thetao_0")
    assert matcher.matches("thetao_12")
    # but not a different variable sharing a prefix
    assert not matcher.matches("thetao_extra")
    assert not matcher.matches("other")


def test_trailing_underscore_prefix_matches_levels_only():
    matcher = NameAndPrefixMatcher(["thetao_"])
    # prefix matches all levels
    assert matcher.matches("thetao_0")
    assert matcher.matches("thetao_12")
    # but not the bare 2D variable
    assert not matcher.matches("thetao")
    # and not a different variable
    assert not matcher.matches("other_0")


def test_explicit_name_level_matches_exactly():
    matcher = NameAndPrefixMatcher(["thetao_0"])
    assert matcher.matches("thetao_0")
    # other levels of the same variable are not matched
    assert not matcher.matches("thetao_1")
    # the bare variable is not matched
    assert not matcher.matches("thetao")


def test_multiple_names_and_prefixes():
    matcher = NameAndPrefixMatcher(["PRESsfc", "so_", "thetao_0"])
    assert matcher.matches("PRESsfc")
    assert matcher.matches("so_3")
    assert matcher.matches("thetao_0")
    assert not matcher.matches("so")
    assert not matcher.matches("thetao_1")

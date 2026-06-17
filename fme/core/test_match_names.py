import pytest

from fme.core.match_names import match_names


def test_exact_match():
    assert match_names(["air_temperature_0"], ["air_temperature_0", "PRESsfc"]) == [
        "air_temperature_0"
    ]


def test_wildcard_matches_in_candidate_order():
    candidates = [
        "specific_total_water_0",
        "PRESsfc",
        "specific_total_water_1",
        "specific_total_water_2",
    ]
    assert match_names(["specific_total_water_*"], candidates) == [
        "specific_total_water_0",
        "specific_total_water_1",
        "specific_total_water_2",
    ]


def test_zero_matches_raises():
    with pytest.raises(ValueError, match="matched none"):
        match_names(["does_not_exist_*"], ["air_temperature_0"])


def test_results_deduplicated_across_overlapping_patterns():
    candidates = ["specific_total_water_0", "specific_total_water_1"]
    result = match_names(
        ["specific_total_water_*", "specific_total_water_0"], candidates
    )
    assert result == ["specific_total_water_0", "specific_total_water_1"]


def test_case_sensitive():
    with pytest.raises(ValueError, match="matched none"):
        match_names(["AIR_TEMPERATURE_0"], ["air_temperature_0"])

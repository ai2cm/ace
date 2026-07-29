import pytest

from fme.ace.requirements import DataRequirements
from fme.core.dataset.schedule import IntMilestone, IntSchedule
from fme.translate.data.requirements import (
    ObjectiveDataRequirements,
    StreamRequirements,
    TranslateDataRequirements,
)


def _objective(**streams: StreamRequirements) -> ObjectiveDataRequirements:
    return ObjectiveDataRequirements(streams=streams)


def test_merges_variable_names_as_ordered_union():
    requirements = TranslateDataRequirements.from_objectives(
        [
            _objective(era5_1deg=StreamRequirements(names=["temp", "humidity"])),
            _objective(era5_1deg=StreamRequirements(names=["humidity", "wind"])),
        ]
    )
    assert requirements.streams["era5_1deg"].names == ["temp", "humidity", "wind"]


def test_merges_n_timesteps_as_pointwise_max_over_schedules():
    requirements = TranslateDataRequirements.from_objectives(
        [
            # constant window of 3
            _objective(era5=StreamRequirements(names=["temp"], n_timesteps=3)),
            # 1 until epoch 4, then 5
            _objective(
                era5=StreamRequirements(
                    names=["temp"],
                    n_timesteps=IntSchedule(
                        start_value=1, milestones=[IntMilestone(epoch=4, value=5)]
                    ),
                )
            ),
        ]
    )
    schedule = requirements.streams["era5"].n_timesteps_schedule
    assert schedule.get_value(0) == 3
    assert schedule.get_value(3) == 3
    assert schedule.get_value(4) == 5
    assert schedule.get_value(10) == 5


def test_streams_in_one_objective_are_one_pairing_group():
    requirements = TranslateDataRequirements.from_objectives(
        [
            _objective(
                fine=StreamRequirements(names=["temp"]),
                coarse=StreamRequirements(names=["temp"]),
            )
        ]
    )
    assert [sorted(group) for group in requirements.pairing_groups] == [
        ["coarse", "fine"]
    ]


def test_pairing_is_transitive_across_objectives():
    """Objectives over (a, b) and (b, c) put all three on one time index."""
    requirements = TranslateDataRequirements.from_objectives(
        [
            _objective(
                era5_1deg=StreamRequirements(names=["temp"]),
                era5_2deg=StreamRequirements(names=["temp"]),
            ),
            _objective(
                era5_2deg=StreamRequirements(names=["temp"]),
                era5_4deg=StreamRequirements(names=["temp"]),
            ),
        ]
    )
    assert [sorted(group) for group in requirements.pairing_groups] == [
        ["era5_1deg", "era5_2deg", "era5_4deg"]
    ]


def test_streams_that_never_co_occur_are_independent_singletons():
    """The transfer-learning shape: no objective reads both streams."""
    requirements = TranslateDataRequirements.from_objectives(
        [
            _objective(era5=StreamRequirements(names=["temp"])),
            _objective(shield_c96=StreamRequirements(names=["temp"])),
        ]
    )
    assert requirements.pairing_groups == [["era5"], ["shield_c96"]]


def test_pairing_groups_are_deterministically_ordered():
    requirements = TranslateDataRequirements.from_objectives(
        [
            _objective(
                b=StreamRequirements(names=["temp"]),
                a=StreamRequirements(names=["temp"]),
            ),
            _objective(c=StreamRequirements(names=["temp"])),
        ]
    )
    assert requirements.pairing_groups == [["b", "a"], ["c"]]


def test_no_objectives_raises():
    with pytest.raises(ValueError, match="At least one objective"):
        TranslateDataRequirements.from_objectives([])


def test_objective_with_no_streams_raises():
    with pytest.raises(ValueError, match="at least one data stream"):
        ObjectiveDataRequirements(streams={})


def test_stream_requirements_with_no_names_raises():
    with pytest.raises(ValueError, match="at least one variable name"):
        StreamRequirements(names=[])


def test_pairing_groups_must_partition_streams():
    requirements = TranslateDataRequirements.from_objectives(
        [_objective(era5=StreamRequirements(names=["temp"]))]
    )
    with pytest.raises(ValueError, match="must partition the streams"):
        TranslateDataRequirements(
            streams=requirements.streams, pairing_groups=[["era5"], ["missing"]]
        )


def test_a_stream_in_two_pairing_groups_is_rejected():
    requirements = TranslateDataRequirements.from_objectives(
        [_objective(era5=StreamRequirements(names=["temp"]))]
    )
    with pytest.raises(ValueError, match="only one pairing group"):
        TranslateDataRequirements(
            streams=requirements.streams, pairing_groups=[["era5"], ["era5"]]
        )


def test_allow_missing_variables_is_rejected():
    """The loader ignores the flag, so carrying it set would load silently wrong."""
    with pytest.raises(ValueError, match="allow_missing_variables is not supported"):
        TranslateDataRequirements(
            streams={
                "era5": DataRequirements(
                    names=["temp"], n_timesteps=1, allow_missing_variables=True
                )
            },
            pairing_groups=[["era5"]],
        )

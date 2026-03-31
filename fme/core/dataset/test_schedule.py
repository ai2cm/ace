import pytest

from fme.core.dataset.schedule import IntMilestone, IntSchedule, ValidatedMilestones


@pytest.mark.parametrize(
    "start_value, milestones, epoch, expected_value",
    [
        pytest.param(10, [], 0, 10, id="constant"),
        pytest.param(10, [], 20, 10, id="constant_later"),
        pytest.param(
            10, [IntMilestone(epoch=1, value=20)], 0, 10, id="before_milestone"
        ),
        pytest.param(10, [IntMilestone(epoch=1, value=20)], 1, 20, id="at_milestone"),
        pytest.param(
            10, [IntMilestone(epoch=1, value=20)], 2, 20, id="after_milestone"
        ),
        pytest.param(
            0,
            [IntMilestone(epoch=5, value=20), IntMilestone(epoch=10, value=30)],
            7,
            20,
            id="multiple_milestones",
        ),
        pytest.param(
            0,
            [IntMilestone(epoch=10, value=30), IntMilestone(epoch=5, value=20)],
            7,
            20,
            id="multiple_milestones_out_of_order",
        ),
    ],
)
def test_validated_milestones_get_value(
    start_value: int, milestones: list[IntMilestone], epoch: int, expected_value: int
):
    schedule = ValidatedMilestones(start_value=start_value, milestones=milestones)
    assert schedule.get_value(epoch) == expected_value


def test_negative_epoch_raises():
    schedule = ValidatedMilestones(
        start_value=0,
        milestones=[IntMilestone(epoch=5, value=20)],
    )
    with pytest.raises(ValueError, match="Epoch must be non-negative"):
        schedule.get_value(-1)


def test_validated_milestones_duplicate_milestone_epochs():
    with pytest.raises(ValueError, match="Milestones must have unique epochs"):
        ValidatedMilestones(
            start_value=0,
            milestones=[
                IntMilestone(epoch=5, value=20),
                IntMilestone(epoch=5, value=30),
            ],
        )


def test_int_schedule_add():
    schedule = IntSchedule(
        start_value=10,
        milestones=[
            IntMilestone(epoch=5, value=20),
            IntMilestone(epoch=10, value=30),
        ],
    ).add(5)
    assert schedule.get_value(0) == 15
    assert schedule.get_value(4) == 15
    assert schedule.get_value(5) == 25
    assert schedule.get_value(9) == 25
    assert schedule.get_value(10) == 35
    assert schedule.get_value(15) == 35


def test_int_schedule_max_value():
    schedule = IntSchedule(
        start_value=10,
        milestones=[
            IntMilestone(epoch=5, value=20),
            IntMilestone(epoch=10, value=15),
        ],
    )
    assert schedule.max_value == 20

    constant_schedule = IntSchedule.from_constant(7)
    assert constant_schedule.max_value == 7


def test_int_schedule_from_constant():
    schedule = IntSchedule.from_constant(42)
    assert schedule.get_value(0) == 42
    assert schedule.get_value(100) == 42

import pytest

from fme.core.dataset.schedule import IntMilestone, IntSchedule


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
def test_int_schedule(
    start_value: int, milestones: list[IntMilestone], epoch: int, expected_value: int
):
    schedule = IntSchedule(start_value=start_value, milestones=milestones)
    assert schedule.get_value(epoch) == expected_value


def test_int_schedule_duplicate_milestone_epochs():
    with pytest.raises(ValueError, match="Milestones must have unique epochs"):
        IntSchedule(
            start_value=0,
            milestones=[
                IntMilestone(epoch=5, value=20),
                IntMilestone(epoch=5, value=30),
            ],
        )

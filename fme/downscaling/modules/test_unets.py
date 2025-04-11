import pytest

from fme.downscaling.modules.unets import check_level_compatibility


@pytest.mark.parametrize(
    "channel_mult, attn_res, passes",
    [
        pytest.param([1], [], True, id="no_division"),
        pytest.param([1] * 3, [], True, id="max_division"),
        pytest.param([1] * 4, [], False, id="too_many_division"),
        pytest.param([1, 1], [], True, id="no_attn_res"),
        pytest.param([1, 1], [2], True, id="single_attn_res"),
        pytest.param([1, 1], [2, 4], True, id="multiple_attn_res"),
        pytest.param([1, 1], [1], False, id="single_attn_res_missing"),
        pytest.param([1, 1], [1, 2], False, id="multiple_attn_res_missing"),
        pytest.param([1, 1], [1, 1], False, id="duplicated_attn"),
    ],
)
def test_check_level_compatibility(channel_mult, attn_res, passes):
    img_res = 4
    if passes:
        check_level_compatibility(img_res, channel_mult, attn_res)
    else:
        with pytest.raises(ValueError):
            check_level_compatibility(img_res, channel_mult, attn_res)

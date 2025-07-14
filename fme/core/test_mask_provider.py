import pytest
import torch

from fme.core.mask_provider import MaskProvider, NullMaskProvider


def test_mask_provider_init_error():
    with pytest.raises(ValueError, match="var_1"):
        _ = MaskProvider(
            masks={
                "mask_0": torch.rand(2, 2),
                "var_1": torch.rand(2, 2),
            }
        )


def test_mask_provider_get_mask_tensor_for():
    mask_0 = torch.rand(2, 2)
    mask_1 = torch.rand(2, 2)
    mask_temp_1 = torch.rand(2, 2)
    mask_2d = torch.rand(2, 2)
    provider = MaskProvider(
        masks={
            "mask_0": mask_0,
            "mask_1": mask_1,
            "mask_temp_1": mask_temp_1,
            "mask_2d": mask_2d,
        }
    )
    # variable specific mask
    assert torch.equal(provider.get_mask_tensor_for("temp_1"), mask_temp_1)
    # level specific masks
    assert torch.equal(provider.get_mask_tensor_for("temp_0"), mask_0)
    assert torch.equal(provider.get_mask_tensor_for("humidity_0"), mask_0)
    assert torch.equal(provider.get_mask_tensor_for("humidity_1"), mask_1)
    # surface mask for non-3D vars
    assert torch.equal(provider.get_mask_tensor_for("surface_var"), mask_2d)


def test_mask_provider_get_mask_tensor_for_no_3d():
    mask_temp_1 = torch.rand(2, 2)
    mask_2d = torch.rand(2, 2)
    provider = MaskProvider(
        masks={
            "mask_temp_1": mask_temp_1,
            "mask_2d": mask_2d,
        }
    )
    # variable specific mask
    assert torch.equal(provider.get_mask_tensor_for("temp_1"), mask_temp_1)
    # 3D var doesn't trigger return of surface mask
    assert provider.get_mask_tensor_for("temp_0") is None
    # surface mask returned
    assert torch.equal(provider.get_mask_tensor_for("surface_var"), mask_2d)


def test_mask_provider_get_mask_tensor_for_var_specific_only():
    mask_temp_1 = torch.rand(2, 2)
    mask_var = torch.rand(2, 2)
    provider = MaskProvider(
        masks={
            "mask_temp_1": mask_temp_1,
            "mask_var": mask_var,
        }
    )
    # variable specific masks
    assert torch.equal(provider.get_mask_tensor_for("temp_1"), mask_temp_1)
    assert torch.equal(provider.get_mask_tensor_for("var"), mask_var)
    # other 3D vars have no mask
    assert provider.get_mask_tensor_for("temp_0") is None
    # other 2D vars have no mask
    assert provider.get_mask_tensor_for("var_2d") is None


def test_mask_provider_empty_init():
    provider = MaskProvider()
    assert provider.get_mask_tensor_for("temp_0") is None
    assert provider.get_mask_tensor_for("var") is None


def test_mask_provider_to_device():
    # test moving masks to a different device
    masks = {"mask_temp": torch.tensor([1, 0], dtype=torch.float32)}
    provider_cpu = MaskProvider(masks=masks)
    # using 'meta' device for testing without needing specific hardware
    provider_meta = provider_cpu.to("meta")
    assert provider_meta.masks["mask_temp"].device == torch.device("meta")
    # original provider should be unchanged
    assert provider_cpu.masks["mask_temp"].device == torch.device("cpu")


def test_mask_provider_update_success():
    # test updating masks from another provider with non-overlapping keys
    masks1 = {"mask_temp": torch.tensor([1, 0])}
    masks2 = {"mask_humidity": torch.tensor([0, 1])}
    provider1 = MaskProvider(masks=masks1)
    provider2 = MaskProvider(masks=masks2)

    provider1.update(provider2)

    expected_masks = {
        "mask_temp": torch.tensor([1, 0]),
        "mask_humidity": torch.tensor([0, 1]),
    }

    assert provider1.masks.keys() == expected_masks.keys()
    for name, mask in provider1.masks.items():
        assert torch.equal(mask, expected_masks[name])


def test_mask_provider_update_failure_overlapping_keys():
    # test updating masks fails when keys overlap
    masks1 = {"mask_temp": torch.tensor([1, 0]), "mask_common": torch.tensor([0, 0])}
    masks2 = {
        "mask_humidity": torch.tensor([0, 1]),
        "mask_common": torch.tensor([1, 1]),
    }
    provider1 = MaskProvider(masks=masks1)
    provider2 = MaskProvider(masks=masks2)

    with pytest.raises(ValueError, match="mask_common"):
        provider1.update(provider2)

    # Ensure original provider is unchanged
    assert provider1.masks.keys() == masks1.keys()
    for name, mask in provider1.masks.items():
        assert torch.equal(mask, masks1[name])


@pytest.mark.parametrize(
    "masks1, masks2, expected_equal",
    [
        pytest.param(
            {"mask_temp": torch.tensor([1, 0])},
            {"mask_temp": torch.tensor([1, 0])},
            True,
            id="providers_equal",
        ),
        pytest.param(
            {"mask_temp": torch.tensor([1, 0])},
            {"mask_temp": torch.tensor([1, 1])},
            False,
            id="providers_unequal",
        ),
        pytest.param(
            {"mask_temp": torch.tensor([1, 1]), "mask_humidity": torch.tensor([1, 1])},
            {"mask_temp": torch.tensor([1, 0])},
            False,
            id="first_provider_extra_key",
        ),
        pytest.param(
            {"mask_temp": torch.tensor([1, 0])},
            {"mask_temp": torch.tensor([1, 1]), "mask_humidity": torch.tensor([1, 1])},
            False,
            id="second_provider_extra_key",
        ),
        pytest.param(
            {"mask_temp": torch.ones(2, 2)},
            {"mask_temp": torch.ones(4, 4)},
            False,
            id="providers_mask_shapes_different",
        ),
    ],
)
def test_mask_provider_eq(masks1, masks2, expected_equal: bool):
    provider1 = MaskProvider(masks=masks1)
    provider2 = MaskProvider(masks=masks2)
    if expected_equal:
        assert provider1 == provider2
    else:
        assert provider1 != provider2


@pytest.mark.parametrize(
    "mask_provider",
    [
        pytest.param(MaskProvider(), id="empty"),
        pytest.param(
            MaskProvider(masks={"mask_0": torch.ones(10, 10)}), id="single_mask"
        ),
        pytest.param(
            MaskProvider(
                masks={
                    "mask_0": torch.ones(10, 10),
                    "mask_temp": torch.zeros(10, 10),
                }
            ),
            id="multiple_masks",
        ),
    ],
)
def test_mask_provider_round_trip(mask_provider: MaskProvider):
    state = mask_provider.to_state()
    reloaded_provider = MaskProvider.from_state(state)
    assert mask_provider == reloaded_provider


def test_null_mask_provider_update_err():
    mask_provider = MaskProvider(masks={"mask_2d": torch.rand(2, 2)})
    with pytest.raises(ValueError, match="mask_2d"):
        NullMaskProvider.update(mask_provider)

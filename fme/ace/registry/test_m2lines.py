import pytest

from fme.ace.registry.m2lines import SamudraBuilder


def test_samudra_builder():
    builder = SamudraBuilder()
    # assuming 5 input (3 prognostic + 2 forcing) and 3 output vars (prognostic)
    model = builder.build(5, 3, (16, 32))
    assert model.layers[0].convblock[0].in_channels == 5
    assert model.layers[-1].out_channels == 3

    with pytest.raises(ValueError, match="norm_kwargs should not have num_features"):
        _ = SamudraBuilder(norm_kwargs={"num_features": 10})

    with pytest.raises(
        ValueError, match="norm_kwargs should not have normalized_shape"
    ):
        _ = SamudraBuilder(norm_kwargs={"normalized_shape": (3, 3)})

import pytest
import torch

from fme.core.weight_ops import overwrite_weights


class SimpleLinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.custom_param = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return self.linear(x) + self.custom_param


class NestedModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = SimpleLinearModule(in_features, out_features)
        self.linear2 = SimpleLinearModule(out_features, in_features)
        self.custom_param = torch.nn.Parameter(torch.randn(5, 5))


@pytest.mark.parametrize(
    "from_module, to_module, expected_exception",
    [
        pytest.param(
            SimpleLinearModule(10, 10),
            SimpleLinearModule(10, 10),
            None,
            id="Matching sizes",
        ),
        pytest.param(
            SimpleLinearModule(10, 10),
            SimpleLinearModule(20, 10),
            None,
            id="to_module larger weights",
        ),
        pytest.param(
            SimpleLinearModule(20, 10),
            SimpleLinearModule(10, 10),
            ValueError,
            id="from_module larger weights",
        ),
        pytest.param(
            NestedModule(10, 20),
            NestedModule(10, 20),
            None,
            id="Complex modules matching sizes",
        ),
        pytest.param(
            NestedModule(10, 20),
            SimpleLinearModule(10, 20),
            ValueError,
            id="Nested modules, mismatched structure",
        ),
    ],
)
def test_overwrite_weights(from_module, to_module, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            overwrite_weights(from_module.state_dict(), to_module)
    else:
        overwrite_weights(from_module.state_dict(), to_module)
        for from_param, to_param in zip(
            from_module.parameters(), to_module.parameters()
        ):
            if len(from_param.shape) == 1:
                assert torch.allclose(
                    from_param.data, to_param.data[: from_param.data.size(0)]
                )
            else:
                assert torch.allclose(
                    from_param.data,
                    to_param.data[: from_param.data.size(0), : from_param.data.size(1)],
                )


def test_overwrite_weights_exclude():
    from_module = NestedModule(10, 20)
    to_module = NestedModule(10, 20)
    overwrite_weights(
        from_module.state_dict(), to_module, exclude_parameters=["linear1.*"]
    )
    assert not torch.allclose(
        from_module.linear1.linear.weight, to_module.linear1.linear.weight
    )
    assert not torch.allclose(
        from_module.linear1.custom_param, to_module.linear1.custom_param
    )
    assert torch.allclose(
        from_module.linear2.linear.weight, to_module.linear2.linear.weight
    )
    assert torch.allclose(
        from_module.linear2.custom_param, to_module.linear2.custom_param
    )
    assert torch.allclose(from_module.custom_param, to_module.custom_param)

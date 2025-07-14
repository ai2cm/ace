import numpy as np
import pytest
import torch
from torch import nn

from fme.core.weight_ops import CopyWeightsConfig, overwrite_weights


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


class NestedModule2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))


class NestedModule1(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))
        self.nested = NestedModule2()


@pytest.mark.parametrize(
    "include, exclude, expected_applied, expected_error",
    [
        pytest.param(["*"], [], ["weight", "nested.weight"], None, id="include all"),
        pytest.param([], ["*"], [], None, id="exclude all"),
        pytest.param(["weight"], ["nested.*"], ["weight"], None, id="weight included"),
        pytest.param(["*"], ["nested.*"], [], ValueError, id="nested param in both"),
        pytest.param(["*"], ["weight"], [], ValueError, id="* include with an exclude"),
        pytest.param([], ["weight"], [], ValueError, id="missing weight using exclude"),
        pytest.param(["weight"], [], [], ValueError, id="missing weight using include"),
        pytest.param(
            ["*.weight"], [], [], ValueError, id="mising weight using wildcard include"
        ),
    ],
)
def test_apply_copy_weights_config(
    include: list[str],
    exclude: list[str],
    expected_applied: list[str],
    expected_error: type[Exception] | None,
):
    source_model = NestedModule1()
    dest_model = NestedModule1()
    original_dest_model_state = dest_model.state_dict()

    if expected_error is not None:
        with pytest.raises(expected_error):
            config = CopyWeightsConfig(
                include=include,
                exclude=exclude,
            )
            config.apply([source_model.state_dict()], [dest_model])
    else:
        config = CopyWeightsConfig(
            include=include,
            exclude=exclude,
        )
        config.apply([source_model.state_dict()], [dest_model])

        for name, param in dest_model.named_parameters():
            if name in expected_applied:
                np.testing.assert_array_equal(
                    param.detach().cpu().numpy(),
                    source_model.state_dict()[name].detach().cpu().numpy(),
                )
            else:
                np.testing.assert_array_equal(
                    param.detach().cpu().numpy(),
                    original_dest_model_state[name].detach().cpu().numpy(),
                )

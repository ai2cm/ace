import torch

from .utils import _force_positive, replace_value_keep_gradient


def test_force_positive():
    data = {
        "foo": torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]]),
        "bar": torch.tensor([[-1.0, 0.0], [0.0, -3.0], [1.0, 2.0]]),
    }
    original_min = torch.min(data["foo"])
    assert original_min < 0.0
    fixed_data = _force_positive(data, ["foo"])
    new_min = torch.min(fixed_data["foo"])
    # Ensure the minimum value of 'foo' is now 0
    torch.testing.assert_close(new_min, torch.tensor(0.0))
    # Ensure other variables are not modified
    torch.testing.assert_close(fixed_data["bar"], data["bar"])


def test_replace_value_keep_gradient():
    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    new_value = torch.clamp(x, min=0.0)
    out = replace_value_keep_gradient(x, new_value)
    # forward: exactly the projected (clamped) value
    torch.testing.assert_close(out, torch.tensor([0.0, 0.5, 2.0]))
    # backward: identity gradient everywhere, including the clamped tail
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_force_positive_keep_gradient_passes_gradient_on_clamped_cells():
    # A clamped-negative input gets zero gradient with a plain clamp, but a
    # nonzero (identity) gradient when keep_gradient=True.
    x_plain = torch.tensor([-2.0, 1.0], requires_grad=True)
    _force_positive({"foo": x_plain}, ["foo"])["foo"].sum().backward()
    torch.testing.assert_close(x_plain.grad, torch.tensor([0.0, 1.0]))

    x_ste = torch.tensor([-2.0, 1.0], requires_grad=True)
    fixed = _force_positive({"foo": x_ste}, ["foo"], keep_gradient=True)
    # forward value is still clamped
    torch.testing.assert_close(fixed["foo"], torch.tensor([0.0, 1.0]))
    fixed["foo"].sum().backward()
    torch.testing.assert_close(x_ste.grad, torch.tensor([1.0, 1.0]))

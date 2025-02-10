import pytest
import torch
from torch import nn

import fme


def test_gradient_accumulation_with_detach():
    model = nn.Linear(1, 1).to(fme.get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    x = torch.randn(10, 1).to(fme.get_device())
    x_series = []
    for i in range(10):
        new_x = model(x)
        loss = new_x.sum()
        loss.backward()
        x_series.append(new_x)
        x = new_x.detach()
    optimizer.step()


def test_gradient_accumulation_without_detach():
    model = nn.Linear(1, 1).to(fme.get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    x = torch.randn(10, 1).to(fme.get_device())
    new_x = model(x)
    loss = new_x.sum()
    loss.backward()
    x = new_x
    new_x = model(x)
    loss = new_x.sum()
    with pytest.raises(RuntimeError) as err:
        loss.backward()
    assert "Trying to backward through the graph a second time" in str(err.value)
    optimizer.step()


def test_gradient_accumulation_with_detach_after_compute():
    model = nn.Linear(1, 1).to(fme.get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    x = torch.randn(10, 1).to(fme.get_device())
    x_series = []
    for i in range(10):
        new_x = model(x)
        x_series.append(new_x)
        x = new_x.detach()
    for i in range(len(x_series)):
        loss = x_series[i].sum()
        loss.backward()
    optimizer.step()


def test_gradient_accumulation_with_detach_and_concat():
    model = nn.Linear(1, 1).to(fme.get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    x = torch.randn(10, 1).to(fme.get_device())
    x_series = []
    for i in range(10):
        new_x = model(x)
        x_series.append(new_x)
        x = new_x.detach()
    output = torch.stack(x_series, dim=1)
    loss = output[:, 0].sum()
    loss.backward()
    loss = output[:, 1].sum()
    with pytest.raises(RuntimeError) as err:
        loss.backward()
    assert "Trying to backward through the graph a second time" in str(err.value)

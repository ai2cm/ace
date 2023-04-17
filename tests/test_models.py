import torch
import networks
import networks.vit

import pytest
import config


@pytest.mark.xfail
def test_get_model(has_registry):
    model = networks.get_model("tfno_no-patching_lr5e-4_full_epochs")
    device = "cuda:0"

    if torch.cuda.is_available():
        with torch.no_grad():
            model = model.to(device)
            xp = torch.zeros((1, 5 * 34 + 2, 720, 1440)).to(device)
            y = model.model(xp)
        assert tuple(y.shape) == (1, 34, 720, 1440)


def test_vit(has_registry):
    model = networks.get_architecture("afno_26ch_v")
    device = "cuda:0"

    if torch.cuda.is_available():
        with torch.no_grad():
            model = model.to(device)
            xp = torch.zeros((1, 26, 720, 1440)).to(device)
            y = model(xp)
        assert tuple(y.shape) == (1, 26, 720, 1440)


@pytest.mark.xfail
def test_modulus_afno(has_registry):
    model = networks.get_architecture("modulus_afno_20")
    device = "cuda:0"

    if torch.cuda.is_available():
        with torch.no_grad():
            model = model.to(device)
            xp = torch.zeros((1, 20, 720, 1440)).to(device)
            y = model.model(xp)
        assert tuple(y.shape) == (1, 20, 720, 1440)


def test_graphcast():

    if not config.MODEL_REGISTRY:
        pytest.skip("Model registry required to instantiate a graphcast model")

    model = networks.get_architecture(
        "graphcast_34ch", model_path=networks.registry.get_model_path("graphcast_34ch")
    )
    device = "cuda:0"

    with torch.no_grad():
        model = model.to(device)
        xp = torch.zeros((1, 34, 721, 1440), dtype=torch.bfloat16).to(device)
        y = model(xp)
    assert tuple(y.shape) == (1, 34, 721, 1440)

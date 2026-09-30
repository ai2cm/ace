import pytest

from fme.translate.domains import DomainConfig, LatentChannels


def test_latent_channels_expand_to_indexed_names():
    block = LatentChannels(name="z", channels=3)
    assert block.names == ["z_0", "z_1", "z_2"]


def test_latent_channels_validation():
    with pytest.raises(ValueError, match="channels >= 1"):
        LatentChannels(name="z", channels=0)
    with pytest.raises(ValueError, match="non-empty name"):
        LatentChannels(name="", channels=2)


def test_domain_mixes_variables_and_latent_blocks_in_order():
    domain = DomainConfig(
        channels=["air_temperature_0", LatentChannels(name="z", channels=2), "co2"]
    )
    assert domain.names == ["air_temperature_0", "z_0", "z_1", "co2"]
    assert domain.n_channels == 4


def test_domain_rejects_empty_channels():
    with pytest.raises(ValueError, match="at least one channel"):
        DomainConfig(channels=[])


def test_domain_rejects_duplicate_names_after_expansion():
    with pytest.raises(ValueError, match="duplicates.*z_0"):
        DomainConfig(channels=["z_0", LatentChannels(name="z", channels=1)])
    with pytest.raises(ValueError, match="duplicates"):
        DomainConfig(
            channels=[
                LatentChannels(name="z", channels=2),
                LatentChannels(name="z", channels=2),
            ]
        )

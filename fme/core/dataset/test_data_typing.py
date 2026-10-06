from fme.core.dataset.data_typing import VariableMetadata


def test_from_attrs_both():
    m = VariableMetadata.from_attrs({"units": "K", "long_name": "Temperature"})
    assert m.units == "K"
    assert m.long_name == "Temperature"


def test_from_attrs_partial():
    m = VariableMetadata.from_attrs({"units": "K"})
    assert m.units == "K"
    assert m.long_name is None


def test_from_attrs_empty():
    m = VariableMetadata.from_attrs({})
    assert m.units is None
    assert m.long_name is None


def test_as_attrs_filters_none():
    assert VariableMetadata().as_attrs() == {}
    assert VariableMetadata(units="K").as_attrs() == {"units": "K"}
    assert VariableMetadata(long_name="T").as_attrs() == {"long_name": "T"}
    assert VariableMetadata("K", "T").as_attrs() == {"units": "K", "long_name": "T"}


def test_display_long_name():
    assert VariableMetadata().display_long_name("fallback") == "fallback"
    assert (
        VariableMetadata(long_name="Temperature").display_long_name("x")
        == "Temperature"
    )
    assert VariableMetadata(long_name="").display_long_name("x") == ""


def test_display_units():
    assert VariableMetadata().display_units() == "unknown_units"
    assert VariableMetadata().display_units("custom") == "custom"
    assert VariableMetadata(units="K").display_units() == "K"
    assert VariableMetadata(units="").display_units() == ""

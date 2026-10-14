from fme.core.compile import compile_enabled, is_compile_enabled


def test_tests_run_with_compile_disabled_by_default():
    """The suite's conftest disables compilation for modules that do not opt in
    via ``fme.core.testing.dynamo_hygiene``."""
    assert not is_compile_enabled()


def test_compile_enabled_restores_previous_setting():
    with compile_enabled(True):
        assert is_compile_enabled()
        with compile_enabled(False):
            assert not is_compile_enabled()
        assert is_compile_enabled()
    assert not is_compile_enabled()

import dataclasses

from .registry import Registry


def test_registry_without_base_type():
    """Test that Registry without a base type works."""

    # Create registry that accepts any class
    model_registry: Registry = Registry()

    # Register a class
    @model_registry.register("model_a")
    @dataclasses.dataclass
    class ModelA:
        pass

    assert isinstance(model_registry.get("model_a", {}), ModelA)


def test_registry_type_checking():
    """Test that Registry enforces type checking on registered classes."""

    # Base class for type checking
    class BaseModel:
        pass

    # Create registry that only accepts BaseModel subclasses
    model_registry: Registry[BaseModel] = Registry[BaseModel]()

    # This should work - ModelA inherits from BaseModel
    @model_registry.register("model_a")
    @dataclasses.dataclass
    class ModelA(BaseModel):
        pass

    try:
        # This would fail in mypy as ModelB is not a subclass of BaseModel
        @model_registry.register("model_b")  # type: ignore[arg-type]
        @dataclasses.dataclass
        class ModelB:
            pass

        assert False, "Expected TypeError - ModelB is not a subclass of BaseModel"
    except TypeError:
        pass

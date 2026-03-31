from collections.abc import Callable, Mapping
from typing import Any, Generic, Protocol, Self, TypeVar


class HasFromState(Protocol):
    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self: ...


T = TypeVar("T", bound=HasFromState)


class Registry(Generic[T]):
    """
    Used to register and initialize multiple types of a dataclass.
    """

    def __init__(self):
        """
        Initialize the registry.

        Args:
            default_type: if given, the "type" key in the config dict is optional
                and by default this type will be used.
        """
        self._types: dict[str, type[T]] = {}

    def register(self, type_name: str) -> Callable[[type[T]], type[T]]:
        """
        Registers a configuration type with the registry.

        When registry.get is called to initialize a dataclass, if the
        "type" argument passed is equal to the type_name you give here,
        then the decorated class will be the one initialized from the data
        in the "config" argument.

        Args:
            type_name: name used in configuration to indicate the decorated
                class as the target type to be initialized when using `get`.
        """

        def register_func(cls: type[T]) -> type[T]:
            base_type = None
            # attribute exists only when type parameter is passed
            # i.e. Registry[BaseClass]()
            if hasattr(self, "__orig_class__"):
                base_type = self.__orig_class__.__args__[0]

            if base_type and not issubclass(cls, base_type):
                raise TypeError(f"{cls} must be a subclass of {base_type}")

            self._types[type_name] = cls
            return cls

        return register_func

    def get(self, type_name: str, config: Mapping[str, Any]) -> T:
        cls = self._types[type_name]
        return cls.from_state(config)

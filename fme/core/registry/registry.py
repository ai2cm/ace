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
        self._compile_unsupported_reasons: dict[str, str] = {}

    def register(
        self, type_name: str, compile_unsupported_reason: str | None = None
    ) -> Callable[[type[T]], type[T]]:
        """
        Registers a configuration type with the registry.

        When registry.get is called to initialize a dataclass, if the
        "type" argument passed is equal to the type_name you give here,
        then the decorated class will be the one initialized from the data
        in the "config" argument.

        Args:
            type_name: name used in configuration to indicate the decorated
                class as the target type to be initialized when using `get`.
            compile_unsupported_reason: if given, why ``torch.compile`` cannot
                be used with what the registered type builds. It is a property
                of the type rather than of a config instance, so it is kept
                here instead of as a dataclass field, keeping it out of
                serialized configs and checkpoints. Read it back with
                `get_compile_unsupported_reason`.
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
            if compile_unsupported_reason is not None:
                self._compile_unsupported_reasons[type_name] = (
                    compile_unsupported_reason
                )
            else:
                self._compile_unsupported_reasons.pop(type_name, None)
            return cls

        return register_func

    def get(self, type_name: str, config: Mapping[str, Any]) -> T:
        cls = self._types[type_name]
        return cls.from_state(config)

    def get_compile_unsupported_reason(self, type_name: str) -> str | None:
        """The reason given at registration for why ``torch.compile`` cannot be
        used with the type, or None if compilation is supported.
        """
        return self._compile_unsupported_reasons.get(type_name)

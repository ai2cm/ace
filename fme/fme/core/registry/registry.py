from typing import Any, Callable, Dict, Generic, Mapping, Optional, Type, TypeVar

import dacite

T = TypeVar("T")
TT = TypeVar("TT", bound=Type)


class Registry(Generic[T]):
    """
    Used to register and initialize multiple types of a dataclass.
    """

    def __init__(self, default_type: Optional[str] = None):
        """
        Initialize the registry.

        Args:
            default_type: if given, the "type" key in the config dict is optional
                and by default this type will be used.
        """
        self._types: Dict[str, Type[T]] = {}
        self.default_type = default_type

    def register(self, type_name: str) -> Callable[[TT], TT]:
        """
        Registers a configuration type with the registry.

        When registry.from_dict is called to initialize a dataclass, if the
        "type" key in that dictionary is equal to the type_name you give here,
        then the decorated class will be the one initialized from the data
        in the "config" key.

        Args:
            type_name: name used in configuration to indicate the decorated
                class as the target type to be initialized when using from_dict.
        """

        def register_func(cls: TT) -> TT:
            self._types[type_name] = cls
            return cls

        return register_func

    def from_dict(self, config: Mapping[str, Any]) -> T:
        """
        Creates a registered type from the given config dict.

        Config should have at least one key, "type", which indicates the type to
        initialize based on its registered type name. This can be omitted if
        this instance was initialized with a default type.

        It can also have a "config" key, which is a dict used to initialize the
        dataclass. By default this is an empty dict.
        """
        config = dict(config)
        config.setdefault("config", {})
        if self.default_type is not None:
            type_name = config.get("type", self.default_type)
        else:
            type_name = config["type"]
        if type_name not in self._types:
            raise ValueError(
                f"Received unexpected type {type_name}, "
                f"expected one of {self._types.keys()}"
            )
        else:
            instance = dacite.from_dict(
                data_class=self._types[type_name],
                data=config["config"],
                config=dacite.Config(strict=True),
            )
            return instance

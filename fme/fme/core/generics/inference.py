import abc
from typing import Generic, Protocol, Tuple, TypeVar

from fme.core.generics.data import DataLoader

PS = TypeVar("PS")  # prognostic state
FD = TypeVar("FD", contravariant=True)  # forcing data
SD = TypeVar("SD", covariant=True)  # stepped data


class PredictFunction(Protocol, Generic[PS, FD, SD]):
    def __call__(
        self,
        initial_condition: PS,
        forcing: FD,
        compute_derived_variables: bool = False,
    ) -> Tuple[SD, PS]: ...


class InferenceDataABC(abc.ABC, Generic[PS, FD]):
    @property
    @abc.abstractmethod
    def initial_condition(self) -> PS: ...

    @property
    @abc.abstractmethod
    def loader(self) -> DataLoader[FD]: ...


class SimpleInferenceData(InferenceDataABC[PS, FD]):
    def __init__(
        self,
        initial_condition: PS,
        loader: DataLoader[FD],
    ):
        self._initial_condition = initial_condition
        self._loader = loader

    @property
    def initial_condition(self) -> PS:
        return self._initial_condition

    @property
    def loader(self) -> DataLoader[FD]:
        return self._loader

import abc
from typing import Generic, Tuple, TypeVar

from fme.core.generics.data import DataLoader

PS = TypeVar("PS")
BD = TypeVar("BD")


class InferenceDataABC(abc.ABC, Generic[PS, BD]):
    @property
    @abc.abstractmethod
    def initial_condition(self) -> PS:
        ...

    @property
    @abc.abstractmethod
    def loader(self) -> DataLoader[BD]:
        ...


class SimpleInferenceData(InferenceDataABC[PS, BD]):
    def __init__(
        self,
        initial_condition: PS,
        loader: DataLoader[BD],
    ):
        self._initial_condition = initial_condition
        self._loader = loader

    @property
    def initial_condition(self) -> PS:
        return self._initial_condition

    @property
    def loader(self) -> DataLoader[BD]:
        return self._loader


class InferenceStepperABC(abc.ABC, Generic[PS, BD]):
    @abc.abstractmethod
    def predict(
        self,
        initial_condition: PS,
        forcing: BD,
        compute_derived_variables: bool = False,
    ) -> Tuple[BD, PS]:
        ...

    @abc.abstractmethod
    def get_forward_data(
        self,
        forcing: BD,
        compute_derived_variables: bool = False,
    ) -> BD:
        ...

import abc
from typing import Generic, Tuple, TypeVar

PS = TypeVar("PS")
BD = TypeVar("BD")


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

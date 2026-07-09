from collections.abc import Callable, Sequence
from typing import TypeVar, cast

import numpy as np

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.core.coordinates import NullDeriveFn
from fme.core.dataset.dataset import DatasetItem
from fme.core.labels import LabelEncoding
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.data_loading.data_typing import CoupledDatasetItem
from fme.coupled.requirements import CoupledPrognosticStateDataRequirements

SelfType = TypeVar("SelfType", bound="CoupledBatchData")


class CoupledPrognosticState:
    """
    Thin typing wrapper around CoupledBatchData to indicate that the data is
    a prognostic state, such as an initial condition or final state when
    evolving forward in time.
    """

    def __init__(
        self,
        ocean_data: PrognosticState | None = None,
        ice_data: PrognosticState | None = None,
        atmosphere_data: PrognosticState | None = None,
    ):
        self._components: dict[str, PrognosticState] = {
            name: val
            for name, val in [
                ("ocean", ocean_data),
                ("ice", ice_data),
                ("atmosphere", atmosphere_data),
            ]
            if val is not None
        }

    @property
    def ocean_data(self) -> PrognosticState | None:
        return self._components.get("ocean")

    @property
    def ice_data(self) -> PrognosticState | None:
        return self._components.get("ice")

    @property
    def atmosphere_data(self) -> PrognosticState | None:
        return self._components.get("atmosphere")

    def to_device(self) -> "CoupledPrognosticState":
        result = CoupledPrognosticState.__new__(CoupledPrognosticState)
        result._components = {k: v.to_device() for k, v in self._components.items()}
        return result

    def as_batch_data(self) -> "CoupledBatchData":
        result = CoupledBatchData.__new__(CoupledBatchData)
        result._components = {k: v.as_batch_data() for k, v in self._components.items()}
        return result


class CoupledBatchData:
    def __init__(
        self,
        ocean_data: BatchData | None = None,
        ice_data: BatchData | None = None,
        atmosphere_data: BatchData | None = None,
    ):
        self._components: dict[str, BatchData] = {
            name: val
            for name, val in [
                ("ocean", ocean_data),
                ("ice", ice_data),
                ("atmosphere", atmosphere_data),
            ]
            if val is not None
        }

    @property
    def ocean_data(self) -> BatchData | None:
        return self._components.get("ocean")

    @property
    def ice_data(self) -> BatchData | None:
        return self._components.get("ice")

    @property
    def atmosphere_data(self) -> BatchData | None:
        return self._components.get("atmosphere")

    def to_device(self) -> "CoupledBatchData":
        result = CoupledBatchData.__new__(CoupledBatchData)
        result._components = {k: v.to_device() for k, v in self._components.items()}
        return result

    @classmethod
    def new_on_device(
        cls,
        ocean_data: BatchData | None = None,
        ice_data: BatchData | None = None,
        atmosphere_data: BatchData | None = None,
    ) -> "CoupledBatchData":
        src = cls(
            ocean_data=ocean_data,
            ice_data=ice_data,
            atmosphere_data=atmosphere_data,
        )
        return src.to_device()

    @classmethod
    def new_on_cpu(
        cls,
        ocean_data: BatchData | None = None,
        ice_data: BatchData | None = None,
        atmosphere_data: BatchData | None = None,
    ) -> "CoupledBatchData":
        tmp = cls(
            ocean_data=ocean_data,
            ice_data=ice_data,
            atmosphere_data=atmosphere_data,
        )
        result = cls.__new__(cls)
        result._components = {k: v.to_cpu() for k, v in tmp._components.items()}
        return result

    @classmethod
    def collate_fn(
        cls,
        samples: Sequence[CoupledDatasetItem],
        component_horizontal_dims: dict[str, list[str]],
        sample_dim_name: str = "sample",
        component_label_encodings: dict[str, LabelEncoding] | None = None,
    ) -> "CoupledBatchData":
        """
        Collate function for use with PyTorch DataLoader. Assembles per-component
        BatchData from a sequence of CoupledDatasetItems.
        """
        if component_label_encodings is None:
            component_label_encodings = {}
        component_data: dict[str, BatchData] = {
            name: BatchData.from_sample_tuples(
                [cast(DatasetItem, getattr(sample, name)) for sample in samples],
                horizontal_dims=horizontal_dims,
                sample_dim_name=sample_dim_name,
                label_encoding=component_label_encodings.get(name),
            )
            for name, horizontal_dims in component_horizontal_dims.items()
        }
        result = cls.__new__(cls)
        result._components = {k: v.to_cpu() for k, v in component_data.items()}
        return result

    def get_start(
        self: SelfType,
        requirements: CoupledPrognosticStateDataRequirements,
    ) -> CoupledPrognosticState:
        """
        Get the initial condition state.
        """
        ic_components: dict[str, PrognosticState] = {}
        for name, data in self._components.items():
            req = getattr(requirements, name)
            assert req is not None
            ic_components[name] = data.get_start(req.names, req.n_timesteps)
        result = CoupledPrognosticState.__new__(CoupledPrognosticState)
        result._components = ic_components
        return result

    def prepend(self: SelfType, initial_condition: CoupledPrognosticState) -> SelfType:
        result = self.__class__.__new__(self.__class__)
        result._components = {
            name: data.prepend(ic)
            for name, data in self._components.items()
            for ic in [initial_condition._components.get(name)]
            if ic is not None
        }
        return result

    def remove_initial_condition(
        self: SelfType,
        n_ic_timesteps: dict[str, int],
    ) -> SelfType:
        result = self.__class__.__new__(self.__class__)
        result._components = {
            name: data.remove_initial_condition(n_ic_timesteps[name])
            for name, data in self._components.items()
        }
        return result

    def compute_derived_variables(
        self: SelfType,
        forcing_data: SelfType,
        derive_funcs: dict[str, Callable[[TensorMapping, TensorMapping], TensorDict]],
    ) -> SelfType:
        result_components: dict[str, BatchData] = {}
        for name, data in self._components.items():
            fn = derive_funcs.get(name)
            if fn is None or isinstance(fn, NullDeriveFn):
                result_components[name] = data
            else:
                forcing = forcing_data._components.get(name)
                assert forcing is not None
                result_components[name] = data.compute_derived_variables(fn, forcing)
        result = self.__class__.__new__(self.__class__)
        result._components = result_components
        return result

    def pin_memory(self: SelfType) -> SelfType:
        result = self.__class__.__new__(self.__class__)
        result._components = {k: v.pin_memory() for k, v in self._components.items()}
        return result


class CoupledPairedData:
    """
    A container for the data and time coordinates of a batch, with paired
    prediction and target data.
    """

    def __init__(
        self,
        ocean_data: PairedData | None = None,
        ice_data: PairedData | None = None,
        atmosphere_data: PairedData | None = None,
    ):
        self._components: dict[str, PairedData] = {
            name: val
            for name, val in [
                ("ocean", ocean_data),
                ("ice", ice_data),
                ("atmosphere", atmosphere_data),
            ]
            if val is not None
        }

    @property
    def ocean_data(self) -> PairedData | None:
        return self._components.get("ocean")

    @property
    def ice_data(self) -> PairedData | None:
        return self._components.get("ice")

    @property
    def atmosphere_data(self) -> PairedData | None:
        return self._components.get("atmosphere")

    @classmethod
    def from_coupled_batch_data(
        cls,
        prediction: CoupledBatchData,
        reference: CoupledBatchData,
    ) -> "CoupledPairedData":
        paired: dict[str, PairedData] = {}
        for name, pred_data in prediction._components.items():
            ref_data = reference._components.get(name)
            assert ref_data is not None
            if not np.all(pred_data.time.values == ref_data.time.values):
                raise ValueError(
                    f"Prediction and target {name} time coordinate must be the same."
                )
            paired[name] = PairedData(
                prediction=pred_data.data,
                reference=ref_data.data,
                time=pred_data.time,
                labels=pred_data.labels,
            )
        result = cls.__new__(cls)
        result._components = paired
        return result

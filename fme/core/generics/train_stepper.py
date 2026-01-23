import abc
from typing import Generic, TypeVar

from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.stepper import BD, FD, PS, SD, StepperABC
from fme.core.training_history import TrainingJob
from fme.core.typing_ import TensorDict

TO = TypeVar("TO", bound="TrainOutputABC")  # train output


class TrainOutputABC(abc.ABC):
    @abc.abstractmethod
    def get_metrics(self) -> TensorDict:
        pass


class TrainStepperABC(StepperABC[PS, BD, FD, SD], Generic[PS, BD, FD, SD, TO]):
    """
    Abstract base class for steppers that support training.

    Inherits inference functionality from StepperABC and adds training-specific
    methods like train_on_batch and update_training_history.

    Type Parameters:
        PS: Prognostic state type
        BD: Batch data type (output of predict)
        FD: Forcing data type (input to predict)
        SD: Stepped data type (output of predict_paired, typically paired data)
        TO: Train output type (output of train_on_batch)
    """

    @abc.abstractmethod
    def train_on_batch(
        self,
        data: BD,
        optimization: OptimizationABC,
        compute_derived_variables: bool = False,
    ) -> TO:
        """
        Train the model on a batch of data.

        Args:
            data: The batch data to train on.
            optimization: The optimization class to use for updating weights.
            compute_derived_variables: Whether to compute derived variables.

        Returns:
            The training output containing metrics and generated data.
        """
        pass

    @abc.abstractmethod
    def update_training_history(self, training_job: TrainingJob) -> None:
        """
        Update the stepper's history of training jobs.

        Args:
            training_job: The training job to add to the history.
        """
        pass

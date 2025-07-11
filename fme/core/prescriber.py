import dataclasses

from fme.core.masking import replace_on_mask
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class PrescriberConfig:
    """
    Configuration for overwriting predictions of 'prescribed_name' by target values.

    If interpolate is False, the data is overwritten in the region where
    'mask_name' == 'mask_value' after values are rounded to integer. If interpolate
    is True, the data is interpolated between the predicted value at 0 and the
    target value at 1 based on the mask variable, and it is assumed the mask variable
    lies in the range from 0 to 1.

    Parameters:
        prescribed_name: Name of the variable to be overwritten.
        mask_name: Name of the mask variable.
        mask_value: Value of the mask variable in the region to be overwritten.
        interpolate: Whether to interpolate linearly between the generated and target
            values in the masked region, where 0 means keep the generated values and
            1 means replace completely with the target values. Requires mask_value
            be set to 1.
    """

    prescribed_name: str
    mask_name: str
    mask_value: int
    interpolate: bool = False

    def __post_init__(self):
        if self.interpolate and self.mask_value != 1:
            raise ValueError(
                "Interpolation requires mask_value to be 1, but it is set to "
                f"{self.mask_value}."
            )

    def build(self, in_names: list[str], out_names: list[str]):
        if not (self.prescribed_name in in_names and self.prescribed_name in out_names):
            raise ValueError(
                "Variables which are being prescribed in masked regions must be in"
                f" in_names and out_names, but {self.prescribed_name} is not."
            )
        return Prescriber(
            prescribed_name=self.prescribed_name,
            mask_name=self.mask_name,
            mask_value=self.mask_value,
            interpolate=self.interpolate,
        )


class Prescriber:
    """
    Class to overwrite 'prescribed_name' by target values in masked regions.
    """

    def __init__(
        self,
        prescribed_name: str,
        mask_name: str,
        mask_value: int,
        interpolate: bool = False,
    ):
        self.prescribed_name = prescribed_name
        self.mask_name = mask_name
        self.mask_value = mask_value
        self.interpolate = interpolate

    def __call__(
        self,
        mask_data: TensorMapping,
        gen: TensorMapping,
        target: TensorMapping,
    ) -> TensorDict:
        """
        Args:
            mask_data: Dictionary of data containing the mask variable.
            gen: Dictionary of data to use outside of mask region.
            target: Dictionary of data to use in mask region.

        Returns:
            Dictionary of data with the prescribed variable overwritten in the mask
            region and other variables unmodified from gen.
        """
        for name, named_tensors in [("gen", gen), ("target", target)]:
            if self.prescribed_name not in named_tensors:
                raise ValueError(
                    f'Prescribed variable "{self.prescribed_name}" '
                    f'is missing from "{name}"'
                )

        if self.interpolate:
            mask = mask_data[self.mask_name]
            # 1 keeps the target values, 0 replaces with the gen values
            output = (
                mask * target[self.prescribed_name]
                + (1 - mask) * gen[self.prescribed_name]
            )
        else:
            # overwrite specified target variable in given mask region
            output = replace_on_mask(
                original=gen[self.prescribed_name],
                replacement=target[self.prescribed_name],
                mask=mask_data[self.mask_name],
                mask_value=self.mask_value,
            )
        return {**gen, self.prescribed_name: output}

    @property
    def prescribed_names(self) -> list[str]:
        return [self.prescribed_name]

    @property
    def mask_names(self) -> list[str]:
        return [self.mask_name]

# flake8: noqa
# Copied from https://github.com/NVIDIA/modulus/commit/89a6091bd21edce7be4e0539cbd91507004faf08
# Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

import pandas as pd
import torch as th
import torch.nn as nn

from .healpix_decoder import UNetDecoderConfig
from .healpix_encoder import UNetEncoderConfig
from .healpix_layers import HEALPixFoldFaces, HEALPixUnfoldFaces


class HEALPixRecUNet(nn.Module):
    """Deep Learning Weather Prediction (DLWP) recurrent UNet model on the HEALPix mesh."""

    def __init__(
        self,
        encoder: UNetEncoderConfig,
        decoder: UNetDecoderConfig,
        input_channels: int,
        output_channels: int,
        prognostic_variables: int,
        n_constants: int,
        decoder_input_channels: int,
        input_time_size: int,
        output_time_size: int,
        delta_time: str = "6h",
        reset_cycle: str = "24h",
        presteps: int = 1,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        couplings: list = [],
    ):
        """
        Initialize the HEALPixRecUNet model.

        Args:
            encoder: UNetEncoderConfig
                ModuleConfig of instantiable parameters for the U-net encoder.
            decoder: UNetDecoderConfig
                ModuleConfig of instantiable parameters for the U-net decoder.
            input_channels: int
                Number of input channels expected in the input array schema. Note this should be the
                number of input variables in the data, NOT including data reshaping for the encoder part.
            output_channels: int
                Number of output channels expected in the output array schema, or output variables.
            n_constants: int
                Number of optional constants expected in the input arrays. If this is zero, no constants
                should be provided as inputs to forward.
            decoder_input_channels: int
                Number of optional prescribed variables expected in the decoder input array
                for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to forward.
            input_time_size: int
                Number of time steps in the input array.
            output_time_size: int
                Number of time steps in the output array.
            delta_time: str, optional
                Hours between two consecutive data points.
            reset_cycle: str, optional
                Hours after which the recurrent states are reset to zero and re-initialized. Set np.infty
                to never reset the hidden states.
            presteps: int, optional
                Number of model steps to initialize recurrent states.
            enable_nhwc: bool, optional
                Model with [N, H, W, C] instead of [N, C, H, W].
            enable_healpixpad: bool, optional
                Enable CUDA HEALPixPadding if installed.
            couplings: list, optional
                Sequence of dictionaries that describe coupling mechanisms. Currently unused in our production model;
                but we want to keep this in the module definition, in case we bring our SST module
                (which subclasses it) into the picture.
        """
        super().__init__()
        self.channel_dim = 2  # Now 2 with [B, F, T*C, H, W]. Was 1 in old data format with [B, T*C, F, H, W]

        self.input_channels = input_channels

        if n_constants == 0 and decoder_input_channels == 0:
            pass
            # raise NotImplementedError(
            #    "support for models with no constant fields and no decoder inputs (TOA insolation) is not available at this time."
            # )
        if couplings is not None:
            if len(couplings) > 0:
                if n_constants == 0:
                    raise NotImplementedError(
                        "support for coupled models with no constant fields is not available at this time."
                    )
                if decoder_input_channels == 0:
                    raise NotImplementedError(
                        "support for coupled models with no decoder inputs (TOA insolation) is not available at this time."
                    )
        else:
            couplings = []

        # add coupled fields to input channels for model initialization
        self.coupled_channels = self._compute_coupled_channels(couplings)
        self.couplings = couplings
        self.train_couplers = None
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.prognostic_variables = prognostic_variables
        self.decoder_input_channels = decoder_input_channels
        self.input_time_size = input_time_size
        self.output_time_size = output_time_size
        self.delta_t = int(pd.Timedelta(delta_time).total_seconds() // 3600)
        self.reset_cycle = int(pd.Timedelta(reset_cycle).total_seconds() // 3600)
        self.presteps = presteps
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_size == 1 and self.input_time_size > 1
        if not self.is_diagnostic and (
            self.output_time_size % self.input_time_size != 0
        ):
            raise ValueError(
                f"'output_time_size' must be a multiple of 'input_time_size' (got "
                f"{self.output_time_size} and {self.input_time_size})"
            )

        # Build the model layers
        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)
        encoder.input_channels = self._compute_input_channels()
        encoder.enable_nhwc = self.enable_nhwc
        encoder.enable_healpixpad = self.enable_healpixpad
        self.encoder = encoder.build()

        self.encoder_depth = len(self.encoder.n_channels)
        decoder.output_channels = self._compute_output_channels()
        decoder.enable_nhwc = self.enable_nhwc
        decoder.enable_healpixpad = self.enable_healpixpad
        self.decoder = decoder.build()

    @property
    def integration_steps(self):
        """Number of integration steps"""
        return max(self.output_time_size // self.input_time_size, 1)

    def _compute_input_channels(self) -> int:
        """
        Calculate total number of input channels in the model.

        Returns:
            int: The total number of input channels.
        """
        return (
            self.input_time_size * (self.input_channels + self.decoder_input_channels)
            + self.n_constants
            + self.coupled_channels
        )

    def _compute_coupled_channels(self, couplings):
        """
        Get the number of coupled channels.

        Args:
            couplings: list
                Sequence of dictionaries that describe coupling mechanisms.

        Returns:
            int: The number of coupled channels.
        """
        c_channels = 0
        for c in couplings:
            c_channels += len(c["params"]["variables"]) * len(
                c["params"]["input_times"]
            )
        return c_channels

    def _compute_output_channels(self) -> int:
        """
        Compute the total number of output channels in the model.

        Returns:
            int: The total number of output channels.
        """
        return (
            1 if self.is_diagnostic else self.input_time_size
        ) * self.output_channels

    # def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> th.Tensor:
    #     """
    #     Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
    #     concatenates in constants and decoder inputs.

    #     Args:
    #         inputs: list of expected input tensors (inputs, decoder_inputs, constants)
    #         step: step number in the sequence of integration_steps

    #     Returns:
    #         reshaped Tensor in expected shape for model encoder
    #     """

    #     # if len(self.couplings) > 0:
    #     #     result = [
    #     #         inputs[0].flatten(
    #     #             start_dim=self.channel_dim, end_dim=self.channel_dim + 1
    #     #         ),
    #     #         inputs[1][
    #     #             :,
    #     #             :,
    #     #             slice(step * self.input_time_size, (step + 1) * self.input_time_size),
    #     #             ...,
    #     #         ].flatten(
    #     #             start_dim=self.channel_dim, end_dim=self.channel_dim + 1
    #     #         ),  # DI
    #     #         inputs[2].expand(
    #     #             *tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])
    #     #         ),  # constants
    #     #         inputs[3].permute(0, 2, 1, 3, 4),  # coupled inputs
    #     #     ]
    #     #     res = th.cat(result, dim=self.channel_dim)

    #     # else:
    #     #     if self.n_constants == 0:
    #     #         result = [  # This logic changes for no insolation layer for the time being
    #     #             inputs[0].flatten(
    #     #                 start_dim=self.channel_dim, end_dim=self.channel_dim + 1
    #     #             ),
    #     #             # inputs[0].flatten(
    #     #             #     start_dim=self.channel_dim, end_dim=self.channel_dim + 1
    #     #             # ),
    #     #             # inputs[1][
    #     #             #     :,
    #     #             #     :,
    #     #             #     slice(
    #     #             #         step * self.input_time_size, (step + 1) * self.input_time_size
    #     #             #     ),
    #     #             #     ...,
    #     #             # ].flatten(
    #     #             #     start_dim=self.channel_dim, end_dim=self.channel_dim + 1
    #     #             # ),  # DI
    #     #         ]
    #     #         res = th.cat(result, dim=self.channel_dim)

    #     #         # fold faces into batch dim
    #     #         res = self.fold(res)

    #     #         return res

    #     #     if self.decoder_input_channels == 0:
    #     #         result = [
    #     #             inputs[0].flatten(
    #     #                 start_dim=self.channel_dim, end_dim=self.channel_dim + 1
    #     #             ),
    #     #             inputs[1].expand(
    #     #                 *tuple([inputs[0].shape[0]] + len(inputs[1].shape) * [-1])
    #     #             ),  # inputs
    #     #         ]
    #     #         print(f"result 1 is {result[0].shape}")
    #     #         print(f"result 2 is {result[1].shape}")

    #     #         res = th.cat(result, dim=self.channel_dim)

    #     #         # fold faces into batch dim
    #     #         res = self.fold(res)

    #     #         return res

    #     #     result = [
    #     #         inputs[0].flatten(
    #     #             start_dim=self.channel_dim, end_dim=self.channel_dim + 1
    #     #         ),
    #     #         inputs[1][
    #     #             :,
    #     #             :,
    #     #             slice(step * self.input_time_size, (step + 1) * self.input_time_size),
    #     #             ...,
    #     #         ].flatten(
    #     #             start_dim=self.channel_dim, end_dim=self.channel_dim + 1
    #     #         ),  # DI
    #     #         inputs[2].expand(
    #     #             *tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])
    #     #         ),  # constants
    #     #     ]
    #     #     res = th.cat(result, dim=self.channel_dim)

    #     # fold faces into batch dim
    #     # res = self.fold(res)
    #     # res = self.fold(inputs)
    #     # return res

    # def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:
    #     """Returns a multiple tensors to from the model decoder.
    #     Splits the time/channel dimensions.

    #     Args:
    #         inputs: list of expected input tensors (inputs, decoder_inputs, constants)
    #         step: step number in the sequence of integration_steps

    #     Returns:
    #         reshaped Tensor in expected shape for model outputs
    #     """
    #     # unfold:
    #     outputs = self.unfold(outputs)

    #     return outputs

    def _initialize_hidden(
        self, inputs: Sequence, outputs: Sequence, step: int
    ) -> None:
        """Initialize the hidden layers

        Args:
            inputs: Inputs to use to initialize the hideen layers
            outputs: Outputs to use to initialize the hideen layers
            step: Current step number of the initialization
        """
        self.reset()
        for prestep in range(self.presteps):
            if step < self.presteps:
                s = step + prestep
                if len(self.couplings) > 0:
                    input_tensor = self.fold(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_size : (s + 1)
                                * self.input_time_size,
                            ]
                        ]
                        + list(inputs[1:3])
                        + [inputs[3][prestep]],
                        step=step + prestep,
                    )
                else:
                    input_tensor = self.fold(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_size : (s + 1)
                                * self.input_time_size,
                            ]
                        ]
                        + list(inputs[1:]),
                        step=step + prestep,
                    )
            else:
                s = step - self.presteps + prestep
                if len(self.couplings) > 0:
                    input_tensor = self.fold(
                        inputs=[outputs[s - 1]]
                        + list(inputs[1:3])
                        + [inputs[3][step - (prestep - self.presteps)]],
                        step=s + 1,
                    )
                else:
                    input_tensor = self.fold(
                        inputs=[outputs[s - 1]] + list(inputs[1:]), step=s + 1
                    )
            self.decoder(self.encoder(input_tensor))

    def forward(self, inputs: th.Tensor, output_only_last=False) -> th.Tensor:
        """
        Forward pass of the HEALPixUnet

        Args:
            inputs: Inputs to the model, which is currently in the form [B, F, C, H, W].
                (We assume that constants have been preprocessed to persist across batch)
                (We also assume for now that the time is implied to be 1)

            Originally, this was expected to be a sequence of the form [prognostics|TISR|constants]:
                [B, F, T, C, H, W] the format for prognostics and TISR
                [F, C, H, W] the format for constants

            output_only_last: If only the last dimension of the outputs should
                be returned

        Returns:
            Predicted outputs
        """
        self.reset()  # will reset every step for now
        # We need to make sure that the new input is the correct size, since we no longer have the ability
        # to differentiate between the inputs, decoder inputs, and constants
        if self._compute_input_channels() != inputs.shape[2]:
            raise ValueError(
                f"Expected input should have channels {self._compute_input_channels()},"
                f" got {inputs.shape[2]}."
            )

        # The input logic gets really changed now that we just have a prognostic vars channel in the inputs.
        # Basically we assume that all the batch-wise expansion for the inputs has already been done.

        # (Re-)initialize recurrent hidden states
        # if (step * (self.delta_t * self.input_time_size)) % self.reset_cycle == 0:
        #     self._initialize_hidden(inputs=inputs, outputs=outputs, step=step)
        # Skipping this for now. We assume a single input time step, and resetting every step.
        # s = self.presteps
        input_tensor = self.fold(
            inputs
        )  # Padding happens in HEALPixPadding, which will
        # unfold this tensor to handle it.

        encodings = self.encoder(input_tensor)
        decodings = self.decoder(encodings)

        # Residual prediction
        n_prognostic_channels = self.prognostic_variables * self.input_time_size
        prognostic_outputs = (
            input_tensor[:, :n_prognostic_channels]
            + decodings[:, :n_prognostic_channels]
        )

        outputs_only = decodings[:, n_prognostic_channels:]

        reshaped = th.cat(
            [self.unfold(prognostic_outputs), self.unfold(outputs_only)],
            dim=self.channel_dim,
        )

        return reshaped

    def reset(self):
        """Resets the state of the network"""
        self.encoder.reset()
        self.decoder.reset()

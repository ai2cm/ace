import copy
from functools import partial

import torch
import torch.nn as nn


class Preprocessor2D(nn.Module):
    def __init__(
        self, n_history, transform_to_nhwc, add_grid, img_shape_x, img_shape_y
    ):
        super(Preprocessor2D, self).__init__()

        self.n_history = n_history
        self.transform_to_nhwc = transform_to_nhwc
        self.add_grid = add_grid

        if self.add_grid:
            tx = torch.linspace(0, 1, img_shape_x + 1, dtype=torch.float32)[0:-1]
            ty = torch.linspace(0, 1, img_shape_y + 1, dtype=torch.float32)[0:-1]

            x_grid, y_grid = torch.meshgrid(tx, ty, indexing="ij")
            x_grid, y_grid = x_grid.unsqueeze(0).unsqueeze(0), y_grid.unsqueeze(
                0
            ).unsqueeze(0)
            grid = torch.cat([x_grid, y_grid], dim=1)
            self.register_buffer("grid", grid)

    def _flatten_history(self, x, y):

        # flatten input
        if x.dim() == 5:
            b_, t_, c_, h_, w_ = x.shape
            x = torch.reshape(x, (b_, t_ * c_, h_, w_))

        # flatten target
        if y.dim() == 5:
            b_, t_, c_, h_, w_ = y.shape
            y = torch.reshape(y, (b_, t_ * c_, h_, w_))

        return x, y

    def _add_grid(self, x, y):
        # we need to replicate the grid for each batch:
        grid = torch.tile(self.grid, dims=(x.shape[0], 1, 1, 1))
        x = torch.cat([x, grid], dim=1)
        return x, y

    def _nchw_to_nhwc(self, x, y):
        x = x.to(memory_format=torch.channels_last)
        y = y.to(memory_format=torch.channels_last)

        return x, y

    def append_history(self, x1, x2):
        """x1 is input, x2 is the next step"""

        # without history, just return the second tensor
        # with grid if requested
        if self.n_history == 0:
            return x2

        # if grid is added, strip it off first
        if self.add_grid:
            x1 = x1[:, :-2, :, :]

        # this is more complicated
        if x1.dim() == 4:
            b_, c_, h_, w_ = x1.shape
            x1 = torch.reshape(
                x1, (b_, (self.n_history + 1), c_ // (self.n_history + 1), h_, w_)
            )

        if x2.dim() == 4:
            b_, c_, h_, w_ = x2.shape
            x2 = torch.reshape(x2, (b_, 1, c_, h_, w_))

        # append
        res = torch.cat([x1[:, 1:, :, :, :], x2], dim=1)

        # flatten again
        b_, t_, c_, h_, w_ = res.shape
        res = torch.reshape(res, (b_, t_ * c_, h_, w_))

        return res

    def forward(self, x, y):
        # we always want to flatten the history, even if its a singleton
        x, y = self._flatten_history(x, y)

        if self.add_grid:
            x, y = self._add_grid(x, y)

        if self.transform_to_nhwc:
            x, y = self._nchw_to_nhwc(x, y)

        return x, y


def get_preprocessor(params):
    if not params.is_3d:
        return Preprocessor2D(params)
    else:
        raise NotImplementedError("Error, no preprocessor for 3D data implemented.")

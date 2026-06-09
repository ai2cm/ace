import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def concat_and_reshape(self, x1, x2):
        """
        x1: upper-air variables with level dimensions.
        x2: surface variables.
        """
        x1 = x1.view(
            x1.shape[0],
            x1.shape[1],
            x1.shape[2] * x1.shape[3],
            x1.shape[4],
            x1.shape[5],
        )
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat.permute(0, 2, 1, 3, 4)

    def reshape_only(self, x1):
        """
        As in "concat_and_reshape", but for upper-air variables only.
        """
        x1 = x1.view(
            x1.shape[0],
            x1.shape[1],
            x1.shape[2] * x1.shape[3],
            x1.shape[4],
            x1.shape[5],
        )
        return x1.permute(0, 2, 1, 3, 4)

    def split_and_reshape(self, tensor):
        tensor1 = tensor[:, : int(self.channels * self.levels), :, :, :]
        tensor2 = tensor[:, -int(self.surface_channels) :, :, :, :]
        tensor1 = tensor1.view(
            tensor1.shape[0],
            self.channels,
            self.levels,
            tensor1.shape[2],
            tensor1.shape[3],
            tensor1.shape[4],
        )
        return tensor1, tensor2

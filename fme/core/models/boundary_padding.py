import torch
import torch.nn.functional as F


class TensorPadding:
    def __init__(self, mode="earth", pad_lat=(40, 40), pad_lon=(40, 40), **kwargs):
        """
        Initialize the TensorPadding class with the specified mode and padding sizes.

        Args:
            mode (str): The padding mode, either 'mirror' or 'earth'.
            pad_lat (list[int]): Padding sizes for the North-South (latitude) dimension [top, bottom].
            pad_lon (list[int]): Padding sizes for the West-East (longitude) dimension [left, right].
        """
        self.mode = mode
        self.pad_NS = pad_lat
        self.pad_WE = pad_lon

    def pad(self, x):
        if self.mode == "mirror":
            return self._mirror_padding(x)
        elif self.mode == "earth":
            return self._earth_padding(x)

    def unpad(self, x):
        if self.mode == "mirror":
            return self._mirror_unpad(x)
        elif self.mode == "earth":
            return self._earth_unpad(x)

    def _earth_padding(self, x):
        """Earth-aware padding: poles use a 180° longitude-shifted reflection,
        longitude uses circular wrapping."""
        if any(p > 0 for p in self.pad_NS):
            # Going over a pole on a sphere shifts longitude by 180°.
            shift_size = int(x.shape[-1] // 2)
            xroll = torch.roll(x, shifts=shift_size, dims=-1)
            xroll_flip_top = torch.flip(xroll[..., : self.pad_NS[0], :], (-2,))
            xroll_flip_bot = torch.flip(xroll[..., -self.pad_NS[1] :, :], (-2,))
            x = torch.cat([xroll_flip_top, x, xroll_flip_bot], dim=-2)

        if any(p > 0 for p in self.pad_WE):
            x = F.pad(x, (self.pad_WE[0], self.pad_WE[1], 0, 0), mode="circular")

        return x

    def _earth_unpad(self, x):
        if any(p > 0 for p in self.pad_NS):
            start_NS = self.pad_NS[0]
            end_NS = -self.pad_NS[1] if self.pad_NS[1] > 0 else None
            x = x[..., start_NS:end_NS, :]

        if any(p > 0 for p in self.pad_WE):
            start_WE = self.pad_WE[0]
            end_WE = -self.pad_WE[1] if self.pad_WE[1] > 0 else None
            x = x[..., :, start_WE:end_WE]

        return x

    def _mirror_padding(self, x):
        """Mirror padding: longitude circular, latitude reflect."""
        if any(p > 0 for p in self.pad_WE):
            x = F.pad(x, (self.pad_WE[0], self.pad_WE[1], 0, 0), mode="circular")

        if any(p > 0 for p in self.pad_NS):
            x = F.pad(x, (0, 0, self.pad_NS[0], self.pad_NS[1]), mode="reflect")

        return x

    def _mirror_unpad(self, x):
        if any(p > 0 for p in self.pad_NS):
            x = x[..., self.pad_NS[0] : -self.pad_NS[1], :]

        if any(p > 0 for p in self.pad_WE):
            x = x[..., :, self.pad_WE[0] : -self.pad_WE[1]]

        return x

"""This code has been vendorized from the PyTorch Image Quality project:

[1]
@misc{kastryulin2022piq,
  title = {PyTorch Image Quality: Metrics for Image Quality Assessment},
  url = {https://arxiv.org/abs/2208.14818},
  author = {Kastryulin, Sergey and Zakirov, Jamil and Prokopenko, Denis and Dylov, Dmitry V.},
  doi = {10.48550/ARXIV.2208.14818},
  publisher = {arXiv},
  year = {2022}
}

[2] https://piq.readthedocs.io/en/latest/index.html
[3] https://github.com/photosynthesis-team/piq
"""

"""
    Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


# fmt: off
# flake8: noqa
# isort: skip_file

r""" This module implements Peak Signal-to-Noise Ratio (PSNR) in PyTorch.
"""
import torch
from typing import List, Optional, Tuple, Union


def _validate_input(
    tensors: List[torch.Tensor],
    dim_range: Tuple[int, int] = (0, -1),
    data_range: Tuple[float, float] = (0.0, -1.0),
    size_range: Optional[Tuple[int, int]] = None,
    check_for_channels_first: bool = False,
) -> None:
    r"""Check that input(-s)  satisfies the requirements
    Args:
        tensors: Tensors to check
        dim_range: Allowed number of dimensions. (min, max)
        data_range: Allowed range of values in tensors. (min, max)
        size_range: Dimensions to include in size comparison. (start_dim, end_dim + 1)
    """

    if not __debug__:
        return

    x = tensors[0]

    for t in tensors:
        assert torch.is_tensor(t), f"Expected torch.Tensor, got {type(t)}"
        assert (
            t.device == x.device
        ), f"Expected tensors to be on {x.device}, got {t.device}"

        if size_range is None:
            assert (
                t.size() == x.size()
            ), f"Expected tensors with same size, got {t.size()} and {x.size()}"
        else:
            assert (
                t.size()[size_range[0] : size_range[1]]
                == x.size()[size_range[0] : size_range[1]]
            ), f"Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}"

        if dim_range[0] == dim_range[1]:
            assert (
                t.dim() == dim_range[0]
            ), f"Expected number of dimensions to be {dim_range[0]}, got {t.dim()}"
        elif dim_range[0] < dim_range[1]:
            assert (
                dim_range[0] <= t.dim() <= dim_range[1]
            ), f"Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}"

        if data_range[0] < data_range[1]:
            assert (
                data_range[0] <= t.min()
            ), f"Expected values to be greater or equal to {data_range[0]}, got {t.min()}"
            assert (
                t.max() <= data_range[1]
            ), f"Expected values to be lower or equal to {data_range[1]}, got {t.max()}"

        if check_for_channels_first:
            channels_last = t.shape[-1] in {1, 2, 3}
            assert (
                not channels_last
            ), "Expected tensor to have channels first format, but got channels last. \
                Please permute channels (e.g. t.permute(0, 3, 1, 2) for 4D tensors) and rerun."


def _reduce(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""Reduce input in batch dimension if needed.

    Args:
        x: Tensor with shape (N, *).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    """
    if reduction == "none":
        return x
    elif reduction == "mean":
        return x.mean(dim=0)
    elif reduction == "sum":
        return x.sum(dim=0)
    else:
        raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'sum'}")


def psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: Union[int, float] = 1.0,
    reduction: str = "mean",
    convert_to_greyscale: bool = False,
) -> torch.Tensor:
    r"""Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        convert_to_greyscale: Convert RGB image to YIQ format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.

    Returns:
        PSNR Index of similarity between two images.

    References:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))

    # Constant for numerical stability
    EPS = 1e-8

    x = x / float(data_range)
    y = y / float(data_range)

    if (x.size(1) == 3) and convert_to_greyscale:
        # Convert RGB image to YIQ and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
        rgb_to_grey = torch.tensor(
            [0.299, 0.587, 0.114], device=x.device, dtype=x.dtype
        ).view(1, -1, 1, 1)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
        y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score: torch.Tensor = -10 * torch.log10(mse + EPS)

    return _reduce(score, reduction)


r""" This module implements Structural Similarity (SSIM) index in PyTorch.

Implementation of classes and functions from this module are inspired by Gongfan Fang's (@VainF) implementation:
https://github.com/VainF/pytorch-msssim

and implementation of one of pull requests to the PyTorch by Kangfu Mei (@MKFMIKU):
https://github.com/pytorch/pytorch/pull/22289/files
"""

from typing import Tuple, Union, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def gaussian_filter(kernel_size: int, sigma: float, device: Optional[str] = None,
                    dtype: Optional[type] = None) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=dtype, device=device)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)



def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int = 11,
    kernel_sigma: float = 1.5,
    data_range: Union[int, float] = 1.0,
    reduction: str = "mean",
    full: bool = False,
    downsample: bool = True,
    k1: float = 0.01,
    k2: float = 0.03,
) -> List[torch.Tensor]:
    r"""Interface of Structural Similarity (SSIM) index.
    Inputs supposed to be in range ``[0, data_range]``.
    To match performance with skimage and tensorflow set ``'downsample' = True``.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        full: Return cs map or not.
        downsample: Perform average pool before SSIM computation. Default: True
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
        as a tensor of size 2.

    References:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI: `10.1109/TIP.2003.819861`
    """
    assert kernel_size % 2 == 1, f"Kernel size must be odd, got [{kernel_size}]"
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))

    x = x / float(data_range)
    y = y / float(data_range)

    # Averagepool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if (f > 1) and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    kernel = gaussian_filter(
        kernel_size, kernel_sigma, device=x.device, dtype=x.dtype
    ).repeat(x.size(1), 1, 1, 1)
    _compute_ssim_per_channel = (
        _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    )
    ssim_map, cs_map = _compute_ssim_per_channel(
        x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2
    )
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    ssim_val = _reduce(ssim_val, reduction)
    cs = _reduce(cs, reduction)

    return ssim_val


class SSIMLoss(_Loss):
    r"""Creates a criterion that measures the structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.

    To match performance with skimage and tensorflow set ``'downsample' = True``.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        SSIM = \{ssim_1,\dots,ssim_{N \times C}\}\\
        ssim_{l}(x, y) = \frac{(2 \mu_x \mu_y + c_1) (2 \sigma_{xy} + c_2)}
        {(\mu_x^2 +\mu_y^2 + c_1)(\sigma_x^2 +\sigma_y^2 + c_2)},

    where :math:`N` is the batch size, `C` is the channel size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        SSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(1 - SSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(1 - SSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    In case of 5D input tensors, complex value is returned as a tensor of size 2.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        downsample: Perform average pool before SSIM computation. Default: True
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).

    Examples:
        >>> loss = SSIMLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI:`10.1109/TIP.2003.819861`
    """
    __constants__ = ["kernel_size", "k1", "k2", "sigma", "kernel", "reduction"]

    def __init__(
        self,
        kernel_size: int = 11,
        kernel_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        downsample: bool = True,
        reduction: str = "mean",
        data_range: Union[int, float] = 1.0,
    ) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.kernel_size = kernel_size

        # This check might look redundant because kernel size is checked within the ssim function anyway.
        # However, this check allows to fail fast when the loss is being initialised and training has not been started.
        assert kernel_size % 2 == 1, f"Kernel size must be odd, got [{kernel_size}]"
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.downsample = downsample
        self.data_range = data_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computation of Structural Similarity (SSIM) index as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
            y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.

        Returns:
            Value of SSIM loss to be minimized, i.e ``1 - ssim`` in [0, 1] range. In case of 5D input tensors,
            complex value is returned as a tensor of size 2.
        """

        score = ssim(
            x=x,
            y=y,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            downsample=self.downsample,
            data_range=self.data_range,
            reduction=self.reduction,
            full=False,
            k1=self.k1,
            k2=self.k2,
        )
        return torch.ones_like(score) - score


def _ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    data_range: Union[float, int] = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        kernel: 2D Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(
            f"Kernel size can't be greater than actual input size. "
            f"Input size: {x.size()}. Kernel size: {kernel.size()}"
        )

    c1 = k1**2
    c2 = k2**2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx = mu_x**2
    mu_yy = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_xx = (
        F.conv2d(x**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    )
    sigma_yy = (
        F.conv2d(y**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    )
    sigma_xy = (
        F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy
    )

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    ssim_val = ss.mean(dim=(-1, -2))
    cs = cs.mean(dim=(-1, -2))
    return ssim_val, cs


def _ssim_per_channel_complex(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    data_range: Union[float, int] = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W, 2)`.
        kernel: 2-D gauss kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Complex Structural Similarity (SSIM) index.
    """
    n_channels = x.size(1)
    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
        raise ValueError(
            f"Kernel size can't be greater than actual input size. Input size: {x.size()}. "
            f"Kernel size: {kernel.size()}"
        )

    c1 = k1**2
    c2 = k2**2

    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]

    mu1_real = F.conv2d(x_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_imag = F.conv2d(x_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_real = F.conv2d(y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_imag = F.conv2d(y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

    compensation = 1.0

    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real

    sigma1_sq = (
        F.conv2d(x_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(y_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq
    )
    sigma12_real = (
        F.conv2d(x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
        - mu1_mu2_real
    )
    sigma12_imag = (
        F.conv2d(x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
        - mu1_mu2_imag
    )
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    # Set alpha = beta = gamma = 1.
    cs_map = (sigma12 * 2 + c2 * compensation) / (
        sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2 * compensation
    )
    ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (
        mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation
    )
    ssim_map = ssim_map * cs_map

    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))

    return ssim_val, cs

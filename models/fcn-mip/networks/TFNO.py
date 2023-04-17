import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch

from .layers import MLPv2 as MLP

import math

import numpy as np
import tensorly as tl

tl.set_backend("pytorch")
from tensorly import tenalg
from tensorly.decomposition import tucker

from tltorch.factorized_tensors.core import FactorizedTensor
from tltorch.utils import FactorList


@torch.jit.script
def compl_mul_fwd_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b.contiguous())
    res = torch.einsum("bixy,ioxy->boxy", ac, bc)
    return torch.view_as_real(res)


@torch.jit.script
def compl_mul_fwd_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ioxyr->srboxy", a, b)
    res = torch.stack(
        [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
    )
    return res


@torch.jit.script
def compl_tuckerc_c_fwd(
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    a0 = torch.view_as_complex(a0)
    a1 = torch.view_as_complex(a1)
    a2 = torch.view_as_complex(a2)
    a3 = torch.view_as_complex(a3)
    c = torch.view_as_complex(c)
    x = torch.view_as_complex(x)

    res = torch.tensordot(a0, x, dims=[[0], [1]], out=None)
    res2 = torch.einsum("gc,abcd->gabd", a2, c)
    res2 = torch.einsum("gabd,hd->gabh", res2, a3)
    res = torch.einsum("gabh,aigh->gbhi", res2, res)
    res = torch.einsum("gbhi,fb->ifgh", res, a1)

    return torch.view_as_real(res)


def contract_tucker(x, tucker):
    a0, a1, a2, a3 = tucker.factors
    c = tucker.core
    return compl_tuckerc_c_fwd(a0, a1, a2, a3, c, x)


@torch.jit.script
def compl_ttc1_c_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    res = torch.einsum("jxk,ky,bcxy->jbcxy", ac, bc, cc)
    return torch.view_as_real(res)


@torch.jit.script
def compl_ttc2_c_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    res = torch.einsum("oi,icj,jbcxy->boxy", ac, bc, cc)
    return torch.view_as_real(res)


def contract_tt(x, tt):
    a, b, c, d = tt.factors
    o1 = compl_ttc1_c_fwd(c, d[..., 0, :], x)
    return compl_ttc2_c_fwd(a[0], b, o1)


class FactorizedSpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes_height,
        modes_embed_dim,
        n_layers=1,
        scale="auto",
        fft_contraction="real",
        fft_norm="backward",
        bias=True,
        mlp_config=None,
        rank=0.5,
        factorization="cp",
        fixed_rank_modes=False,
        decomposition_kwargs=dict(),
    ):
        """
        Parameters
        ----------
        fft_contraction : {'real', 'complex'}
            if real, we use that (a,b)*(c,d) = (ac-bd, ad+cb)
            if complex, the weights are first cast to complex before contraction and the result cast back to float
        fft_norm : {'backward', 'forward', 'ortho'}, default is 'backward'
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_embed_dim
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                fixed_rank_modes = [5]
            else:
                fixed_rank_modes = None

        if mlp_config is not None:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=out_channels,
                        hidden_features=int(
                            round(out_channels * mlp_config["expansion"])
                        ),
                        drop=mlp_config["dropout"],
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.mlp = None

        self.fft_contraction = fft_contraction
        self.fft_norm = fft_norm
        if fft_contraction == "complex" or fft_contraction == "real_complex":
            self._contract = compl_mul_fwd_complex
        elif fft_contraction == "real":
            self._contract = compl_mul_fwd_real
        else:
            raise ValueError()

        if factorization is None:
            self.weight = nn.Parameter(
                scale
                * torch.randn(
                    2 * n_layers,
                    in_channels,
                    out_channels,
                    self.modes_height,
                    self.modes_width,
                    2,
                    dtype=torch.float32,
                )
            )
            self._get_weight = self._get_weight_dense
        elif factorization.lower() == "tucker":
            self.weight = ComplexTuckerTensor.new(
                (
                    2 * n_layers,
                    in_channels,
                    out_channels,
                    self.modes_height,
                    self.modes_width,
                ),
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                **decomposition_kwargs,
            )
            self.weight.normal_(0, scale)
            # self._get_weight = self._get_weight_factorized_tt
            self._get_weight = self._get_weight_factorized
            self._contract = contract_tucker

        elif factorization.lower() == "tucker_rec":
            self.weight = ComplexTuckerTensor.new(
                (
                    2 * n_layers,
                    in_channels,
                    out_channels,
                    self.modes_height,
                    self.modes_width,
                ),
                rank=self.rank,
                factorization="tucker",
                fixed_rank_modes=fixed_rank_modes,
                **decomposition_kwargs,
            )
            self.weight.normal_(0, scale)
            self._get_weight = self._get_weight_reconstructed

        elif factorization.lower() == "tt":
            self.weight = ComplexTTTensor.new(
                (
                    2 * n_layers,
                    in_channels,
                    out_channels,
                    self.modes_height,
                    self.modes_width,
                ),
                rank=self.rank,
                factorization=factorization,
            )
            self.weight.normal_(0, scale)

            self._get_weight = self._get_weight_factorized
            self._contract = contract_tt

        else:
            raise ValueError(
                f"Warning - only Tucker and TT supported currently, got {factorization}"
            )

        if bias:
            self.bias = nn.Parameter(
                scale * torch.randn(n_layers, self.out_channels, 1, 1)
            )
        else:
            self.bias = 0

    def _get_weight_factorized(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return self.weight()[2 * layer_index + corner_index]

    def _get_weight_reconstructed(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return self.weight()[2 * layer_index + corner_index].to_tensor().contiguous()

    def _get_weight_dense(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return self.weight[2 * layer_index + corner_index, :, :, :, :, :]

    def forward(self, x, indices=0):
        with torch.autocast(device_type="cuda", enabled=False):

            batchsize, channels, height, width = x.shape

            # Compute Fourier coeffcients
            x = torch.fft.rfft2(x.float(), norm=self.fft_norm)

            # Multiply relevant Fourier modes
            x = torch.view_as_real(x)
            # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
            # out_fft = torch.zeros(x.shape, device=x.device)
            out_fft = torch.zeros(
                [batchsize, self.out_channels, height, width // 2 + 1, 2],
                device=x.device,
            )

            # upper block (truncate high freq)
            out_fft[:, :, : self.modes_height, : self.modes_width] = self._contract(
                x[:, :, : self.modes_height, : self.modes_width],
                self._get_weight(indices, 0),
            )
            # Lower block
            out_fft[:, :, -self.modes_height :, : self.modes_width] = self._contract(
                x[:, :, -self.modes_height :, : self.modes_width],
                self._get_weight(indices, 1),
            )

            out_fft = torch.view_as_complex(out_fft)
            x = torch.fft.irfft2(
                out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm
            )  # (x.size(-2), x.size(-1)))

            x = x + self.bias[indices, ...]

            if self.mlp is not None:
                x = self.mlp[indices](x)

            return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution
        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single convolution is parametrized, directly use the main class."
            )

        return SubConv2d(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


class SubConv2d(nn.Module):
    """Class representing one of the convolutions from the mother joint factorized convolution
    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data,
    which is shared.
    """

    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x):
        return self.main_conv.forward(x, self.indices)


class ComplexTTTensor(FactorizedTensor, name="ComplexTT"):
    """Tensor-Train (Matrix-Product-State) Factorization"""

    def __init__(self, factors, shape=None, rank=None):
        super().__init__()
        if shape is None or rank is None:
            self.shape, self.rank = tl.tt_tensor._validate_tt_tensor(
                [torch.view_as_complex(f) for f in factors]
            )
        else:
            self.shape, self.rank = shape, rank

        self.order = len(self.shape)
        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)

        # Avoid the issues with ParameterList
        factors = [
            nn.Parameter(
                torch.empty((rank[i], s, rank[i + 1], 2), device=device, dtype=dtype)
            )
            for i, s in enumerate(shape)
        ]

        return cls(factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        shape = tensor.shape
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)

        with torch.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, rank)

        return cls([nn.Parameter(f.contiguous()) for f in factors])

    def init_from_tensor(self, tensor, **kwargs):
        with torch.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, self.rank)

        self.factors = FactorList([nn.Parameter(f.contiguous()) for f in factors])
        self.rank = tuple([f.shape[0] for f in factors] + [1])
        return self

    @property
    def decomposition(self):
        return self.factors

    @property
    def decomposition_as_complex(self):
        return [torch.view_as_complex(f) for f in self.factors]

    def to_tensor(self):
        return torch.view_as_real(tl.tt_to_tensor(self.decomposition_as_complex))

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

        r = np.prod(self.rank)
        std_factors = (std / r) ** (1 / self.order)
        with torch.no_grad():
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Select one dimension of one mode
            factor, next_factor, *factors = self.factors
            next_factor = tenalg.mode_dot(
                torch.view_as_complex(next_factor),
                torch.view_as_complex(factor[:, indices, :, :].squeeze(1)),
                0,
            )
            return ComplexTTTensor([torch.view_as_real(next_factor), *factors])

        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[:, indices, :, :], *factors]
            return ComplexTTTensor(factors)

        else:
            factors = []
            complex_factors = self.decomposition_as_complex
            all_contracted = True
            for i, index in enumerate(indices):
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}."
                    )
                if isinstance(index, int):
                    if i:
                        factor = tenalg.mode_dot(
                            factor, complex_factors[i][:, index, :].T, -1
                        )
                    else:
                        factor = complex_factors[i][:, index, :]
                else:
                    if i:
                        if all_contracted:
                            factor = tenalg.mode_dot(
                                complex_factors[i][:, index, :], factor, 0
                            )
                        else:
                            factors.append(factor)
                            factor = complex_factors[i][:, index, :]
                    else:
                        factor = complex_factors[i][:, index, :]
                    all_contracted = False

            if factor.ndim == 2:  # We have contracted all cores, so have a 2D matrix
                if self.order == (i + 1):
                    # No factors left
                    return factor.squeeze()
                else:
                    next_factor, *factors = complex_factors[i + 1 :]
                    factor = tenalg.mode_dot(next_factor, factor, 0)
                    return ComplexTTTensor(
                        [torch.view_as_real(factor)]
                        + [torch.view_as_real(f) for f in factors]
                    )
            else:
                return ComplexTTTensor(
                    [torch.view_as_real(f) for f in factors]
                    + [torch.view_as_real(factor)]
                    + self.factors[i + 1 :]
                )


class ComplexTuckerTensor(FactorizedTensor, name="ComplexTucker"):
    """Tucker Factorization
    Parameters
    ----------
    core
    factors
    shape
    rank
    """

    def __init__(self, core, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = tl.tucker_tensor._validate_tucker_tensor(
                (
                    torch.view_as_complex(core),
                    [torch.view_as_complex(f) for f in factors],
                )
            )

        self.order = len(self.shape)
        setattr(self, "core", core)
        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, fixed_rank_modes=None, device=None, dtype=None, **kwargs):
        rank = tl.tucker_tensor.validate_tucker_rank(
            shape, rank, fixed_modes=fixed_rank_modes
        )

        # Register the parameters
        core = nn.Parameter(torch.empty((*rank, 2), device=device, dtype=dtype))
        # Avoid the issues with ParameterList
        factors = [
            nn.Parameter(torch.empty((s, r, 2), device=device, dtype=dtype))
            for (s, r) in zip(shape, rank)
        ]

        return cls(core, factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", fixed_rank_modes=None, **kwargs):
        shape = tensor.shape
        rank = tl.tucker_tensor.validate_tucker_rank(
            shape, rank, fixed_modes=fixed_rank_modes
        )

        with torch.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)

        return cls(
            nn.Parameter(core.contiguous()),
            [nn.Parameter(f.contiguous()) for f in factors],
        )

    def init_from_tensor(
        self, tensor, unsqueezed_modes=None, unsqueezed_init="average", **kwargs
    ):
        """Initialize the tensor factorization from a tensor
        Parameters
        ----------
        tensor : torch.Tensor
            full tensor to decompose
        unsqueezed_modes : int list
            list of modes for which the rank is 1 that don't correspond to a mode in the full tensor
            essentially we are adding a new dimension for which the core has dim 1,
            and that is not initialized through decomposition.
            Instead first `tensor` is decomposed into the other factors.
            The `unsqueezed factors` are then added and  initialized e.g. with 1/dim[i]
        unsqueezed_init : 'average' or float
            if unsqueezed_modes, this is how the added "unsqueezed" factors will be initialized
            if 'average', then unsqueezed_factor[i] will have value 1/tensor.shape[i]
        """
        if unsqueezed_modes is not None:
            unsqueezed_modes = sorted(unsqueezed_modes)
            for mode in unsqueezed_modes[::-1]:
                if self.rank[mode] != 1:
                    msg = "It is only possible to initialize by averagig over mode for which rank=1."
                    msg += f"However, got unsqueezed_modes={unsqueezed_modes} but rank[{mode}]={self.rank[mode]} != 1."
                    raise ValueError(msg)

            rank = tuple(
                r for (i, r) in enumerate(self.rank) if i not in unsqueezed_modes
            )
        else:
            rank = self.rank

        with torch.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)

            if unsqueezed_modes is not None:
                # Initialise with 1/shape[mode] or given value
                for mode in unsqueezed_modes:
                    size = self.shape[mode]
                    factor = torch.ones(size, 1)
                    if unsqueezed_init == "average":
                        factor /= size
                    else:
                        factor *= unsqueezed_init
                    factors.insert(mode, factor)
                    core = core.unsqueeze(mode)

        self.core = nn.Parameter(torch.view_as_real(core).contiguous())
        self.factors = FactorList(
            [nn.Parameter(torch.view_as_real(f).contiguous()) for f in factors]
        )
        return self

    @property
    def decomposition(self):
        return self.core, self.factors

    @property
    def decomposition_as_complex(self):
        return torch.view_as_complex(self.core), [
            torch.view_as_complex(f) for f in self.factors
        ]

    def to_tensor(self):
        return torch.view_as_real(tl.tucker_to_tensor(self.decomposition_as_complex))

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

        r = np.prod([math.sqrt(r) for r in self.rank])
        std_factors = (std / r) ** (1 / (self.order + 1))

        with torch.no_grad():
            self.core.data.normal_(0, std_factors)
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        core, factors = self.decomposition_as_complex
        if isinstance(indices, int):
            # Select one dimension of one mode
            mixing_factor, *factors = factors
            core = tenalg.mode_dot(core, mixing_factor[indices, :], 0)
            return self.__class__(torch.view_as_real(core), self.factors[1:])

        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices, :, :], *factors]
            return self.__class__(self.core, factors)

        else:
            # Index multiple dimensions
            modes = []
            factors = []
            factors_contract = []
            for i, (index, factor) in enumerate(zip(indices, self.factors)):
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}."
                    )
                if isinstance(index, int):
                    modes.append(i)
                    factors_contract.append(torch.view_as_complex(factor)[index, :])
                else:
                    factors.append(torch.view_as_complex(factor)[index, :])

            core = tenalg.multi_mode_dot(
                torch.view_as_complex(self.core), factors_contract, modes=modes
            )
            core = torch.view_as_real(core)
            factors = [torch.view_as_real(f) for f in factors] + self.factors[i + 1 :]

            if factors:
                return self.__class__(core, factors)

            # Fully contracted tensor
            return core


class FactorizedFNO2d(nn.Module):
    def __init__(
        self,
        params,
        modes_height=40,
        modes_embed_dim=24,
        embed_dim=64,
        N_in_channels=3,
        N_out_channels=3,
        fc_channels=256,
        n_layers=4,
        levels=0,
        joint_factorization=False,
        non_linearity=F.gelu,
        rank=1,
        factorization="cp",
        fixed_rank_modes=False,
        domain_padding=9,
        fft_contraction="real",
        fft_norm="backward",
        mlp_config=None,
        verbose=True,
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        super().__init__()
        if params is None:
            self.modes_height = modes_height
            self.modes_embed_dim = modes_embed_dim
            self.embed_dim = embed_dim
            self.fc_channels = fc_channels
            self.n_layers = n_layers
            self.levels = levels
            self.joint_factorization = joint_factorization
            self.non_linearity = non_linearity
            self.rank = rank
            self.factorization = factorization
            self.fixed_rank_modes = fixed_rank_modes
            self.domain_padding = domain_padding
            self.in_channels = N_in_channels
            self.out_channels = N_out_channels
            self.decomposition_kwargs = decomposition_kwargs
            self.fft_norm = fft_norm
            self.fft_contraction = fft_contraction
            self.verbose = verbose
            self.mlp_config = mlp_config
        else:
            self.modes_height = params.modes_height
            self.modes_embed_dim = params.modes_embed_dim
            self.embed_dim = params.embed_dim
            self.fc_channels = (
                params.fc_channels if hasattr(params, "fc_channels") else fc_channels
            )
            self.n_layers = params.n_layers if hasattr(params, "n_layers") else n_layers
            self.levels = params.levels
            self.joint_factorization = (
                params.joint_factorization
                if hasattr(params, "joint_factorization")
                else joint_factorization
            )
            self.non_linearity = (
                params.non_linearity
                if hasattr(params, "non_linearity")
                else non_linearity
            )
            self.rank = params.rank if hasattr(params, "rank") else rank
            self.factorization = (
                params.factorization
                if hasattr(params, "factorization")
                else factorization
            )
            self.fixed_rank_modes = (
                params.fixed_rank_modes
                if hasattr(params, "fixed_rank_modes")
                else fixed_rank_modes
            )
            self.domain_padding = (
                params.domain_padding
                if hasattr(params, "domain_padding")
                else domain_padding
            )
            self.in_channels = (
                params.N_in_channels
                if hasattr(params, "N_in_channels")
                else N_in_channels
            )
            self.out_channels = (
                params.N_out_channels
                if hasattr(params, "N_out_channels")
                else N_out_channels
            )
            self.decomposition_kwargs = (
                params.decomposition_kwargs
                if hasattr(params, "decomposition_kwargs")
                else decomposition_kwargs
            )
            self.fft_norm = params.fft_norm if hasattr(params, "fft_norm") else fft_norm
            self.fft_contraction = (
                params.fft_contraction
                if hasattr(params, "fft_contraction")
                else fft_contraction
            )
            self.verbose = params.verbose if hasattr(params, "verbose") else verbose
            if hasattr(params, "use_mlp") and params.use_mlp:
                self.mlp_config = {}
                self.mlp_config["expansion"] = (
                    params.mlp_expansion if hasattr(params, "mlp_expansion") else 1.0
                )
                self.mlp_config[
                    "dropout"
                ] = False  # params.mlp_dropout if hasattr(params, 'mlp_dropout') else False
            else:
                self.mlp_config = None

        if joint_factorization:
            self.convs = FactorizedSpectralConv2d(
                self.embed_dim,
                self.embed_dim,
                self.modes_height,
                self.modes_embed_dim,
                rank=self.rank,
                fft_contraction=self.fft_contraction,
                fft_norm=self.fft_norm,
                factorization=self.factorization,
                fixed_rank_modes=self.fixed_rank_modes,
                decomposition_kwargs=self.decomposition_kwargs,
                mlp_config=self.mlp_config,
                n_layers=self.n_layers,
            )
        else:
            self.convs = nn.ModuleList(
                [
                    FactorizedSpectralConv2d(
                        self.embed_dim,
                        self.embed_dim,
                        self.modes_height,
                        self.modes_embed_dim,
                        fft_contraction=self.fft_contraction,
                        rank=self.rank,
                        factorization=self.factorization,
                        fixed_rank_modes=self.fixed_rank_modes,
                        decomposition_kwargs=self.decomposition_kwargs,
                        mlp_config=self.mlp_config,
                        n_layers=1,
                    )
                    for _ in range(self.n_layers)
                ]
            )
        self.linears = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, self.embed_dim, 1) for _ in range(self.n_layers)]
        )

        self.fc0 = nn.Conv2d(self.in_channels, self.embed_dim, 1)
        self.fc1 = nn.Linear(self.embed_dim, self.fc_channels)
        self.fc2 = nn.Linear(self.fc_channels, self.out_channels)

    def forward(self, x, super_res=1):
        x = self.fc0(x)
        # x = x.permute(0,3,1,2)

        x = F.pad(x, [0, self.domain_padding, 0, self.domain_padding])

        for i in range(self.n_layers):
            if super_res > 1 and i == (self.n_layers - 1):
                super_res = super_res
            else:
                super_res = 1

            x1 = self.convs[i](x)
            x2 = self.linears[i](x)
            x = x1 + x2
            if i < (self.n_layers - 1):
                x = self.non_linearity(x)

        x = x[..., : -self.domain_padding, : -self.domain_padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

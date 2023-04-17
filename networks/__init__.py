import pathlib
from typing import Optional

from networks.TFNO import FactorizedFNO2d
from networks import preprocessor
from networks import afnonet
from networks import geometric
from networks import vit
from networks import geometric_v1
from networks import graphcast
from networks.YParams import YParams
import torch
import einops
import numpy as np
import contextlib

from fcn_mip import registry
from fcn_mip import filesystem
import modulus
from modulus.internal.utils.graphcast.graph import Graph
from modulus.internal.models.gnn.graphcast.graph_cast_net import GraphCastNet
from modulus.internal.utils.graphcast.data_utils import StaticData
from modulus.distributed.manager import DistributedManager

from fcn_mip import schema


TFNO_JEAN = "tfno_no-patching_lr5e-4_full_epochs"


class Wrapper(torch.nn.Module):
    """Makes sure the parameter names are the same as the checkpoint"""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        """x: (batch, history, channel, x, y)"""

        return self.module(x)


class Inference(torch.nn.Module):
    n_history = 0

    def __init__(
        self,
        model,
        channels,
        center_path,
        scale_path,
        grid,
        channel_set: schema.ChannelSet,
    ):
        super().__init__()
        self.channel_set = channel_set
        if isinstance(model, modulus.Module) or isinstance(
            model, graphcast.GraphCastWrapper
        ):
            self.model = model
        else:
            self.model = Wrapper(model)
        self.channels = channels
        self.channel_set = channel_set
        self.grid = grid
        self.graph = None  # Cuda graph
        self.iteration = 0
        self.out = None

        center = np.load(center_path)
        scale = np.load(scale_path)

        center = torch.from_numpy(np.squeeze(center)).float()
        scale = torch.from_numpy(np.squeeze(scale)).float()

        self.register_buffer("scale_org", scale)
        self.register_buffer("center_org", center)

        self.register_buffer("scale", scale[self.channels, None, None])
        self.register_buffer("center", center[self.channels, None, None])

    def normalize(self, x):
        return (x - self.center_org[None, :, None, None]) / self.scale_org[
            None, :, None, None
        ]

    def load_checkpoint(self, path):
        if isinstance(self.model, modulus.Module):
            # Use native model load utility if modulus.Module
            self.model.load(path)
        elif isinstance(self.model, graphcast.GraphCastWrapper):
            checkpoint = torch.load(path)
            weights = checkpoint["model_state_dict"]
            weights = fix_state_dict_keys(weights, add_module=False)
            self.model.model.load_state_dict(weights, strict=True)
        else:
            checkpoint = torch.load(path)
            weights = checkpoint["model_state"]
            drop_vars = ["module.norm.weight", "module.norm.bias"]
            weights = {k: v for k, v in weights.items() if k not in drop_vars}
            # need to use strict = False to avoid this error message when
            # using sfno_76ch::
            # RuntimeError: Error(s) in loading state_dict for Wrapper:
            # Missing key(s) in state_dict: "module.trans_down.weights",
            # "module.itrans_up.pct",
            self.model.load_state_dict(weights, strict=False)

    def run_steps(self, x, n, normalize=True, cuda_graphs=False, autocast_fp16=False):
        with torch.no_grad():
            # drop all but the last time point
            # remove channels

            _, n_time_levels, n_channels, _, _ = x.shape
            assert n_time_levels == self.n_history + 1

            if normalize:
                x = self.normalize(x)
            x = x[:, -1, self.channels].clone()

            for i in range(n):
                if not cuda_graphs:
                    with (
                        torch.cuda.amp.autocast()
                        if autocast_fp16
                        else contextlib.nullcontext()
                    ):
                        y = self.model(x)
                    self.out = self.scale * y + self.center
                    x = y
                # CUDA graphs
                else:
                    if self.iteration < 11:  # For DDP if needed (idk)
                        warmup_stream = torch.cuda.Stream()
                        with torch.cuda.stream(warmup_stream):
                            with (
                                torch.cuda.amp.autocast()
                                if autocast_fp16
                                else contextlib.nullcontext()
                            ):
                                y = self.model(x)
                            self.out = self.scale * y + self.center
                            x.copy_(y)
                    elif self.iteration == 11:
                        self.graph = torch.cuda.CUDAGraph()
                        x = x.detach().clone()
                        with torch.cuda.graph(self.graph):
                            print("Recording graph!")
                            with (
                                torch.cuda.amp.autocast()
                                if autocast_fp16
                                else contextlib.nullcontext()
                            ):
                                y = self.model(x)
                            self.out = self.scale * y + self.center
                            x.copy_(y)
                    else:
                        self.graph.replay()

                self.iteration += 1
                out = self.out
                yield out


class TFNOJean(torch.nn.Module):
    n_history = 4
    channels = list(range(34))
    grid = schema.Grid.grid_720x1440
    channel_set = schema.ChannelSet.var34

    def __init__(self, center_path, scale_path):
        super().__init__()
        self.model = Wrapper(
            FactorizedFNO2d(
                params=None,
                modes_height=64,
                modes_embed_dim=50,
                embed_dim=192,
                N_in_channels=172,
                N_out_channels=34,
                fc_channels=1024,
                n_layers=4,
                levels=0,
                joint_factorization=False,
                rank=[2, 96, 96, 32, 32],
                factorization="tucker",
                fixed_rank_modes=False,
                domain_padding=27,
                fft_contraction="complex",
                fft_norm="forward",
                mlp_config={"expansion": 1.0, "dropout": False},
                verbose=True,
                decomposition_kwargs=dict(),
            )
        )
        self.preprocess = preprocessor.Preprocessor2D(
            n_history=4,
            add_grid=True,
            transform_to_nhwc=False,
            img_shape_x=720,
            img_shape_y=1440,
        )
        self.scales = Scales(center_path, scale_path)
        self.scales.normalize(torch.zeros(1, 34, 32, 32)).shape == (
            1,
            34,
            32,
            32,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        weights = checkpoint["model_state"]
        self.model.load_state_dict(weights)

    def normalize(self, x):
        return self.scales.normalize(x)

    def run_steps(self, x, n, normalize=True, **kw):
        with torch.no_grad():
            if normalize:
                x = self.scales.normalize(x).float()
            else:
                x = x.float()

            x = einops.rearrange(x, "t w c x y -> t ( w c) x y")
            y = None
            for i in range(n):
                x, _ = self.preprocess(x, x)
                y = self.model(x)
                x = self.preprocess.append_history(x, y)
                denormalized = self.scales.denormalize(y)
                yield denormalized


class Scales(torch.nn.Module):
    def __init__(self, center_path, scale_path):
        super().__init__()

        center = np.load(center_path)
        scale = np.load(scale_path)

        center = np.load(center_path)
        scale = np.load(scale_path)

        center = np.squeeze(center)
        scale = np.squeeze(scale)

        center = torch.from_numpy(center)
        scale = torch.from_numpy(scale)

        assert center.ndim == 1
        assert scale.ndim == 1

        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def normalize(self, x):
        return (x - self.center[:, None, None]) / self.scale[:, None, None]

    def denormalize(self, x):
        return x * self.scale[:, None, None] + self.center[:, None, None]


def _create_73_ch_sfno():
    config_path = pathlib.Path(__file__).parent / "geometric_v1" / "sfnonet.yaml"
    params = YParams(config_path.as_posix(), "sfno_73ch")
    params.img_crop_shape_x = 721
    params.img_crop_shape_y = 1440
    params.N_in_channels = 73
    params.N_out_channels = 73
    return geometric_v1.FourierNeuralOperatorNet(params)


def get_architecture(architecture: str, model_path: Optional[str] = None) -> torch.nn.Module:
    """
    Args:
        architecture: name of the architecture
        model_path: path to a directory containing static data required to
            instantiate the model. For example, the "graph" of graphcast.
    
    Returns:
        An untrained pytorch model
    """

    if architecture == "baseline_afno_26":
        return afnonet.AFNONet(
            img_size=(720, 1440),
            patch_size=(8, 8),
            in_chans=26,
            out_chans=26,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            num_blocks=8,
        )
    elif architecture == "afno_26ch_v":
        return vit.get_model()
    elif architecture == "hafno_baseline_26ch_edim512_mlp2":
        return geometric.get_model(
            grid=schema.Grid.grid_721x1440,
            config_key="hafno_baseline_26ch_edim512_mlp2",
            config_file="hfnonet.yaml",
        )
    elif architecture == "modulus_afno_20":
        from modulus.models.afno import AFNO
        return AFNO(
            img_size=(720, 1440),
            in_channels=20,
            out_channels=20,
            patch_size=(8, 8),
            embed_dim=768,
            depth=12,
            num_blocks=8,
        )
    elif architecture == "sfno_73ch":
        return _create_73_ch_sfno()
    elif architecture == "graphcast_34ch":
        num_channels = 34
        icospheres_path = (
            model_path
            + registry.SEPERATOR
            + "icospheres.pickle"
        )

        icospheres_path = filesystem.download_cached(icospheres_path)

        static_data_path = (
            model_path + registry.SEPERATOR + "static"
        )
        static_data_path = filesystem.download_cached(static_data_path, recursive=True)

        dist = DistributedManager()

        # instantiate the model, set dtype and move to device
        base_model = (
            GraphCastNet(
                meshgraph_path=icospheres_path,
                static_dataset_path=static_data_path,
                input_dim_grid_nodes=num_channels,
                input_dim_mesh_nodes=3,
                input_dim_edges=4,
                output_dim_grid_nodes=num_channels,
                processor_layers=16,
                hidden_dim=512,
                do_concat_trick=True,
            )
            .to(dtype=torch.bfloat16)
            .to(dist.device)
        )

        # set model to inference mode
        base_model.eval()

        model = graphcast.GraphCastWrapper(
            base_model,
            torch.bfloat16
        )
        return model
    raise NotImplementedError(architecture)


def get_model(model):
    """
    Function to construct an inference model and load the appropriate
    checkpoints from the model registry

    Parameters
    ----------
    model : String describing the required model config


    Returns
    -------
    Inference model


    """
    path = registry.get_weight_path(model)
    scale_path = registry.get_scale_path(model)
    center_path = registry.get_center_path(model)
    metadata = registry.get_metadata(model)

    # download cached
    path = filesystem.download_cached(path)
    scale_path = filesystem.download_cached(scale_path)
    center_path = filesystem.download_cached(center_path)

    if model == TFNO_JEAN:
        inference = TFNOJean(center_path, scale_path)
    else:
        inference = Inference(
            model=get_architecture(metadata.architecture, model_path=registry.get_model_path(model)),
            channels=metadata.in_channels,
            center_path=center_path,
            scale_path=scale_path,
            grid=metadata.grid,
            channel_set=metadata.channel_set,
        )

    inference.load_checkpoint(path)
    return inference


def fix_state_dict_keys(state_dict, add_module=False):
    """Add or remove 'module.' from state_dict keys

    Parameters
    ----------
    state_dict : Dict
        Model state_dict
    add_module : bool, optional
        If True, will add 'module.' to keys, by default False

    Returns
    -------
    Dict
        Model state_dict with fixed keys
    """
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if add_module:
            new_key = 'module.' + key
        else:
            new_key = key.replace('module.', '')
        fixed_state_dict[new_key] = value
    return fixed_state_dict

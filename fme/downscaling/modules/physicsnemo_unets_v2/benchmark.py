import dataclasses
from dataclasses import dataclass

import torch

from fme.core.benchmark.benchmark import BenchmarkABC, register_benchmark
from fme.core.device import get_device
from fme.core.typing_ import TensorDict

from .group_norm import apex_available
from .unets import SongUNetv2


def _is_channels_last(t: torch.Tensor) -> bool:
    return t.ndim == 4 and t.is_contiguous(memory_format=torch.channels_last)


def _is_contiguous_only(t: torch.Tensor) -> bool:
    return (
        t.ndim == 4
        and t.is_contiguous()
        and not t.is_contiguous(memory_format=torch.channels_last)
    )


@dataclass
class ConversionEvent:
    module_name: str
    module_type: str
    direction: str
    tensor_role: str


def _attach_format_hooks(
    model: torch.nn.Module,
) -> tuple[list[ConversionEvent], list]:
    """Attach forward hooks that record memory-format transitions.

    Returns the (shared) events list and the hook handles so the caller can
    remove them after the forward pass.
    """
    events: list[ConversionEvent] = []
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, outputs):
            in_tensors = []
            if isinstance(inputs, torch.Tensor):
                in_tensors.append(("input_0", inputs))
            elif isinstance(inputs, (tuple, list)):
                for i, t in enumerate(inputs):
                    if isinstance(t, torch.Tensor):
                        in_tensors.append((f"input_{i}", t))

            out_tensors = []
            if isinstance(outputs, torch.Tensor):
                out_tensors.append(("output_0", outputs))
            elif isinstance(outputs, (tuple, list)):
                for i, t in enumerate(outputs):
                    if isinstance(t, torch.Tensor):
                        out_tensors.append((f"output_{i}", t))

            for in_role, in_t in in_tensors:
                for out_role, out_t in out_tensors:
                    if _is_channels_last(in_t) and _is_contiguous_only(out_t):
                        events.append(
                            ConversionEvent(
                                module_name=name,
                                module_type=type(module).__name__,
                                direction="from_channels_last",
                                tensor_role=f"{in_role}->{out_role}",
                            )
                        )
                    elif _is_contiguous_only(in_t) and _is_channels_last(out_t):
                        events.append(
                            ConversionEvent(
                                module_name=name,
                                module_type=type(module).__name__,
                                direction="to_channels_last",
                                tensor_role=f"{in_role}->{out_role}",
                            )
                        )

        return hook

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(make_hook(name)))

    return events, handles


def check_weights_channels_last(model: torch.nn.Module) -> dict[str, bool]:
    """Return a dict mapping 4-D parameter names to whether they are
    channels_last."""
    results = {}
    for name, param in model.named_parameters():
        if param.ndim == 4:
            results[name] = param.is_contiguous(memory_format=torch.channels_last)
    return results


@dataclass
class ChannelsLastDiagnostics:
    n_channels_last_weights: int
    n_total_4d_weights: int
    n_conversions_to_channels_last: int
    n_conversions_from_channels_last: int
    conversion_layers: list[dict[str, str]]

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        return {
            "n_channels_last_weights": torch.tensor(self.n_channels_last_weights),
            "n_total_4d_weights": torch.tensor(self.n_total_4d_weights),
            "n_conversions_to_channels_last": torch.tensor(
                self.n_conversions_to_channels_last
            ),
            "n_conversions_from_channels_last": torch.tensor(
                self.n_conversions_from_channels_last
            ),
        }

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _run_channels_last_diagnostics(
    model: SongUNetv2,
    x: torch.Tensor,
    noise_labels: torch.Tensor,
    class_labels: torch.Tensor | None,
) -> ChannelsLastDiagnostics:
    """Run a single-sample forward pass with hooks to collect channels_last
    diagnostics.  Uses B=1 slice and ``torch.no_grad`` to minimise memory."""
    weight_info = check_weights_channels_last(model)
    n_cl_weights = sum(v for v in weight_info.values())
    n_total_weights = len(weight_info)

    probe_x = x[:1]
    probe_noise = noise_labels[:1]
    probe_cls = class_labels[:1] if class_labels is not None else None

    events, handles = _attach_format_hooks(model)
    with torch.no_grad():
        model(probe_x, probe_noise, probe_cls)
    for h in handles:
        h.remove()

    to_cl = sum(1 for e in events if e.direction == "to_channels_last")
    from_cl = sum(1 for e in events if e.direction == "from_channels_last")

    conversion_layers = [
        {
            "module_name": e.module_name,
            "module_type": e.module_type,
            "direction": e.direction,
            "tensor_role": e.tensor_role,
        }
        for e in events
    ]

    return ChannelsLastDiagnostics(
        n_channels_last_weights=n_cl_weights,
        n_total_4d_weights=n_total_weights,
        n_conversions_to_channels_last=to_cl,
        n_conversions_from_channels_last=from_cl,
        conversion_layers=conversion_layers,
    )


@dataclass(frozen=True)
class _SongUNetv2BenchmarkParams:
    img_resolution: int
    in_channels: int
    out_channels: int
    label_dim: int
    model_channels: int
    channel_mult: list[int]
    use_apex_gn: bool = False
    use_amp_bf16: bool = False


def _make_song_unet_benchmark(
    *,
    new_params: _SongUNetv2BenchmarkParams,
    regression_params: _SongUNetv2BenchmarkParams | None,
) -> type[BenchmarkABC]:
    class _SongUNetv2BenchmarkVariant(BenchmarkABC):
        def __init__(
            self,
            model: SongUNetv2,
            x: torch.Tensor,
            noise_labels: torch.Tensor,
            class_labels: torch.Tensor | None,
            use_amp_bf16: bool,
        ):
            self.model = model
            self.x = x
            self.noise_labels = noise_labels
            self.class_labels = class_labels
            self.use_amp_bf16 = use_amp_bf16
            self._channels_last_diagnostics = _run_channels_last_diagnostics(
                model, x, noise_labels, class_labels
            )

        def _get_amp_context(self):
            if self.use_amp_bf16:
                return torch.amp.autocast(get_device().type, dtype=torch.bfloat16)
            return torch.amp.autocast(get_device().type, enabled=False)

        def run_instance(self, timer) -> TensorDict:
            with self._get_amp_context():
                result = self.model(
                    self.x,
                    self.noise_labels,
                    self.class_labels,
                    timer=timer,
                )
            return {
                "output": result.detach(),
                "diagnostics": self._channels_last_diagnostics.to_tensor_dict(),
            }

        @classmethod
        def _build(
            cls, params: _SongUNetv2BenchmarkParams
        ) -> "_SongUNetv2BenchmarkVariant":
            device = get_device()
            model = SongUNetv2(
                img_resolution=params.img_resolution,
                in_channels=params.in_channels,
                out_channels=params.out_channels,
                label_dim=params.label_dim,
                model_channels=params.model_channels,
                channel_mult=params.channel_mult,
                num_blocks=4,
                attn_resolutions=[],
                dropout=0.0,
                use_apex_gn=params.use_apex_gn,
            ).to(device)
            model.eval()

            B = 1
            H = W = params.img_resolution
            x = torch.randn(B, params.in_channels, H, W, device=device)
            noise_labels = torch.rand(B, device=device)
            class_labels = (
                torch.randn(B, params.label_dim, device=device)
                if params.label_dim > 0
                else None
            )
            if params.use_apex_gn:
                if not apex_available():
                    raise ValueError(
                        "'apex' is not installed, set `use_apex_gn=False`"
                    )
                x = x.to(memory_format=torch.channels_last)

            return cls(
                model=model,
                x=x,
                noise_labels=noise_labels,
                class_labels=class_labels,
                use_amp_bf16=params.use_amp_bf16,
            )

        @classmethod
        def new(cls) -> "_SongUNetv2BenchmarkVariant":
            return cls._build(new_params)

        @classmethod
        def new_for_regression(cls) -> "_SongUNetv2BenchmarkVariant | None":
            if regression_params is None:
                return None
            return cls._build(regression_params)

    return _SongUNetv2BenchmarkVariant


SongUNetv2Benchmark = _make_song_unet_benchmark(
    new_params=_SongUNetv2BenchmarkParams(
        img_resolution=64,
        in_channels=3,
        out_channels=2,
        label_dim=0,
        model_channels=16,
        channel_mult=[1, 2, 2, 2],
    ),
    regression_params=_SongUNetv2BenchmarkParams(
        img_resolution=16,
        in_channels=3,
        out_channels=2,
        label_dim=0,
        model_channels=16,
        channel_mult=[1, 2],
    ),
)

SongUNetv2BenchmarkBf16 = _make_song_unet_benchmark(
    new_params=_SongUNetv2BenchmarkParams(
        img_resolution=64,
        in_channels=3,
        out_channels=2,
        label_dim=0,
        model_channels=16,
        channel_mult=[1, 2, 2, 2],
        use_amp_bf16=True,
    ),
    regression_params=_SongUNetv2BenchmarkParams(
        img_resolution=16,
        in_channels=3,
        out_channels=2,
        label_dim=0,
        model_channels=16,
        channel_mult=[1, 2],
        use_amp_bf16=True,
    ),
)

SongUNetv2BenchmarkApex = _make_song_unet_benchmark(
    new_params=_SongUNetv2BenchmarkParams(
        img_resolution=512,
        in_channels=6,
        out_channels=4,
        label_dim=0,
        model_channels=128,  # min for apex gn
        channel_mult=[1, 2, 2, 2],
        use_apex_gn=True,
    ),
    regression_params=None,
)

SongUNetv2BenchmarkApexBf16 = _make_song_unet_benchmark(
    new_params=_SongUNetv2BenchmarkParams(
        img_resolution=512,
        in_channels=6,
        out_channels=4,
        label_dim=0,
        model_channels=128,  # min for apex gn
        channel_mult=[1, 2, 2, 2],
        use_apex_gn=True,
        use_amp_bf16=True,
    ),
    regression_params=None,
)


register_benchmark("songunetv2")(SongUNetv2Benchmark)
register_benchmark("songunetv2_bf16")(SongUNetv2BenchmarkBf16)
register_benchmark("songunetv2_apex")(SongUNetv2BenchmarkApex)
register_benchmark("songunetv2_apex_bf16")(SongUNetv2BenchmarkApexBf16)

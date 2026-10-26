"""Debug-only instrumentation to localize non-finite training losses.

Enabled with the environment variable ``FME_NAN_DEBUG=1``. Records every
grad-enabled network call since the last optimizer step (the packed input, the
labels and the RNG states before the call), checks each stage of
``step_with_adjustments`` for non-finite values, and when a non-finite loss is
seen:

- logs which stage first went non-finite and per-variable counts,
- replays the recorded calls through the *eager* module with forward hooks to
  find the first submodule emitting a non-finite value,
- dumps the records plus the eager module (pickled) to
  ``$FME_NAN_DEBUG_DIR`` (default ``/results/nan_dumps``) for offline replay.
"""

import logging
import os
from typing import Any

import torch
from torch import nn

ENABLED = os.environ.get("FME_NAN_DEBUG", "0") == "1"
DUMP_DIR = os.environ.get("FME_NAN_DEBUG_DIR", "/results/nan_dumps")
MAX_DUMPS = int(os.environ.get("FME_NAN_DEBUG_MAX_DUMPS", "3"))

logger = logging.getLogger("fme.nan_debug")

_records: list[dict[str, Any]] = []
_stage_flags: list[dict[str, Any]] = []
_module_ref: Any = None
_n_dumps = 0
_n_events = 0
_n_calls = 0


def register_module(module: Any) -> None:
    """Register the fme ``Module`` (with ``_module`` and ``forward_module``)."""
    global _module_ref
    _module_ref = module


def _stats(t: torch.Tensor) -> dict[str, float]:
    t = t.detach().float()
    finite = torch.isfinite(t)
    tf = t[finite]
    out = {
        "n": float(t.numel()),
        "n_nonfinite": float((~finite).sum().item()),
        "n_nan": float(torch.isnan(t).sum().item()),
    }
    if tf.numel() > 0:
        out.update(
            min=tf.min().item(),
            max=tf.max().item(),
            absmax=tf.abs().max().item(),
            mean=tf.mean().item(),
            std=tf.std().item() if tf.numel() > 1 else 0.0,
        )
    return out


def record_call(
    input_tensor: torch.Tensor,
    labels: torch.Tensor | None,
    cuda_rng: torch.Tensor | None,
    cpu_rng: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    global _n_calls
    _n_calls += 1
    finite = bool(torch.isfinite(output_tensor).all().item())
    rec = {
        "call_index": _n_calls,
        "input": input_tensor.detach().cpu().clone(),
        "labels": None if labels is None else labels.detach().cpu().clone(),
        "cuda_rng": cuda_rng,
        "cpu_rng": cpu_rng,
        "output_finite": finite,
        "training": None,
    }
    if not finite:
        rec["output"] = output_tensor.detach().cpu().clone()
        logger.warning(
            f"NAN_DEBUG: network output non-finite at call {_n_calls}: "
            f"out={_stats(output_tensor)} in={_stats(input_tensor)}"
        )
    _records.append(rec)


_pre: dict[str, Any] = {}

_PRE_NAMES = ("PRATEsfc", "LHTFLsfc", "tendency_of_total_water_path_due_to_advection")


def note_pre_corrector(output: Any, normalizer: Any, input: Any = None) -> None:
    """Keep references to the pre-corrector output and normalizer."""
    if not ENABLED or not torch.is_grad_enabled():
        return
    _pre["output"] = output
    _pre["normalizer"] = normalizer
    _pre["input"] = input


def _per_sample(t: torch.Tensor) -> str:
    t = t.detach().float()
    if t.dim() < 3:
        return f"scalar-ish {t.flatten().tolist()[:16]}"
    flat = t.reshape(t.shape[0], -1)
    rows = []
    for b in range(flat.shape[0]):
        r = flat[b]
        rows.append(
            f"[b{b} mean={r.mean().item():.4g} min={r.min().item():.4g} "
            f"max={r.max().item():.4g} frac>0={(r > 0).float().mean().item():.3f}]"
        )
    return " ".join(rows)


def _log_pre_corrector() -> None:
    out = _pre.get("output")
    norm = _pre.get("normalizer")
    if out is None:
        return
    for name in _PRE_NAMES:
        if name in out:
            logger.warning(f"NAN_DEBUG: pre-corrector {name}: {_per_sample(out[name])}")
        if norm is not None:
            m = norm.means.get(name)
            sd = norm.stds.get(name)
            if m is not None:
                logger.warning(
                    f"NAN_DEBUG: normalizer {name} mean={m.detach().flatten().tolist()} "
                    f"std={sd.detach().flatten().tolist()}"
                )
    inp = _pre.get("input")
    if inp is not None and "PRATEsfc" in inp:
        logger.warning(f"NAN_DEBUG: input PRATEsfc: {_per_sample(inp['PRATEsfc'])}")


def check_stage(stage: str, data: Any) -> None:
    """Record which variables are non-finite after a stage."""
    if not ENABLED or not torch.is_grad_enabled():
        return
    bad = {}
    if isinstance(data, torch.Tensor):
        items = {"tensor": data}
    else:
        items = dict(data)
    for name, value in items.items():
        if not isinstance(value, torch.Tensor):
            continue
        n_bad = int((~torch.isfinite(value)).sum().item())
        if n_bad > 0:
            bad[name] = n_bad
    _stage_flags.append({"stage": stage, "bad": bad})
    if bad:
        logger.warning(f"NAN_DEBUG: stage {stage} non-finite vars: {bad}")
        for name, value in items.items():
            if name in bad and isinstance(value, torch.Tensor) and value.dim() >= 3:
                per = (~torch.isfinite(value)).reshape(value.shape[0], -1).sum(-1)
                logger.warning(f"NAN_DEBUG: {name} non-finite per sample {per.tolist()}")
        if stage == "corrected":
            _log_pre_corrector()


def _unwrap(m: nn.Module) -> nn.Module:
    m = getattr(m, "_orig_mod", m)
    # DDP / DummyWrapper
    while hasattr(m, "module") and isinstance(m.module, nn.Module):
        m = m.module
    return m


def _replay_eager(rec: dict[str, Any], train: bool) -> dict[str, Any]:
    """Replay one call in eager with hooks; return first non-finite module."""
    inner = _unwrap(_module_ref._module)
    device = next(inner.parameters()).device
    names = {m: n for n, m in inner.named_modules()}
    events: list[tuple[str, dict, list]] = []
    first_bad: list[Any] = []

    def hook(mod, inputs, output):
        outs = output if isinstance(output, tuple | list) else (output,)
        for o in outs:
            if isinstance(o, torch.Tensor) and not torch.isfinite(o).all():
                in_stats = [
                    _stats(i) for i in inputs if isinstance(i, torch.Tensor)
                ]
                events.append((names.get(mod, "?"), _stats(o), in_stats))
                return

    handles = [m.register_forward_hook(hook) for m in inner.modules()]
    was_training = inner.training
    try:
        inner.train(train)
        if rec["cuda_rng"] is not None:
            torch.cuda.set_rng_state(rec["cuda_rng"])
        torch.set_rng_state(rec["cpu_rng"])
        with torch.no_grad():
            labels = None if rec["labels"] is None else rec["labels"].to(device)
            out = inner(rec["input"].to(device), labels=labels)
        result = {
            "train": train,
            "output_finite": bool(torch.isfinite(out).all().item()),
            "output_stats": _stats(out),
            # hooks fire in post-order (children before parents), so the
            # first event is the innermost first-failing module
            "first_events": events[:3],
        }
    finally:
        for h in handles:
            h.remove()
        inner.train(was_training)
    del first_bad
    return result


def on_non_finite_loss(loss: torch.Tensor) -> None:
    if not ENABLED:
        return
    try:
        _on_non_finite_loss(loss)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"NAN_DEBUG: handler failed: {e!r}")


def _on_non_finite_loss(loss: torch.Tensor) -> None:
    global _n_dumps, _n_events
    _n_events += 1
    rank = int(os.environ.get("RANK", "0"))
    logger.warning(
        f"NAN_DEBUG: non-finite loss {loss.item()} event {_n_events} rank {rank}; "
        f"{len(_records)} recorded calls, output_finite="
        f"{[r['output_finite'] for r in _records]}; stage flags="
        f"{[s for s in _stage_flags if s['bad']]}"
    )
    replays = []
    if _module_ref is not None and _n_events <= 20:
        for rec in _records:
            for train in (True, True, False):
                try:
                    r = _replay_eager(rec, train)
                except Exception as e:  # noqa: BLE001
                    r = {"error": repr(e)}
                r["call_index"] = rec["call_index"]
                replays.append(r)
                logger.warning(f"NAN_DEBUG: eager replay: {r}")
    if _n_dumps < MAX_DUMPS and _module_ref is not None:
        _n_dumps += 1
        os.makedirs(DUMP_DIR, exist_ok=True)
        path = os.path.join(DUMP_DIR, f"rank{rank}_event{_n_events}.pt")
        inner = _unwrap(_module_ref._module)
        torch.save(
            {
                "records": _records,
                "stage_flags": _stage_flags,
                "replays": replays,
                "state_dict": {k: v.cpu() for k, v in inner.state_dict().items()},
                "loss": loss.item(),
            },
            path,
        )
        module_path = os.path.join(DUMP_DIR, "module.pt")
        if not os.path.exists(module_path):
            torch.save(inner, module_path)
        logger.warning(f"NAN_DEBUG: dumped to {path}")


def reset() -> None:
    _records.clear()
    _stage_flags.clear()

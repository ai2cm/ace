from typing import Optional, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist

from fme.ace.utils import comm

from fme.ace.models.makani_utils.checkpoint_helpers import  gather_model_state_dict, prepend_prefix_to_state_dict, scatter_model_state_dict

def _save_checkpoint_flexible(
    checkpoint_path: str,
    model: nn.Module,
    loss: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[lr_scheduler.LRScheduler] = None,
    counters: Optional[Dict[str, int]] = None,
):
    # checkpoint name
    checkpoint_fname = checkpoint_path.format(mp_rank=0)

    # iterate over parameters and gather them from the ranks
    if comm.get_size("model") > 1:
        state_dict = gather_model_state_dict(model)
    else:
        state_dict = model.state_dict()

    # drop module prefix in case if DDP is being used
    if isinstance(model, nn.parallel.DistributedDataParallel):
        nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")

    store_dict = {"model_state": state_dict}

    if loss is not None:
        store_dict["loss_state_dict"] = loss.state_dict()

    if optimizer is not None:
        if comm.get_size("model") > 1:
            store_dict["optimizer_state_dict"] = gather_optimizer_state_dict(model, optimizer)
        else:
            store_dict["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        store_dict["scheduler_state_dict"] = scheduler.state_dict()

    if counters is not None:
        store_dict["iters"] = counters["iters"]
        store_dict["epoch"] = counters["epoch"]

    # in flexible mode only rank 0 needs to save the data to disk
    if comm.get_world_rank() == 0:
        torch.save(store_dict, checkpoint_fname)

    return

def _restore_checkpoint_flexible(
    checkpoint_path: str,
    model: nn.Module,
    loss: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[lr_scheduler.LRScheduler] = None,
    counters: Optional[Dict[str, int]] = None,
    strict: bool = True,
):
    # when loading the weights in flexble mode we exclusively use mp_rank=0 and load them onto the cpu
    checkpoint_fname = checkpoint_path.format(mp_rank=0)
    checkpoint = torch.load(checkpoint_fname, map_location="cpu", weights_only=False)

    # this is reworked to avoid loading modules related to the SHT
    state_dict = checkpoint["model_state"]

    if isinstance(model, nn.parallel.DistributedDataParallel):
        # prepend module prefix to state dict:
        prepend_prefix_to_state_dict(state_dict, "module.")

    if comm.get_size("model") > 1:
        state_dict = scatter_model_state_dict(model, state_dict, strict)

    # load state dict
    model.load_state_dict(state_dict, strict=strict)

    # the loss is also restored in the case that it has a state
    if loss is not None:
        loss.load_state_dict(checkpoint["loss_state_dict"])

    # If finetuning, restore checkpoint does not load optimizer state, instead uses config specified lr.
    if optimizer is not None:
        if comm.get_size("model") > 1:
            checkpoint["optimizer_state_dict"] = scatter_optimizer_state_dict(model, optimizer, checkpoint["optimizer_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if counters is not None:
        counters["iters"] = checkpoint["iters"]
        counters["start_epoch"] = checkpoint["epoch"]

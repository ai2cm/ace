import torch


def get_device():
    return torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

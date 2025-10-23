import torch
from typing import Tuple


def get_device() -> Tuple[torch.device, str]:
    """
    Detects and sets up the appropriate PyTorch device (MPS, CUDA, or CPU).

    The function checks for Apple Silicon (MPS) first, then NVIDIA (CUDA),
    and defaults to CPU. It also applies the manual seed for MPS if available.

    Parameters
    ----------
    seed : int
        The random seed value to use for torch.mps.manual_seed().

    Returns
    -------
    Tuple[torch.device, str]
        A tuple containing:
        - The selected torch.device object.
        - A string describing the selected device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        status = "Using Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        status = "Using NVIDIA CUDA GPU"
    else:
        device = torch.device("cpu")
        status = "Using CPU"

    return device, status

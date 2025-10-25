import torch

def get_device() -> tuple[torch.device, str]:
    """
    Detects and sets up the appropriate PyTorch device (MPS, CUDA, or CPU).

    The function checks for Apple Silicon (MPS) first, then NVIDIA (CUDA),
    and defaults to CPU. It also applies the manual seed for MPS if available.

    Args:
        seed_value : Optional[int], default=42
            The integer value used to set the random seeds.

    Returns:
        tuple[torch.device, str]
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

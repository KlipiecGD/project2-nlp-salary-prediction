import os
import torch
import random
import numpy as np
from typing import Optional


def set_seed(seed_value: Optional[int] = 42) -> None:
    """
      Sets seeds for reproducibility across Python's random, NumPy, and PyTorch.

    Args:
        seed_value : Optional[int], default=42
            The integer value used to set the random seeds.
    """
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)

    # PyTorch seeds and deterministic settings
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.mps.benchmark = False

    # Apple Silicon GPU (MPS) seed
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)


def seed_worker(worker_id: int) -> None:
    """
    Sets separate, deterministic seeds for PyTorch DataLoader workers.

    Args:
        worker_id : int
            The unique ID assigned to the DataLoader worker process.
    """
    worker_seed: int = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)

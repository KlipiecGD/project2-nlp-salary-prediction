import os
import torch
import random
import numpy as np
from typing import Optional
from config.config import RANDOM_SEED


def set_seed(seed_value: Optional[int] = None) -> None:
    """
    Sets seeds for reproducibility across Python's random, NumPy, and PyTorch.

    Parameters
    ----------
    seed_value : Optional[int], default=RANDOM_SEED
        The integer value used to set the random seeds.
    """
    seed = seed_value if seed_value is not None else RANDOM_SEED

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeds and deterministic settings
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.mps.benchmark = False

    # Apple Silicon GPU (MPS) seed
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """
    Sets separate, deterministic seeds for PyTorch DataLoader workers.

    Parameters
    ----------
    worker_id : int
        The unique ID assigned to the DataLoader worker process.
    """
    worker_seed: int = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)

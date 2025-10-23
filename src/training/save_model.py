import os
import logging
import torch
from torch import nn
from typing import Optional
from config.config import MODELS_DIR

def save_model(
    model: nn.Module,
    model_name: str,
    model_dir: str = MODELS_DIR,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Save the trained model to the specified directory.

    Args:
        model: Trained PyTorch model to be saved.
        model_name: Name of the model file (without extension).
        model_dir: Directory where the model will be saved.
        logger: Optional logger for logging information.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        if logger:
            logger.info(f"Created directory {model_dir} for saving the model.")

    model_path = os.path.join(model_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)

    if logger:
        logger.info(f"Model saved to {model_path}.")
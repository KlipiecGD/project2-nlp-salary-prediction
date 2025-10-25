import os
import logging
import torch
from torch import nn
from typing import Optional


def save_model(
    model: nn.Module,
    model_name: str,
    model_dir: str = "trained_models",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Save the trained model to the specified directory.

    Args:
        model: nn.Module, Trained PyTorch model to be saved.
        model_name: str, Name of the model file (without extension).
        model_dir: str, Directory where the model will be saved, default 'trained_models'.
        logger: Optional[logging.Logger], Optional logger for logging information.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        if logger:
            logger.info(f"Created directory {model_dir} for saving the model.")

    model_path = os.path.join(model_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)

    if logger:
        logger.info(f"Model saved to {model_path}.")

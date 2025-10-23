import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Dict
import logging
from config.config import LOSS_FUNCTION


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_model_metrics_manual(
    model: nn.Module,
    model_name: str,
    train_losses_scaled: list,
    valid_losses_scaled: list,
    train_losses_real: list,
    valid_losses_real: list,
    test_metrics: Dict,
    elapsed_time: float,
    loss_fn: str = LOSS_FUNCTION,
) -> Dict:
    """
    Collect metrics from manually trained models (not using train_model function).
    Use this for models trained with custom training loops.

    Args:
        model: Trained PyTorch model
        model_name: Name/identifier for the model
        train_losses_scaled: List of training losses (scaled) per epoch
        valid_losses_scaled: List of validation losses (scaled) per epoch
        train_losses_real: List of training losses (real scale) per epoch - MSE or MAE
        valid_losses_real: List of validation losses (real scale) per epoch - MSE or MAE
        test_metrics: Dictionary with test metrics (e.g., {'Test MSE (scaled)': ..., 'Test MSE (real)': ..., 'Test RMSE (real)': ...})
        elapsed_time: Training time in seconds
        loss_fn: Loss function used ('mse' or 'mae')

    Returns:
        Dictionary containing all model information and metrics
    """

    # Count trainable parameters
    n_params = count_parameters(model)

    # Find best validation epoch
    best_valid_idx = np.argmin(valid_losses_real)

    # Create history dict in the format expected by other functions
    if loss_fn == "mse":
        history = {
            "train_loss_scaled": train_losses_scaled,
            "valid_loss_scaled": valid_losses_scaled,
            "train_mse_real": train_losses_real,
            "valid_mse_real": valid_losses_real,
            "train_rmse": [np.sqrt(x) for x in train_losses_real],
            "valid_rmse": [np.sqrt(x) for x in valid_losses_real],
        }
    else:  # mae
        history = {
            "train_loss_scaled": train_losses_scaled,
            "valid_loss_scaled": valid_losses_scaled,
            "train_mae_real": train_losses_real,
            "valid_mae_real": valid_losses_real,
        }

    # Build comprehensive results dictionary
    results = {
        "model_name": model_name,
        "n_trainable_params": n_params,
        "training_time_seconds": elapsed_time,
        "training_time_minutes": elapsed_time / 60,
        "epochs_trained": len(train_losses_scaled),
        "best_epoch": best_valid_idx + 1,
        # Test metrics
        **test_metrics,
        # Full training history
        "history": history,
    }

    # Add loss-specific metrics (best epoch only)
    if loss_fn == "mse":
        results["train_mse_real"] = train_losses_real[best_valid_idx]
        results["train_rmse_real"] = np.sqrt(train_losses_real[best_valid_idx])
        results["valid_mse_real"] = valid_losses_real[best_valid_idx]
        results["valid_rmse_real"] = np.sqrt(valid_losses_real[best_valid_idx])
    else:  # mae
        results["train_mae_real"] = train_losses_real[best_valid_idx]
        results["valid_mae_real"] = valid_losses_real[best_valid_idx]

    return results


def collect_model_metrics(
    model: nn.Module,
    history: Dict,
    elapsed_time: float,
    test_metrics: Dict,
    model_name: str,
    loss_fn: str = LOSS_FUNCTION,
) -> Dict:
    """
    Collect all model metrics into a single dictionary.

    Args:
        model: Trained PyTorch model
        history: History dictionary from train_model function
        elapsed_time: Training time in seconds from train_model function
        test_metrics: Test metrics dictionary from evaluate_model function
        model_name: Name/identifier for the model
        loss_fn: Loss function used ('mse' or 'mae')

    Returns:
        Dictionary containing all model information and metrics
    """

    # Count trainable parameters
    n_params = count_parameters(model)

    # Get number of epochs trained
    epochs_trained = len(history["train_loss_scaled"])

    # Find best validation epoch (minimum validation loss)
    if loss_fn == "mse":
        best_valid_idx = np.argmin(history["valid_rmse"])
        best_train_rmse = history["train_rmse"][best_valid_idx]
        best_valid_rmse = history["valid_rmse"][best_valid_idx]
        best_train_mse = history["train_mse_real"][best_valid_idx]
        best_valid_mse = history["valid_mse_real"][best_valid_idx]
    else:  # mae
        best_valid_idx = np.argmin(history["valid_mae_real"])
        best_train_mae = history["train_mae_real"][best_valid_idx]
        best_valid_mae = history["valid_mae_real"][best_valid_idx]

    # Build comprehensive results dictionary
    results = {
        "model_name": model_name,
        "n_trainable_params": n_params,
        "training_time_seconds": elapsed_time,
        "training_time_minutes": elapsed_time / 60,
        "epochs_trained": epochs_trained,
        "best_epoch": best_valid_idx + 1,
        # Test metrics
        **test_metrics,
        # Full training history
        "history": history,
    }

    # Add loss-specific metrics (best epoch only)
    if loss_fn == "mse":
        results["train_mse_real"] = best_train_mse
        results["train_rmse_real"] = best_train_rmse
        results["valid_mse_real"] = best_valid_mse
        results["valid_rmse_real"] = best_valid_rmse
    else:  # mae
        results["train_mae_real"] = best_train_mae
        results["valid_mae_real"] = best_valid_mae

    return results


def results_to_dataframe(results_dict: Dict):
    """
    Convert results dictionary to a pandas DataFrame for easy comparison.

    Args:
        results_dict: Dictionary with model results (each value is output from collect_model_metrics)

    Returns:
        pandas DataFrame with key metrics for each model
    """

    summary_data = []
    for model_name, metrics in results_dict.items():
        row = {
            "Model": model_name,
            "Parameters": metrics["n_trainable_params"],
            "Training Time (min)": round(metrics["training_time_minutes"], 2),
            "Epochs": metrics["epochs_trained"],
            "Best Epoch": metrics["best_epoch"],
        }

        # Add loss-specific metrics (best epoch only)
        if "Test RMSE (real)" in metrics:
            row["Train RMSE"] = round(metrics["train_rmse_real"], 2)
            row["Valid RMSE"] = round(metrics["valid_rmse_real"], 2)
            row["Test RMSE"] = round(metrics["Test RMSE (real)"], 2)
        elif "Test MAE (real)" in metrics:
            row["Train MAE"] = round(metrics["train_mae_real"], 2)
            row["Valid MAE"] = round(metrics["valid_mae_real"], 2)
            row["Test MAE"] = round(metrics["Test MAE (real)"], 2)

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def print_model_parameters_summary(
    model: nn.Module, logger: logging.Logger = None
) -> None:
    """
    Prints the parameter count for each named layer and the total sum.
    Args:
        model: The PyTorch model whose parameters are to be summarized.
    """
    total_params = 0

    # Print layer-wise details
    for name, param in model.named_parameters():
        num_params = param.numel()
        if logger:
            logger.info(
                f"{name}: {num_params:,} parameters, trainable={param.requires_grad}"
            )
        total_params += num_params

    # Print summed total
    if logger:
        logger.info("-" * 40)
        logger.info(f"Total Parameters: {total_params:,}")

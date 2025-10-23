import torch
import torch.nn as nn
from typing import Dict
import logging
import numpy as np
from config.config import LOSS_FUNCTION


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    training_time: float = None,
    target_scaler=None,
    device: torch.device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
    loss_fn: str = LOSS_FUNCTION,
    model_name: str = None,
    results_dict: Dict = None,
    log: bool = False,
    multi_input: bool = False,
    logger: logging.Logger = None,
) -> Dict:
    """
    Evaluates a PyTorch regression model and stores results in a dictionary.
    Supports both single-input and multi-input architectures.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test data.
        target_scaler: The fitted scaler for inverse transforming predictions (None if log=True).
        device: torch device (mps, cuda or cpu).
        loss_fn_str: The metric to calculate ('mse' or 'mae').
        model_name: Name/identifier for the model (optional).
        results_dict: Dictionary to store results (optional, will update in-place).
        log: If True, y is log-transformed; real metrics computed using expm1.
        multi_input: If True, expects (embeddings, tabular, target) batches;
                     if False, expects (features, target) batches.
        logger: Optional logger for logging information.

    Returns:
        A dictionary containing the calculated metrics.
    """
    model.eval()

    # Select the appropriate loss function
    if loss_fn == "mse":
        loss_function = nn.MSELoss()
    elif loss_fn == "mae":
        loss_function = nn.L1Loss()
    else:
        logger.error("Unsupported loss function. Use 'mse' or 'mae'.")
        raise ValueError("Unsupported loss function. Use 'mse' or 'mae'.")

    # Metrics for evaluation
    test_loss_scaled = 0.0
    all_predictions_real = []
    all_targets_real = []

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch based on architecture type
            if multi_input:
                embeddings_batch, tabular_batch, y_batch = batch
                embeddings_batch = embeddings_batch.to(device)
                tabular_batch = tabular_batch.to(device)
                y_batch = y_batch.to(device)
                batch_size = embeddings_batch.size(0)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                batch_size = X_batch.size(0)

            # Forward pass
            if multi_input:
                predictions = model(embeddings_batch, tabular_batch).squeeze()
            else:
                predictions = model(X_batch).squeeze()

            # Accumulate loss in scaled domain
            loss = loss_function(predictions, y_batch)
            test_loss_scaled += loss.item() * batch_size

            # Inverse transform to real scale
            if log:
                # For log1p-transformed targets: use expm1 to get back to original scale
                predictions_real = np.expm1(predictions.cpu().numpy())
                y_batch_real = np.expm1(y_batch.cpu().numpy())
            else:
                # For scaled targets: inverse transform using scaler
                if target_scaler is None:
                    logger.error("target_scaler must be provided when log=False")
                    raise ValueError("target_scaler must be provided when log=False")
                predictions_real = target_scaler.inverse_transform(
                    predictions.cpu().numpy().reshape(-1, 1)
                ).ravel()
                y_batch_real = target_scaler.inverse_transform(
                    y_batch.cpu().numpy().reshape(-1, 1)
                ).ravel()

            # Store predictions and targets for real-scale metrics
            all_predictions_real.extend(predictions_real)
            all_targets_real.extend(y_batch_real)

    # Average loss over the entire dataset
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)

    # Calculate real-scale metrics over the entire dataset
    all_predictions_real = np.array(all_predictions_real)
    all_targets_real = np.array(all_targets_real)

    # Build metrics dictionary based on loss function
    metrics = {}
    if loss_fn == "mse":
        mse_real = np.mean((all_predictions_real - all_targets_real) ** 2)
        metrics["Test MSE (scaled)"] = test_loss_scaled_avg
        metrics["Test MSE (real)"] = mse_real
        metrics["Test RMSE (real)"] = np.sqrt(mse_real)
    elif loss_fn == "mae":
        mae_real = np.mean(np.abs(all_predictions_real - all_targets_real))
        metrics["Test MAE (scaled)"] = test_loss_scaled_avg
        metrics["Test MAE (real)"] = mae_real

    if training_time is not None:
        metrics["Training Time (minutes)"] = training_time / 60

    # Store in results_dict if provided
    if results_dict is not None and model_name is not None:
        results_dict[model_name] = metrics

    if logger:
        logger.info("Evaluation metrics stored successfully.")

    return metrics

import time
import logging
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config.config import (
    RANDOM_SEED,
    LOSS_FUNCTION,
    OPTIMIZER,
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    DELTA,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
)
from src.utils.seed_utils import set_seed
from src.training.early_stopping import EarlyStopping


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    target_scaler=None,
    device: torch.device = "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu",
    n_epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    loss_fn: str = LOSS_FUNCTION,
    optimizer_fn: str = OPTIMIZER,
    patience: int = EARLY_STOPPING_PATIENCE,
    delta: float = DELTA,
    early_stopping=None,
    use_lr_scheduler: bool = False,
    scheduler_patience: int = SCHEDULER_PATIENCE,
    scheduler_factor: float = SCHEDULER_FACTOR,
    seed: Optional[int] = RANDOM_SEED,
    log: bool = False,
    multi_input: bool = False,
    logger: logging.Logger = None,
) -> Tuple[nn.Module, dict, float]:
    """
    Train a PyTorch regression model with early stopping.
    Supports both single-input and multi-input architectures.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        target_scaler: Fitted scaler for inverse transforming predictions (None if log=True)
        device: torch device (mps, cuda or cpu)
        n_epochs: Maximum number of epochs
        lr: Learning rate
        loss_fn: Loss function ('mse' for Mean Squared Error, 'mae' for Mean Absolute Error)
        optimizer_fn: Optimizer function ('adam' or 'sgd')
        patience: Early stopping patience
        delta: Minimum change to qualify as improvement
        early_stopping: EarlyStopping object (if None, creates one)
        use_lr_scheduler: Whether to use ReduceLROnPlateau scheduler
        scheduler_patience: Patience for learning rate scheduler
        scheduler_factor: Factor to reduce learning rate by
        seed: Random seed for reproducibility (None to skip seeding)
        log: If True, y is log-transformed; real metrics computed using expm1
        multi_input: If True, expects (embeddings, tabular, target) batches;
                     if False, expects (features, target) batches
        logger: Optional logger for logging information

    Returns:
        Tuple of (trained_model, history_dict, elapsed_time)
    """
    # Set seed for reproducibility
    if seed is not None:
        set_seed(seed)

    # Move model to device
    model = model.to(device)

    # Setup loss function
    if loss_fn == "mse":
        loss_function = nn.MSELoss()
    elif loss_fn == "mae":
        loss_function = nn.L1Loss()
    else:
        logger.error("Unsupported loss function. Use 'mse' or 'mae'.")
        raise ValueError("Unsupported loss function. Use 'mse' or 'mae'.")

    # Setup optimizer
    if optimizer_fn == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_fn == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        logger.error("Unsupported optimizer. Use 'adam' or 'sgd'.")
        raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")

    # Initialize learning rate scheduler if requested
    scheduler = None
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
        )

    # Initialize early stopping if not provided
    if early_stopping is None:
        early_stopping = EarlyStopping(
            patience=patience,
            delta=delta,
            verbose=True,
            restore_best_weights=True,
            logger=logger,
        )

    # Initialize history dict based on loss function
    if loss_fn == "mse":
        history = {
            "train_loss_scaled": [],
            "valid_loss_scaled": [],
            "train_mse_real": [],
            "valid_mse_real": [],
            "train_rmse": [],
            "valid_rmse": [],
        }
    else:  # mae
        history = {
            "train_loss_scaled": [],
            "valid_loss_scaled": [],
            "train_mae_real": [],
            "valid_mae_real": [],
        }

    # Logger info
    if logger:
        logger.info(f"Starting training for {n_epochs} epochs on device: {device}")
        logger.info(
            f"Using loss function: {loss_fn}, optimizer: {optimizer_fn} with lr={lr}"
        )
        if use_lr_scheduler:
            logger.info(
                f"Learning rate scheduler with patience {scheduler_patience} and factor {scheduler_factor} enabled."
            )
        logger.info(f"Early stopping patience: {patience}, delta: {delta}")
        if log:
            logger.info("Log mode: Real metrics will be computed using expm1.")
        else:
            logger.info(
                "Real metrics will be computed using target_scaler inverse transform."
            )
        if multi_input:
            logger.info("Multi-input architecture detected.")
        else:
            logger.info("Single-input architecture detected.")

    start_time = time.time()

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss_scaled = 0.0
        train_loss_real = 0.0

        for batch in train_loader:
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

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            if multi_input:
                predictions = model(embeddings_batch, tabular_batch).squeeze()
            else:
                predictions = model(X_batch).squeeze()

            # Compute loss
            loss = loss_function(predictions, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss_scaled += loss.item() * batch_size

            # Compute metric in real scale
            if log:
                predictions_real = np.expm1(predictions.detach().cpu().numpy())
                y_batch_real = np.expm1(y_batch.detach().cpu().numpy())
            else:
                if target_scaler is None:
                    raise ValueError("target_scaler must be provided when log=False")
                predictions_real = target_scaler.inverse_transform(
                    predictions.detach().cpu().numpy().reshape(-1, 1)
                ).ravel()
                y_batch_real = target_scaler.inverse_transform(
                    y_batch.detach().cpu().numpy().reshape(-1, 1)
                ).ravel()

            if loss_fn == "mse":
                metric_real = np.mean((predictions_real - y_batch_real) ** 2)
            else:  # mae
                metric_real = np.mean(np.abs(predictions_real - y_batch_real))

            train_loss_real += metric_real * batch_size

        # Calculate average training losses
        train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
        train_loss_real_avg = train_loss_real / len(train_loader.dataset)

        if loss_fn == "mse":
            train_rmse = np.sqrt(train_loss_real_avg)

        # Validation phase
        model.eval()
        valid_loss_scaled = 0.0
        valid_loss_real = 0.0

        with torch.no_grad():
            for batch in valid_loader:
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

                # Compute loss
                loss = loss_function(predictions, y_batch)
                valid_loss_scaled += loss.item() * batch_size

                # Compute metric in real scale
                if log:
                    predictions_real = np.expm1(predictions.detach().cpu().numpy())
                    y_batch_real = np.expm1(y_batch.detach().cpu().numpy())
                else:
                    if target_scaler is None:
                        logger.error("target_scaler must be provided when log=False")
                        raise ValueError(
                            "target_scaler must be provided when log=False"
                        )
                    predictions_real = target_scaler.inverse_transform(
                        predictions.detach().cpu().numpy().reshape(-1, 1)
                    ).ravel()
                    y_batch_real = target_scaler.inverse_transform(
                        y_batch.detach().cpu().numpy().reshape(-1, 1)
                    ).ravel()

                if loss_fn == "mse":
                    metric_real = np.mean((predictions_real - y_batch_real) ** 2)
                else:  # mae
                    metric_real = np.mean(np.abs(predictions_real - y_batch_real))

                valid_loss_real += metric_real * batch_size

        # Calculate average validation losses
        valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
        valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)

        if loss_fn == "mse":
            valid_rmse = np.sqrt(valid_loss_real_avg)

        # Store metrics
        history["train_loss_scaled"].append(train_loss_scaled_avg)
        history["valid_loss_scaled"].append(valid_loss_scaled_avg)

        if loss_fn == "mse":
            history["train_mse_real"].append(train_loss_real_avg)
            history["valid_mse_real"].append(valid_loss_real_avg)
            history["train_rmse"].append(train_rmse)
            history["valid_rmse"].append(valid_rmse)
        else:  # mae
            history["train_mae_real"].append(train_loss_real_avg)
            history["valid_mae_real"].append(valid_loss_real_avg)

        # Log epoch results
        if logger:
            logger.info(f"Epoch {epoch + 1}/{n_epochs}:")
            if loss_fn == "mse":
                logger.info(
                    f"  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {train_rmse:.2f}"
                )
                logger.info(
                    f"  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {valid_rmse:.2f}"
                )
            else:  # mae
                logger.info(
                    f"  Train - MAE: {train_loss_scaled_avg:.4f}, Real MAE: {train_loss_real_avg:.2f}"
                )
                logger.info(
                    f"  Valid - MAE: {valid_loss_scaled_avg:.4f}, Real MAE: {valid_loss_real_avg:.2f}"
                )

        # Step the learning rate scheduler if enabled
        if scheduler is not None:
            scheduler.step(valid_loss_scaled_avg)
            current_lr = optimizer.param_groups[0]["lr"]
            if logger:
                logger.info(f"  Learning Rate: {current_lr:.6f}")

        # Check early stopping
        early_stopping.check_early_stop(valid_loss_scaled_avg, model)
        if early_stopping.stop_training:
            if logger:
                logger.info("Early stopping triggered. Restoring best model weights.")
            break

    elapsed_time = time.time() - start_time
    if logger:
        logger.info(f"Training completed in {elapsed_time / 60:.2f} minutes.")

    return model, history, elapsed_time

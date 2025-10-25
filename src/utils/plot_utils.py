import torch.nn as nn
from pathlib import Path
from typing import  Optional
import matplotlib.pyplot as plt
import logging
import numpy as np


def plot_losses_from_lists(
    train_losses: list[float],
    valid_losses: list[float],
    train_real_losses: Optional[list[float]] = None,
    valid_real_losses: Optional[list[float]] = None,
    save_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Plot training and validation losses.

    This function can plot either the scaled losses or both scaled and real-scale
    losses side-by-side, depending on whether the real-scale losses are provided.

    Args:
        train_losses: list[float], A list of scaled training loss values, one for each epoch.
        valid_losses: list[float], A list of scaled validation loss values, one for each epoch.
        train_real_losses: Optional[list[float]], An optional list of real-scale training loss values.
                           Defaults to None.
        valid_real_losses: Optional[list[float]], An optional list of real-scale validation loss values.
                           Defaults to None.
        save_path: Optional[str], path to save the plot image. If None, the plot is shown
                   but not saved.
        logger: Optional[logging.Logger], logger instance for output.
    """
    if train_real_losses is None or valid_real_losses is None:
        # Only plot scaled losses
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, color="blue", label="Train Loss", linewidth=2)
        ax.plot(
            epochs, valid_losses, color="orange", label="Validation Loss", linewidth=2
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (Scaled MSE)", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    else:
        # Plot both scaled and real losses
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        epochs = range(1, len(train_losses) + 1)

        # Plot 1: Scaled MSE
        axes[0].plot(
            epochs, train_losses, color="blue", label="Train Loss", linewidth=2
        )
        axes[0].plot(
            epochs, valid_losses, color="orange", label="Validation Loss", linewidth=2
        )
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss (Scaled MSE)", fontsize=11)
        axes[0].set_title("Scaled MSE Loss", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Real MSE
        axes[1].plot(
            epochs, train_real_losses, color="blue", label="Train Loss", linewidth=2
        )
        axes[1].plot(
            epochs,
            valid_real_losses,
            color="orange",
            label="Validation Loss",
            linewidth=2,
        )
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Loss (Real MSE)", fontsize=11)
        axes[1].set_title("Real Scale MSE Loss", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if logger:
            logger.info(f"Loss plot saved to {save_path}.")
    plt.show()


def plot_losses_curves(
    history_dict: dict,
    loss_fn: str = "mse",
    save_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Plot training and validation losses from a history dictionary.
    Left plot shows scaled losses, right plot shows real-scale losses.

    Args:
        history_dict: dict, Dictionary containing loss histories
        loss_fn: str, 'mse' or 'mae' - determines which losses to plot
        save_path: Optional[str], path to save the plot image. If None, the plot is shown
                   but not saved.
        logger: Optional[logging.Logger], logger instance for output.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if loss_fn == "mse":
        # Plot 1: Scaled MSE (left)
        epochs = range(1, len(history_dict["train_loss_scaled"]) + 1)
        axes[0].plot(
            epochs,
            history_dict["train_loss_scaled"],
            color="blue",
            label="Train Loss",
            linewidth=2,
        )
        axes[0].plot(
            epochs,
            history_dict["valid_loss_scaled"],
            color="orange",
            label="Validation Loss",
            linewidth=2,
        )
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss (Scaled MSE)", fontsize=11)
        axes[0].set_title("Scaled MSE Loss", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Real MSE (right)
        axes[1].plot(
            epochs,
            history_dict["train_mse_real"],
            color="blue",
            label="Train Loss",
            linewidth=2,
        )
        axes[1].plot(
            epochs,
            history_dict["valid_mse_real"],
            color="orange",
            label="Validation Loss",
            linewidth=2,
        )
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Loss (Real MSE)", fontsize=11)
        axes[1].set_title("Real Scale MSE Loss", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    else:  # mae
        # Plot 1: Scaled MAE (left)
        epochs = range(1, len(history_dict["train_loss_scaled"]) + 1)
        axes[0].plot(
            epochs,
            history_dict["train_loss_scaled"],
            color="blue",
            label="Train Loss",
            linewidth=2,
        )
        axes[0].plot(
            epochs,
            history_dict["valid_loss_scaled"],
            color="orange",
            label="Validation Loss",
            linewidth=2,
        )
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss (Scaled MAE)", fontsize=11)
        axes[0].set_title("Scaled MAE Loss", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Real MAE (right)
        axes[1].plot(
            epochs,
            history_dict["train_mae_real"],
            color="blue",
            label="Train Loss",
            linewidth=2,
        )
        axes[1].plot(
            epochs,
            history_dict["valid_mae_real"],
            color="orange",
            label="Validation Loss",
            linewidth=2,
        )
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Loss (Real MAE)", fontsize=11)
        axes[1].set_title("Real Scale MAE Loss", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if logger:
            logger.info(f"Loss curves plot saved to {save_path}.")
    plt.show()


def plot_top_models_comparison(
    results_dict: dict,
    top_n: int = 10,
    figsize: tuple = (16, 6),
    save_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Creates side-by-side bar plots comparing models by Test MSE (scaled) and Test RMSE (real).

    Args:
        results_dict: dict, Dictionary with model names as keys and metrics as values
        top_n: int, Number of top models to display (default: 10)
        figsize: tuple, Figure size as (width, height) tuple
        save_path: Optional[str], path to save the plot image. If None, the plot is shown
                   but not saved.
        logger: Optional[logging.Logger], logger instance for output (default: None, no output)
    """
    # Extract data from results dictionary
    model_names = list(results_dict.keys())
    mse_scaled = [results_dict[name]["Test MSE (scaled)"] for name in model_names]
    rmse_real = [float(results_dict[name]["Test RMSE (real)"]) for name in model_names]

    # Create dataframe-like structure for sorting
    data = list(zip(model_names, mse_scaled, rmse_real))

    # Sort by MSE (scaled) and get top N
    data_sorted_mse = sorted(data, key=lambda x: x[1])[:top_n]
    top_names_mse = [x[0] for x in data_sorted_mse]
    top_mse_scaled = [x[1] for x in data_sorted_mse]
    top_mse_real = [results_dict[x]["Test MSE (real)"] for x in top_names_mse]
    top_mse_time = [results_dict[x]["Training Time (minutes)"] for x in top_names_mse]

    # Sort by RMSE (real) and get top N
    data_sorted_rmse = sorted(data, key=lambda x: x[2])[:top_n]
    top_names_rmse = [x[0] for x in data_sorted_rmse]
    top_rmse_real = [x[2] for x in data_sorted_rmse]
    top_rmse_time = [results_dict[x]["Training Time (minutes)"] for x in top_names_rmse]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Top 10 by Test MSE (scaled)
    y_pos_1 = np.arange(len(top_names_mse))
    bars1 = ax1.barh(y_pos_1, top_mse_scaled, color="steelblue", alpha=0.8)
    ax1.set_yticks(y_pos_1)
    # Add training time to labels
    labels_mse = [
        f"{name} ({time:.2f}min)" for name, time in zip(top_names_mse, top_mse_time)
    ]
    ax1.set_yticklabels(labels_mse, fontsize=9)
    ax1.invert_yaxis()  # Best model at top
    ax1.set_xlabel("Test MSE (scaled)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Top {top_n} Models by Test MSE (Scaled)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, top_mse_scaled)):
        ax1.text(val, i, f" {val:.4f}", va="center", fontsize=8)

    # Right plot: Top 10 by Test RMSE (real)
    y_pos_2 = np.arange(len(top_names_rmse))
    bars2 = ax2.barh(y_pos_2, top_rmse_real, color="coral", alpha=0.8)
    ax2.set_yticks(y_pos_2)
    # Add training time to labels
    labels_rmse = [
        f"{name} ({time:.2f}min)" for name, time in zip(top_names_rmse, top_rmse_time)
    ]
    ax2.set_yticklabels(labels_rmse, fontsize=9)
    ax2.invert_yaxis()  # Best model at top
    ax2.set_xlabel("Test RMSE (real)", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Top {top_n} Models by Test RMSE (Real Scale)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax2.grid(axis="x", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, top_rmse_real)):
        ax2.text(val, i, f" {val:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if logger:
            logger.info(f"Top models comparison plot saved to {save_path}.")
    plt.show()

    # Print summary statistics only if logger is provided
    if logger is not None:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"TOP {top_n} MODELS SUMMARY")
        logger.info(f"{'=' * 70}")

        # Best model by scaled MSE
        best_mse_name = top_names_mse[0]
        best_mse_metrics = results_dict[best_mse_name]
        logger.info(f"\nBest Model (by scaled MSE): {best_mse_name}")
        logger.info(f"  All Metrics:")
        for metric_name, metric_value in best_mse_metrics.items():
            if isinstance(metric_value, (np.floating, np.integer)):
                metric_value = float(metric_value)
            if "scaled" in metric_name.lower():
                logger.info(f"    - {metric_name}: {metric_value:.6f}")
            elif "time" in metric_name.lower():
                logger.info(f"    - {metric_name}: {metric_value:.2f} min")
            else:
                logger.info(f"    - {metric_name}: {metric_value:,.2f}")

        # Best model by real RMSE
        best_rmse_name = top_names_rmse[0]
        best_rmse_metrics = results_dict[best_rmse_name]
        logger.info(f"\nBest Model (by real RMSE): {best_rmse_name}")
        logger.info(f"  All Metrics:")
        for metric_name, metric_value in best_rmse_metrics.items():
            if isinstance(metric_value, (np.floating, np.integer)):
                metric_value = float(metric_value)
            if "scaled" in metric_name.lower():
                logger.info(f"    - {metric_name}: {metric_value:.6f}")
            elif "time" in metric_name.lower():
                logger.info(f"    - {metric_name}: {metric_value:.2f} min")
            else:
                logger.info(f"    - {metric_name}: {metric_value:,.2f}")

        logger.info(f"{'=' * 70}\n")


def plot_single_model_report(
    model: nn.Module,
    history: dict,
    elapsed_time: float,
    test_metrics: dict,
    model_name: str,
    loss_fn: str = "mse",
    save_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create comprehensive visualization report for a single model.
    Generates 4 plots: RMSE comparison, parameters, training time, and overfitting analysis.

    Args:
        model: nn.Module, Trained PyTorch model
        history: dict, History dictionary from train_model function
        elapsed_time: float, Training time in seconds from train_model function
        test_metrics: dict, Test metrics dictionary from evaluate_model function
        model_name: str, Name/identifier for the model
        loss_fn: str, Loss function used ('mse' or 'mae')
        save_dir: Optional[str], Directory to save plots. If None, plots are only displayed.
        logger: Optional[logging.Logger], logger instance for output
    """

    # Count trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get number of epochs trained
    epochs_trained = len(history["train_loss_scaled"])

    # Find best validation epoch
    if loss_fn == "mse":
        best_valid_idx = np.argmin(history["valid_rmse"])
        train_rmse = history["train_rmse"][best_valid_idx]
        valid_rmse = history["valid_rmse"][best_valid_idx]
        test_rmse = test_metrics["Test RMSE (real)"]
        metric_name = "RMSE"
    else:  # mae
        best_valid_idx = np.argmin(history["valid_mae_real"])
        train_rmse = history["train_mae_real"][best_valid_idx]
        valid_rmse = history["valid_mae_real"][best_valid_idx]
        test_rmse = test_metrics["Test MAE (real)"]
        metric_name = "MAE"

    best_epoch = best_valid_idx + 1
    training_time_min = elapsed_time / 60

    # Create save directory if specified
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        if logger:
            logger.info(f"Saving plots to: {save_path}")

    # Plot 1: Train/Valid/Test RMSE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array([0])
    width = 0.25

    bars1 = ax.bar(
        x - width,
        [train_rmse],
        width,
        label=f"Train {metric_name}",
        alpha=0.8,
        color="skyblue",
    )
    bars2 = ax.bar(
        x,
        [valid_rmse],
        width,
        label=f"Valid {metric_name}",
        alpha=0.8,
        color="lightcoral",
    )
    bars3 = ax.bar(
        x + width,
        [test_rmse],
        width,
        label=f"Test {metric_name}",
        alpha=0.8,
        color="lightgreen",
    )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{model_name}: Train/Valid/Test {metric_name} Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks([0])
    ax.set_xticklabels([model_name])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    ax.text(
        x - width,
        train_rmse,
        f"{train_rmse:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        x,
        valid_rmse,
        f"{valid_rmse:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        x + width,
        test_rmse,
        f"{test_rmse:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    if save_dir:
        plt.savefig(
            save_path / f"{model_name}_rmse_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        if logger:
            logger.info(f"Saved: {model_name}_rmse_comparison.png")
    plt.show()

    # Plot 2: Model Complexity (Parameters)
    fig, ax = plt.subplots(figsize=(10, 6))

    bar = ax.barh([model_name], [n_params], color="steelblue", alpha=0.7)
    ax.set_xlabel("Number of Parameters", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{model_name}: Model Complexity", fontsize=12, fontweight="bold", pad=15
    )
    ax.grid(axis="x", alpha=0.3)

    # Add value label
    ax.text(
        n_params,
        0,
        f" {n_params:,}",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    if save_dir:
        plt.savefig(
            save_path / f"{model_name}_parameters.png", dpi=300, bbox_inches="tight"
        )
        if logger:
            logger.info(f"Saved: {model_name}_parameters.png")
    plt.show()

    # Plot 3: Training Time
    fig, ax = plt.subplots(figsize=(10, 6))

    bar = ax.barh([model_name], [training_time_min], color="coral", alpha=0.7)
    ax.set_xlabel("Training Time (minutes)", fontsize=11, fontweight="bold")
    ax.set_title(f"{model_name}: Training Time", fontsize=12, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    # Add value label
    ax.text(
        training_time_min,
        0,
        f" {training_time_min:.2f} min",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    if save_dir:
        plt.savefig(
            save_path / f"{model_name}_training_time.png", dpi=300, bbox_inches="tight"
        )
        if logger:
            logger.info(f"Saved: {model_name}_training_time.png")
    plt.show()

    # Plot 4: Overfitting Analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    overfit_gap = valid_rmse - train_rmse
    bar_color = "coral" if overfit_gap > 0 else "lightgreen"

    bar = ax.barh([model_name], [overfit_gap], color=bar_color, alpha=0.7)
    ax.set_xlabel(
        f"Overfitting Gap (Valid {metric_name} - Train {metric_name})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"{model_name}: Overfitting Analysis", fontsize=12, fontweight="bold", pad=15
    )
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add value label
    text_ha = "left" if overfit_gap > 0 else "right"
    ax.text(
        overfit_gap,
        0,
        f" {overfit_gap:.2f}" if overfit_gap > 0 else f"{overfit_gap:.2f} ",
        va="center",
        ha=text_ha,
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    if save_dir:
        plt.savefig(
            save_path / f"{model_name}_overfitting.png", dpi=300, bbox_inches="tight"
        )
        if logger:
            logger.info(f"Saved: {model_name}_overfitting.png")
    plt.show()

    # Summary Statistics
    if logger:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"MODEL REPORT: {model_name}")
        logger.info(f"{'=' * 70}")
        logger.info(f"Parameters: {n_params:,}")
        logger.info(f"Training Time: {training_time_min:.2f} minutes")
        logger.info(f"Epochs Trained: {epochs_trained}")
        logger.info(f"Best Epoch: {best_epoch}")
        logger.info(f"Train {metric_name}: {train_rmse:.2f}")
        logger.info(f"Valid {metric_name}: {valid_rmse:.2f}")
        logger.info(f"Test {metric_name}: {test_rmse:.2f}")
        logger.info(f"Overfitting Gap: {overfit_gap:.2f}")
        logger.info(f"{'=' * 70}\n")

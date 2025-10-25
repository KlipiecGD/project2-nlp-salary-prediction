import logging
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.datasets.salary_dataset import SalaryDataset
from src.datasets.multi_input_dataset import MultiInputDataset


def combine_features(
    text_embeddings: dict[str, np.ndarray],
    categorical_train: np.ndarray,
    categorical_valid: np.ndarray,
    categorical_test: np.ndarray,
    logger: logging.Logger = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]:
    """
    Combines text embeddings and categorical features.

    Args:
        text_embeddings: dict[str, np.ndarray], Dictionary with text embeddings.
        categorical_train: np.ndarray, Processed categorical features for training.
        categorical_valid: np.ndarray, Processed categorical features for validation.
        categorical_test: np.ndarray, Processed categorical features for test.
        multi_input: bool, Whether to track tabular start index for multi-input models.
        logger: Optional[logging.Logger], Optional logger for logging information.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]],
            Tuple of (X_train, X_valid, X_test, tabular_start_index).
    """
    X_train = np.hstack([text_embeddings["X_train_text"], categorical_train])
    X_valid = np.hstack([text_embeddings["X_valid_text"], categorical_valid])
    X_test = np.hstack([text_embeddings["X_test_text"], categorical_test])

    tabular_start_index = text_embeddings["X_train_text"].shape[1]

    if logger:
        logger.info("Features combined successfully.")

    return X_train, X_valid, X_test, tabular_start_index


def create_datasets(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    multi_input: bool = False,
    tabular_start_index: Optional[int] = None,
    logger: logging.Logger = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Creates PyTorch datasets.

    Args:
        X_train: np.ndarray, Training features.
        X_valid: np.ndarray, Validation features.
        X_test: np.ndarray, Test features.
        y_train: np.ndarray, Training targets.
        y_valid: np.ndarray, Validation targets.
        y_test: np.ndarray, Test targets.
        multi_input: bool, Whether to create MultiInputDataset or standard SalaryDataset, default False.
        tabular_start_index: Optional[int], Index where tabular features start in X (required if multi_input is True).
        logger: Optional[logging.Logger], Optional logger for logging information.

    Returns:
        tuple[Dataset, Dataset, Dataset]
            Tuple of (train_dataset, valid_dataset, test_dataset).
    """
    if multi_input:
        train_dataset = MultiInputDataset(
            X_train, y_train, tabular_start_index=tabular_start_index
        )
        valid_dataset = MultiInputDataset(
            X_valid, y_valid, tabular_start_index=tabular_start_index
        )
        test_dataset = MultiInputDataset(
            X_test, y_test, tabular_start_index=tabular_start_index
        )

        if logger:
            logger.info(
                f"MultiInputDataset created with tabular_start_index={tabular_start_index}."
            )
    else:
        train_dataset = SalaryDataset(X_train, y_train)
        valid_dataset = SalaryDataset(X_valid, y_valid)
        test_dataset = SalaryDataset(X_test, y_test)
        if logger:
            logger.info("Standard SalaryDataset created.")

    return train_dataset, valid_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    seed_worker: Optional[callable] = None,
    generator: Optional[torch.Generator] = None,
    logger: logging.Logger = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch dataloaders.

    Args:
        train_dataset: Dataset, Training dataset.
        valid_dataset: Dataset, Validation dataset.
        test_dataset: Dataset, Test dataset.
        batch_size: int, Batch size for dataloaders, default 64.
        num_workers: int, Number of worker processes.
        seed_worker: Optional[callable], Function to seed workers for reproducibility.
        generator: Optional[torch.Generator], Random generator for reproducibility.
        logger: Optional[logging.Logger], Optional logger for logging information.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]
            Tuple of (train_loader, valid_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    if logger:
        logger.info(f"DataLoaders created with batch_size={batch_size}")

    return train_loader, valid_loader, test_loader

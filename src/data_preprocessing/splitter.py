import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    valid_size: float = 0.5,
    random_state: int = 42,
    log: bool = False,
    logger: logging.Logger = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits data into training, validation, and test sets.

    Args:
        df: pd.DataFrame, Input dataframe.
        test_size: float, Proportion of the dataset to include in the test split, default is 0.2.
        valid_size: float, Proportion of the temporary split to include in the validation split, default is 0.5.
        random_state: int, Seed for reproducible train/test splits, default is 42.
        log: bool, Whether to apply log transformation to the target variable, default is False.
        logger: Optional[logging.Logger], Optional logger for logging information.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        A tuple containing:
        - X_train (pd.DataFrame): Training features.
        - X_valid (pd.DataFrame): Validation features.
        - X_test (pd.DataFrame): Test features.
        - y_train (pd.Series): Training target.
        - y_valid (pd.Series): Validation target.
        - y_test (pd.Series): Test target.
    """
    # Separate features (X) and target (y)
    X = df[
        [
            "Title",
            "FullDescription",
            "Category",
            "Company",
            "LocationNormalized",
            "ContractType",
            "ContractTime",
            "SourceName",
        ]
    ]

    if log:
        y = np.log1p(
            df["SalaryNormalized"]
        )  # log(1 + x) to handle zero salaries if any
    else:
        y = df["SalaryNormalized"]

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=valid_size, random_state=random_state
    )

    if logger:
        logger.info(f"Data split into train, validation, and test sets.")
        logger.info(f"Training set shape: {X_train.shape}, {y_train.shape}")
        logger.info(f"Validation set shape: {X_valid.shape}, {y_valid.shape}")
        logger.info(f"Test set shape: {X_test.shape}, {y_test.shape}")
        if log:
            logger.info("Target variable 'SalaryNormalized' has been log-transformed.")

    return X_train, X_valid, X_test, y_train, y_valid, y_test

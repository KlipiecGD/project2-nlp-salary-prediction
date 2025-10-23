import logging
import os
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder


def preprocess_categorical_data(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    categorical_columns: List[str] = ["Category", "ContractType", "ContractTime"],
    high_cardinality_columns: List[str] = [
        "Company",
        "LocationNormalized",
        "SourceName",
    ],
    categorical_na_strategy: str = "constant",
    categorical_fill_value: str = "unknown",
    high_card_na_strategy: str = "constant",
    high_card_fill_value: str = "unknown",
    logger: logging.Logger = None,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses categorical features using one-hot encoding and target encoding.

    Args:
        X_train: Training data.
        X_valid: Validation data.
        X_test: Test data.
        y_train: Training target.
        categorical_columns: Columns for one-hot encoding.
        high_cardinality_columns: Columns for target encoding.
        categorical_na_strategy: Imputation strategy for categorical columns.
        categorical_fill_value: Fill value for categorical columns.
        high_card_na_strategy: Imputation strategy for high-cardinality columns.
        high_card_fill_value: Fill value for high-cardinality columns.
        logger: Optional logger for logging information.

    Returns:
        Tuple of (transformed X_train, X_valid, X_test, fitted ColumnTransformer).
    """
    # Setup categorical imputer
    if categorical_na_strategy == "constant":
        cat_imputer = SimpleImputer(
            strategy="constant", fill_value=categorical_fill_value
        )
    else:
        cat_imputer = SimpleImputer(strategy="most_frequent")

    one_hot_pipeline = Pipeline(
        [
            ("imputer", cat_imputer),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )

    # Setup high cardinality imputer
    if high_card_na_strategy == "constant":
        high_card_imputer = SimpleImputer(
            strategy="constant", fill_value=high_card_fill_value
        )
    else:
        high_card_imputer = SimpleImputer(strategy="most_frequent")

    target_pipeline = Pipeline(
        [
            ("imputer", high_card_imputer),
            ("target_enc", TargetEncoder()),
            ("scaler", StandardScaler()),
        ]
    )

    # Combine into ColumnTransformer
    categorical_preprocessor = ColumnTransformer(
        [
            ("onehot", one_hot_pipeline, categorical_columns),
            ("target_scaled", target_pipeline, high_cardinality_columns),
        ]
    )

    # Fit and transform
    X_train_cat = categorical_preprocessor.fit_transform(X_train, y_train)
    X_valid_cat = categorical_preprocessor.transform(X_valid)
    X_test_cat = categorical_preprocessor.transform(X_test)

    if logger:
        logger.info("Categorical features preprocessed successfully.")

    return X_train_cat, X_valid_cat, X_test_cat, categorical_preprocessor

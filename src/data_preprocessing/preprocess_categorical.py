import logging
import os
from typing import Optional
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
    categorical_columns: Optional[list[str]] = ["Category", "ContractType", "ContractTime"],
    high_cardinality_columns: Optional[list[str]] = [
        "Company",
        "LocationNormalized",
        "SourceName",
    ],
    categorical_na_strategy: Optional[str] = "constant",
    categorical_fill_value: Optional[str] = "unknown",
    high_card_na_strategy: Optional[str] = "constant",
    high_card_fill_value: Optional[str] = "unknown",
    logger: logging.Logger = None,
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses categorical features using one-hot encoding and target encoding.

    Args:
        X_train (pd.DataFrame): Training feature dataframe.
        X_valid (pd.DataFrame): Validation feature dataframe.
        X_test (pd.DataFrame): Test feature dataframe.
        y_train (pd.Series): Training target series.
        categorical_columns: Optional[list[str]]: Columns to apply one-hot encoding, default: ["Category", "ContractType", "ContractTime"].
        high_cardinality_columns: Optional[list[str]]: Columns to apply target encoding, default: ["Company", "LocationNormalized", "SourceName"].
        categorical_na_strategy: Optional[str]: Imputation strategy for categorical_columns.
            If "constant", uses `categorical_fill_value`; otherwise uses "most_frequent", default: "constant".
        categorical_fill_value: Optional[str]: Fill value used when categorical_na_strategy is "constant", default: "unknown".
        high_card_na_strategy: Optional[str]: Imputation strategy for high_cardinality_columns.
            If "constant", uses `high_card_fill_value`; otherwise uses "most_frequent", default: "constant".
        high_card_fill_value: Optional[str]: Fill value used when high_card_na_strategy is "constant", default: "unknown".
        logger: Optional[logging.Logger]: Logger for informational messages. Default: None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
            A tuple containing:
            - X_train_cat (np.ndarray): Transformed training categorical features.
            - X_valid_cat (np.ndarray): Transformed validation categorical features.
            - X_test_cat (np.ndarray): Transformed test categorical features.
            - categorical_preprocessor (ColumnTransformer): Fitted ColumnTransformer used for preprocessing.
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

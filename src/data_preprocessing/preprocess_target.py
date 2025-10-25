import os
import logging
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Any, Optional


def preprocess_target(
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    log: Optional[bool] = False,
    save_artifacts: Optional[bool] = True,
    preprocessor_dir: Optional[str] = "fitted_preprocessors",
    artifact_prefix: Optional[str] = "",
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """
    Preprocess target values for salary prediction.

    Args:
        y_train: (pd.Series), Training target values.
        y_valid: (pd.Series), Validation target values.
        y_test: (pd.Series), Test target values.
        log: Optional[bool], If True, indicates that y is already log-transformed,
            so skips StandardScaler and artifact saving. Defaults to False.
        save_artifacts: Optional[bool], Whether to save the target scaler. Defaults to True.
        preprocessor_dir: Optional[str], Directory to save preprocessor artifacts.
            Defaults to "fitted_preprocessors".
        artifact_prefix: Optional[str], Prefix for saved artifact filenames. Defaults to "".
        logger: Optional[logging.Logger], Optional logger for logging information.

    Returns:
        dict, dict containing:
            - y_train_scaled: Scaled training targets
            - y_valid_scaled: Scaled validation targets
            - y_test_scaled: Scaled test targets
            - target_scaler: Fitted StandardScaler (None if log=True)
    """

    if log:
        # y is already log-transformed, don't apply StandardScaler
        y_train_scaled = y_train.values
        y_valid_scaled = y_valid.values
        y_test_scaled = y_test.values
        target_scaler = None
        if logger:
            logger.info("Log mode: Skipping StandardScaler fitting and artifact saving")
    else:
        # Fit StandardScaler on training targets
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(
            y_train.values.reshape(-1, 1)
        ).ravel()
        y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # Save target scaler artifact (skip if log=True)
    if save_artifacts and not log:
        # Create directory if it doesn't exist
        os.makedirs(preprocessor_dir, exist_ok=True)

        scaler_path = os.path.join(
            preprocessor_dir, f"{artifact_prefix}target_scaler.pkl"
        )
        joblib.dump(target_scaler, scaler_path)

        if logger:
            logger.info(f"Target scaler saved to: {scaler_path}")

    return (y_train_scaled, y_valid_scaled, y_test_scaled, target_scaler)

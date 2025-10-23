import os
import logging
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
from config.config import PREPROCESSORS_DIR


def preprocess_target(
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    log: bool = False,
    save_artifacts: bool = True,
    preprocessor_dir: str = PREPROCESSORS_DIR,
    artifact_prefix: str = "",
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """
    Preprocess target values for salary prediction.

    Args:
        y_train: Training target values
        y_valid: Validation target values
        y_test: Test target values
        log: If True, indicates that y is already log-transformed,
             so skips StandardScaler and artifact saving
        save_artifacts: Whether to save the target scaler
        preprocessor_dir: Directory to save preprocessor artifacts
        artifact_prefix: Prefix for saved artifact filenames
        logger: Optional logger for logging information

    Returns:
        dict containing:
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
        
        scaler_path = os.path.join(preprocessor_dir, f"{artifact_prefix}target_scaler.pkl")
        joblib.dump(target_scaler, scaler_path)
        
        if logger:
            logger.info(f"Target scaler saved to: {scaler_path}")
    
    return (y_train_scaled, y_valid_scaled, y_test_scaled, target_scaler)
import logging
import pandas as pd


def load_data(
    file_path: str = "data/Train_rev1.csv", logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Loads data from a CSV and handles missing values in 'Title'.

    Args:
        file_path: str, Path to the raw CSV data file. Defaults to "data/Train_rev1.csv".
        logger: Optional[logging.Logger], Optional logger for logging information.

    Returns:
        pd.DataFrame: Loaded dataset with imputed Title values.
    """
    # Load data from the specified path
    df = pd.read_csv(file_path)

    # Impute missing value in 'Title' based on EDA findings.
    df.loc[df["Title"].isnull(), "Title"] = "Quality Improvement Manager"

    if logger:
        logger.info(f"Data successfully loaded from {file_path}. Shape: {df.shape}")

    return df

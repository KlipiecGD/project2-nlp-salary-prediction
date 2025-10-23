import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from typing import Optional


class TextEmbedder(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer to generate text embeddings using a pre-trained model.

    This class leverages the `sentence-transformers` library to convert text data into
    numerical vector representations (embeddings) suitable for machine learning models.
    """

    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        """
        Initializes the TextEmbedder.

        Args:
            model_name: The name of the pre-trained SentenceTransformer model to use.
                        Defaults to 'all-MiniLM-L12-v2'.
        """
        self.model_name = model_name
        self.model = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # The model is pre-trained, so we just load it here.
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Convert input to a list of strings and handle potential NaNs.
        X = [str(x) if x is not None else "" for x in X.iloc[:, 0].tolist()]

        # Generate the embeddings.
        embeddings = self.model.encode(X, show_progress_bar=False)
        return embeddings

    def _check_n_features(self, X, reset):
        """Ensure compatibility with sklearn's feature validation"""
        pass

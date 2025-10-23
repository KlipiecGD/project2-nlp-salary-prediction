import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import Optional, List
from config.config import RANDOM_SEED, TITLE_COLUMN


class TfidfTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn compatible transformer for applying TF-IDF vectorization
    and optional Singular Value Decomposition (SVD) for dimensionality reduction.

    This transformer is designed to process a specific text column in a pandas DataFrame,
    convert it into a TF-IDF matrix, and can optionally reduce its dimensions using SVD.
    """

    def __init__(
        self,
        text_column: str = TITLE_COLUMN,
        max_features: int = 50,
        use_svd: bool = False,
        n_components: int = 10,
        stop_words: Optional[List[str] | str] = None,
        random_state: int = RANDOM_SEED,
    ):
        """
        Initializes the TfidfTransformer.

        Args:
            text_column: The name of the DataFrame column containing the text data.
                         Defaults to 'Title'.
            max_features: The maximum number of features (tokens) to be considered by
                          the TfidfVectorizer. Defaults to 50.
            use_svd: If True, applies TruncatedSVD for dimensionality reduction.
                     Defaults to False.
            n_components: The number of components to keep after SVD. This is only
                          used if `use_svd` is True. Defaults to 10.
            stop_words: A list of stop words or a string indicating a language
                        (e.g., 'english'). Passed directly to TfidfVectorizer.
                        Defaults to None.
            random_state: Random seed for reproducibility, used in SVD.
                           Defaults to RANDOM_SEED.
        """
        self.text_column = text_column
        self.max_features = max_features
        self.use_svd = use_svd
        self.n_components = n_components
        self.stop_words = stop_words
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            ngram_range=(1, 2),
            min_df=5,
        )

        text_data = X[self.text_column].fillna("")
        tfidf_matrix = self.vectorizer_.fit_transform(text_data)

        if self.use_svd:
            self.svd_ = TruncatedSVD(
                n_components=self.n_components, random_state=self.random_state
            )
            self.svd_.fit(tfidf_matrix)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        # Check for fitted attribute
        if not hasattr(self, "vectorizer_"):
            raise RuntimeError(
                "This TfidfTransformer instance is not fitted yet. Call 'fit' first."
            )

        text_data = X[self.text_column].fillna("")
        tfidf_matrix = self.vectorizer_.transform(text_data)

        if self.use_svd:
            tfidf_matrix = self.svd_.transform(tfidf_matrix)
            return tfidf_matrix
        else:
            return tfidf_matrix.toarray()

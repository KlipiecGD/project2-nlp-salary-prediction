import re
import string
from typing import Optional, List, Set
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn compatible transformer for cleaning and preprocessing text data.

    This class handles various text cleaning steps such as converting to lowercase,
    removing URLs, emails, HTML tags, and punctuation. It also includes optional steps
    for removing numbers, stopwords, and lemmatization. It is designed to work with
    pandas DataFrames containing 'Title' and 'FullDescription' columns.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        remove_numbers: bool = False,
        lemmatize: bool = False,
    ):
        """
        Initializes the TextPreprocessor with various preprocessing options.

        Args:
            lowercase: If True, converts all text to lowercase. Defaults to True.
            remove_punctuation: If True, removes all punctuation from the text.
                                Defaults to False.
            remove_stopwords: If True, removes common English stopwords. Defaults to False.
            remove_numbers: If True, removes all numbers from the text. Defaults to False.
            lemmatize: If True, reduces words to their base or root form. Defaults to False.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words("english"))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Cleans a single text string based on the initialized parameters.

        Args:
            text: The text string to be cleaned.

        Returns:
            The cleaned text string.
        """
        if pd.isnull(text) or text.strip() == "":
            return ""

        text = str(text)

        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove HTML tags (just in case)
        text = re.sub(r"<.*?>", "", text)

        # Remove multiple asterisks (e.g., *****)
        text = re.sub(r"\*{2,}", " ", text)

        # Remove extra whitespaces
        text = re.sub(r"\s+", " ", text).strip()

        # Remove special characters
        text = re.sub(r"[^\s\w+#.+-]", "", text)

        # Remove numbers - optional
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove punctuation - optional
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenization
        tokens = text.split()

        # Remove stopwords - optional
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatization - optional
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        # Join tokens back into a single string
        text = " ".join(tokens)

        return text.strip()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        if "Title" in X_copy.columns:
            X_copy["Title"] = X_copy["Title"].apply(self.clean_text)

        if "FullDescription" in X_copy.columns:
            X_copy["FullDescription"] = X_copy["FullDescription"].apply(self.clean_text)

        return X_copy

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.transform(X)


class MinimalTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn compatible transformer for minimal text cleaning.

    This class performs a streamlined set of text cleaning operations that are not
    typically handled by TF-IDF vectorizers, such as removing URLs, email addresses,
    HTML tags, and multiple asterisks. It is designed to work with pandas DataFrames
    containing 'Title' and 'FullDescription' columns.
    """

    def clean_text(self, text: str) -> str:
        """
        Cleans a single text string by removing specific patterns.

        Args:
            text: The text string to be cleaned.

        Returns:
            The cleaned text string.
        """
        if pd.isnull(text):
            return ""
        text = str(text).strip()
        if text == "":
            return ""

        # Remove URLs, emails, HTML, asterisks
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\*{2,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if "Title" in X_copy.columns:
            X_copy["Title"] = X_copy["Title"].apply(self.clean_text)
        if "FullDescription" in X_copy.columns:
            X_copy["FullDescription"] = X_copy["FullDescription"].apply(self.clean_text)
        return X_copy


def clean_tokens(
    tokens: List[str],
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = False,
    remove_special_chars: bool = True,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    min_token_length: int = 1,
    max_token_length: Optional[int] = None,
    stop_words: Optional[Set[str]] = None,
) -> List[str]:
    """
    Clean a list of tokens with multiple configurable options.

    Args:
        tokens: List of string tokens to clean
        remove_urls: Remove tokens that are URLs (http/https/www)
        remove_emails: Remove tokens that are email addresses
        remove_numbers: Remove tokens that are pure numbers
        remove_punctuation: Remove tokens that are pure punctuation
        remove_special_chars: Remove special characters from tokens
        remove_stopwords: Remove common stopwords
        lemmatize: Apply lemmatization to tokens
        min_token_length: Minimum length for tokens
        max_token_length: Maximum length for tokens (None = no limit)
        stop_words: Custom set of stopwords (uses NLTK English stopwords if None)

    Returns:
        Cleaned list of tokens
    """
    if not tokens:
        return []

    # Initialize lemmatizer and stopwords if needed
    lemmatizer = None
    if lemmatize:
        lemmatizer = WordNetLemmatizer()

    if remove_stopwords and stop_words is None:
        stop_words = set(stopwords.words("english"))

    cleaned_tokens = []

    for token in tokens:
        if not token or not isinstance(token, str):
            continue

        # Skip empty tokens
        if not token.strip():
            continue

        # 1. Remove URLs
        if remove_urls:
            if re.match(r"http\S+|www\S+|https\S+", token):
                continue

        # 2. Remove emails
        if remove_emails:
            if re.match(r"\S+@\S+", token):
                continue

        # 3. Remove pure numbers
        if remove_numbers:
            if token.isdigit():
                continue

        # 4. Remove special characters from token
        if remove_special_chars:
            token = re.sub(r"[^\w+#.+-]", "", token)
            if not token:  # Skip if token becomes empty
                continue

        # 5. Filter by token length
        if len(token) < min_token_length:
            continue
        if max_token_length and len(token) > max_token_length:
            continue

        # 6. Remove stopwords
        if remove_stopwords:
            if token in stop_words:
                continue

        # 7. Lemmatization
        if lemmatize:
            token = lemmatizer.lemmatize(token)

        cleaned_tokens.append(token)

    return cleaned_tokens

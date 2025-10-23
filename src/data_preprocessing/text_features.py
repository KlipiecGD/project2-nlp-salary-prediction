import os
import logging
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from preprocessors.text_preprocessors import MinimalTextPreprocessor
from preprocessors.tfidf_transformer import TfidfTransformer
from preprocessors.text_embedder import TextEmbedder
from config.config import (
    TITLE_COLUMN,
    DESC_COLUMN,
    TFIDF_FEATURES_DIR,
    EMBEDDINGS_DIR,
    SENTENCE_TRANSFORMER_MODEL,
)


def preprocess_text_data_tfidf(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    title_column: str = TITLE_COLUMN,
    desc_column: str = DESC_COLUMN,
    title_max_features: int = 50,
    desc_max_features: int = 800,
    title_use_svd: bool = False,
    title_n_components: int = 10,
    desc_use_svd: bool = False,
    desc_n_components: int = 50,
    title_stop_words: Optional[Union[str, List[str]]] = None,
    desc_stop_words: Optional[Union[str, List[str]]] = None,
    save_features: bool = True,
    feature_prefix: str = "",
    features_dir: str = TFIDF_FEATURES_DIR,
    logger: logging.Logger = None,
) -> Tuple[Dict[str, np.ndarray], int, int, Any]:
    """
    Generates TF-IDF features and optionally saves them to disk.

    Args:
        X_train: Training data.
        X_valid: Validation data.
        X_test: Test data.
        title_column: Name of the title column.
        desc_column: Name of the description column.
        title_max_features: Max TF-IDF features for title.
        desc_max_features: Max TF-IDF features for description.
        title_use_svd: Whether to apply SVD to title features.
        title_n_components: Number of SVD components for title.
        desc_use_svd: Whether to apply SVD to description features.
        desc_n_components: Number of SVD components for description.
        title_stop_words: Stop words for title vectorizer.
        desc_stop_words: Stop words for description vectorizer.
        save_features: Whether to save features to disk.
        feature_prefix: Prefix for saved feature files.
        features_dir: Directory to save features.
        logger: Optional logger for logging information.

    Returns:
        Tuple of (dict with all features, title_dim, desc_dim, text_preprocessor).
    """
    # Step 1: Clean text
    text_preprocessor = MinimalTextPreprocessor()
    X_train_clean = text_preprocessor.fit_transform(X_train)
    X_valid_clean = text_preprocessor.transform(X_valid)
    X_test_clean = text_preprocessor.transform(X_test)

    # Step 2: Build TF-IDF pipelines
    title_pipeline = make_pipeline(
        TfidfTransformer(
            text_column=title_column,
            max_features=title_max_features,
            use_svd=title_use_svd,
            n_components=title_n_components,
            stop_words=title_stop_words,
        )
    )

    desc_pipeline = make_pipeline(
        TfidfTransformer(
            text_column=desc_column,
            max_features=desc_max_features,
            use_svd=desc_use_svd,
            n_components=desc_n_components,
            stop_words=desc_stop_words,
        )
    )
    if logger:
        logger.info("Text preprocessing pipelines created.")

    # Generate title TF-IDF features
    title_train = title_pipeline.fit_transform(X_train_clean[[title_column]])
    title_valid = title_pipeline.transform(X_valid_clean[[title_column]])
    title_test = title_pipeline.transform(X_test_clean[[title_column]])

    # Generate description TF-IDF features
    desc_train = desc_pipeline.fit_transform(X_train_clean[[desc_column]])
    desc_valid = desc_pipeline.transform(X_valid_clean[[desc_column]])
    desc_test = desc_pipeline.transform(X_test_clean[[desc_column]])

    # Combine TF-IDF features
    text_train = np.hstack([title_train, desc_train])
    text_valid = np.hstack([title_valid, desc_valid])
    text_test = np.hstack([title_test, desc_test])

    if logger:
        logger.info("TF-IDF features generated successfully.")

    # Calculate dimensions
    title_dim = title_n_components if title_use_svd else title_max_features
    desc_dim = desc_n_components if desc_use_svd else desc_max_features

    print(f"Title TF-IDF dimension: {title_dim}")
    print(f"Description TF-IDF dimension: {desc_dim}")

    features = {
        "X_train_text": text_train,
        "X_valid_text": text_valid,
        "X_test_text": text_test,
        "title_train": title_train,
        "title_valid": title_valid,
        "title_test": title_test,
        "desc_train": desc_train,
        "desc_valid": desc_valid,
        "desc_test": desc_test,
    }

    # Save features if requested
    if save_features:
        os.makedirs(features_dir, exist_ok=True)
        for name, array in features.items():
            filepath = os.path.join(features_dir, f"{feature_prefix}{name}.npy")
            np.save(filepath, array)
        if logger:
            logger.info(f"TF-IDF features saved to directory: {features_dir}")

    return features, title_dim, desc_dim, text_preprocessor


def load_tfidf_features(
    feature_prefix: str = "",
    features_dir: str = TFIDF_FEATURES_DIR,
    logger: logging.Logger = None,
) -> Dict[str, np.ndarray]:
    """
    Loads pre-computed TF-IDF features from disk.

    Args:
        feature_prefix: Prefix used when saving features.
        features_dir: Directory where features are stored.
        logger: Optional logger for logging information.

    Returns:
        Dictionary containing all loaded features.
    """
    feature_names = [
        "X_train_text",
        "X_valid_text",
        "X_test_text",
        "title_train",
        "title_valid",
        "title_test",
        "desc_train",
        "desc_valid",
        "desc_test",
    ]

    features = {}
    for name in feature_names:
        filepath = os.path.join(features_dir, f"{feature_prefix}{name}.npy")
        features[name] = np.load(filepath)

    if logger:
        logger.info(f"TF-IDF features loaded from directory: {features_dir}")
    return features


def preprocess_text_data_embeddings(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    title_column: str = TITLE_COLUMN,
    desc_column: str = DESC_COLUMN,
    embedding_model_name: str = SENTENCE_TRANSFORMER_MODEL,
    save_embeddings: bool = True,
    embedding_prefix: str = "",
    embeddings_dir: str = EMBEDDINGS_DIR,
    logger: logging.Logger = None,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Generates text embeddings using SentenceTransformers and optionally saves them to disk.

    Args:
        X_train: Training data.
        X_valid: Validation data.
        X_test: Test data.
        title_column: Name of the title column.
        desc_column: Name of the description column.
        embedding_model_name: SentenceTransformer model name.
        save_embeddings: Whether to save embeddings to disk.
        embedding_prefix: Prefix for saved embedding files.
        embeddings_dir: Directory to save embeddings.
        logger: Optional logger for logging information.

    Returns:
        Tuple of (dict with all embeddings, embedding dimension).
    """
    # Build embedding pipelines
    title_embedding_pipeline = Pipeline(
        [("embedder", TextEmbedder(model_name=embedding_model_name))]
    )

    desc_embedding_pipeline = Pipeline(
        [("embedder", TextEmbedder(model_name=embedding_model_name))]
    )

    if logger:
        logger.info("Text embedding pipelines created.")
        logger.info("Using embedding model: " + embedding_model_name)

    # Generate title embeddings
    title_train = title_embedding_pipeline.fit_transform(X_train[[title_column]])
    title_valid = title_embedding_pipeline.transform(X_valid[[title_column]])
    title_test = title_embedding_pipeline.transform(X_test[[title_column]])

    # Generate description embeddings
    desc_train = desc_embedding_pipeline.fit_transform(X_train[[desc_column]])
    desc_valid = desc_embedding_pipeline.transform(X_valid[[desc_column]])
    desc_test = desc_embedding_pipeline.transform(X_test[[desc_column]])

    # Combine embeddings
    text_train = np.hstack([title_train, desc_train])
    text_valid = np.hstack([title_valid, desc_valid])
    text_test = np.hstack([title_test, desc_test])

    if logger:
        logger.info("Embeddings generated successfully.")

    # Get embedding dimension
    embedding_dim = title_embedding_pipeline[
        "embedder"
    ].model.get_sentence_embedding_dimension()

    embeddings = {
        "X_train_text": text_train,
        "X_valid_text": text_valid,
        "X_test_text": text_test,
        "title_train": title_train,
        "title_valid": title_valid,
        "title_test": title_test,
        "desc_train": desc_train,
        "desc_valid": desc_valid,
        "desc_test": desc_test,
    }

    # Save embeddings if requested
    if save_embeddings:
        os.makedirs(embeddings_dir, exist_ok=True)
        for name, array in embeddings.items():
            filepath = os.path.join(embeddings_dir, f"{embedding_prefix}{name}.npy")
            np.save(filepath, array)
        if logger:
            logger.info(f"Embeddings saved to directory: {embeddings_dir}")

    return embeddings, embedding_dim


def load_embeddings(
    embedding_prefix: str = "",
    embeddings_dir: str = EMBEDDINGS_DIR,
    logger: logging.Logger = None,
) -> Dict[str, np.ndarray]:
    """
    Loads pre-computed embeddings from disk.

    Args:
        embedding_prefix: Prefix used when saving embeddings.
        embeddings_dir: Directory where embeddings are stored.

    Returns:
        Dictionary containing all loaded embeddings.
    """
    embedding_names = [
        "X_train_text",
        "X_valid_text",
        "X_test_text",
        "title_train",
        "title_valid",
        "title_test",
        "desc_train",
        "desc_valid",
        "desc_test",
    ]

    embeddings = {}
    for name in embedding_names:
        filepath = os.path.join(embeddings_dir, f"{embedding_prefix}{name}.npy")
        embeddings[name] = np.load(filepath)

    if logger:
        logger.info(f"Embeddings loaded from directory: {embeddings_dir}")
    return embeddings

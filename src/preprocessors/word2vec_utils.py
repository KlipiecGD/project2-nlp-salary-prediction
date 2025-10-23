import logging
from typing import List, Optional, Dict
from gensim.models import Word2Vec
import numpy as np
from nltk.tokenize import word_tokenize


def train_word2vec_model(
    texts: List[str],
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 20,
    sg: int = 1,  # 1=Skip-gram, 0=CBOW
    epochs: int = 10,
    logger: Optional[logging.Logger] = None,
) -> Word2Vec:
    """
    Train Word2Vec model on given corpus.

    Args:
        texts: List of text strings
        vector_size: Dimension of word embeddings
        window: Context window size
        min_count: Minimum word frequency
        sg: Algorithm (1=Skip-gram, 0=CBOW)
        epochs: Training epochs
        logger: Optional logger for logging information
    """
    # Tokenize texts
    tokenized = [word_tokenize(text.lower()) for text in texts]

    # Train Word2Vec
    w2v_model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=4,
    )

    if logger:
        logger.info(f"Word2Vec trained with {len(w2v_model.wv)} words")
    return w2v_model


def encode_text_with_w2v(
    text: str,
    w2v_model: Word2Vec,
    method: str = "mean",
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Convert text to a fixed-size vector using Word2Vec.

    Args:
        text: Input text
        w2v_model: Trained Word2Vec model
        method: Aggregation method ('mean', 'max', 'sum', 'mean+max')

    Returns:
        Fixed-size vector representing the text
    """
    tokens = word_tokenize(text.lower())

    vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]

    if not vectors:
        if method == "mean+max":
            return np.zeros(2 * w2v_model.vector_size)
        return np.zeros(w2v_model.vector_size)

    vectors = np.array(vectors)

    if method == "mean":
        return np.mean(vectors, axis=0)
    elif method == "max":
        return np.max(vectors, axis=0)
    elif method == "mean+max":
        mean_vec = np.mean(vectors, axis=0)
        max_vec = np.max(vectors, axis=0)
        return np.concatenate([mean_vec, max_vec])  # Double the size
    elif method == "sum":
        return np.sum(vectors, axis=0)
    else:
        logger.error(f"Unknown aggregation method: {method}")
        raise ValueError("Method must be 'mean', 'max', 'sum', or 'mean+max'")


def create_embedding_matrix_w2v(
    vocab: Dict[str, int], w2v_model: Word2Vec, logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Create embedding matrix from Word2Vec for vocabulary.
    Words not in Word2Vec get small random values.
    Args:
        vocab: Vocabulary dictionary mapping tokens to indices
        w2v_model: Trained Word2Vec model
        logger: Optional logger for logging information
    Returns:
        Embedding matrix as numpy array
    """

    vocab_size = len(vocab)
    embedding_dim = w2v_model.vector_size

    # Initialize with small random values
    embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))

    # Set PAD to zeros
    embedding_matrix[vocab["<PAD>"]] = np.zeros(embedding_dim)

    # Fill with Word2Vec vectors
    found = 0
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            found += 1

    if logger:
        logger.info(
            f"Embedding matrix created: {found}/{vocab_size} words found in Word2Vec."
        )
    return embedding_matrix


def build_vocab_from_w2v(
    w2v_model: Word2Vec, logger: Optional[logging.Logger] = None
) -> Dict[str, int]:
    """Build vocabulary from Word2Vec model.
    Args:
        w2v_model: Trained Word2Vec model
        logger: Optional logger for logging information
    Returns:
        Vocabulary dictionary mapping tokens to indices
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, word in enumerate(w2v_model.wv.index_to_key, start=2):
        vocab[word] = idx
    if logger:
        logger.info(f"Vocabulary built from Word2Vec: {len(vocab)} words")
    return vocab

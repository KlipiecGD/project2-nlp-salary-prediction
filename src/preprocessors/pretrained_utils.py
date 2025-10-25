import logging
from typing import Optional
import gensim.downloader as api
import numpy as np
from nltk.tokenize import word_tokenize


def load_pretrained_embeddings(
    model_name: str = "fasttext", logger: Optional[logging.Logger] = None
) -> object:
    """
    Load pre-trained embeddings.

    Args:
        model_name: str, The name of the pre-trained model to use ('word2vec', 'glove', or 'fasttext'), default is 'fasttext'.
        logger: Logger object for logging information

    Returns:
        object, Loaded embedding model
    """
    if logger:
        logger.info(f"Loading {model_name} pre-trained embeddings...")

    if model_name == "word2vec":
        # Google's Word2Vec (3M words, 300-dim)
        model = api.load("word2vec-google-news-300")

    elif model_name == "glove":
        # GloVe trained on Wikipedia (400K words, 100-dim)
        model = api.load("glove-wiki-gigaword-100")
        # For 300-dim: api.load('glove-wiki-gigaword-300')

    elif model_name == "fasttext":
        # FastText with subword info (1M words, 300-dim)
        model = api.load("fasttext-wiki-news-subwords-300")

    else:
        if logger:
            logger.error(f"Unknown model name: {model_name}")
        raise ValueError(f"Unknown model: {model_name}")

    if logger:
        logger.info(
            f"Loaded {len(model)} words with {model.vector_size}-dimensional vectors"
        )
    return model

def encode_text_pretrained(text: str, model: object, method="mean") -> np.ndarray:
    """
    Encode text using pre-trained embeddings.

    Args:
        text: str, Input text
        model: object, Pre-trained embedding model (Word2Vec, GloVe, FastText)
        method: str, Aggregation method ('mean', 'max', 'sum', 'mean+max')

    Returns:
        np.ndarray, Fixed-size vector representing the text
    """
    tokens = word_tokenize(text.lower())

    # Get vectors for tokens in vocabulary
    vectors = []
    for token in tokens:
        try:
            vectors.append(model[token])
        except KeyError:
            # Token not in vocabulary - skip it
            continue

    if not vectors:
        if method == "mean+max":
            return np.zeros(2 * model.vector_size)
        return np.zeros(model.vector_size)

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
        return np.mean(vectors, axis=0)


def create_embedding_matrix_from_pretrained(
    vocab: dict[str, int],
    pretrained_model,
    embedding_dim: int = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Create embedding matrix from pre-trained model for your vocabulary.

    Args:
        vocab: dict[str, int], Vocabulary mapping words to indices.
        pretrained_model: Pre-trained embedding model (e.g., Word2Vec, GloVe, FastText).
        embedding_dim: int, Dimension of embeddings; if None, uses model's vector size.
        logger: Optional[logging.Logger], Optional logger for logging information.
    Returns:
        np.ndarray, Embedding matrix.
    """
    if embedding_dim is None:
        embedding_dim = pretrained_model.vector_size

    vocab_size = len(vocab)

    # Initialize with small random values
    embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))

    # Set PAD token to zeros
    embedding_matrix[vocab["<PAD>"]] = np.zeros(embedding_dim)

    # Fill with pre-trained vectors where available
    found = 0
    missing = []

    for word, idx in vocab.items():
        if word in ["<PAD>", "<UNK>"]:
            continue

        try:
            embedding_matrix[idx] = pretrained_model[word]
            found += 1
        except KeyError:
            # Word not in pre-trained model - keep random initialization
            missing.append(word)

    coverage = (found / vocab_size) * 100
    if logger:
        logger.info(
            f"Embedding matrix created: {found}/{vocab_size} words found in pre-trained model."
        )
        logger.info(f"Coverage: {coverage:.1f}%")
        logger.info(f"Missing from pre-trained: {len(missing)} words")

    return embedding_matrix

from collections import Counter
from typing import List, Dict, Optional
from nltk.tokenize import word_tokenize
from src.preprocessors.text_preprocessors import clean_tokens


def build_vocab(
    texts: List[str],
    min_freq: int = 20,
    clean_tokens_use: bool = False,
    clean_tokens_config: Optional[Dict] = None,
) -> Dict[str, int]:
    """
    Build a vocabulary dictionary mapping tokens to indices.

    Args:
        texts: List of text strings.
        min_freq: Minimum frequency for a token to be included in the vocab.
        clean_tokens: Whether to clean tokens using clean_tokens().
        clean_tokens_config: Dictionary of cleaning parameters to pass to clean_tokens().
                            Example: {'remove_urls': True, 'remove_numbers': True}
                            If None, default cleaning is applied.

    Returns:
        A dictionary mapping tokens to indices, including <PAD> and <UNK>.
    """
    all_tokens = []

    # Default cleaning config
    if clean_tokens_config is None and clean_tokens_use:
        clean_tokens_config = {
            "remove_urls": True,
            "remove_emails": True,
            "remove_numbers": False,
            "remove_special_chars": True,
            "remove_stopwords": False,
            "lemmatize": False,
            "min_token_length": 1,
            "max_token_length": None,
            "stop_words": None,
        }

    for text in texts:
        # 1. Tokenize first
        tokens = word_tokenize(text.lower())

        # 2. Clean the tokens if required
        if clean_tokens_use:
            tokens = clean_tokens(tokens, **clean_tokens_config)

        all_tokens.extend(tokens)

    # Count token frequencies
    token_counts = Counter(all_tokens)

    # Initialize vocabulary with special tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2

    # Add tokens that meet the minimum frequency threshold
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab


def text_to_sequence(
    text: str,
    vocab: Dict[str, int],
    max_length: int,
    clean_tokens_use: bool = False,
    clean_tokens_config: Optional[Dict] = None,
) -> List[int]:
    """
    Convert a text string to a sequence of token indices based on the provided vocabulary.

    Args:
        text: Input text string
        vocab: Vocabulary dictionary mapping tokens to indices
        max_length: Maximum length of the output sequence (for padding/truncating)
        clean_tokens: Whether to clean tokens using clean_tokens()
        clean_tokens_config: Dictionary of cleaning parameters to pass to clean_tokens()
                            Should match the config used in build_vocab()

    Returns:
        A list of token indices representing the input text
    """
    # Default cleaning config (should match build_vocab)
    if clean_tokens_config is None and clean_tokens_use:
        clean_tokens_config = {
            "remove_urls": True,
            "remove_emails": True,
            "remove_numbers": False,
            "remove_special_chars": True,
            "remove_stopwords": False,
            "lemmatize": False,
            "min_token_length": 1,
            "max_token_length": None,
            "stop_words": None,
        }

    # 1. Tokenize first
    tokens = word_tokenize(text.lower())

    # 2. Clean the tokens if required
    if clean_tokens_use:
        tokens = clean_tokens(tokens, **clean_tokens_config)

    # 3. Map tokens to indices (using <UNK> for unseen words)
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    # 4. Pad or truncate the sequence to max_length
    if len(sequence) < max_length:
        sequence += [vocab["<PAD>"]] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]

    return sequence

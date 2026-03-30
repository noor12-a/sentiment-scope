"""Text preprocessing utilities for the text classification project.

This module provides a collection of simple, pure‑Python functions that
clean raw text data before it is fed to a model.  The functions are kept
tiny on purpose – they are easy to understand, unit‑test and combine.
"""

import re
import string
from typing import List


def to_lower(text: str) -> str:
    """Convert text to lower‑case.

    Args:
        text: Input string.
    Returns:
        Lower‑cased string.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters.

    Uses ``string.punctuation`` which covers the most common ASCII symbols.
    """
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Strip all numeric characters from the string.
    """
    return re.sub(r"\d+", "", text)


def strip_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and trim.
    """
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    """Tokenize using NLTK's TreebankWordTokenizer.
    """
    from nltk.tokenize import TreebankWordTokenizer
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)


def preprocess(text: str) -> str:
    """Apply the full preprocessing pipeline.

    The order is:
    1. lower‑case
    2. remove punctuation
    3. remove numbers
    4. strip extra whitespace
    """
    text = to_lower(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = strip_whitespace(text)
    return text

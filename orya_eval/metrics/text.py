"""Text evaluation metrics."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Lowercase and normalize whitespace for robust string comparisons."""
    text = "" if value is None else str(value)
    text = text.strip().lower()
    return _WHITESPACE_RE.sub(" ", text)


def contains_match(reference: object, prediction: object) -> float:
    """Return 1.0 when either normalized string contains the other."""
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    return float(ref in pred or pred in ref)


def token_f1(reference: object, prediction: object) -> float:
    """Compute a simple token-level F1 score on normalized whitespace tokens."""
    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum((ref_counts & pred_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def normalized_similarity(reference: object, prediction: object) -> float:
    """Return a normalized string similarity ratio after text normalization."""
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    return SequenceMatcher(a=ref, b=pred).ratio()

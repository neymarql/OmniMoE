"""Evaluation metrics for multimodal tasks."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List

import numpy as np


def exact_match(pred: str, references: Iterable[str]) -> float:
    return float(pred.strip().lower() in {ref.strip().lower() for ref in references})


def bleu_score(prediction: str, references: List[str], n: int = 4) -> float:
    prediction_tokens = prediction.split()
    reference_tokens = [ref.split() for ref in references]
    precisions = []
    for k in range(1, n + 1):
        pred_ngrams = Counter(tuple(prediction_tokens[i : i + k]) for i in range(len(prediction_tokens) - k + 1))
        max_counts = Counter()
        for ref in reference_tokens:
            ref_ngrams = Counter(tuple(ref[i : i + k]) for i in range(len(ref) - k + 1))
            for ngram, count in ref_ngrams.items():
                max_counts[ngram] = max(max_counts[ngram], count)
        overlap = sum(min(count, max_counts.get(ngram, 0)) for ngram, count in pred_ngrams.items())
        total = max(sum(pred_ngrams.values()), 1)
        precisions.append(overlap / total)
    geometric_mean = np.exp(sum(np.log(p + 1e-8) for p in precisions) / n)
    brevity_penalty = np.exp(1 - len(reference_tokens[0]) / max(len(prediction_tokens), 1)) if len(prediction_tokens) < len(reference_tokens[0]) else 1
    return brevity_penalty * geometric_mean


__all__ = ["exact_match", "bleu_score"]

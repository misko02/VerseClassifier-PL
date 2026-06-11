from __future__ import annotations

from typing import Iterable


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    raise NotImplementedError

from __future__ import annotations

from typing import Iterable

from ..data.schemas import TextChunk


class BaselineTextClassifier:
    def fit(self, samples: Iterable[TextChunk]) -> "BaselineTextClassifier":
        raise NotImplementedError

    def predict(self, samples: Iterable[TextChunk]) -> list[int]:
        raise NotImplementedError

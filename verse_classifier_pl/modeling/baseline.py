from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..data.schemas import TextChunk


@dataclass(slots=True)
class BaselineConfig:
    max_features: int = 50_000
    min_df: int = 2
    ngram_min: int = 1
    ngram_max: int = 2
    random_seed: int = 42


class BaselineTextClassifier:
    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents=None,
                        analyzer="word",
                        ngram_range=(self.config.ngram_min, self.config.ngram_max),
                        min_df=self.config.min_df,
                        max_features=self.config.max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=self.config.random_seed,
                    ),
                ),
            ]
        )

    def fit(self, samples: Iterable[TextChunk]) -> "BaselineTextClassifier":
        sample_list = list(samples)
        self.pipeline.fit(_texts(sample_list), _labels(sample_list))
        return self

    def predict(self, samples: Iterable[TextChunk]) -> list[int]:
        sample_list = list(samples)
        return [int(label) for label in self.pipeline.predict(_texts(sample_list))]

    def predict_texts(self, texts: Iterable[str]) -> list[int]:
        return [int(label) for label in self.pipeline.predict(list(texts))]

    def predict_text_probabilities(self, texts: Iterable[str]) -> list[dict[int, float]]:
        probabilities = self.pipeline.predict_proba(list(texts))
        classes = [int(label) for label in self.pipeline.classes_]
        return [
            {label: float(probability) for label, probability in zip(classes, row)}
            for row in probabilities
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "pipeline": self.pipeline}, path)

    @classmethod
    def load(cls, path: Path) -> "BaselineTextClassifier":
        payload = joblib.load(path)
        model = cls(config=payload["config"])
        model.pipeline = payload["pipeline"]
        return model


def _texts(samples: Iterable[TextChunk]) -> list[str]:
    return ["\n".join(sample.lines) for sample in samples]


def _labels(samples: Iterable[TextChunk]) -> list[int]:
    return [sample.label for sample in samples]

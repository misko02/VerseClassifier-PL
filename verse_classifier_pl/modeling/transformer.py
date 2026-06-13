from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..data.schemas import TextChunk
from ..evaluation.metrics import compute_classification_metrics


DEFAULT_HERBERT_MODEL = "allegro/herbert-base-cased"


@dataclass(slots=True)
class TransformerConfig:
    model_name: str = DEFAULT_HERBERT_MODEL
    max_length: int = 128
    learning_rate: float = 2e-5
    epochs: float = 3.0
    train_batch_size: int = 8
    eval_batch_size: int = 16
    weight_decay: float = 0.01
    random_seed: int = 42


class TextChunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        chunks: Iterable[TextChunk],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.chunks = list(chunks)
        self.encodings = tokenizer(
            ["\n".join(chunk.lines) for chunk in self.chunks],
            truncation=True,
            max_length=max_length,
        )
        self.labels = [chunk.label for chunk in self.chunks]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class TransformerTextClassifier:
    def __init__(self, config: TransformerConfig | None = None) -> None:
        self.config = config or TransformerConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            id2label={0: "poetry", 1: "rap"},
            label2id={"poetry": 0, "rap": 1},
        )

    def fit(self, samples: Iterable[TextChunk]) -> "TransformerTextClassifier":
        sample_list = list(samples)
        self.train(
            train_samples=sample_list,
            val_samples=[],
            output_dir=Path(".artifacts/transformer"),
        )
        return self

    def predict(self, samples: Iterable[TextChunk]) -> list[int]:
        dataset = TextChunkDataset(samples, self.tokenizer, self.config.max_length)
            
        trainer = Trainer(
            model=self.model,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer) 
        )
            
        predictions = trainer.predict(dataset).predictions
        return [int(label) for label in np.argmax(predictions, axis=1)]
    
    def predict_probabilities(self, samples: Iterable[TextChunk]) -> list[dict[int, float]]:
        dataset = TextChunkDataset(samples, self.tokenizer, self.config.max_length)
        trainer = Trainer(
            model=self.model,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
        
        logits = trainer.predict(dataset).predictions
        
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        
        return [{0: float(prob[0]), 1: float(prob[1])} for prob in probs]

    def train(
        self,
        *,
        train_samples: Iterable[TextChunk],
        val_samples: Iterable[TextChunk],
        output_dir: Path,
    ) -> dict[str, float]:
        set_seed(self.config.random_seed)
        output_dir = Path(output_dir)
        train_dataset = TextChunkDataset(
            train_samples,
            self.tokenizer,
            self.config.max_length,
        )
        val_dataset = TextChunkDataset(
            val_samples,
            self.tokenizer,
            self.config.max_length,
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            eval_strategy="epoch" if len(val_dataset) else "no",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            load_best_model_at_end=bool(len(val_dataset)),
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config.random_seed,
            report_to=[],
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if len(val_dataset) else None,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=_compute_trainer_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate() if len(val_dataset) else {}
        self.save(output_dir / "model")
        return {key: float(value) for key, value in metrics.items()}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: Path, max_length: int = 128) -> "TransformerTextClassifier":
        config = TransformerConfig(model_name=str(path), max_length=max_length)
        return cls(config=config)


def _compute_trainer_metrics(eval_prediction: object) -> dict[str, float]:
    logits = getattr(eval_prediction, "predictions", None)
    labels = getattr(eval_prediction, "label_ids", None)
    if logits is None or labels is None:
        logits, labels = eval_prediction  # type: ignore[misc]
    predictions = np.argmax(logits, axis=1)
    return compute_classification_metrics(labels, predictions)

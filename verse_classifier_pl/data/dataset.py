from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split

from .schemas import TextChunk


def load_chunks_jsonl(path: Path) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_no, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            lines = payload.get("lines") or payload.get("text", "").splitlines()
            if len(lines) != 4:
                raise ValueError(f"Expected 4 lines at {path}:{line_no}, got {len(lines)}")
            chunks.append(
                TextChunk(
                    source=str(payload["source"]),
                    label=int(payload["label"]),
                    title=str(payload["title"]),
                    author=str(payload["author"]),
                    chunk_index=int(payload["chunk_index"]),
                    lines=tuple(str(line) for line in lines),  # type: ignore[arg-type]
                )
            )
    return chunks


def save_chunks_jsonl(chunks: Iterable[TextChunk], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            payload = asdict(chunk)
            payload["text"] = "\n".join(chunk.lines)
            output_file.write(json.dumps(payload, ensure_ascii=False))
            output_file.write("\n")


def split_chunks_by_work(
    chunks: list[TextChunk],
    *,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> tuple[list[TextChunk], list[TextChunk], list[TextChunk]]:
    if not chunks:
        raise ValueError("Cannot split an empty dataset.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be lower than 1.")

    group_to_label = _group_labels(chunks)
    groups = sorted(group_to_label)
    labels = [group_to_label[group] for group in groups]

    train_val_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )

    if val_size:
        train_val_labels = [group_to_label[group] for group in train_val_groups]
        relative_val_size = val_size / (1 - test_size)
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=relative_val_size,
            random_state=random_seed,
            stratify=train_val_labels,
        )
    else:
        train_groups = train_val_groups
        val_groups = []

    return (
        _select_chunks(chunks, set(train_groups)),
        _select_chunks(chunks, set(val_groups)),
        _select_chunks(chunks, set(test_groups)),
    )


def sample_per_class(
    chunks: list[TextChunk],
    *,
    max_samples_per_class: int | None,
    random_seed: int,
) -> list[TextChunk]:
    if max_samples_per_class is None:
        return chunks

    by_label: dict[int, list[TextChunk]] = defaultdict(list)
    for chunk in chunks:
        by_label[chunk.label].append(chunk)

    rng = random.Random(random_seed)
    sampled: list[TextChunk] = []
    for label_chunks in by_label.values():
        if len(label_chunks) <= max_samples_per_class:
            sampled.extend(label_chunks)
        else:
            sampled.extend(rng.sample(label_chunks, max_samples_per_class))

    return sorted(sampled, key=_chunk_sort_key)


def class_counts(chunks: Iterable[TextChunk]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for chunk in chunks:
        counts[chunk.label] += 1
    return dict(sorted(counts.items()))


def _group_labels(chunks: list[TextChunk]) -> dict[str, int]:
    group_to_label: dict[str, int] = {}
    for chunk in chunks:
        group = _work_group_id(chunk)
        if group in group_to_label and group_to_label[group] != chunk.label:
            raise ValueError(f"Conflicting labels for group {group}")
        group_to_label[group] = chunk.label
    return group_to_label


def _select_chunks(chunks: list[TextChunk], groups: set[str]) -> list[TextChunk]:
    selected = [chunk for chunk in chunks if _work_group_id(chunk) in groups]
    return sorted(selected, key=_chunk_sort_key)


def _work_group_id(chunk: TextChunk) -> str:
    return f"{chunk.source}\0{chunk.author}\0{chunk.title}"


def _chunk_sort_key(chunk: TextChunk) -> tuple[str, str, str, int]:
    return (chunk.source, chunk.author, chunk.title, chunk.chunk_index)

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TextSample:
    source: str
    label: int
    title: str
    author: str
    text: str


@dataclass(slots=True)
class TextChunk:
    source: str
    label: int
    title: str
    author: str
    chunk_index: int
    lines: tuple[str, str, str, str]

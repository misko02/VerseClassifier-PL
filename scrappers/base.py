from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class ScrapedText:
    source: str
    title: str
    author: str
    text: str
    url: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Scraper(Protocol):
    source_name: str

    def scrape(self) -> list[ScrapedText]:
        ...

def write_jsonl(samples: list[ScrapedText], output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as output_file:
        for sample in samples:
            output_file.write(sample.to_json())
            output_file.write("\n")

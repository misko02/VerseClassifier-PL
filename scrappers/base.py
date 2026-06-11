from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ScrapedText:
    source: str
    title: str
    author: str
    text: str


class Scraper(Protocol):
    source_name: str

    def scrape(self) -> list[ScrapedText]:
        ...

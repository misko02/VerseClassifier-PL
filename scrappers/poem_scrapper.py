from __future__ import annotations

from .base import ScrapedText


class PoemScraper:
    source_name = "poetry"

    def scrape(self) -> list[ScrapedText]:
        raise NotImplementedError
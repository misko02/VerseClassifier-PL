from __future__ import annotations

from .base import ScrapedText


class RapGeniusScraper:
    source_name = "rap_genius"

    def scrape(self) -> list[ScrapedText]:
        raise NotImplementedError
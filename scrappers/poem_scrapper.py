from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

import requests

from .base import ScrapedText

logger = logging.getLogger(__name__)


DEFAULT_POET_SLUGS = (
    "adam-mickiewicz",
    "juliusz-slowacki",
    "cyprian-kamil-norwid",
    "jan-kochanowski",
    "julian-tuwim",
    "boleslaw-lesmian",
    "krzysztof-kamil-baczynski",
    "adam-asnyk"
)


@dataclass(slots=True)
class PoemScraper:
    """Scraper for Wolne Lektury (Polish poems, lyric only).
    
    Filters to only include 'liryka' (lyric poetry), excluding drama/theater.
    Uses Wolne Lektury API with /api/kinds/liryka/ endpoint.
    """
    source_name: str = "poetry"
    author_slugs: tuple[str, ...] = DEFAULT_POET_SLUGS
    limit_per_author: int = 20
    timeout_seconds: int = 30
    session: requests.Session = field(default_factory=requests.Session)

    api_base_url: str = "https://wolnelektury.pl/api"
    book_kind: str = "liryka"

    def scrape(self) -> list[ScrapedText]:
        samples: list[ScrapedText] = []
        for author_slug in self.author_slugs:
            try:
                books = self._get_author_lyric_books(author_slug)
                for book in books[: self.limit_per_author]:
                    sample = self._scrape_book(book)
                    if sample is not None:
                        samples.append(sample)
            except Exception as e:
                logger.warning("Failed to scrape author %s: %s", author_slug, e)
        return samples

    def _get_author_lyric_books(self, author_slug: str) -> list[dict[str, Any]]:
        """Fetch only lyric books for an author (no dramas)."""
        # API endpoint: /api/authors/{slug}/kinds/{kind}/books/
        url = f"{self.api_base_url}/authors/{author_slug}/kinds/{self.book_kind}/books/"
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            logger.warning(f"Unexpected response structure for {url}")
            return []
        return payload

    def _scrape_book(self, book: dict[str, Any]) -> ScrapedText | None:
        """Fetch book details and extract text."""
        detail_url = book.get("href")
        if not isinstance(detail_url, str):
            return None

        try:
            detail_response = self.session.get(detail_url, timeout=self.timeout_seconds)
            detail_response.raise_for_status()
            details = detail_response.json()

            text_url = details.get("txt")
            if not text_url or not isinstance(text_url, str):
                return None

            text_response = self.session.get(text_url, timeout=self.timeout_seconds)
            text_response.raise_for_status()
            text_response.encoding = text_response.encoding or "utf-8"

            title = str(details.get("title") or book.get("title") or "")
            author = self._format_authors(details.get("authors")) or str(book.get("author") or "")
            source_url = str(details.get("url") or detail_url)

            return ScrapedText(
                source=self.source_name,
                title=title,
                author=author,
                text=self._strip_wolne_lektury_footer(text_response.text),
                url=source_url,
            )
        except Exception as e:
            logger.warning(f"Failed to scrape book {book.get('title')}: {e}")
            return None

    @staticmethod
    def _format_authors(authors: Any) -> str:
        if not isinstance(authors, list):
            return ""
        names = [author.get("name") for author in authors if isinstance(author, dict)]
        return ", ".join(name for name in names if isinstance(name, str))

    @staticmethod
    def _strip_wolne_lektury_footer(text: str) -> str:
        """Remove Wolne Lektury footer metadata."""
        footer_markers = (
            "-----",
            "Ten utwór jest",
            "Wszystkie zasoby Wolnych Lektur",
        )
        lines = text.replace("\r\n", "\n").split("\n")
        for index, line in enumerate(lines):
            if any(line.startswith(marker) for marker in footer_markers):
                return "\n".join(lines[:index]).strip()
        return text.strip()


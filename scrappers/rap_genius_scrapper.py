from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import lyricsgenius
except Exception:  # pragma: no cover - optional dependency
    lyricsgenius = None

from .base import ScrapedText

try:
    from verse_classifier_pl.config import RAW_DATA_DIR
except Exception:
    RAW_DATA_DIR = Path(".data/raw")


DEFAULT_RAP_ARTISTS = (
    "Taco Hemingway",
    "Pezet",
    "Łona",
    "Eldo",
    "Bisz",
    "O.S.T.R.",
    "Sokół",
    "KęKę",
    "Quebonafide",
    "Golin",
    "Fisz",
    "Malik Montana",
    "Bedoes",
    "Sarius",
    "Kali",
    "Pro8l3m",
)


@dataclass(slots=True)
class RapGeniusScraper:
    """Scraper for Genius (rap).

    - Requires GENIUS_ACCESS_TOKEN environment variable or `access_token` argument.
    - Uses lyricsgenius library; run locally to fetch data into `.data/raw`.
    """

    source_name: str = "rap_genius"
    artists: tuple[str, ...] = DEFAULT_RAP_ARTISTS
    limit_per_artist: int = 20
    access_token: Optional[str] = None
    timeout_seconds: int = 30
    _client: object = field(default=None, init=False)

    def __post_init__(self) -> None:
        token = self.access_token or os.getenv("GENIUS_ACCESS_TOKEN")
        if not token:
            raise RuntimeError("Missing Genius token. Set GENIUS_ACCESS_TOKEN in env.")
        if lyricsgenius is None:
            raise RuntimeError(
                "lyricsgenius package is not installed. Install it before running the scraper."
            )

        self._client = lyricsgenius.Genius(
            token,
            timeout=self.timeout_seconds,
            retries=3,
            remove_section_headers=True,
            skip_non_songs=True,
            excluded_terms=["Remix", "Live", "Demo", "Instrumental"],
        )
        self._client.verbose = False

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, *parts: str) -> str:
        name = "_".join(p.strip().replace("/", "_") for p in parts if p)
        safe = "".join(c for c in name if c.isalnum() or c in "-_. ")
        return safe.replace(" ", "_")[:200]

    def _clean_genius_lyrics(self, lyrics: str) -> str:
        if not lyrics:
            return ""
        text = lyrics.replace("\r\n", "\n")
        text = re.sub(r"^\d+\s*Contributors?.*?Lyrics\s*", "", text, flags=re.S)
        text = re.sub(r"\s*Embed\s*$", "", text)
        text = re.sub(r"\[[^\]]+\]", "", text)
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line and not line.endswith("Lyrics")]
        return "\n".join(lines).strip()

    def _has_minimum_lines(self, lyrics: str, min_lines: int = 4) -> bool:
        lines = [line for line in (raw_line.strip() for raw_line in lyrics.splitlines()) if line]
        return len(lines) >= min_lines

    def scrape(
        self,
        save: bool = True,
        out_dir: Optional[Path] = None,
    ) -> List[ScrapedText]:
        """Scrape configured artists and optionally save raw JSON files locally.

        Returns list of ScrapedText objects.
        """
        if out_dir is None:
            out_dir = Path(RAW_DATA_DIR)
        out_dir = Path(out_dir)
        self._ensure_dir(out_dir)

        samples: List[ScrapedText] = []
        for artist_name in self.artists:
            logging.info("Fetching artist: %s", artist_name)
            artist = self._client.search_artist(
                artist_name,
                max_songs=self.limit_per_artist,
                sort="popularity",
            )
            if artist is None:
                logging.warning("No artist found for %s", artist_name)
                continue

            for song in getattr(artist, "songs", []) or []:
                lyrics = self._clean_genius_lyrics(getattr(song, "lyrics", "") or "")
                if not lyrics:
                    continue
                # Optional guard: ensure chunking rule later (4 lines) can apply
                if not self._has_minimum_lines(lyrics, min_lines=4):
                    continue

                title = getattr(song, "title", "")
                author = getattr(artist, "name", artist_name)
                url = getattr(song, "url", None)

                scraped = ScrapedText(
                    source=self.source_name,
                    title=title,
                    author=author,
                    text=lyrics,
                    url=url,
                )
                samples.append(scraped)

                if save:
                    fname = f"genius_{self._safe_filename(author, title)}.json"
                    path = out_dir / fname
                    payload = {
                        "title": title,
                        "author": author,
                        "url": url,
                        "lyrics": lyrics,
                    }
                    with open(path, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, ensure_ascii=False, indent=2)

        return samples

    def scrape_song_ids(
        self,
        song_ids: Iterable[int],
        save: bool = True,
        out_dir: Optional[Path] = None,
    ) -> List[ScrapedText]:
        if out_dir is None:
            out_dir = Path(RAW_DATA_DIR)
        out_dir = Path(out_dir)
        self._ensure_dir(out_dir)

        samples: List[ScrapedText] = []
        for sid in song_ids:
            data = self._client.song(sid)
            song = data.get("song") if isinstance(data, dict) else data
            if not song:
                continue
            title = song.get("title") or ""
            artist = (song.get("primary_artist") or {}).get("name") or ""
            lyrics = song.get("lyrics") or ""
            lyrics = self._clean_genius_lyrics(lyrics)
            if not lyrics or not self._has_minimum_lines(lyrics):
                continue
            scraped = ScrapedText(
                source=self.source_name,
                title=title,
                author=artist,
                text=lyrics,
                url=song.get("url"),
            )
            samples.append(scraped)
            if save:
                fname = f"genius_{sid}_{self._safe_filename(artist, title)}.json"
                path = out_dir / fname
                payload = {
                    "title": title,
                    "author": artist,
                    "url": song.get("url"),
                    "lyrics": lyrics,
                }
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)

        return samples

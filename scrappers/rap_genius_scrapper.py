from __future__ import annotations

from dataclasses import dataclass
import os
import re

import lyricsgenius

from .base import ScrapedText


DEFAULT_RAP_ARTISTS = (
    "Taco Hemingway",
    "Pezet",
    "Łona",
    "Eldo",
)


@dataclass(slots=True)
class RapGeniusScraper:
    source_name = "rap_genius"
    artists: tuple[str, ...] = DEFAULT_RAP_ARTISTS
    limit_per_artist: int = 20
    access_token: str | None = None
    timeout_seconds: int = 30

    def scrape(self) -> list[ScrapedText]:
        token = self.access_token or os.getenv("GENIUS_ACCESS_TOKEN")
        if not token:
            raise RuntimeError(
                "Missing Genius token. Set GENIUS_ACCESS_TOKEN before running the rap scraper."
            )

        genius = lyricsgenius.Genius(
            token,
            timeout=self.timeout_seconds,
            retries=3,
            remove_section_headers=True,
            skip_non_songs=True,
            excluded_terms=["Remix", "Live", "Demo", "Instrumental"],
        )
        genius.verbose = False

        samples: list[ScrapedText] = []
        for artist_name in self.artists:
            artist = genius.search_artist(
                artist_name,
                max_songs=self.limit_per_artist,
                sort="popularity",
            )
            if artist is None:
                continue
            for song in artist.songs:
                lyrics = self._clean_genius_lyrics(song.lyrics or "")
                if lyrics:
                    samples.append(
                        ScrapedText(
                            source=self.source_name,
                            title=song.title,
                            author=artist.name or artist_name,
                            text=lyrics,
                            url=getattr(song, "url", None),
                        )
                    )
        return samples

    @staticmethod
    def _clean_genius_lyrics(lyrics: str) -> str:
        text = lyrics.replace("\r\n", "\n")
        text = re.sub(r"^\d+\s*Contributors?.*?Lyrics\s*", "", text, flags=re.S)
        text = re.sub(r"\s*Embed\s*$", "", text)
        text = re.sub(r"\[[^\]]+\]", "", text)
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line and not line.endswith("Lyrics")]
        return "\n".join(lines).strip()

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

from .base import ScrapedText

logger = logging.getLogger(__name__)

DEFAULT_MODERN_AUTHORS = (
    "Wislawa_Szymborska",
    "Czeslaw_Milosz",
    "Tadeusz_Rozewicz",
    "Andrzej_Bursa", 
    "Rafal_Wojaczek"
)

@dataclass(slots=True)
class ModernPoetryScraper:
    """Scraper dla poezji współczesnej z portalu poezja.org (w tym poeci wyklęci)"""
    """Scraper for modern poetry from poezja.org"""
    
    source_name: str = "poetry"
    author_slugs: tuple[str, ...] = DEFAULT_MODERN_AUTHORS
    limit_per_author: int = 15
    timeout_seconds: int = 10
    session: requests.Session = field(default_factory=requests.Session)
    
    base_url: str = "https://poezja.org/wz/"

    def __post_init__(self):
        """Sets global session headers to mimic a real browser."""
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "pl-PL,pl;q=0.9",
        })

    def scrape(self) -> list[ScrapedText]:
        samples: list[ScrapedText] = []
        for author_slug in self.author_slugs:
            logger.info(f"Rozpoczynam pobieranie: {author_slug.replace('_', ' ')}")
            try:
                links = self._get_poem_links(author_slug)
                logger.info(f"Znaleziono {len(links)} wierszy dla {author_slug}.")
                
                for i, link in enumerate(links[:self.limit_per_author], 1):
                    logger.info(f"[{i}/{min(self.limit_per_author, len(links))}] Pobieranie: {link}")
                    try:
                        sample = self._scrape_poem(link, author_slug)
                        if sample is not None:
                            samples.append(sample)
                    except Exception as e:
                        logger.warning(f"Błąd przy pobieraniu wiersza {link}: {e}")
            except Exception as e:
                logger.error(f"Błąd przy pobieraniu linków dla {author_slug}: {e}")
                
        return samples

    def _get_poem_links(self, author_slug: str) -> list[str]:
        author_url = f"{self.base_url}{author_slug}/"
        response = self.session.get(author_url, timeout=self.timeout_seconds)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith(author_url) and len(href) > len(author_url) + 2:
                links.add(href)
                
        links_list = list(links)
        random.shuffle(links_list)
            
        return links_list

    def _scrape_poem(self, url: str, author_slug: str) -> ScrapedText | None:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else url.split('/')[-1].replace('_', ' ')
        
        content_div = soup.find('div', class_=lambda c: c and ('content' in c.lower() or 'text' in c.lower() or 'poem' in c.lower()))
        if not content_div:
            content_div = soup.find('article')
            
        if not content_div:
            return None
            
        for a in content_div.find_all('a'):
            a.decompose()
            
        poem_lines = []
        text = content_div.get_text(separator="\n").strip()
        
        for line in text.split('\n'):
            cleaned_line = line.strip()
            if cleaned_line and "Czytaj dalej" not in cleaned_line and "Spis treści" not in cleaned_line:
                poem_lines.append(cleaned_line)
                
        full_text = "\n".join(poem_lines)
        if not full_text:
            return None
            
        return ScrapedText(
            source=self.source_name,
            title=title,
            author=author_slug.replace('_', ' '),
            text=full_text,
            url=url
        )
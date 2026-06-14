# DISCLAIMER
# Wiersze pochodzące z podanego scrappera są całkowiście autorstwa Zbigniewa Herberta i zostały użyte wyłącznie w celach edukacyjnych.
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

from .base import ScrapedText

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class HerbertScraper:
    """Scrapper for Zbigniew Herbert's poetry from fundacjaherberta.com"""
    
    source_name: str = "poetry"
    limit: int | None = 30
    randomize: bool = True
    timeout_seconds: int = 10
    session: requests.Session = field(default_factory=requests.Session)
    
    base_url: str = "https://fundacjaherberta.com/biblioteka-herberta/wiersze/"
    
    def __post_init__(self):
        """Sets global session headers to mimic a real browser."""
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1"
        })

    def scrape(self) -> list[ScrapedText]:
        samples: list[ScrapedText] = []
        try:
            links = self._get_poem_links()
            logger.info(f"Znaleziono {len(links)} wierszy Herberta do pobrania.")
            
            for i, link in enumerate(links, 1):
                logger.info(f"[{i}/{len(links)}] Pobieranie: {link}")
                try:
                    sample = self._scrape_poem(link)
                    if sample is not None:
                        samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to scrape poem {link}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch poem links: {e}")
            
        return samples

    def _get_poem_links(self) -> list[str]:
        response = self.session.get(self.base_url, timeout=self.timeout_seconds)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith(self.base_url) and len(href) > len(self.base_url):
                links.add(href)
                
        links_list = list(links)
        
        if self.randomize:
            random.shuffle(links_list)
        if self.limit:
            links_list = links_list[:self.limit]
            
        return links_list

    def _scrape_poem(self, url: str) -> ScrapedText | None:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else url.rstrip('/').split('/')[-1].replace('-', ' ').title()
        
        content_div = soup.find('div', class_=lambda c: c and 'fusion-content-tb' in c)
        if not content_div:
            logger.warning(f"Missing fusion-content-tb div at {url}")
            return None
            
        poem_lines = []
        for p in content_div.find_all('p'):
            style = p.get('style', '')
            if 'text-align: right' in style.lower():
                continue
            text = p.get_text(separator="\n").strip()
            if text:
                poem_lines.append(text)
                
        full_text = "\n".join(poem_lines)
        if not full_text:
            return None
            
        return ScrapedText(
            source=self.source_name,
            title=title,
            author="Zbigniew Herbert",
            text=full_text,
            url=url
        )
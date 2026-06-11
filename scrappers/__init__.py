"""Data fetchers for poetry and rap sources."""

from .poem_scrapper import PoemScraper
from .rap_genius_scrapper import RapGeniusScraper

__all__ = ["PoemScraper", "RapGeniusScraper"]

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """Normalize multiple spaces/tabs to single space."""
    return " ".join(text.split())


def normalize_line_whitespace(text: str) -> str:
    """Normalize spaces inside lines without flattening line breaks."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    # Remove http(s) URLs
    text = re.sub(r"https?://[^\s]+", "", text)
    # Remove www. URLs
    text = re.sub(r"www\.[^\s]+", "", text)
    return text


def remove_stage_directions(text: str) -> str:
    """Remove stage directions (didaskalia) in square brackets.
    
    Handles:
    - [Verse 1], [Chorus], [Bridge], [Intro], [Outro]
    - [instrumental], [beat drop]
    - Any other bracketed annotation
    """
    text = re.sub(r"\[[^\]]+\]", "", text)
    return text.strip()


def remove_empty_lines(text: str) -> str:
    """Remove empty lines, keeping only non-blank lines."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def clean_lyrics(text: str, remove_urls_flag: bool = True, remove_directions: bool = True) -> str:
    """Pipeline: clean lyrics by removing URLs, stage directions, normalizing whitespace.
    
    Args:
        text: Raw lyrics text.
        remove_urls_flag: Strip URLs if True.
        remove_directions: Strip stage directions/didaskalia if True.
    
    Returns:
        Cleaned text.
    """
    if not text or not isinstance(text, str):
        return ""
    
    if remove_urls_flag:
        text = remove_urls(text)
    if remove_directions:
        text = remove_stage_directions(text)
    
    # Normalize whitespace while preserving verse boundaries.
    text = normalize_line_whitespace(text)
    text = remove_empty_lines(text)
    
    return text.strip()


def count_lines(text: str) -> int:
    """Count non-empty lines in text."""
    return len([line for line in text.splitlines() if line.strip()])

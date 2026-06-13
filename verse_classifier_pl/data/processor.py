from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..core.cleaning import clean_lyrics, count_lines
from ..core.chunking import split_into_four_line_chunks
from .schemas import TextChunk

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process scraped raw JSONL texts into labeled chunks suitable for training.
    
    Pipeline:
    1. Load raw JSONL files from scrapers (rap_genius.jsonl, poetry.jsonl).
    2. Clean text (remove URLs, didaskalia).
    3. Split into 4-line chunks.
    4. Assign labels (0=poetry, 1=rap) based on source.
    5. Save processed dataset as JSON Lines.
    """

    RAP_LABEL = 1
    POETRY_LABEL = 0

    def __init__(self, raw_dir: Path, processed_dir: Path, min_lines: int = 4):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.min_lines = min_lines
        self._processed_count = 0
        self._skipped_count = 0

    def process_raw_files(self, output_file: Optional[Path] = None) -> List[TextChunk]:
        """Load all raw JSONL files and process into chunks."""
        chunks: List[TextChunk] = []
        
        if not self.raw_dir.exists():
            logger.warning(f"Raw data directory not found: {self.raw_dir}")
            return chunks

        rap_jsonl = self.raw_dir / "rap_genius.jsonl"
        if rap_jsonl.exists():
            logger.info(f"Processing {rap_jsonl}")
            try:
                chunks.extend(self._process_jsonl_file(rap_jsonl, label=self.RAP_LABEL))
            except Exception as e:
                logger.error(f"Failed to process {rap_jsonl}: {e}")

        poetry_jsonl = self.raw_dir / "poetry.jsonl"
        if poetry_jsonl.exists():
            logger.info(f"Processing {poetry_jsonl}")
            try:
                chunks.extend(self._process_jsonl_file(poetry_jsonl, label=self.POETRY_LABEL))
            except Exception as e:
                logger.error(f"Failed to process {poetry_jsonl}: {e}")

        if output_file:
            self._save_jsonl(chunks, output_file)

        logger.info(f"Processed {self._processed_count} chunks, skipped {self._skipped_count}")
        return chunks

    def _process_jsonl_file(self, fpath: Path, label: int) -> List[TextChunk]:
        """Load JSONL file, clean, chunk, and label."""
        chunks: List[TextChunk] = []
        source_name = "rap_genius" if label == self.RAP_LABEL else "poetry"

        with open(fpath, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at {fpath}:{line_no}: {e}")
                    self._skipped_count += 1
                    continue

                title = data.get("title", "")
                author = data.get("author", "")
                lyrics = data.get("text", "")

                if not title or not lyrics:
                    self._skipped_count += 1
                    continue

                cleaned = clean_lyrics(lyrics, remove_urls_flag=True, remove_directions=True)
                if not cleaned or count_lines(cleaned) < self.min_lines:
                    self._skipped_count += 1
                    continue

                text_chunks = split_into_four_line_chunks(cleaned)
                if not text_chunks:
                    self._skipped_count += 1
                    continue

                for chunk_idx, chunk_text in enumerate(text_chunks):
                    lines = tuple(chunk_text.split("\n"))
                    chunk_obj = TextChunk(
                        source=source_name,
                        label=label,
                        title=title,
                        author=author,
                        chunk_index=chunk_idx,
                        lines=lines,
                    )
                    chunks.append(chunk_obj)
                    self._processed_count += 1

        return chunks

    def _save_jsonl(self, chunks: List[TextChunk], output_file: Path) -> None:
        """Save chunks as JSON Lines format."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as fh:
            for chunk in chunks:
                line = json.dumps({
                    "source": chunk.source,
                    "label": chunk.label,
                    "title": chunk.title,
                    "author": chunk.author,
                    "chunk_index": chunk.chunk_index,
                    "lines": chunk.lines,
                    "text": "\n".join(chunk.lines),
                }, ensure_ascii=False)
                fh.write(line + "\n")

        logger.info(f"Saved {len(chunks)} chunks to {output_file}")

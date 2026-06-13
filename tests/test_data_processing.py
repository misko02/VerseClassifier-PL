from __future__ import annotations

import json

from verse_classifier_pl.core.cleaning import clean_lyrics, count_lines
from verse_classifier_pl.data.processor import DataProcessor


def test_clean_lyrics_preserves_line_boundaries() -> None:
    raw = " pierwszy   wers \n\ndrugi\twers\n[Refren]\ntrzeci wers\nczwarty wers"

    cleaned = clean_lyrics(raw)

    assert count_lines(cleaned) == 4
    assert cleaned.splitlines() == [
        "pierwszy wers",
        "drugi wers",
        "trzeci wers",
        "czwarty wers",
    ]


def test_processor_reads_repo_raw_jsonl_and_creates_four_line_chunks(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    sample = {
        "source": "poetry",
        "title": "Test",
        "author": "Anon",
        "text": "a\nb\nc\nd\ne\nf\ng\nh",
    }
    (raw_dir / "poetry.jsonl").write_text(
        json.dumps(sample, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    processor = DataProcessor(raw_dir=raw_dir, processed_dir=processed_dir)
    chunks = processor.process_raw_files(output_file=processed_dir / "combined.jsonl")

    assert len(chunks) == 2
    assert chunks[0].label == DataProcessor.POETRY_LABEL
    assert chunks[0].lines == ("a", "b", "c", "d")
    assert (processed_dir / "combined.jsonl").read_text(encoding="utf-8").count("\n") == 2

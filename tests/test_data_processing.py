from __future__ import annotations

import json

from verse_classifier_pl.core.cleaning import clean_lyrics, count_lines
from verse_classifier_pl.data.dataset import split_chunks_by_work
from verse_classifier_pl.data.processor import DataProcessor
from verse_classifier_pl.data.schemas import TextChunk


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


def test_split_chunks_by_work_keeps_one_title_in_one_split() -> None:
    chunks = []
    for label, source in [(0, "poetry"), (1, "rap_genius")]:
        for work_idx in range(10):
            for chunk_idx in range(2):
                chunks.append(
                    TextChunk(
                        source=source,
                        label=label,
                        title=f"title-{label}-{work_idx}",
                        author=f"author-{label}",
                        chunk_index=chunk_idx,
                        lines=("a", "b", "c", "d"),
                    )
                )

    train, val, test = split_chunks_by_work(
        chunks,
        test_size=0.2,
        val_size=0.2,
        random_seed=42,
    )

    split_names_by_work = {}
    for split_name, split_chunks in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        for chunk in split_chunks:
            work = (chunk.source, chunk.author, chunk.title)
            split_names_by_work.setdefault(work, split_name)
            assert split_names_by_work[work] == split_name

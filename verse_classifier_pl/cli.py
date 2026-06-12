from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_RANDOM_SEED
from .data.paths import RAW_DATA_DIR
from scrappers.base import write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-classifier",
        description="Scaffold CLI for the VerseClassifier-PL project.",
    )
    subparsers = parser.add_subparsers(dest="command")

    scrape_parser = subparsers.add_parser("scrape", help="Fetch raw texts locally.")
    scrape_parser.add_argument(
        "source",
        choices=("poetry", "rap", "all"),
        help="Data source to fetch.",
    )
    scrape_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory for generated JSONL files.",
    )
    scrape_parser.add_argument(
        "--limit-per-author",
        type=int,
        default=20,
        help="Maximum poems fetched for each Wolne Lektury author.",
    )
    scrape_parser.add_argument(
        "--limit-per-artist",
        type=int,
        default=20,
        help="Maximum songs fetched for each Genius artist.",
    )
    scrape_parser.add_argument(
        "--poet",
        action="append",
        dest="poets",
        help="Wolne Lektury author slug. Defaults are configured in the poetry scraper.",
    )
    scrape_parser.add_argument(
        "--artist",
        action="append",
        dest="artists",
        help="Genius artist name. Defaults are configured in the rap scraper.",
    )
    subparsers.add_parser("prepare-data", help="Clean and chunk raw texts.")
    subparsers.add_parser("train-baseline", help="Train the baseline model.")
    subparsers.add_parser("train-transformer", help="Fine-tune the transformer.")
    subparsers.add_parser("evaluate", help="Run evaluation and XAI reports.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "scrape":
        return run_scrape(args)

    print(f"{args.command}: scaffold placeholder (seed={DEFAULT_RANDOM_SEED})")
    return 0


def run_scrape(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir

    if args.source in {"poetry", "all"}:
        from scrappers.poem_scrapper import DEFAULT_POET_SLUGS, PoemScraper

        poets = tuple(args.poets) if args.poets else DEFAULT_POET_SLUGS
        poetry = PoemScraper(author_slugs=poets, limit_per_author=args.limit_per_author)
        poetry_samples = poetry.scrape()
        poetry_path = output_dir / "poetry.jsonl"
        write_jsonl(poetry_samples, poetry_path)
        print(f"Saved {len(poetry_samples)} poetry texts to {poetry_path}")

    if args.source in {"rap", "all"}:
        from scrappers.rap_genius_scrapper import DEFAULT_RAP_ARTISTS, RapGeniusScraper

        artists = tuple(args.artists) if args.artists else DEFAULT_RAP_ARTISTS
        rap = RapGeniusScraper(artists=artists, limit_per_artist=args.limit_per_artist)
        rap_samples = rap.scrape()
        rap_path = output_dir / "rap_genius.jsonl"
        write_jsonl(rap_samples, rap_path)
        print(f"Saved {len(rap_samples)} rap texts to {rap_path}")

    return 0

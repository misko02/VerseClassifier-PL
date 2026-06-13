from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_RANDOM_SEED
from .data.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from scrappers.base import write_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-classifier",
        description="VerseClassifier-PL: Polish rap vs poetry classifier.",
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
        help="Directory for raw JSON files (default: .data/raw).",
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
        help="Wolne Lektury author slug. Repeat to add multiple.",
    )
    scrape_parser.add_argument(
        "--artist",
        action="append",
        dest="artists",
        help="Genius artist name. Repeat to add multiple.",
    )

    # Prepare-data subcommand
    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Clean, chunk, and label raw texts.",
    )
    prepare_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory with raw JSON files (default: .data/raw).",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Output directory for processed JSONL (default: .data/processed).",
    )
    prepare_parser.add_argument(
        "--output-file",
        type=str,
        default="combined.jsonl",
        help="Output filename (default: combined.jsonl).",
    )

    subparsers.add_parser("train-baseline", help="Train the baseline (TF-IDF + LR) model.")
    subparsers.add_parser("train-transformer", help="Fine-tune HerBERT transformer.")
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
    elif args.command == "prepare-data":
        return run_prepare_data(args)

    logger.info(f"{args.command}: scaffold placeholder (seed={DEFAULT_RANDOM_SEED})")
    return 0


def run_scrape(args: argparse.Namespace) -> int:
    """Run scrapers for poetry and/or rap."""
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source in {"poetry", "all"}:
        try:
            from scrappers.poem_scrapper import DEFAULT_POET_SLUGS, PoemScraper

            poets = tuple(args.poets) if args.poets else DEFAULT_POET_SLUGS
            logger.info(f"Scraping poetry from authors: {poets}")
            scraper = PoemScraper(
                author_slugs=poets,
                limit_per_author=args.limit_per_author,
            )
            samples = scraper.scrape()
            output_path = output_dir / "poetry.jsonl"
            write_jsonl(samples, output_path)
            logger.info(f"Scraped {len(samples)} poetry texts")
            logger.info(f"Saved poetry raw data to {output_path}")
        except Exception as e:
            logger.error(f"Poetry scraper failed: {e}")
            return 1

    if args.source in {"rap", "all"}:
        try:
            from scrappers.rap_genius_scrapper import DEFAULT_RAP_ARTISTS, RapGeniusScraper

            artists = tuple(args.artists) if args.artists else DEFAULT_RAP_ARTISTS
            logger.info(f"Scraping rap from artists: {artists}")
            scraper = RapGeniusScraper(
                artists=artists,
                limit_per_artist=args.limit_per_artist,
            )
            samples = scraper.scrape(save=False)
            output_path = output_dir / "rap_genius.jsonl"
            write_jsonl(samples, output_path)
            logger.info(f"Scraped {len(samples)} rap texts")
            logger.info(f"Saved rap raw data to {output_path}")
        except Exception as e:
            logger.error(f"Rap scraper failed: {e}")
            return 1

    logger.info(f"Raw data saved to {output_dir}")
    return 0


def run_prepare_data(args: argparse.Namespace) -> int:
    """Process raw JSON files into labeled chunks."""
    try:
        from .data.processor import DataProcessor

        raw_dir = Path(args.raw_dir)
        output_dir = Path(args.output_dir)
        output_file = output_dir / args.output_file

        logger.info(f"Processing raw data from {raw_dir}")
        processor = DataProcessor(raw_dir=raw_dir, processed_dir=output_dir, min_lines=4)
        chunks = processor.process_raw_files(output_file=output_file)
        logger.info(f"Processed {len(chunks)} chunks, saved to {output_file}")
        return 0
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return 1

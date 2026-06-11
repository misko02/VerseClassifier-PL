from __future__ import annotations

import argparse

from .config import DEFAULT_RANDOM_SEED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-classifier",
        description="Scaffold CLI for the VerseClassifier-PL project.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("scrape", help="Prepare data fetchers.")
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

    print(f"{args.command}: scaffold placeholder (seed={DEFAULT_RANDOM_SEED})")
    return 0

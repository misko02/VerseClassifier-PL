from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import ARTIFACTS_DIR, DEFAULT_RANDOM_SEED
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

    baseline_parser = subparsers.add_parser(
        "train-baseline",
        help="Train the baseline (TF-IDF + Logistic Regression) model.",
    )
    baseline_parser.add_argument(
        "--dataset",
        type=Path,
        default=PROCESSED_DATA_DIR / "combined.jsonl",
        help="Processed JSONL dataset produced by prepare-data.",
    )
    baseline_parser.add_argument(
        "--model-dir",
        type=Path,
        default=ARTIFACTS_DIR / "baseline",
        help="Directory for baseline model artifacts.",
    )
    baseline_parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "splits",
        help="Directory for train/val/test split JSONL files.",
    )
    baseline_parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of works reserved for test split.",
    )
    baseline_parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of works reserved for validation split.",
    )
    baseline_parser.add_argument(
        "--max-train-samples-per-class",
        type=int,
        default=None,
        help="Optional class-balanced cap for faster baseline experiments.",
    )
    baseline_parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for splits and model training.",
    )
    baseline_parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Maximum number of TF-IDF features.",
    )
    baseline_parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF terms.",
    )
    baseline_parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum word n-gram size for TF-IDF.",
    )
    subparsers.add_parser("train-transformer", help="Fine-tune HerBERT transformer.")
    predict_parser = subparsers.add_parser(
        "predict",
        help="Classify a custom text with a trained baseline model.",
    )
    predict_parser.add_argument(
        "--model",
        type=Path,
        default=ARTIFACTS_DIR / "baseline" / "model.joblib",
        help="Path to trained baseline model.joblib.",
    )
    input_group = predict_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Text to classify.",
    )
    input_group.add_argument(
        "--file",
        type=Path,
        help="UTF-8 text file to classify.",
    )
    predict_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
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
    elif args.command == "train-baseline":
        return run_train_baseline(args)
    elif args.command == "predict":
        return run_predict(args)

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


def run_train_baseline(args: argparse.Namespace) -> int:
    """Train the first TF-IDF + Logistic Regression baseline."""
    try:
        from .data.dataset import (
            class_counts,
            load_chunks_jsonl,
            sample_per_class,
            save_chunks_jsonl,
            split_chunks_by_work,
        )
        from .evaluation.metrics import compute_classification_metrics
        from .modeling.baseline import BaselineConfig, BaselineTextClassifier

        chunks = load_chunks_jsonl(args.dataset)
        logger.info(f"Loaded {len(chunks)} chunks from {args.dataset}")

        train_chunks, val_chunks, test_chunks = split_chunks_by_work(
            chunks,
            test_size=args.test_size,
            val_size=args.val_size,
            random_seed=args.random_seed,
        )
        train_chunks = sample_per_class(
            train_chunks,
            max_samples_per_class=args.max_train_samples_per_class,
            random_seed=args.random_seed,
        )

        args.split_dir.mkdir(parents=True, exist_ok=True)
        save_chunks_jsonl(train_chunks, args.split_dir / "train.jsonl")
        save_chunks_jsonl(val_chunks, args.split_dir / "val.jsonl")
        save_chunks_jsonl(test_chunks, args.split_dir / "test.jsonl")

        logger.info(f"Train chunks: {len(train_chunks)} {class_counts(train_chunks)}")
        logger.info(f"Val chunks: {len(val_chunks)} {class_counts(val_chunks)}")
        logger.info(f"Test chunks: {len(test_chunks)} {class_counts(test_chunks)}")

        model = BaselineTextClassifier(
            BaselineConfig(
                max_features=args.max_features,
                min_df=args.min_df,
                ngram_min=1,
                ngram_max=args.ngram_max,
                random_seed=args.random_seed,
            )
        )
        model.fit(train_chunks)

        val_predictions = model.predict(val_chunks)
        test_predictions = model.predict(test_chunks)
        val_metrics = compute_classification_metrics(
            [chunk.label for chunk in val_chunks],
            val_predictions,
        )
        test_metrics = compute_classification_metrics(
            [chunk.label for chunk in test_chunks],
            test_predictions,
        )

        args.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.model_dir / "model.joblib"
        metrics_path = args.model_dir / "metrics.json"
        model.save(model_path)
        metrics = {
            "dataset": str(args.dataset),
            "model": "tfidf_logistic_regression",
            "random_seed": args.random_seed,
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
            "test_chunks": len(test_chunks),
            "train_class_counts": class_counts(train_chunks),
            "val_class_counts": class_counts(val_chunks),
            "test_class_counts": class_counts(test_chunks),
            "validation": val_metrics,
            "test": test_metrics,
        }
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(f"Saved baseline model to {model_path}")
        logger.info(f"Saved metrics to {metrics_path}")
        logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        return 0
    except Exception as e:
        logger.error(f"Baseline training failed: {e}")
        return 1


def run_predict(args: argparse.Namespace) -> int:
    """Classify a custom text with a saved baseline model."""
    try:
        from .core.cleaning import clean_lyrics
        from .modeling.baseline import BaselineTextClassifier

        text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
        cleaned_text = clean_lyrics(text)
        if not cleaned_text:
            raise ValueError("Input text is empty after cleaning.")

        model = BaselineTextClassifier.load(args.model)
        label = model.predict_texts([cleaned_text])[0]
        probabilities = model.predict_text_probabilities([cleaned_text])[0]
        label_name = _label_name(label)
        poetry_probability = probabilities.get(0, 0.0)
        rap_probability = probabilities.get(1, 0.0)

        if args.json:
            print(
                json.dumps(
                    {
                        "label": label,
                        "label_name": label_name,
                        "probabilities": {
                            "poetry": poetry_probability,
                            "rap": rap_probability,
                        },
                        "cleaned_text": cleaned_text,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Prediction: {label_name} (label={label})")
            print(f"poetry: {poetry_probability:.4f}")
            print(f"rap: {rap_probability:.4f}")
        return 0
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1


def _label_name(label: int) -> str:
    return "rap" if label == 1 else "poetry"

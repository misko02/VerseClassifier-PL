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
        choices=("poetry", "rap", "herbert", "modern", "all"),
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
    scrape_parser.add_argument(
        "--limit-herbert",
        type=int,
        default=30,
        help="Maximum poems fetched from Herbert Foundation.",
    )
    scrape_parser.add_argument(
        "--limit-modern",
        type=int,
        default=15,
        help="Maximum poems fetched per modern author from poezja.org.",
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
    transformer_parser = subparsers.add_parser(
        "train-transformer",
        help="Fine-tune HerBERT transformer.",
    )
    transformer_parser.add_argument(
        "--dataset",
        type=Path,
        default=PROCESSED_DATA_DIR / "combined.jsonl",
        help="Processed JSONL dataset produced by prepare-data.",
    )
    transformer_parser.add_argument(
        "--model-dir",
        type=Path,
        default=ARTIFACTS_DIR / "transformer",
        help="Directory for transformer model artifacts.",
    )
    transformer_parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "splits_transformer",
        help="Directory for train/val/test split JSONL files.",
    )
    transformer_parser.add_argument(
        "--pretrained-model",
        type=str,
        default="allegro/herbert-base-cased",
        help="Hugging Face model name or local path.",
    )
    transformer_parser.add_argument("--epochs", type=float, default=3.0)
    transformer_parser.add_argument("--learning-rate", type=float, default=2e-5)
    transformer_parser.add_argument("--train-batch-size", type=int, default=8)
    transformer_parser.add_argument("--eval-batch-size", type=int, default=16)
    transformer_parser.add_argument("--max-length", type=int, default=128)
    transformer_parser.add_argument("--weight-decay", type=float, default=0.01)
    transformer_parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of works reserved for test split.",
    )
    transformer_parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of works reserved for validation split.",
    )
    transformer_parser.add_argument(
        "--max-train-samples-per-class",
        type=int,
        default=None,
        help="Optional class-balanced cap for quick transformer experiments.",
    )
    transformer_parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for splits and model training.",
    )
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
    predict_parser.add_argument(
        "--model-type",
        choices=("baseline", "transformer"),
        default="baseline",
        help="Type of the model to use for prediction.",
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
    elif args.command == "train-transformer":
        return run_train_transformer(args)
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
        
    if args.source in {"herbert", "all"}:
        try:
            from scrappers.herbert_scrapper import HerbertScraper
            logger.info("Scraping Zbigniew Herbert poetry...")
            scraper = HerbertScraper(limit=args.limit_herbert)
            samples = scraper.scrape()
            
            output_path = output_dir / "poetry.jsonl"
            write_jsonl(samples, output_path, append=True)
            
            logger.info(f"Scraped {len(samples)} Herbert poetry texts")
            logger.info(f"Appended Herbert poetry to {output_path}")
        except Exception as e:
            logger.error(f"Herbert poetry scraper failed: {e}")
            return 1
    if args.source in {"modern", "all"}:
        try:
            from scrappers.modern_scrapper import ModernPoetryScraper

            logger.info("Scraping modern poetry (poezja.org)...")
            scraper = ModernPoetryScraper(limit_per_author=args.limit_modern)
            samples = scraper.scrape()
            
            output_path = output_dir / "poetry.jsonl"
            write_jsonl(samples, output_path, append=True)
            
            logger.info(f"Scraped {len(samples)} modern poetry texts")
            logger.info(f"Appended modern poetry to {output_path}")
        except Exception as e:
            logger.error(f"Modern poetry scraper failed: {e}")
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
    """Classify a custom text with a saved model."""
    try:
        from .core.cleaning import clean_lyrics

        text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
        cleaned_text = clean_lyrics(text)
        if not cleaned_text:
            raise ValueError("Input text is empty after cleaning.")

        if args.model_type == "baseline":
            from .modeling.baseline import BaselineTextClassifier
            model = BaselineTextClassifier.load(args.model)
            label = model.predict_texts([cleaned_text])[0]
            probabilities = model.predict_text_probabilities([cleaned_text])[0]
            poetry_probability = probabilities.get(0, 0.0)
            rap_probability = probabilities.get(1, 0.0)

        elif args.model_type == "transformer":
                    from .modeling.transformer import TransformerTextClassifier
                    from .data.schemas import TextChunk
                    
                    model = TransformerTextClassifier.load(args.model)
                    dummy_chunk = TextChunk(
                        source="predict",
                        title="custom",
                        author="custom",
                        lines=cleaned_text.split("\n"),
                        label=0,
                        chunk_index=0
                    )
                    
                    probabilities = model.predict_probabilities([dummy_chunk])[0]
                    poetry_probability = probabilities.get(0, 0.0)
                    rap_probability = probabilities.get(1, 0.0)
                    
                    label = 1 if rap_probability > poetry_probability else 0
        label_name = _label_name(label)

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


def run_train_transformer(args: argparse.Namespace) -> int:
    """Fine-tune HerBERT on processed 4-line chunks."""
    try:
        from .data.dataset import (
            class_counts,
            load_chunks_jsonl,
            sample_per_class,
            save_chunks_jsonl,
            split_chunks_by_work,
        )
        from .evaluation.metrics import compute_classification_metrics
        from .modeling.transformer import TransformerConfig, TransformerTextClassifier

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

        model = TransformerTextClassifier(
            TransformerConfig(
                model_name=args.pretrained_model,
                max_length=args.max_length,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                weight_decay=args.weight_decay,
                random_seed=args.random_seed,
            )
        )
        val_metrics = model.train(
            train_samples=train_chunks,
            val_samples=val_chunks,
            output_dir=args.model_dir,
        )
        test_predictions = model.predict(test_chunks)
        test_metrics = compute_classification_metrics(
            [chunk.label for chunk in test_chunks],
            test_predictions,
        )

        metrics = {
            "dataset": str(args.dataset),
            "model": "herbert_sequence_classification",
            "pretrained_model": args.pretrained_model,
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
        args.model_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.model_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(f"Saved transformer model to {args.model_dir / 'model'}")
        logger.info(f"Saved metrics to {metrics_path}")
        if "eval_f1" in val_metrics:
            logger.info(f"Validation F1: {val_metrics['eval_f1']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        return 0
    except Exception as e:
        logger.error(f"Transformer training failed: {e}")
        return 1

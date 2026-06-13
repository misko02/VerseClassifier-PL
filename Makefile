.PHONY: help install lint test scrape-poetry scrape-rap scrape-all prepare-data train-baseline predict train-transformer evaluate

PYTHON ?= poetry run python
POETRY_LIMIT ?= 20
RAP_LIMIT ?= 20
OUTPUT_DIR ?= .data/raw
DATASET ?= .data/processed/combined.jsonl
BASELINE_DIR ?= .artifacts/baseline
SPLIT_DIR ?= .data/processed/splits
TEST_SIZE ?= 0.15
VAL_SIZE ?= 0.15
MAX_TRAIN_SAMPLES_PER_CLASS ?=
TEXT ?=
MODEL ?= .artifacts/baseline/model.joblib

help:
	@printf "VerseClassifier-PL commands:\n"
	@printf "  make install            Install dependencies with Poetry\n"
	@printf "  make lint               Run Ruff checks\n"
	@printf "  make test               Run pytest\n"
	@printf "  make scrape-poetry      Fetch poems from Wolne Lektury into .data/raw/poetry.jsonl\n"
	@printf "  make scrape-rap         Fetch rap lyrics from Genius into .data/raw/rap_genius.jsonl\n"
	@printf "  make scrape-all         Fetch both poetry and rap raw datasets\n"
	@printf "  make prepare-data       Build .data/processed/combined.jsonl\n"
	@printf "  make train-baseline     Train TF-IDF + Logistic Regression baseline\n"
	@printf "  make predict TEXT='...' Classify a custom text with baseline model\n"
	@printf "\nConfig variables:\n"
	@printf "  POETRY_LIMIT=20 RAP_LIMIT=20 OUTPUT_DIR=.data/raw\n"
	@printf "  DATASET=.data/processed/combined.jsonl BASELINE_DIR=.artifacts/baseline\n"
	@printf "  MODEL=.artifacts/baseline/model.joblib TEXT='custom text'\n"
	@printf "  TEST_SIZE=0.15 VAL_SIZE=0.15 MAX_TRAIN_SAMPLES_PER_CLASS=\n"
	@printf "  GENIUS_ACCESS_TOKEN is required for rap scraping\n"

install:
	poetry install

lint:
	poetry run ruff check .

test:
	poetry run pytest

scrape-poetry:
	$(PYTHON) -m verse_classifier_pl scrape poetry --limit-per-author $(POETRY_LIMIT) --output-dir $(OUTPUT_DIR)

scrape-rap:
	$(PYTHON) -m verse_classifier_pl scrape rap --limit-per-artist $(RAP_LIMIT) --output-dir $(OUTPUT_DIR)

scrape-all:
	$(PYTHON) -m verse_classifier_pl scrape all --limit-per-author $(POETRY_LIMIT) --limit-per-artist $(RAP_LIMIT) --output-dir $(OUTPUT_DIR)

prepare-data:
	$(PYTHON) -m verse_classifier_pl prepare-data

train-baseline:
	$(PYTHON) -m verse_classifier_pl train-baseline --dataset $(DATASET) --model-dir $(BASELINE_DIR) --split-dir $(SPLIT_DIR) --test-size $(TEST_SIZE) --val-size $(VAL_SIZE) $(if $(MAX_TRAIN_SAMPLES_PER_CLASS),--max-train-samples-per-class $(MAX_TRAIN_SAMPLES_PER_CLASS),)

predict:
	$(PYTHON) -m verse_classifier_pl predict --model $(MODEL) --text "$(TEXT)"

train-transformer:
	$(PYTHON) -m verse_classifier_pl train-transformer

evaluate:
	$(PYTHON) -m verse_classifier_pl evaluate

.PHONY: help install lint test scrape-poetry scrape-rap scrape-all prepare-data train-baseline predict train-transformer train-transformer-smoke evaluate

PYTHON ?= poetry run python
RAP_LIMIT ?= 15
POETRY_LIMIT ?= 20
HERBERT_LIMIT ?= 30
MODERN_LIMIT ?= 20
OUTPUT_DIR ?= .data/raw
DATASET ?= .data/processed/combined.jsonl
BASELINE_DIR ?= .artifacts/baseline
TRANSFORMER_DIR ?= .artifacts/transformer
SPLIT_DIR ?= .data/processed/splits
TRANSFORMER_SPLIT_DIR ?= .data/processed/splits_transformer
TEST_SIZE ?= 0.15
VAL_SIZE ?= 0.15
MAX_TRAIN_SAMPLES_PER_CLASS ?= 1500
TEXT ?=
FILE ?=
MODEL ?= .artifacts/baseline/model.joblib
PRETRAINED_MODEL ?= allegro/herbert-base-cased
TRANSFORMER_MODEL ?= .artifacts/transformer/model
EPOCHS ?= 3
TRAIN_BATCH_SIZE ?= 8
EVAL_BATCH_SIZE ?= 16
MAX_LENGTH ?= 128
LEARNING_RATE ?= 2e-5

help:
	@printf "VerseClassifier-PL commands:\n"
	@printf "  make install            Install dependencies with Poetry\n"
	@printf "  make lint               Run Ruff checks\n"
	@printf "  make test               Run pytest\n"
	@printf "  make scrape-poetry      Fetch poems from Wolne Lektury into .data/raw/poetry.jsonl\n"
	@printf "  make scrape-herbert     Fetch Herbert's poems from fundacjaherberta.com into .data/raw/herbert_poetry.jsonl\n"
	@printf "  make scrape-modern      Fetch poems from modern authors on poezja.org into .data/raw/modern_poetry.jsonl\n"
	@printf "  make scrape-rap         Fetch rap lyrics from Genius into .data/raw/rap_genius.jsonl\n"
	@printf "  make scrape-all         Fetch both poetry and rap raw datasets\n"
	@printf "  make prepare-data       Build .data/processed/combined.jsonl\n"
	@printf "  make train-baseline     Train TF-IDF + Logistic Regression baseline\n"
	@printf "  make predict TEXT='...' Classify a custom text with baseline model\n"
	@printf "  make train-transformer  Fine-tune HerBERT transformer\n"
	@printf "  make train-transformer-smoke  Run a tiny transformer training smoke-test\n"
	@printf "\nConfig variables:\n"
	@printf "  POETRY_LIMIT=20 RAP_LIMIT=15 OUTPUT_DIR=.data/raw\n"
	@printf "  DATASET=.data/processed/combined.jsonl BASELINE_DIR=.artifacts/baseline\n"
	@printf "  MODEL=.artifacts/baseline/model.joblib TEXT='custom text'\n"
	@printf "  PRETRAINED_MODEL=allegro/herbert-base-cased TRANSFORMER_DIR=.artifacts/transformer\n"
	@printf "  TEST_SIZE=0.15 VAL_SIZE=0.15 MAX_TRAIN_SAMPLES_PER_CLASS=1500\n"
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

scrape-herbert:
	$(PYTHON) -m verse_classifier_pl scrape herbert --limit-herbert $(HERBERT_LIMIT) --output-dir $(OUTPUT_DIR)

scrape-modern:
	$(PYTHON) -m verse_classifier_pl scrape modern --limit-modern $(MODERN_LIMIT) --output-dir $(OUTPUT_DIR)

scrape-all:
	$(PYTHON) -m verse_classifier_pl scrape all --limit-per-author $(POETRY_LIMIT) --limit-per-artist $(RAP_LIMIT) --limit-herbert $(HERBERT_LIMIT) --limit-modern $(MODERN_LIMIT) --output-dir $(OUTPUT_DIR)
prepare-data:
	$(PYTHON) -m verse_classifier_pl prepare-data

train-baseline:
	$(PYTHON) -m verse_classifier_pl train-baseline --dataset $(DATASET) --model-dir $(BASELINE_DIR) --split-dir $(SPLIT_DIR) --test-size $(TEST_SIZE) --val-size $(VAL_SIZE) $(if $(MAX_TRAIN_SAMPLES_PER_CLASS),--max-train-samples-per-class $(MAX_TRAIN_SAMPLES_PER_CLASS),)

predict:
	$(PYTHON) -m verse_classifier_pl predict --model $(MODEL) --text "$(TEXT)"

train-transformer:
	$(PYTHON) -m verse_classifier_pl train-transformer --dataset $(DATASET) --model-dir $(TRANSFORMER_DIR) --split-dir $(TRANSFORMER_SPLIT_DIR) --pretrained-model $(PRETRAINED_MODEL) --epochs $(EPOCHS) --learning-rate $(LEARNING_RATE) --train-batch-size $(TRAIN_BATCH_SIZE) --eval-batch-size $(EVAL_BATCH_SIZE) --max-length $(MAX_LENGTH) --test-size $(TEST_SIZE) --val-size $(VAL_SIZE) $(if $(MAX_TRAIN_SAMPLES_PER_CLASS),--max-train-samples-per-class $(MAX_TRAIN_SAMPLES_PER_CLASS),)

train-transformer-smoke:
	$(MAKE) train-transformer EPOCHS=1 MAX_TRAIN_SAMPLES_PER_CLASS=32 TRAIN_BATCH_SIZE=4 EVAL_BATCH_SIZE=8

predict-transformer:
	$(PYTHON) -m verse_classifier_pl predict --model-type transformer --model $(TRANSFORMER_MODEL) $(if $(TEXT),--text "$(TEXT)",) $(if $(FILE),--file $(FILE),)

evaluate:
	$(PYTHON) -m verse_classifier_pl evaluate

.PHONY: help install lint test scrape-poetry scrape-rap scrape-all prepare-data train-baseline train-transformer evaluate

PYTHON ?= poetry run python
POETRY_LIMIT ?= 20
RAP_LIMIT ?= 20
OUTPUT_DIR ?= .data/raw

help:
	@printf "VerseClassifier-PL commands:\n"
	@printf "  make install            Install dependencies with Poetry\n"
	@printf "  make lint               Run Ruff checks\n"
	@printf "  make test               Run pytest\n"
	@printf "  make scrape-poetry      Fetch poems from Wolne Lektury into .data/raw/poetry.jsonl\n"
	@printf "  make scrape-rap         Fetch rap lyrics from Genius into .data/raw/rap_genius.jsonl\n"
	@printf "  make scrape-all         Fetch both poetry and rap raw datasets\n"
	@printf "\nConfig variables:\n"
	@printf "  POETRY_LIMIT=20 RAP_LIMIT=20 OUTPUT_DIR=.data/raw\n"
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
	$(PYTHON) -m verse_classifier_pl train-baseline

train-transformer:
	$(PYTHON) -m verse_classifier_pl train-transformer

evaluate:
	$(PYTHON) -m verse_classifier_pl evaluate

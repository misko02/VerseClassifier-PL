# VerseClassifier-PL
## 🇵🇱 O projekcie
Klasyfikator AI wykorzystujący uczenie głębokie do oceny, czy podany fragment tekstu jest wersem z polskiego rapu, czy fragmentem wiersza. 

Projekt składa się ze scraperów tworzących autorski zbiór danych na podstawie API strony Genius.com oraz publicznie dostępnych w internecie wierszy, a także z samego klasyfikatora opartego o architekturę Transformer (model HerBERT).

## 🇬🇧 About the project
An AI classifier that uses deep learning to determine whether a given text snippet is a verse from a Polish rap song or an excerpt from a poem.

The project consists of scrapers that generate datasets using the Genius.com API and publicly available poems found onlone, as well as the classifier itself, which is based on the HerBERT model. 

---

## ⚠️ Nota prawna / Legal Disclaimer (Data Collection)
Ze względu na ochronę praw autorskich twórców współczesnych (zarówno raperów, jak i poetów), to repozytorium **nie zawiera** gotowego, surowego zbioru tekstów (plików `.json` / `.csv`). 
W repozytorium udostępniono jedynie skrypty pobierające (`scrapers/`), które pozwalają na samodzielne odtworzenie zbioru w celach edukacyjnych i badawczych na własnym komputerze.

---

## 🧱 Startowa struktura repozytorium
- `main.py` - lokalny entrypoint do uruchamiania projektu.
- `verse_classifier_pl/` - właściwy pakiet aplikacji.
- `scrappers/` - fetchery danych dla rapu i poezji.
- `tests/` - jednostkowe i integracyjne testy.
- `.data/` - **prywatne, lokalne zbiory robocze** (`raw`, `interim`, `processed`) – poza publicznym repozytorium ze względu na prawa autorskie.
- `.artifacts/` - **lokalne wyniki eksperymentów**: checkpointy modeli, metryki, raporty, wykresy – poza repozytorium.

---

## 🕸️ Uruchamianie scraperów
Scrapery zapisują dane wyłącznie lokalnie do `.data/raw/`, które jest ignorowane przez Git. Format wyjściowy to JSONL: jeden utwór na linię z polami `source`, `title`, `author`, `text`, `url`.

### Instalacja zależności
```bash
poetry install
```

Jeżeli pracujesz bez Poetry, możesz wskazać własny interpreter w Makefile:
```bash
make scrape-poetry PYTHON=.venv/bin/python
```

### Poezja: Wolne Lektury
Domyślnie pobierani są wybrani autorzy klasyczni z publicznego API Wolnych Lektur.

```bash
make scrape-poetry
```

Wariant z mniejszym limitem:
```bash
make scrape-poetry POETRY_LIMIT=5
```

Bez Makefile:
```bash
poetry run python -m verse_classifier_pl scrape poetry --limit-per-author 5
```

Możesz też podać konkretne slug-i autorów:
```bash
poetry run python -m verse_classifier_pl scrape poetry \
  --poet adam-mickiewicz \
  --poet juliusz-slowacki \
  --limit-per-author 10
```

Wynik: `.data/raw/poetry.jsonl`.

### Rap: Genius
Scraper rapu używa pakietu `lyricsgenius` i wymaga tokena API Genius:

```bash
export GENIUS_ACCESS_TOKEN="tu_wklej_token"
make scrape-rap
```

Wariant z własnymi artystami:
```bash
poetry run python -m verse_classifier_pl scrape rap \
  --artist "Taco Hemingway" \
  --artist "Łona" \
  --limit-per-artist 10
```

Wynik: `.data/raw/rap_genius.jsonl`.

### Wszystkie źródła
```bash
export GENIUS_ACCESS_TOKEN="tu_wklej_token"
make scrape-all POETRY_LIMIT=10 RAP_LIMIT=10
```

Pozostałe pomocnicze komendy:
```bash
make help
make lint
make test
make prepare-data
make train-baseline
make train-transformer
make evaluate
```

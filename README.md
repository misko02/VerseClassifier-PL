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

## Startowa struktura repozytorium
- `main.py` - lokalny entrypoint do uruchamiania projektu.
- `verse_classifier_pl/` - właściwy pakiet aplikacji.
- `scrappers/` - fetchery danych dla rapu i poezji.
- `tests/` - jednostkowe i integracyjne testy.
- `.data/` - **prywatne, lokalne zbiory robocze** (`raw`, `interim`, `processed`) – poza publicznym repozytorium ze względu na prawa autorskie.
- `.artifacts/` - **lokalne wyniki eksperymentów**: checkpointy modeli, metryki, raporty, wykresy – poza repozytorium.

---

## Pipeline: Scrapowanie i postprocessing danych

Pełny workflow tworzenia datasetu:

### 1. Instalacja zależności

```bash
poetry install
```

### 2. Scrapowanie surowych danych

Dane są zapisywane wyłącznie lokalnie w `.data/raw/` (ignorowane przez Git). Format wyjściowy: JSONL (jeden utwór na linię).

#### Poeta: Wolne Lektury

Domyślnie pobierani są wybrani autorzy klasyczni i współcześni z publicznego API Wolnych Lektur.

```bash
poetry run python -m verse_classifier_pl scrape poetry
```

Z własnym limitem:
```bash
poetry run python -m verse_classifier_pl scrape poetry --limit-per-author 10
```

Z konkretnym autorem:
```bash
poetry run python -m verse_classifier_pl scrape poetry \
  --poet adam-mickiewicz \
  --poet juliusz-slowacki \
  --limit-per-author 15
```

Wynik: `.data/raw/` (pliki `wolne_*.json`).

#### Rap: Genius API

Wymaga tokena API Genius (zdobądź na https://genius.com/api-clients):

```bash
export GENIUS_ACCESS_TOKEN="tu_wklej_token"
poetry run python -m verse_classifier_pl scrape rap
```

Z własnymi artystami:
```bash
poetry run python -m verse_classifier_pl scrape rap \
  --artist "Taco Hemingway" \
  --artist "Łona" \
  --limit-per-artist 15
```

Wynik: `.data/raw/` (pliki `genius_*.json`).

#### Wszystkie źródła

```bash
export GENIUS_ACCESS_TOKEN="tu_wklej_token"
poetry run python -m verse_classifier_pl scrape all --limit-per-author 10 --limit-per-artist 10
```

### 3. Postprocessing: czyszczenie i chunking

Po scrapowaniu wykonaj postprocessing:

```bash
poetry run python -m verse_classifier_pl prepare-data
```

Opcjonalnie z własnymi ścieżkami:
```bash
poetry run python -m verse_classifier_pl prepare-data \
  --raw-dir .data/raw \
  --output-dir .data/processed \
  --output-file dataset.jsonl
```

**Co się dzieje:**
- Ładuje wszystkie pliki JSON z `.data/raw/`
- Czyści teksty: usuwa URL-e, didaskalia `[...]`
- Dzieli na 4-linijkowe chunki (zgodnie z wymogiem projektu)
- Etykietuje: `0=poetry`, `1=rap`
- Zapisuje do JSONL w `.data/processed/`

**Wynik:** `.data/processed/combined.jsonl` — dataset gotowy do treningu.

### 4. Struktura wyjściowego datasetu

Każdy wiersz w `combined.jsonl` to JSON-object:
```json
{
  "source": "rap_genius" | "poetry",
  "label": 0 | 1,
  "title": "Tytuł utworu",
  "author": "Autor",
  "chunk_index": 0,
  "lines": ["linia 1", "linia 2", "linia 3", "linia 4"],
  "text": "linia 1\nlinia 2\nlinia 3\nlinia 4"
}
```

---

## Makefile (opcjonalnie)

Jeśli masz skonfigurowany `Makefile`, możesz używać skrótów:

```bash
make scrape-poetry POETRY_LIMIT=10
make scrape-rap RAP_LIMIT=10
make scrape-all
make prepare-data
make train-baseline
make predict TEXT="tu wpisz własny tekst"
make train-transformer-smoke
make train-transformer
```

Szczegóły w `Makefile`.

---

## Pierwszy model: baseline TF-IDF + Logistic Regression

Po utworzeniu `.data/processed/combined.jsonl` możesz wytrenować pierwszy model:

```bash
make train-baseline
```

Albo bez Makefile:

```bash
poetry run python -m verse_classifier_pl train-baseline
```

Domyślny trening:
- ładuje `.data/processed/combined.jsonl`
- dzieli dane na train/validation/test w proporcji `70/15/15`
- robi split po całych utworach (`source + author + title`), więc chunki z jednego utworu nie przeciekają między zbiorami
- trenuje `TfidfVectorizer` + `LogisticRegression`
- zapisuje splity do `.data/processed/splits/`
- zapisuje model do `.artifacts/baseline/model.joblib`
- zapisuje metryki do `.artifacts/baseline/metrics.json`

Przykład szybkiego eksperymentu z mniejszą próbką treningową:

```bash
make train-baseline MAX_TRAIN_SAMPLES_PER_CLASS=1000
```

Przykład z własnymi ścieżkami:

```bash
poetry run python -m verse_classifier_pl train-baseline \
  --dataset .data/processed/combined.jsonl \
  --model-dir .artifacts/baseline \
  --split-dir .data/processed/splits \
  --test-size 0.15 \
  --val-size 0.15
```

### Predykcja własnego tekstu

Po wytrenowaniu baseline możesz sprawdzić własny fragment:

```bash
make predict TEXT="idę przez miasto i liczę światła na mokrym asfalcie"
```

Albo bez Makefile:

```bash
poetry run python -m verse_classifier_pl predict \
  --text "idę przez miasto i liczę światła na mokrym asfalcie"
```

Tekst z pliku:

```bash
poetry run python -m verse_classifier_pl predict --file sample.txt
```

Wynik zawiera klasę oraz prawdopodobieństwa:
- `poetry` = poezja
- `rap` = rap

JSON dla skryptów:

```bash
poetry run python -m verse_classifier_pl predict \
  --text "noc rozlewa atrament po bruku" \
  --json
```

---

## Model transformerowy: HerBERT

Docelowy model projektu bazuje na HerBERT-cie z Hugging Face:
`allegro/herbert-base-cased`.

Pierwsze uruchomienie pobierze tokenizer i wagi modelu, więc wymaga dostępu do
internetu albo wcześniejszego cache Hugging Face.

Szybki smoke-test na małej próbce:

```bash
make train-transformer-smoke
```

Pełniejszy trening:

```bash
make train-transformer
```

Albo bez Makefile:

```bash
poetry run python -m verse_classifier_pl train-transformer \
  --dataset .data/processed/combined.jsonl \
  --model-dir .artifacts/transformer \
  --split-dir .data/processed/splits_transformer \
  --pretrained-model allegro/herbert-base-cased \
  --epochs 3 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --max-length 128
```

Domyślny trening:
- używa tych samych 4-linijkowych chunków co baseline
- dzieli dane po całych utworach (`source + author + title`)
- zapisuje splity do `.data/processed/splits_transformer/`
- zapisuje model i tokenizer do `.artifacts/transformer/model/`
- zapisuje checkpointy do `.artifacts/transformer/checkpoints/`
- zapisuje metryki do `.artifacts/transformer/metrics.json`

Przy słabszym sprzęcie zacznij od:

```bash
make train-transformer MAX_TRAIN_SAMPLES_PER_CLASS=500 TRAIN_BATCH_SIZE=4 EPOCHS=1
```

Jeżeli masz GPU z większą pamięcią, możesz zwiększyć batch:

```bash
make train-transformer TRAIN_BATCH_SIZE=16 EVAL_BATCH_SIZE=32
```

### Predykcja dla transformera

Gdy posiadasz już wytrenowany model, możesz użyc transformera do skuteczniejszej predykcji:


```bash
make predict-transformer TEXT="idę przez miasto i liczę światła na mokrym asfalcie"
```

Albo bez Makefile:

```bash
poetry run python -m verse_classifier_pl predict \
	--model-type transformer \
	--model .artifacts/transformer/model \
	--text "$(TEXT)"
```

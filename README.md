# Learning Market Signals from Financial News Headlines Using Interpretable Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An end-to-end interpretable NLP pipeline that extracts macroeconomic signals from **292,196 WSJ headlines** (2018–2026) and predicts daily S&P 500 directional movement using a Random Forest classifier with SHAP explainability.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Data](#data)
- [Methodology](#methodology)
- [Technologies](#technologies)
- [Author](#author)

---

## Project Overview

Financial news encodes macroeconomic expectations before they manifest in prices. This project builds a fully interpretable pipeline to:

1. **Scrape** daily WSJ archive headlines (8+ years, ~292K headlines)
2. **Model topics** using LDA and BERTopic to discover latent macroeconomic themes
3. **Engineer features** via a validated signal map (12 thematic categories) + FinBERT/BERT sentiment scoring
4. **Predict** S&P 500 daily direction (Up/Down) with a Random Forest classifier
5. **Explain** model decisions using SHAP values and permutation importance

The result is a production-ready risk modeling feature set grounded in both statistical NLP and domain financial knowledge.

---

## Pipeline Architecture

```
┌─────────────────────┐
│  WSJ Archive Scraper │  292,196 headlines · 2018-01-01 → 2026-04-04
└────────┬────────────┘
         │  wsj_headlines.csv
         ▼
┌─────────────────────┐
│  Topic Modeling      │  3-layer stop-word engineering → LDA coherence
│  (Notebook 02)       │  tuning → BERTopic neural discovery
└────────┬────────────┘
         │  validated signal taxonomy (12 categories)
         ▼
┌─────────────────────┐
│  Feature Extraction  │  Keyword-count signal map · FinBERT sentiment
│  (Notebook 03)       │  · BERT SST-2 sentiment · VIX · Oil
└────────┬────────────┘
         │  wsj_market_features.csv  (2,059 trading days × 16 features)
         ▼
┌─────────────────────┐
│  RF Classifier       │  SimpleImputer → MaxAbsScaler → RandomForest
│  (Notebook 04)       │  GridSearchCV (G-Mean) · 80/20 stratified split
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Interpretability    │  SHAP · Permutation Importance · Complexity
│                      │  Analysis (problexity)
└─────────────────────┘
```

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | See notebook |
| G-Mean (Up × Down Recall) | Tuned objective |
| SHAP Top Feature | `FED_POLICY` / `FEAR_CRISIS` |
| Dataset Size | 2,059 trading days |
| NLP Features | 12 signal-map + 2 BERT sentiment |

> Full evaluation metrics, confusion matrices, ROC curves, and SHAP beeswarm plots are in [`notebooks/04_rf_sp500_prediction.ipynb`](notebooks/04_rf_sp500_prediction.ipynb).

---

## Project Structure

```
wsj-market-signals/
│
├── README.md                     ← You are here
├── LICENSE
├── .gitignore
├── requirements.txt              ← Pinned dependencies
├── Makefile                      ← Convenience commands
│
├── data/
│   ├── raw/                      ← Original scraped data (not tracked by git)
│   ├── processed/                ← Feature matrices ready for modeling
│   └── external/                 ← VIX, Oil price data from yfinance
│
├── notebooks/
│   ├── 01_data_collection.ipynb  ← WSJ scraper walkthrough & EDA
│   ├── 02_topic_modeling.ipynb   ← LDA + BERTopic + signal taxonomy
│   ├── 03_feature_extraction.ipynb ← Signal map + FinBERT/BERT pipeline
│   └── 04_rf_sp500_prediction.ipynb ← RF model + SHAP interpretability
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── scraper.py            ← WSJ archive scraper (refactored)
│   │   └── make_dataset.py       ← Build daily corpora from raw headlines
│   ├── features/
│   │   ├── __init__.py
│   │   ├── signal_map.py         ← 12-category keyword signal map
│   │   ├── sentiment.py          ← FinBERT + BERT inference wrappers
│   │   └── build_features.py     ← Assemble final feature matrix
│   └── models/
│       ├── __init__.py
│       ├── train_model.py        ← RF training + GridSearchCV
│       └── predict_model.py      ← Inference on new feature vectors
│
├── reports/
│   └── figures/                  ← Generated charts (word clouds, SHAP plots, etc.)
│
├── scripts/
│   ├── run_scraper.sh            ← Kick off headline collection
│   └── run_pipeline.sh           ← End-to-end pipeline runner
│
├── tests/
│   ├── test_signal_map.py
│   └── test_features.py
│
└── docs/
    └── methodology.md            ← Detailed methodology write-up
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- ~4 GB RAM (for BERTopic / FinBERT inference)
- macOS / Linux (MPS acceleration supported on Apple Silicon)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/wsj-market-signals.git
cd wsj-market-signals
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Register the kernel for Jupyter

```bash
python -m ipykernel install --user --name wsj-signals --display-name "WSJ Market Signals"
jupyter lab
```

---

## Usage

### Run the full pipeline end-to-end

```bash
make all
```

### Or step by step

```bash
# Step 1 — Collect headlines (requires WSJ access)
python src/data/scraper.py

# Step 2 — Topic modeling
jupyter nbconvert --to notebook --execute notebooks/02_topic_modeling.ipynb

# Step 3 — Feature extraction
jupyter nbconvert --to notebook --execute notebooks/03_feature_extraction.ipynb

# Step 4 — Train and evaluate the RF model
jupyter nbconvert --to notebook --execute notebooks/04_rf_sp500_prediction.ipynb
```

### Using pre-built features (skip scraping)

If you have `wsj_market_features.csv`, you can jump directly to modeling:

```python
import pandas as pd
from src.models.train_model import train_rf_pipeline

df = pd.read_csv("data/processed/wsj_market_features.csv", parse_dates=["date"], index_col="date")
results = train_rf_pipeline(df)
```

---

## Data

| File | Description | Rows | Size |
|------|-------------|------|------|
| `wsj_headlines.csv` | Raw scraped WSJ headlines | 292,196 | ~50 MB |
| `wsj_market_features.csv` | Final feature matrix + target | 2,059 | ~200 KB |

> **Note:** Raw headline data is not tracked in this repository due to WSJ copyright. The scraper script (`src/data/scraper.py`) and feature extraction notebook can reproduce the dataset. The processed feature matrix `wsj_market_features.csv` is included in `data/processed/`.

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `TRADE_WAR` | int | Daily headline count mentioning tariffs, trade deals, sanctions |
| `FED_POLICY` | int | Fed rate signals, inflation language, treasury yields |
| `FEAR_CRISIS` | int | Crash, recession, plunge, contagion language |
| `EARNINGS` | int | Earnings beats/misses, guidance language |
| `EUPHORIA_MOMENTUM` | int | Rally, surge, record high language |
| `GEOPOLITICAL` | int | War, conflict, election uncertainty |
| `SECTOR_TRIGGER` | int | Tech, healthcare, finance sector-specific signals |
| `ENERGY_COMMODITY` | int | Oil, gas, commodity price signals |
| `LABOR_CONSUMER` | int | Jobs, unemployment, consumer spending |
| `TRUMP_POLICY` | int | Executive order, policy shift mentions |
| `CRYPTO_FINTECH` | int | Bitcoin, crypto, fintech mentions |
| `AMPLIFIER_MOD` | int | Sentiment amplifiers (record, historic, unprecedented) |
| `bert_sentiment` | float | BERT SST-2 daily sentiment score |
| `finbert_sentiment` | float | FinBERT domain-specific sentiment score |
| `oil_close` | float | WTI crude oil closing price |
| `vix_close` | float | CBOE VIX volatility index close |
| `target` | int | **Label**: 1 = S&P 500 Up, 0 = S&P 500 Down |

---

## Methodology

See [`docs/methodology.md`](docs/methodology.md) for a detailed write-up covering:

- Stop-word engineering (3-layer strategy preserving financial signal words)
- LDA coherence-based hyperparameter tuning
- BERTopic clustering approach
- Signal map validation against topic model outputs
- Data leakage prevention protocol
- G-Mean as evaluation metric for imbalanced financial classification
- SHAP interpretation of Random Forest decisions

---

## Technologies

| Category | Tools |
|----------|-------|
| **Data Collection** | `requests`, `BeautifulSoup4`, `curl_cffi` |
| **NLP / Topic Modeling** | `BERTopic`, `scikit-learn` (LDA, NMF), `spaCy`, `NLTK` |
| **Transformers** | `transformers`, `sentence-transformers`, `FinBERT` (ProsusAI) |
| **ML / Modeling** | `scikit-learn`, `imbalanced-learn`, `problexity` |
| **Explainability** | `shap`, permutation importance |
| **Visualization** | `matplotlib`, `seaborn`, `plotly`, `wordcloud` |
| **Data** | `pandas`, `numpy`, `yfinance` |
| **Acceleration** | PyTorch MPS (Apple Silicon), CUDA |

---

## Author

**Jishan Ahmed**
Senior Data Scientist · NLP & Interpretable ML

- [LinkedIn](https://linkedin.com/in/your-profile)
- [GitHub](https://github.com/your-username)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

> **Disclaimer:** This project is for research and educational purposes only. It does not constitute financial advice. Past headline patterns do not guarantee future market performance.

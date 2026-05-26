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

> Full evaluation metrics, confusion matrices, ROC curves, and SHAP beeswarm plots are in [`notebooks/03_rf_sp500_prediction.ipynb`](notebooks/03_rf_sp500_prediction.ipynb).

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

- [LinkedIn](https://www.linkedin.com/in/jishan-ahmed-689aa63b)
- [GitHub](https://github.com/JishanBSU2018)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

> **Disclaimer:** This project is for research and educational purposes only. It does not constitute financial advice. Past headline patterns do not guarantee future market performance.

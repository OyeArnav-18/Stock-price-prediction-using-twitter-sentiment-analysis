# Stock Price Prediction Using Twitter Sentiment Analysis

**MVSREC · BE CSIT · Batch 23**
**Team:** Arnav Bhardwaj (245124751003) · Suraj Pratap Singh (245124751061)
**Guide:** Neeraj Sharma, Asst. Professor, Dept. of CSIT, MVSREC

---

## Project Overview

A machine learning system that predicts stock price movements by combining
Twitter sentiment analysis with historical price data. Built and tested on
Microsoft (MSFT) stock data from 2020–2024.

## Key Results

| Model | RMSE | Improvement |
|---|---|---|
| ARIMA (baseline) | $57.74 | — |
| Random Forest | $18.80 | 67% better |
| LSTM (main model) | $12.00 | 79% better |

## What We Built

- **Data Pipeline** — Tweet collection (5,791 tweets) + stock price data (987 trading days)
- **NLP Module** — VADER sentiment analyser enhanced with a custom financial lexicon
- **3 Predictive Models** — ARIMA, Random Forest, LSTM neural network
- **Streamlit Web App** — Interactive dashboard with live sentiment analysis and Buy/Hold/Sell signals

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras |
| ML | Scikit-learn |
| NLP | NLTK, VADER Sentiment |
| Time Series | Statsmodels (ARIMA) |
| Data | Pandas, NumPy, yfinance |
| Visualisation | Matplotlib |
| Web App | Streamlit |

## How to Run

pip install -r requirements.txt
streamlit run app.py

## Project Structure

app.py, requirements.txt, runtime.txt, .streamlit/config.toml, README.md

## Core Finding

Adding Twitter sentiment to a stock prediction model reduces prediction error by 79.2% compared to using price data alone (ARIMA baseline). This validates that public opinion on social media contains real signal about future stock price movements.

## Limitations

- Tweet dataset has no date stamps — sentiment sampled rather than date-matched
- Real-time Twitter API requires paid access ($100/month)
- Direction accuracy: 52.9% (beats random 50%, not investment-grade)
- Tested on MSFT only

## Future Scope

- Replace VADER with FinBERT for financial-specific sentiment
- Integrate real-time Twitter/Reddit APIs
- Extend to 50+ stocks with portfolio-level aggregation
- Intraday minute-level prediction pipeline

## Recent Fixes

- **Sentiment scoring bug** — VADER's default lexicon scored finance slang like "crushed" and "load up" as negative/neutral, when in financial context they signal strongly positive sentiment (e.g. "crushed earnings" = beat expectations). Fixed by expanding the custom lexicon to 50+ finance-specific terms, including `crushed`, `smashed`, `load up`, `revenue`, `buy`, `profit`, and more. A tweet like "AAPL crushed it, revenue up 15%, time to load up on shares" now correctly scores positive.
- **Sidebar toggle disappearing** — custom CSS was unintentionally hiding Streamlit's collapsed-sidebar control button. Fixed by explicitly targeting `[data-testid="stSidebarCollapsedControl"]` and forcing it to stay visible, fixed-position, and styled, whether the sidebar is open or collapsed.

---
Theme-based Project · Sem IV 2025-2026 · MVSREC

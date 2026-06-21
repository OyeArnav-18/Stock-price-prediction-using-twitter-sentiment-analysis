# Stock Price Prediction Using Twitter Sentiment Analysis

**MVSREC · BE CSE Sec A · Batch 22**
**Team:** Arnav Bhardwaj (245124751003) · Suraj Pratap Singh (245124751061)
**Guide:** Neeraj Sharma, Asst. Professor, Dept. of CSE, MVSREC

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
- **NLP Module** — VADER sentiment analyser enhanced with financial lexicon
- **3 Predictive Models** — ARIMA, Random Forest, LSTM neural network
- **Streamlit Web App** — Interactive dashboard with live sentiment analysis and Buy/Hold/Sell signals

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras |
| ML | Scikit-learn |
| NLP | NLTK, VADER Sentiment |
| Time Series | Statsmodels (ARIMA) |
| Data | Pandas, NumPy, yfinance |
| Visualisation | Matplotlib, Seaborn |
| Web App | Streamlit |

## How to Run

pip install -r requirements.txt
streamlit run app.py

## Project Structure

app.py, requirements.txt, .streamlit/config.toml, README.md

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

---
Theme-based Project · Sem III 2025-2026 · MVSREC

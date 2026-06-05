# 📊 Zambia Inflation Forecasting
### Author: Given Chinyama | Data Scientist

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Models](https://img.shields.io/badge/Models-ARIMAX%20%7C%20XGBoost%20%7C%20Prophet%20%7C%20LSTM-green)
![Data](https://img.shields.io/badge/Data-World%20Bank%20%7C%20Yahoo%20Finance-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 🎯 Project Overview
A hybrid macroeconomic forecasting framework that predicts Zambia's 
annual inflation rate by combining classical statistical models with 
modern machine learning and deep learning architectures.

Rather than relying on a single model, this project benchmarks four 
distinct approaches — ARIMAX, XGBoost, Facebook Prophet, and LSTM — 
against the same test set to identify the most accurate forecasting 
strategy for Zambia's economic context.

---

## 📈 Final Results

| Rank | Model | RMSE | MAE | MAPE |
|------|-------|------|-----|------|
| 🥇 1st | ARIMAX | 5.65 | 4.18 | 23.68% |
| 🥈 2nd | LSTM | 6.29 | 4.54 | 25.12% |
| 🥉 3rd | XGBoost | 5.72 | 4.52 | 29.10% |
| 4th | Prophet | 11.50 | 8.03 | 57.35% |

### 3-Year Forward Forecast (ARIMAX)
| Year | Forecast | 95% Confidence Interval |
|------|----------|------------------------|
| 2025 | 14.34% | 5.76% – 22.92% |
| 2026 | 13.97% | 4.02% – 23.91% |
| 2027 | 13.75% | 3.38% – 24.11% |

---

## 🗃️ Data Sources

| Source | Data | API |
|--------|------|-----|
| World Bank | Zambia CPI Inflation (FP.CPI.TOTL.ZG) | `wbgapi` |
| World Bank | USD/ZMW Exchange Rate (PA.NUS.FCRF) | `wbgapi` |
| Yahoo Finance | Copper Spot Prices (HG=F) | `yfinance` |

**Dataset:** 25 years (2000–2024) · 3 features

---

## 🔬 Methodology

### Phase 0 — Data Collection
- Pulled 3 macroeconomic indicators from World Bank API and Yahoo Finance
- Merged into a single aligned annual time series (2000–2024)

### Phase 1 — EDA & Preprocessing
- Visualized all 3 variables over time
- Correlation analysis: Copper prices show -0.66 correlation with inflation
- Augmented Dickey-Fuller stationarity tests
- Chronological train/test split (2000–2019 train | 2020–2024 test)

### Phase 2 — Modeling
- **ARIMAX** — ARIMA with exogenous regressors; auto-tuned via AIC
- **XGBoost** — Lag features + rolling windows + exogenous variables
- **Prophet** — Trend + exogenous regressors
- **LSTM** — Sliding window sequences with 2-layer architecture + Dropout

### Phase 3 — Benchmarking
- Compared all 4 models on RMSE, MAE and MAPE
- Walk-forward chronological validation (no data leakage)

### Phase 4 — Forward Forecast
- Retrained ARIMAX on full dataset
- Generated 3-year outlook (2025–2027) with 95% confidence intervals

---

## 💡 Key Findings

- **ARIMAX won** because Zambia's inflation series is already stationary 
  (ADF p=0.0000) and the exogenous variables add meaningful signal
- **Copper prices** are the strongest external predictor (r = -0.66) — 
  confirming Zambia's economic sensitivity as one of the world's largest 
  copper producers
- **Prophet underperformed** due to insufficient training data (19 points) 
  for its changepoint detection algorithm
- **LSTM ranked 2nd** despite the small dataset, suggesting non-linear 
  dynamics exist in Zambia's inflation patterns

---

## 🛠️ Tech Stack


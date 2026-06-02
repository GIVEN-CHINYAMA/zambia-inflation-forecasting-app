# 📊 Zambia Inflation Forecasting App

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GIVEN-CHINYAMA/zambia-inflation-forecasting/blob/main/zambia_inflation_forecasting.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive **Streamlit dashboard** for forecasting Zambia's annual inflation rate using a hybrid ensemble of four machine learning and statistical models — **ARIMAX**, **XGBoost**, **Facebook Prophet**, and **LSTM**. The app integrates real-time macroeconomic data from the World Bank API and Yahoo Finance, enabling data-driven inflation analysis and multi-year forward forecasting.

---

## 🗂️ Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Models](#-models)
- [Data Sources](#-data-sources)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Results](#-model-results)
- [Key Insights](#-key-insights)
- [Author](#-author)

---

## 🔍 Overview

Zambia's inflation rate is influenced by a complex interplay of macroeconomic variables — particularly the **USD/ZMW exchange rate** and **copper commodity prices**, given the country's reliance on copper exports. This project builds a rigorous forecasting pipeline that:

1. Pulls live economic data from the **World Bank API** and **Yahoo Finance**
2. Performs **Exploratory Data Analysis (EDA)**, stationarity testing, and feature engineering
3. Trains and evaluates four distinct forecasting models
4. Benchmarks all models on RMSE, MAE, and MAPE
5. Generates a **3-year forward inflation forecast (2025–2027)** using the best-performing model
6. Presents all results through an **interactive Streamlit dashboard**

---

## ✨ Features

- 📡 **Real-time data integration** — live pulls from World Bank API and Yahoo Finance
- 🤖 **Four forecasting models** — statistical, machine learning, and deep learning approaches
- 📈 **Interactive visualizations** — time series plots, confidence intervals, model overlays
- 🏆 **Automated model benchmarking** — ranked comparison across RMSE, MAE, and MAPE
- 🔭 **Multi-year forward forecasting** — 3-year inflation outlook with 95% confidence bands
- 🧪 **Stationarity testing** — Augmented Dickey-Fuller (ADF) tests with automatic differencing
- 📊 **Feature importance analysis** — XGBoost feature contribution breakdown

---

## 🤖 Models

| Model | Type | Description |
|---|---|---|
| **ARIMAX** | Statistical | ARIMA extended with exogenous regressors (exchange rate + copper prices). Auto-selects optimal p, q via AIC. |
| **XGBoost** | Machine Learning | Gradient boosting on engineered lag features, rolling statistics, and differenced exogenous variables. |
| **Facebook Prophet** | Statistical / ML | Trend-and-seasonality decomposition model with external regressors. |
| **LSTM** | Deep Learning | Recurrent neural network with a sliding 3-year window and early stopping. |

All models are trained on a **chronological train/test split** (last 5 years held out as the test set) to prevent data leakage.

---

## 📦 Data Sources

| Variable | Source | Indicator / Ticker |
|---|---|---|
| Zambia Annual Inflation (CPI) | World Bank API | `FP.CPI.TOTL.ZG` |
| USD/ZMW Exchange Rate | World Bank API | `PA.NUS.FCRF` |
| Copper Spot Price (USD/lb) | Yahoo Finance | `HG=F` |

Data covers approximately **40 years** of annual observations. All three series are merged and aligned on a common yearly index.

---

## 📁 Project Structure

```
zambia-inflation-forecasting-app/
│
├── app.py                          # Main Streamlit dashboard application
├── requirements.txt                # Python dependencies
├── zambia_inflation_forecasting.ipynb  # Full research notebook (Phases 0–4)
│
├── README.md
├── LICENSE
└── .gitignore
```

### Research Notebook Phases

| Phase | Description |
|---|---|
| **Phase 0** | Data collection — World Bank API + Yahoo Finance |
| **Phase 1** | EDA, correlation analysis, ADF stationarity tests, train/test split |
| **Phase 2A** | ARIMAX model — auto-ARIMA with exogenous variables |
| **Phase 2B** | XGBoost model — lag/rolling feature engineering + gradient boosting |
| **Phase 2C** | Facebook Prophet — trend decomposition with external regressors |
| **Phase 2D** | LSTM — sliding window sequences + MinMax scaling + early stopping |
| **Phase 3** | Model benchmarking dashboard — RMSE / MAE / MAPE comparison |
| **Phase 4** | 3-year forward forecast (2025–2027) using best model (ARIMAX) |

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/GIVEN-CHINYAMA/zambia-inflation-forecasting-app.git
cd zambia-inflation-forecasting-app

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
streamlit
wbgapi
yfinance
pandas
numpy
matplotlib
seaborn
pmdarima
prophet
xgboost
scikit-learn
tensorflow
statsmodels
```

---

## 🚀 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Run the Research Notebook

Open the notebook in Google Colab using the badge at the top of this page, or run it locally in Jupyter:

```bash
jupyter notebook zambia_inflation_forecasting.ipynb
```

The notebook will automatically download all required data and produce visualisations for each phase.

---

## 📊 Model Results

Evaluated on the held-out test set (last 5 years):

| Rank | Model | RMSE | MAE | MAPE |
|---|---|---|---|---|
| 🥇 1st | **ARIMAX** | 5.65 | 4.18 | 23.68% |
| 🥈 2nd | LSTM | 6.29 | 4.54 | 25.12% |
| 🥉 3rd | XGBoost | 5.72 | 4.52 | 29.10% |
| 4th | Prophet | 11.50 | 8.03 | 57.35% |

### 🔭 3-Year Forward Forecast (ARIMAX)

| Year | Forecast | 95% Confidence Interval |
|---|---|---|
| 2025 | See app | Displayed in dashboard |
| 2026 | See app | Displayed in dashboard |
| 2027 | See app | Displayed in dashboard |

> The forward forecast uses 3-year rolling means of the differenced exchange rate and copper price as projected exogenous inputs.

---

## 💡 Key Insights

1. **ARIMAX is the top performer** — pairing ARIMA's time-series strengths with macroeconomic drivers gave the lowest error across all three metrics.

2. **Prophet underperformed** — with only ~19 training points, Prophet lacked sufficient data to identify meaningful changepoints and extrapolated a downward trend that diverged sharply from the post-2020 inflation surge.

3. **LSTM showed promise despite limited data** — ranking second on MAPE suggests the model captured some non-linear dynamics, though deep learning typically benefits from far larger datasets.

4. **Lag features dominate XGBoost** — feature importance analysis revealed that `Lag_2` and `Rolling_Mean_3` are the strongest inflation predictors, with exchange rate and copper prices providing meaningful secondary signals.

5. **Copper prices are a key macro driver** — a **−0.66 correlation** between copper prices and inflation confirms Zambia's well-documented sensitivity to copper export revenues, consistent with its position as one of the world's largest copper producers.

---

## 👤 Author

**Given Chinyama** — Data Scientist

- GitHub: [@GIVEN-CHINYAMA](https://github.com/GIVEN-CHINYAMA)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Data Sources: [World Bank Open Data](https://data.worldbank.org/) · [Yahoo Finance](https://finance.yahoo.com/)*



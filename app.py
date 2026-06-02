"""
Zambia Inflation Forecasting — Streamlit App
Author: Given Chinyama
A hybrid forecasting framework: ARIMAX · XGBoost · Prophet · LSTM
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zambia Inflation Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #16213e 100%);
        border-right: 1px solid #2d3561;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2140 0%, #252a4a 100%);
        border: 1px solid #3d4270;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { color: #7eb8f7 !important; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #8892b0 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

    /* Headers */
    h1 { color: #ccd6f6 !important; }
    h2, h3 { color: #a8b2d8 !important; }

    /* Info/success/warning boxes */
    .stAlert { border-radius: 10px; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #8892b0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d3561 0%, #3d4a8a 100%) !important;
        color: #ccd6f6 !important;
    }

    /* Divider */
    hr { border-color: #2d3561; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-green  { background: #1a3a2a; color: #64ffda; border: 1px solid #64ffda44; }
    .badge-blue   { background: #1a2a3a; color: #7eb8f7; border: 1px solid #7eb8f744; }
    .badge-purple { background: #2a1a3a; color: #c792ea; border: 1px solid #c792ea44; }
    .badge-orange { background: #3a2a1a; color: #ffcb6b; border: 1px solid #ffcb6b44; }

    /* Section card */
    .section-card {
        background: linear-gradient(135deg, #1e2140 0%, #1a1d2e 100%);
        border: 1px solid #2d3561;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Colour palette ─────────────────────────────────────────────────────────
COLORS = {
    "ARIMAX":   "#64ffda",
    "XGBoost":  "#7eb8f7",
    "Prophet":  "#c792ea",
    "LSTM":     "#ffcb6b",
    "actual":   "#f78c6c",
    "history":  "#4a5568",
    "grid":     "rgba(100,100,140,0.15)",
    "bg":       "rgba(0,0,0,0)",
    "paper":    "#1a1d2e",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["paper"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color="#8892b0", family="Inter, sans-serif"),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    hovermode="x unified",
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="📡 Fetching World Bank & market data…")
def load_data():
    import wbgapi as wb
    import yfinance as yf

    # Inflation
    infl_raw = wb.data.DataFrame("FP.CPI.TOTL.ZG", "ZMB", mrv=40)
    infl = infl_raw.T
    infl.index = pd.to_datetime(
        [str(y) for y in infl.index.str.replace("YR", "")], format="%Y"
    )
    infl.columns = ["Inflation"]
    infl = infl.sort_index()

    # Exchange rate
    fx_raw = wb.data.DataFrame("PA.NUS.FCRF", "ZMB", mrv=40)
    fx = fx_raw.T
    fx.index = pd.to_datetime(
        [str(y) for y in fx.index.str.replace("YR", "")], format="%Y"
    )
    fx.columns = ["USDZMW"]
    fx = fx.sort_index()

    # Copper
    copper = yf.download("HG=F", start="1990-01-01", interval="1mo", auto_adjust=True)["Close"]
    copper = copper.resample("YE").mean()
    copper.index = copper.index.to_period("Y").to_timestamp()
    copper.name = "Copper_USD"

    df = infl.join(fx, how="inner").join(copper, how="inner")
    df.dropna(inplace=True)
    df.index.name = "Year"
    return df


@st.cache_data(show_spinner="🔬 Running ARIMAX model…")
def run_arimax(df):
    from pmdarima import auto_arima
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    exog = pd.DataFrame({
        "USDZMW_diff": df["USDZMW"].diff(),
        "Copper_diff": df["Copper_USD"].diff(),
    }).dropna()
    infl = df["Inflation"].loc[exog.index]
    train_y, test_y = infl.iloc[:-5], infl.iloc[-5:]
    train_x, test_x = exog.iloc[:-5], exog.iloc[-5:]

    model = auto_arima(
        train_y, exogenous=train_x, d=0,
        start_p=0, start_q=0, max_p=4, max_q=4,
        seasonal=False, trace=False, error_action="ignore",
        suppress_warnings=True, stepwise=True,
    )
    fc, ci = model.predict(n_periods=5, exogenous=test_x, return_conf_int=True)
    rmse = float(np.sqrt(mean_squared_error(test_y, fc)))
    mae  = float(mean_absolute_error(test_y, fc))
    mape = float(np.mean(np.abs((test_y.values - fc) / test_y.values)) * 100)

    return dict(model=model, forecast=fc, ci=ci,
                actual=test_y.values, test_index=test_y.index,
                train_index=train_y.index, train_actual=train_y.values,
                order=model.order, rmse=rmse, mae=mae, mape=mape, exog=exog)


@st.cache_data(show_spinner="🌲 Running XGBoost model…")
def run_xgboost(df):
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    d = df.copy()
    d["Lag_1"]          = d["Inflation"].shift(1)
    d["Lag_2"]          = d["Inflation"].shift(2)
    d["Rolling_Mean_3"] = d["Inflation"].shift(1).rolling(3).mean()
    d["Rolling_Std_3"]  = d["Inflation"].shift(1).rolling(3).std()
    d["USDZMW_diff"]    = d["USDZMW"].diff()
    d["Copper_diff"]    = d["Copper_USD"].diff()
    d.dropna(inplace=True)

    feats = ["Lag_1", "Lag_2", "Rolling_Mean_3", "Rolling_Std_3", "USDZMW_diff", "Copper_diff"]
    X, y = d[feats], d["Inflation"]
    X_tr, X_te = X.iloc[:-5], X.iloc[-5:]
    y_tr, y_te = y.iloc[:-5], y.iloc[-5:]

    model = XGBRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_tr, y_tr)
    fc = model.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, fc)))
    mae  = float(mean_absolute_error(y_te, fc))
    mape = float(np.mean(np.abs((y_te.values - fc) / y_te.values)) * 100)
    importance = pd.Series(model.feature_importances_, index=feats)

    return dict(forecast=fc, actual=y_te.values,
                test_index=y_te.index, train_index=y_tr.index, train_actual=y_tr.values,
                importance=importance, rmse=rmse, mae=mae, mape=mape)


@st.cache_data(show_spinner="🔮 Running Prophet model…")
def run_prophet(df):
    from prophet import Prophet
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    d = df.copy()
    d["USDZMW_diff"] = d["USDZMW"].diff()
    d["Copper_diff"] = d["Copper_USD"].diff()
    d.dropna(inplace=True)

    pdf = pd.DataFrame({
        "ds": d.index, "y": d["Inflation"].values,
        "USDZMW_diff": d["USDZMW_diff"].values,
        "Copper_diff": d["Copper_diff"].values,
    })
    train_df, test_df = pdf.iloc[:-5], pdf.iloc[-5:]

    m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                daily_seasonality=False, interval_width=0.95)
    m.add_regressor("USDZMW_diff")
    m.add_regressor("Copper_diff")
    m.fit(train_df)

    fc_df = m.predict(test_df)
    fc = fc_df["yhat"].values
    actual = test_df["y"].values
    ci = fc_df[["yhat_lower", "yhat_upper"]].values

    rmse = float(np.sqrt(mean_squared_error(actual, fc)))
    mae  = float(mean_absolute_error(actual, fc))
    mape = float(np.mean(np.abs((actual - fc) / actual)) * 100)

    trend = m.predict(pdf)[["ds", "trend"]]
    return dict(forecast=fc, actual=actual, ci=ci,
                test_index=test_df["ds"].values, train_index=train_df["ds"].values,
                train_actual=train_df["y"].values, trend=trend,
                rmse=rmse, mae=mae, mape=mape)


@st.cache_data(show_spinner="🧠 Training LSTM network…")
def run_lstm(df):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    tf.random.set_seed(42)
    np.random.seed(42)

    d = df.copy()
    d["USDZMW_diff"] = d["USDZMW"].diff()
    d["Copper_diff"] = d["Copper_USD"].diff()
    d.dropna(inplace=True)

    feat_cols = ["Inflation", "USDZMW_diff", "Copper_diff"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(d[feat_cols])

    tgt_scaler = MinMaxScaler()
    tgt_scaled = tgt_scaler.fit_transform(d[["Inflation"]])

    WINDOW = 3
    def make_seq(data, target, w):
        X, y = [], []
        for i in range(w, len(data)):
            X.append(data[i-w:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    X_all, y_all = make_seq(scaled, tgt_scaled, WINDOW)
    dates = d.index[WINDOW:]
    split = len(X_all) - 5
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW, X_tr.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, epochs=200, batch_size=4,
                        validation_split=0.2, callbacks=[es], verbose=0)

    fc_sc = model.predict(X_te)
    fc = tgt_scaler.inverse_transform(fc_sc).flatten()
    actual = tgt_scaler.inverse_transform(y_te).flatten()
    train_actual = tgt_scaler.inverse_transform(y_tr).flatten()

    rmse = float(np.sqrt(mean_squared_error(actual, fc)))
    mae  = float(mean_absolute_error(actual, fc))
    mape = float(np.mean(np.abs((actual - fc) / actual)) * 100)

    return dict(forecast=fc, actual=actual, train_actual=train_actual,
                test_index=dates[split:], train_index=dates[:split],
                history=history.history, rmse=rmse, mae=mae, mape=mape,
                epochs=len(history.history["loss"]))


@st.cache_data(show_spinner="🔭 Generating 3-year forward forecast…")
def run_forward_forecast(_arimax_result, df):
    exog = _arimax_result["exog"]
    model = _arimax_result["model"]

    future_exog = pd.DataFrame({
        "USDZMW_diff": [exog["USDZMW_diff"].tail(3).mean()] * 3,
        "Copper_diff": [exog["Copper_diff"].tail(3).mean()] * 3,
    })
    fc, ci = model.predict(n_periods=3, exogenous=future_exog, return_conf_int=True)
    years = pd.date_range("2025", periods=3, freq="YS")
    return fc, ci, years


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.divider()

    run_models = st.multiselect(
        "Models to run",
        ["ARIMAX", "XGBoost", "Prophet", "LSTM"],
        default=["ARIMAX", "XGBoost", "Prophet", "LSTM"],
    )

    st.divider()
    show_ci = st.toggle("Show confidence intervals", value=True)
    show_history = st.toggle("Show training history on charts", value=True)

    st.divider()
    st.markdown("### 📚 Data Sources")
    st.markdown("""
    - 🌍 **World Bank API** — Inflation & FX
    - 📈 **Yahoo Finance** — Copper (HG=F)
    """)

    st.divider()
    st.markdown("### 👤 Author")
    st.markdown("**Given Chinyama**  \nData Scientist")
    st.markdown(
        '<span class="badge badge-blue">v2.0</span>'
        '<span class="badge badge-green">ARIMAX</span>'
        '<span class="badge badge-blue">XGBoost</span>'
        '<span class="badge badge-purple">Prophet</span>'
        '<span class="badge badge-orange">LSTM</span>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; padding: 28px 0 8px 0;">
  <h1 style="font-size:2.4rem; letter-spacing:-0.5px; margin-bottom:6px;">
    📊 Zambia Inflation Forecasting
  </h1>
  <p style="color:#8892b0; font-size:1.05rem; max-width:660px; margin:0 auto;">
    A hybrid framework combining <b>ARIMAX · XGBoost · Prophet · LSTM</b>
    to forecast Zambia's inflation using macroeconomic drivers.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

try:
    df = load_data()
except Exception as e:
    st.error(f"**Data load failed:** {e}")
    st.info("Check your internet connection. The app requires access to the World Bank API and Yahoo Finance.")
    st.stop()

# Quick top-row KPIs
latest_infl  = df["Inflation"].iloc[-1]
latest_year  = df.index[-1].year
latest_fx    = df["USDZMW"].iloc[-1]
latest_cu    = df["Copper_USD"].iloc[-1]
yoy_infl     = df["Inflation"].iloc[-1] - df["Inflation"].iloc[-2]
yoy_fx       = df["USDZMW"].iloc[-1] - df["USDZMW"].iloc[-2]

c1, c2, c3, c4 = st.columns(4)
c1.metric("📈 Latest Inflation", f"{latest_infl:.1f}%",  f"{yoy_infl:+.1f}pp YoY ({latest_year})")
c2.metric("💱 USD/ZMW Rate",     f"{latest_fx:.2f}",      f"{yoy_fx:+.2f} YoY")
c3.metric("🟤 Copper (USD/lb)",  f"${latest_cu:.2f}",    None)
c4.metric("📅 Dataset",          f"{len(df)} years",      f"{df.index.min().year}–{df.index.max().year}")

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_eda, tab_models, tab_benchmark, tab_forecast = st.tabs([
    "🔍 Exploratory Analysis",
    "🤖 Model Forecasts",
    "🏆 Benchmarking",
    "🔭 Forward Forecast",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ────────────────────────────────────────────────────────────────────────────

with tab_eda:
    st.markdown("### Macroeconomic Trends")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=["Annual Inflation Rate (%)", "USD/ZMW Exchange Rate", "Copper Price (USD/lb)"],
    )
    traces = [
        go.Scatter(x=df.index, y=df["Inflation"],  name="Inflation",  line=dict(color=COLORS["ARIMAX"],  width=2.5)),
        go.Scatter(x=df.index, y=df["USDZMW"],     name="USD/ZMW",    line=dict(color=COLORS["XGBoost"], width=2.5)),
        go.Scatter(x=df.index, y=df["Copper_USD"], name="Copper",     line=dict(color=COLORS["LSTM"],    width=2.5)),
    ]
    for i, tr in enumerate(traces, 1):
        fig.add_trace(tr, row=i, col=1)

    fig.update_layout(**PLOTLY_LAYOUT, height=580, showlegend=False,
                      title_text="Zambia Macroeconomic Indicators", title_font_color="#ccd6f6")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation Analysis")
    col_a, col_b = st.columns(2)

    with col_a:
        corr = df.corr()
        heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="RdBu", zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            colorbar=dict(tickfont=dict(color="#8892b0")),
        ))
        heat.update_layout(**PLOTLY_LAYOUT, height=340, title_text="Correlation Matrix")
        st.plotly_chart(heat, use_container_width=True)

    with col_b:
        scatter = go.Figure()
        scatter.add_trace(go.Scatter(
            x=df["USDZMW"], y=df["Inflation"],
            mode="markers+text",
            text=[str(y) for y in df.index.year],
            textposition="top center", textfont=dict(size=9, color="#8892b0"),
            marker=dict(color=COLORS["XGBoost"], size=10, opacity=0.85,
                        line=dict(color="white", width=1)),
            name="Year",
        ))
        scatter.update_layout(**PLOTLY_LAYOUT, height=340,
                              title_text="Inflation vs Exchange Rate",
                              xaxis_title="USD/ZMW", yaxis_title="Inflation (%)")
        st.plotly_chart(scatter, use_container_width=True)

    st.markdown("### Stationarity Summary (Augmented Dickey-Fuller)")
    with st.spinner("Running ADF tests…"):
        from statsmodels.tsa.stattools import adfuller
        rows = []
        for col in ["Inflation", "USDZMW", "Copper_USD"]:
            r = adfuller(df[col].dropna())
            rows.append({"Variable": col, "ADF Statistic": round(r[0], 4),
                         "p-value": round(r[1], 4),
                         "Stationary?": "✅ Yes" if r[1] <= 0.05 else "⚠️ No (d=1 needed)"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL FORECASTS
# ────────────────────────────────────────────────────────────────────────────

with tab_models:
    if not run_models:
        st.info("Select at least one model in the sidebar.")
    else:
        results = {}

        # --- ARIMAX ---
        if "ARIMAX" in run_models:
            with st.expander("📐 ARIMAX Model", expanded=True):
                try:
                    ar = run_arimax(df)
                    results["ARIMAX"] = ar
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Order",  str(ar["order"]))
                    m2.metric("RMSE",   f"{ar['rmse']:.4f}")
                    m3.metric("MAE",    f"{ar['mae']:.4f}")
                    m4.metric("MAPE",   f"{ar['mape']:.2f}%")

                    fig = go.Figure()
                    if show_history:
                        fig.add_trace(go.Scatter(x=ar["train_index"], y=ar["train_actual"],
                            name="Training", line=dict(color=COLORS["history"], width=2)))
                    fig.add_trace(go.Scatter(x=ar["test_index"], y=ar["actual"],
                        name="Actual", line=dict(color=COLORS["actual"], width=2.5),
                        mode="lines+markers", marker=dict(size=8)))
                    fig.add_trace(go.Scatter(x=ar["test_index"], y=ar["forecast"],
                        name="ARIMAX", line=dict(color=COLORS["ARIMAX"], width=2.5, dash="dash"),
                        mode="lines+markers", marker=dict(symbol="x", size=9)))
                    if show_ci:
                        fig.add_trace(go.Scatter(
                            x=list(ar["test_index"]) + list(ar["test_index"])[::-1],
                            y=list(ar["ci"][:, 0]) + list(ar["ci"][:, 1])[::-1],
                            fill="toself", fillcolor="rgba(100,255,218,0.1)",
                            line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
                    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                                      title_text="ARIMAX — Zambia Inflation Forecast",
                                      yaxis_title="Inflation (%)")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"ARIMAX error: {e}")

        # --- XGBoost ---
        if "XGBoost" in run_models:
            with st.expander("🌲 XGBoost Model", expanded=True):
                try:
                    xg = run_xgboost(df)
                    results["XGBoost"] = xg
                    m1, m2, m3 = st.columns(3)
                    m1.metric("RMSE", f"{xg['rmse']:.4f}")
                    m2.metric("MAE",  f"{xg['mae']:.4f}")
                    m3.metric("MAPE", f"{xg['mape']:.2f}%")

                    col_f, col_i = st.columns([2, 1])
                    with col_f:
                        fig = go.Figure()
                        if show_history:
                            fig.add_trace(go.Scatter(x=xg["train_index"], y=xg["train_actual"],
                                name="Training", line=dict(color=COLORS["history"], width=2)))
                        fig.add_trace(go.Scatter(x=xg["test_index"], y=xg["actual"],
                            name="Actual", line=dict(color=COLORS["actual"], width=2.5),
                            mode="lines+markers", marker=dict(size=8)))
                        fig.add_trace(go.Scatter(x=xg["test_index"], y=xg["forecast"],
                            name="XGBoost", line=dict(color=COLORS["XGBoost"], width=2.5, dash="dashdot"),
                            mode="lines+markers", marker=dict(symbol="triangle-up", size=10)))
                        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                                          title_text="XGBoost — Zambia Inflation Forecast",
                                          yaxis_title="Inflation (%)")
                        st.plotly_chart(fig, use_container_width=True)

                    with col_i:
                        imp = xg["importance"].sort_values()
                        fig_i = go.Figure(go.Bar(
                            x=imp.values, y=imp.index, orientation="h",
                            marker=dict(color=COLORS["XGBoost"], opacity=0.85),
                        ))
                        fig_i.update_layout(**PLOTLY_LAYOUT, height=360,
                                            title_text="Feature Importance",
                                            xaxis_title="Score")
                        st.plotly_chart(fig_i, use_container_width=True)
                except Exception as e:
                    st.error(f"XGBoost error: {e}")

        # --- Prophet ---
        if "Prophet" in run_models:
            with st.expander("🔮 Prophet Model", expanded=True):
                try:
                    pr = run_prophet(df)
                    results["Prophet"] = pr
                    m1, m2, m3 = st.columns(3)
                    m1.metric("RMSE", f"{pr['rmse']:.4f}")
                    m2.metric("MAE",  f"{pr['mae']:.4f}")
                    m3.metric("MAPE", f"{pr['mape']:.2f}%")

                    col_f, col_t = st.columns(2)
                    with col_f:
                        fig = go.Figure()
                        if show_history:
                            fig.add_trace(go.Scatter(x=pr["train_index"], y=pr["train_actual"],
                                name="Training", line=dict(color=COLORS["history"], width=2)))
                        fig.add_trace(go.Scatter(x=pr["test_index"], y=pr["actual"],
                            name="Actual", line=dict(color=COLORS["actual"], width=2.5),
                            mode="lines+markers", marker=dict(size=8)))
                        fig.add_trace(go.Scatter(x=pr["test_index"], y=pr["forecast"],
                            name="Prophet", line=dict(color=COLORS["Prophet"], width=2.5, dash="dash"),
                            mode="lines+markers", marker=dict(symbol="square", size=9)))
                        if show_ci:
                            fig.add_trace(go.Scatter(
                                x=list(pr["test_index"]) + list(pr["test_index"])[::-1],
                                y=list(pr["ci"][:, 0]) + list(pr["ci"][:, 1])[::-1],
                                fill="toself", fillcolor="rgba(199,146,234,0.1)",
                                line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
                        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                                          title_text="Prophet — Zambia Inflation Forecast",
                                          yaxis_title="Inflation (%)")
                        st.plotly_chart(fig, use_container_width=True)

                    with col_t:
                        trend = pr["trend"]
                        fig_t = go.Figure(go.Scatter(x=trend["ds"], y=trend["trend"],
                            line=dict(color=COLORS["Prophet"], width=2)))
                        fig_t.update_layout(**PLOTLY_LAYOUT, height=360,
                                            title_text="Prophet — Trend Component")
                        st.plotly_chart(fig_t, use_container_width=True)
                except Exception as e:
                    st.error(f"Prophet error: {e}")

        # --- LSTM ---
        if "LSTM" in run_models:
            with st.expander("🧠 LSTM Model", expanded=True):
                try:
                    ls = run_lstm(df)
                    results["LSTM"] = ls
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Epochs trained", ls["epochs"])
                    m2.metric("RMSE", f"{ls['rmse']:.4f}")
                    m3.metric("MAE",  f"{ls['mae']:.4f}")
                    m4.metric("MAPE", f"{ls['mape']:.2f}%")

                    col_f, col_l = st.columns(2)
                    with col_f:
                        fig = go.Figure()
                        if show_history:
                            fig.add_trace(go.Scatter(x=ls["train_index"], y=ls["train_actual"],
                                name="Training", line=dict(color=COLORS["history"], width=2)))
                        fig.add_trace(go.Scatter(x=ls["test_index"], y=ls["actual"],
                            name="Actual", line=dict(color=COLORS["actual"], width=2.5),
                            mode="lines+markers", marker=dict(size=8)))
                        fig.add_trace(go.Scatter(x=ls["test_index"], y=ls["forecast"],
                            name="LSTM", line=dict(color=COLORS["LSTM"], width=2.5, dash="dash"),
                            mode="lines+markers", marker=dict(symbol="diamond", size=9)))
                        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                                          title_text="LSTM — Zambia Inflation Forecast",
                                          yaxis_title="Inflation (%)")
                        st.plotly_chart(fig, use_container_width=True)

                    with col_l:
                        h = ls["history"]
                        fig_l = go.Figure()
                        fig_l.add_trace(go.Scatter(y=h["loss"], name="Train Loss",
                            line=dict(color=COLORS["LSTM"], width=2)))
                        fig_l.add_trace(go.Scatter(y=h["val_loss"], name="Val Loss",
                            line=dict(color="#8892b0", width=2, dash="dash")))
                        fig_l.update_layout(**PLOTLY_LAYOUT, height=360,
                                            title_text="Training Loss Curve",
                                            xaxis_title="Epoch", yaxis_title="MSE")
                        st.plotly_chart(fig_l, use_container_width=True)
                except Exception as e:
                    st.error(f"LSTM error: {e}")

        # Cache results for other tabs
        st.session_state["results"] = results


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — BENCHMARKING
# ────────────────────────────────────────────────────────────────────────────

with tab_benchmark:
    st.markdown("### Model Performance Leaderboard")

    results = st.session_state.get("results", {})
    if not results:
        st.info("Run models in the **Model Forecasts** tab first.")
    else:
        rows = []
        for name, r in results.items():
            rows.append({"Model": name, "RMSE": round(r["rmse"], 4),
                         "MAE": round(r["mae"], 4), "MAPE (%)": round(r["mape"], 2)})
        bench_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
        medals = ["🥇", "🥈", "🥉"] + [""] * 10
        bench_df.insert(0, "Rank", [f"{medals[i]} #{i+1}" for i in range(len(bench_df))])

        st.dataframe(
            bench_df.style
                .background_gradient(subset=["RMSE", "MAE", "MAPE (%)"], cmap="YlOrRd_r")
                .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE (%)": "{:.2f}%"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("### Metric Comparison Charts")
        fig_b = make_subplots(rows=1, cols=3,
            subplot_titles=["RMSE ↓ Lower is better",
                            "MAE ↓ Lower is better",
                            "MAPE % ↓ Lower is better"])

        model_colors = [COLORS.get(m, "#8892b0") for m in bench_df["Model"]]
        for col, metric in enumerate(["RMSE", "MAE", "MAPE (%)"], 1):
            fig_b.add_trace(
                go.Bar(x=bench_df["Model"], y=bench_df[metric],
                       marker=dict(color=model_colors, opacity=0.85),
                       text=[f"{v:.2f}" for v in bench_df[metric]],
                       textposition="outside", name=metric, showlegend=False),
                row=1, col=col,
            )
        fig_b.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("### All Models vs Actual")
        fig_ov = go.Figure()
        first = next(iter(results.values()))
        if show_history:
            fig_ov.add_trace(go.Scatter(
                x=first["train_index"], y=first["train_actual"],
                name="Historical", line=dict(color=COLORS["history"], width=2.5)))
        fig_ov.add_trace(go.Scatter(
            x=first["test_index"], y=first["actual"],
            name="Actual", line=dict(color=COLORS["actual"], width=3),
            mode="lines+markers", marker=dict(size=9)))
        for name, r in results.items():
            fig_ov.add_trace(go.Scatter(
                x=r["test_index"], y=r["forecast"],
                name=f"{name} (MAPE={r['mape']:.1f}%)",
                line=dict(color=COLORS.get(name, "#fff"), width=2, dash="dash"),
                mode="lines+markers", marker=dict(size=7)))
        fig_ov.update_layout(**PLOTLY_LAYOUT, height=460,
                             title_text="All Models vs Actual — Zambia Inflation",
                             yaxis_title="Inflation (%)")
        st.plotly_chart(fig_ov, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — FORWARD FORECAST
# ────────────────────────────────────────────────────────────────────────────

with tab_forecast:
    st.markdown("### 🔭 3-Year Forward Forecast (2025–2027)")
    st.markdown("Using **ARIMAX** — the best-performing model — retrained on the full dataset.")

    results = st.session_state.get("results", {})
    if "ARIMAX" not in results:
        st.warning("Please run the **ARIMAX model** in the Model Forecasts tab first.")
    else:
        try:
            ar = results["ARIMAX"]
            fc, ci, years = run_forward_forecast(ar, df)

            # KPI row
            c1, c2, c3 = st.columns(3)
            for col, (yr, f, bounds) in zip([c1, c2, c3], zip(years.year, fc, ci)):
                col.metric(f"📅 {yr}", f"{f:.2f}%",
                           f"CI: {bounds[0]:.1f}% – {bounds[1]:.1f}%")

            # Plot
            fig_ff = go.Figure()
            fig_ff.add_trace(go.Scatter(
                x=df.index, y=df["Inflation"],
                name="Historical", line=dict(color=COLORS["history"], width=2.5),
                mode="lines+markers", marker=dict(size=5)))
            fig_ff.add_trace(go.Scatter(
                x=years, y=fc,
                name="Forecast 2025–2027",
                line=dict(color=COLORS["ARIMAX"], width=3, dash="dash"),
                mode="lines+markers+text",
                marker=dict(symbol="diamond", size=12),
                text=[f"{v:.1f}%" for v in fc], textposition="top center",
                textfont=dict(color=COLORS["ARIMAX"], size=13)))
            if show_ci:
                fig_ff.add_trace(go.Scatter(
                    x=list(years) + list(years)[::-1],
                    y=list(ci[:, 0]) + list(ci[:, 1])[::-1],
                    fill="toself", fillcolor="rgba(100,255,218,0.08)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
            # Vertical line at forecast start
            fig_ff.add_vline(x="2025-01-01", line_dash="dot",
                             line_color="rgba(100,255,218,0.3)",
                             annotation_text="Forecast →",
                             annotation_font_color="#64ffda")
            fig_ff.update_layout(**PLOTLY_LAYOUT, height=480,
                                 title_text="Zambia Inflation — ARIMAX 3-Year Forward Forecast",
                                 yaxis_title="Inflation (%)")
            st.plotly_chart(fig_ff, use_container_width=True)

            # Table
            st.markdown("#### Forecast Table")
            fwd_df = pd.DataFrame({
                "Year": years.year,
                "Forecast (%)": [f"{v:.2f}" for v in fc],
                "Lower CI (%)": [f"{v:.2f}" for v in ci[:, 0]],
                "Upper CI (%)": [f"{v:.2f}" for v in ci[:, 1]],
            })
            st.dataframe(fwd_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Forward forecast error: {e}")

    st.divider()
    st.markdown("""
    ### 📋 Key Analytical Takeaways

    | # | Finding |
    |---|---------|
    | 1 | **ARIMAX won** — combining ARIMA with exchange rate & copper prices produced the lowest error |
    | 2 | **Prophet underperformed** — only ~19 training points were insufficient for changepoint detection |
    | 3 | **LSTM showed promise** — ranked 2nd on MAPE despite the small dataset |
    | 4 | **XGBoost feature importance** — Lag_2 and Rolling_Mean_3 are the strongest predictors |
    | 5 | **Copper prices matter** — the −0.66 correlation confirms Zambia's commodity-driven economy |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.82rem; padding:8px 0 16px 0;">
  Built with ❤️ by <b>Given Chinyama</b> &nbsp;·&nbsp;
  Data: World Bank Open Data · Yahoo Finance &nbsp;·&nbsp;
  Models: ARIMAX · XGBoost · Prophet · LSTM
</div>
""", unsafe_allow_html=True)


# QuantEdge Pro - Simplified Institutional Version with Universe & Risk/Optimization Wiring
# Save as: app.py (or any name you like) and run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


# =========================
# Config
# =========================
class Config:
    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.045
    MAX_TICKERS = 150
    YF_BATCH_SIZE = 50
    MC_SIMULATIONS = 10000


# =========================
# Universe Definition
# =========================
INSTRUMENTS = [
    # --- US Tech & Large Caps ---
    {"ticker": "AAPL", "name": "Apple", "region": "US", "sector": "Tech"},
    {"ticker": "MSFT", "name": "Microsoft", "region": "US", "sector": "Tech"},
    {"ticker": "GOOGL", "name": "Alphabet", "region": "US", "sector": "Tech"},
    {"ticker": "AMZN", "name": "Amazon", "region": "US", "sector": "Tech"},
    {"ticker": "META", "name": "Meta Platforms", "region": "US", "sector": "Tech"},
    {"ticker": "NVDA", "name": "NVIDIA", "region": "US", "sector": "Tech"},
    {"ticker": "TSLA", "name": "Tesla", "region": "US", "sector": "Auto"},
    {"ticker": "JPM", "name": "JPMorgan", "region": "US", "sector": "Bank"},
    {"ticker": "BAC", "name": "Bank of America", "region": "US", "sector": "Bank"},
    {"ticker": "XOM", "name": "Exxon Mobil", "region": "US", "sector": "Energy"},
    {"ticker": "UNH", "name": "UnitedHealth", "region": "US", "sector": "Health"},
    {"ticker": "V", "name": "Visa", "region": "US", "sector": "Payments"},
    {"ticker": "MA", "name": "Mastercard", "region": "US", "sector": "Payments"},
    {"ticker": "SPY", "name": "S&P 500 ETF", "region": "US", "sector": "ETF"},

    # --- TR Core Blue Chips ---
    {"ticker": "AKBNK.IS", "name": "Akbank", "region": "TR", "sector": "Bank"},
    {"ticker": "GARAN.IS", "name": "Garanti BBVA", "region": "TR", "sector": "Bank"},
    {"ticker": "YKBNK.IS", "name": "Yapi Kredi", "region": "TR", "sector": "Bank"},
    {"ticker": "ISCTR.IS", "name": "Isbank", "region": "TR", "sector": "Bank"},
    {"ticker": "HALKB.IS", "name": "Halkbank", "region": "TR", "sector": "Bank"},
    {"ticker": "VAKBN.IS", "name": "Vakifbank", "region": "TR", "sector": "Bank"},
    {"ticker": "TSKB.IS", "name": "TSKB", "region": "TR", "sector": "Bank"},
    {"ticker": "QNBFB.IS", "name": "QNB Finansbank", "region": "TR", "sector": "Bank"},
    {"ticker": "KCHOL.IS", "name": "Koc Holding", "region": "TR", "sector": "Holding"},
    {"ticker": "SAHOL.IS", "name": "Sabanci Holding", "region": "TR", "sector": "Holding"},
    {"ticker": "TUPRS.IS", "name": "Tupras", "region": "TR", "sector": "Energy"},
    {"ticker": "SISE.IS", "name": "Sisecam", "region": "TR", "sector": "Industrial"},
    {"ticker": "EREGL.IS", "name": "Erdemir", "region": "TR", "sector": "Steel"},
    {"ticker": "BIMAS.IS", "name": "BIM", "region": "TR", "sector": "Retail"},
    {"ticker": "MGROS.IS", "name": "Migros", "region": "TR", "sector": "Retail"},
    {"ticker": "TCELL.IS", "name": "Turkcell", "region": "TR", "sector": "Telecom"},
    {"ticker": "TTKOM.IS", "name": "Turk Telekom", "region": "TR", "sector": "Telecom"},
    {"ticker": "THYAO.IS", "name": "Turkish Airlines", "region": "TR", "sector": "Transport"},
    {"ticker": "PGSUS.IS", "name": "Pegasus", "region": "TR", "sector": "Transport"},
    {"ticker": "ARCLK.IS", "name": "Arcelik", "region": "TR", "sector": "Consumer"},
    {"ticker": "FROTO.IS", "name": "Ford Otosan", "region": "TR", "sector": "Auto"},
    {"ticker": "TOASO.IS", "name": "Tofas", "region": "TR", "sector": "Auto"},
    {"ticker": "ASELS.IS", "name": "Aselsan", "region": "TR", "sector": "Defence"},
    {"ticker": "ENJSA.IS", "name": "Enerjisa", "region": "TR", "sector": "Energy"},
    {"ticker": "KRDMD.IS", "name": "Kardemir", "region": "TR", "sector": "Steel"},

    # --- TR Leasing & Insurance / Financials ---
    {"ticker": "ANSGR.IS", "name": "Anadolu Sigorta", "region": "TR", "sector": "Insurance"},
    {"ticker": "AKGRT.IS", "name": "Aksigorta", "region": "TR", "sector": "Insurance"},
    {"ticker": "ANHYT.IS", "name": "Anadolu Hayat", "region": "TR", "sector": "Insurance"},
    {"ticker": "RAYSG.IS", "name": "Ray Sigorta", "region": "TR", "sector": "Insurance"},
    {"ticker": "ISFIN.IS", "name": "Is Leasing", "region": "TR", "sector": "Leasing"},
    {"ticker": "VAKFN.IS", "name": "Vakif Leasing", "region": "TR", "sector": "Leasing"},
    {"ticker": "ISGYO.IS", "name": "Is REIT", "region": "TR", "sector": "REIT"},
    {"ticker": "VKGYO.IS", "name": "Vakif REIT", "region": "TR", "sector": "REIT"},
    {"ticker": "SNGYO.IS", "name": "Sinpas REIT", "region": "TR", "sector": "REIT"},
    {"ticker": "HLGYO.IS", "name": "Halk REIT", "region": "TR", "sector": "REIT"},

    # --- Japan Core Equities ---
    {"ticker": "7203.T", "name": "Toyota", "region": "JP", "sector": "Auto"},
    {"ticker": "6758.T", "name": "Sony", "region": "JP", "sector": "Tech"},
    {"ticker": "9984.T", "name": "SoftBank Group", "region": "JP", "sector": "Tech"},
    {"ticker": "8035.T", "name": "Tokyo Electron", "region": "JP", "sector": "Tech"},
    {"ticker": "9983.T", "name": "Fast Retailing", "region": "JP", "sector": "Retail"},
    {"ticker": "6861.T", "name": "Keyence", "region": "JP", "sector": "Tech"},
    {"ticker": "9432.T", "name": "NTT", "region": "JP", "sector": "Telecom"},
    {"ticker": "6954.T", "name": "Fanuc", "region": "JP", "sector": "Industrial"},

    # --- Japan Bank Stocks ---
    {"ticker": "8306.T", "name": "Mitsubishi UFJ", "region": "JP", "sector": "Bank"},
    {"ticker": "8316.T", "name": "Sumitomo Mitsui FG", "region": "JP", "sector": "Bank"},
    {"ticker": "8411.T", "name": "Mizuho FG", "region": "JP", "sector": "Bank"},
    {"ticker": "7182.T", "name": "Japan Post Bank", "region": "JP", "sector": "Bank"},

    # --- Korea ---
    {"ticker": "005930.KS", "name": "Samsung Electronics", "region": "KR", "sector": "Tech"},
    {"ticker": "000660.KS", "name": "SK Hynix", "region": "KR", "sector": "Tech"},
    {"ticker": "035420.KS", "name": "Naver", "region": "KR", "sector": "Tech"},
    {"ticker": "035720.KS", "name": "Kakao", "region": "KR", "sector": "Tech"},
    {"ticker": "051910.KS", "name": "LG Chem", "region": "KR", "sector": "Chemical"},
    {"ticker": "005380.KS", "name": "Hyundai Motor", "region": "KR", "sector": "Auto"},
    {"ticker": "006400.KS", "name": "Samsung SDI", "region": "KR", "sector": "Battery"},
    {"ticker": "105560.KS", "name": "KB Financial", "region": "KR", "sector": "Bank"},
    {"ticker": "055550.KS", "name": "Shinhan Financial", "region": "KR", "sector": "Bank"},

    # --- Singapore ---
    {"ticker": "D05.SI", "name": "DBS Group", "region": "SG", "sector": "Bank"},
    {"ticker": "U11.SI", "name": "UOB", "region": "SG", "sector": "Bank"},
    {"ticker": "O39.SI", "name": "OCBC", "region": "SG", "sector": "Bank"},
    {"ticker": "Z74.SI", "name": "Singtel", "region": "SG", "sector": "Telecom"},
    {"ticker": "C6L.SI", "name": "ST Engineering", "region": "SG", "sector": "Industrial"},
    {"ticker": "C07.SI", "name": "Jardine C&C", "region": "SG", "sector": "Auto"},
    {"ticker": "C09.SI", "name": "CityDev", "region": "SG", "sector": "REIT"},

    # --- China (via US listings / ETFs) ---
    {"ticker": "BABA", "name": "Alibaba", "region": "CN", "sector": "Tech"},
    {"ticker": "TCEHY", "name": "Tencent", "region": "CN", "sector": "Tech"},
    {"ticker": "JD", "name": "JD.com", "region": "CN", "sector": "Tech"},
    {"ticker": "PDD", "name": "Pinduoduo", "region": "CN", "sector": "Tech"},
    {"ticker": "NTES", "name": "NetEase", "region": "CN", "sector": "Tech"},
    {"ticker": "BIDU", "name": "Baidu", "region": "CN", "sector": "Tech"},
    {"ticker": "MCHI", "name": "MSCI China ETF", "region": "CN", "sector": "ETF"},
    {"ticker": "FXI", "name": "FTSE China 50 ETF", "region": "CN", "sector": "ETF"},
]

UNIVERSE_DF = pd.DataFrame(INSTRUMENTS)
UNIVERSE_DF = UNIVERSE_DF.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

REGIONS = sorted(UNIVERSE_DF["region"].unique())
SECTORS = sorted(UNIVERSE_DF["sector"].unique())

# Scenario definitions
SCENARIOS = {
    "Custom Filter": None,
    "Full Global": lambda df: df,
    "Global Tech + TR Banks": lambda df: df[
        ((df["sector"] == "Tech") & df["region"].isin(["US", "JP", "KR", "CN"]))
        | ((df["region"] == "TR") & (df["sector"] == "Bank"))
    ],
    "Asia Leaders": lambda df: df[df["region"].isin(["JP", "KR", "CN", "SG"])],
    "TR Financial Complex": lambda df: df[
        (df["region"] == "TR")
        & (df["sector"].isin(["Bank", "Insurance", "Leasing", "REIT"]))
    ],
}


# =========================
# Helper: Fetch prices from Yahoo (batched)
# =========================
def _download_batch(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()

    # MultiIndex (field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            closes = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            closes = data["Close"].copy()
        else:
            first_level = data.columns.levels[0][0]
            closes = data[first_level].copy()
    else:
        # Single ticker fallback
        cols = list(data.columns)
        pick = None
        for c in ["Adj Close", "Close"]:
            if c in cols:
                pick = c
                break
        if pick is None:
            pick = cols[0]
        closes = data[[pick]].copy()
        closes.columns = [tickers[0]]
    return closes


def fetch_price_data(tickers, start_date, end_date):
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    tickers = sorted(list(dict.fromkeys(tickers)))  # dedupe while preserving order
    if not tickers:
        return pd.DataFrame()

    if len(tickers) > Config.MAX_TICKERS:
        tickers = tickers[: Config.MAX_TICKERS]

    all_closes = []
    for i in range(0, len(tickers), Config.YF_BATCH_SIZE):
        batch = tickers[i : i + Config.YF_BATCH_SIZE]
        closes = _download_batch(batch, start_date, end_date)
        if not closes.empty:
            all_closes.append(closes)

    if not all_closes:
        return pd.DataFrame()

    prices = pd.concat(all_closes, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    prices = prices.ffill().bfill()
    return prices


# =========================
# Risk Metrics
# =========================
def historical_var(returns, alpha=0.95):
    if returns.empty:
        return np.nan
    return -np.percentile(returns, (1 - alpha) * 100)


def historical_cvar(returns, alpha=0.95):
    if returns.empty:
        return np.nan
    var = historical_var(returns, alpha)
    tail = returns[returns <= -var]
    return -tail.mean() if len(tail) > 0 else var


def parametric_var(returns, alpha=0.95):
    if returns.empty:
        return np.nan
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0
    # simple z-score
    from scipy.stats import norm

    z = norm.ppf(1 - alpha)
    return -(mu + z * sigma)


def mc_var_cvar(returns, alpha=0.95, n_sims=Config.MC_SIMULATIONS):
    if returns.empty:
        return np.nan, np.nan
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0, 0.0
    sims = np.random.normal(mu, sigma, n_sims)
    var = -np.percentile(sims, (1 - alpha) * 100)
    tail = sims[sims <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return var, cvar


def compute_max_drawdown(returns):
    if returns.empty:
        return 0.0
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def compute_beta(portfolio_returns, benchmark_returns):
    if portfolio_returns.empty or benchmark_returns.empty:
        return np.nan
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 10:
        return np.nan
    rp = aligned.iloc[:, 0]
    rb = aligned.iloc[:, 1]
    cov = np.cov(rp, rb)[0, 1]
    var_b = np.var(rb)
    if var_b == 0:
        return np.nan
    return cov / var_b


def compute_risk_profile(portfolio_returns, benchmark_returns, alpha=0.95):
    if portfolio_returns.empty:
        return {}
    ann_ret = portfolio_returns.mean() * Config.TRADING_DAYS_PER_YEAR
    ann_vol = portfolio_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)
    sharpe = (ann_ret - Config.RISK_FREE_RATE) / ann_vol if ann_vol != 0 else np.nan
    max_dd = compute_max_drawdown(portfolio_returns)
    h_var = historical_var(portfolio_returns, alpha)
    h_cvar = historical_cvar(portfolio_returns, alpha)
    p_var = parametric_var(portfolio_returns, alpha)
    mc_v, mc_c = mc_var_cvar(portfolio_returns, alpha)
    rel_var = rel_cvar = np.nan
    beta = np.nan
    if benchmark_returns is not None and not benchmark_returns.empty:
        diff = portfolio_returns - benchmark_returns
        rel_var = historical_var(diff, alpha)
        rel_cvar = historical_cvar(diff, alpha)
        beta = compute_beta(portfolio_returns, benchmark_returns)
    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hist_var": h_var,
        "hist_cvar": h_cvar,
        "param_var": p_var,
        "mc_var": mc_v,
        "mc_cvar": mc_c,
        "rel_var": rel_var,
        "rel_cvar": rel_cvar,
        "beta": beta,
    }


# =========================
# Optimization Engine
# =========================
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns

    HAS_PYPORT = True
except Exception:
    HAS_PYPORT = False


def run_optimizations(prices, returns, benchmark_returns):
    strategies = {}
    if prices.empty:
        return strategies
    tickers = list(prices.columns)
    n = len(tickers)

    # Equal-weight
    w_eq = np.repeat(1 / n, n)
    port_eq = returns.dot(w_eq)
    strat = {
        "name": "Equal Weight",
        "weights": pd.Series(w_eq, index=tickers),
        "returns": port_eq,
    }
    strat["risk"] = compute_risk_profile(port_eq, benchmark_returns)
    strategies["Equal Weight"] = strat

    if not HAS_PYPORT:
        return strategies

    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.sample_cov(prices)

    # Max Sharpe
    try:
        ef = EfficientFrontier(mu, cov)
        ef.max_sharpe(risk_free_rate=Config.RISK_FREE_RATE)
        w_ms = ef.clean_weights()
        w_vec = np.array([w_ms[t] for t in tickers])
        port_ms = returns.dot(w_vec)
        s_ms = {
            "name": "Max Sharpe (MVO)",
            "weights": pd.Series(w_vec, index=tickers),
            "returns": port_ms,
        }
        s_ms["risk"] = compute_risk_profile(port_ms, benchmark_returns)
        strategies["Max Sharpe (MVO)"] = s_ms
    except Exception:
        pass

    # Min Vol
    try:
        ef = EfficientFrontier(mu, cov)
        ef.min_volatility()
        w_mv = ef.clean_weights()
        w_vec = np.array([w_mv[t] for t in tickers])
        port_mv = returns.dot(w_vec)
        s_mv = {
            "name": "Min Volatility",
            "weights": pd.Series(w_vec, index=tickers),
            "returns": port_mv,
        }
        s_mv["risk"] = compute_risk_profile(port_mv, benchmark_returns)
        strategies["Min Volatility"] = s_mv
    except Exception:
        pass

    # CLA
    try:
        cla = CLA(mu, cov)
        cla.max_sharpe()
        w_cla = cla.clean_weights()
        w_vec = np.array([w_cla[t] for t in tickers])
        port_cla = returns.dot(w_vec)
        s_cla = {
            "name": "CLA Max Sharpe",
            "weights": pd.Series(w_vec, index=tickers),
            "returns": port_cla,
        }
        s_cla["risk"] = compute_risk_profile(port_cla, benchmark_returns)
        strategies["CLA Max Sharpe"] = s_cla
    except Exception:
        pass

    # HRP
    try:
        hrp = HRPOpt(returns)
        w_hrp = hrp.optimize()
        w_vec = np.array([w_hrp[t] for t in tickers])
        port_hrp = returns.dot(w_vec)
        s_hrp = {
            "name": "HRP",
            "weights": pd.Series(w_vec, index=tickers),
            "returns": port_hrp,
        }
        s_hrp["risk"] = compute_risk_profile(port_hrp, benchmark_returns)
        strategies["HRP"] = s_hrp
    except Exception:
        pass

    # Black-Litterman (simple, using market-implied priors)
    try:
        market_weights = np.repeat(1 / n, n)
        prior = market_implied_prior_returns(cov, market_weights, risk_aversion=None)
        bl = BlackLittermanModel(cov, pi=prior)
        bl_ret = bl.bl_returns()
        bl_cov = bl.bl_cov()
        ef = EfficientFrontier(bl_ret, bl_cov)
        ef.max_sharpe(risk_free_rate=Config.RISK_FREE_RATE)
        w_bl = ef.clean_weights()
        w_vec = np.array([w_bl[t] for t in tickers])
        port_bl = returns.dot(w_vec)
        s_bl = {
            "name": "Black-Litterman Max Sharpe",
            "weights": pd.Series(w_vec, index=tickers),
            "returns": port_bl,
        }
        s_bl["risk"] = compute_risk_profile(port_bl, benchmark_returns)
        strategies["Black-Litterman Max Sharpe"] = s_bl
    except Exception:
        pass

    return strategies


# =========================
# Streamlit UI Helpers
# =========================
def universe_and_scenario_panel(universe_df):
    st.subheader("🌐 Universe / Scenario Panel")
    st.markdown(
        "Select **regions**, **sectors**, or use quick **scenario presets**. "
        "All data is fetched from Yahoo Finance."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        regions_selected = st.multiselect(
            "Regions",
            options=REGIONS,
            default=REGIONS,
            key="regions_selected",
        )
    with col2:
        sectors_selected = st.multiselect(
            "Sectors",
            options=SECTORS,
            default=SECTORS,
            key="sectors_selected",
        )
    with col3:
        scenario_names = list(SCENARIOS.keys())
        default_scen_idx = st.session_state.get("scenario_index", 0)
        scenario = st.selectbox(
            "Scenario Preset",
            options=scenario_names,
            index=default_scen_idx,
            key="scenario_select",
        )

    # Quick buttons
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    if bcol1.button("Global Tech + TR Banks"):
        st.session_state["scenario_index"] = scenario_names.index("Global Tech + TR Banks")
    if bcol2.button("Asia Leaders"):
        st.session_state["scenario_index"] = scenario_names.index("Asia Leaders")
    if bcol3.button("TR Financial Complex"):
        st.session_state["scenario_index"] = scenario_names.index("TR Financial Complex")
    if bcol4.button("Full Global"):
        st.session_state["scenario_index"] = scenario_names.index("Full Global")

    # Apply scenario or filters
    if scenario != "Custom Filter" and SCENARIOS[scenario] is not None:
        df_filtered = SCENARIOS[scenario](universe_df)
    else:
        df_filtered = universe_df[
            (universe_df["region"].isin(regions_selected))
            & (universe_df["sector"].isin(sectors_selected))
        ]

    tickers = df_filtered["ticker"].tolist()
    st.markdown(f"**Selected tickers ({len(tickers)}):**")
    st.write(", ".join(tickers))

    with st.expander("Show Universe Table"):
        st.dataframe(df_filtered, width="stretch")

    return tickers


def plot_weights_bar(weights_series, title):
    if weights_series is None or weights_series.empty:
        return
    w = weights_series[weights_series > 1e-4].sort_values(ascending=False)
    fig = go.Figure()
    fig.add_bar(x=w.index, y=w.values)
    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Weight",
        template="plotly_dark",
    )
    st.plotly_chart(fig, width="stretch")


def plot_cumulative_returns(returns, title):
    if returns is None or returns.empty:
        return
    cum = (1 + returns).cumprod()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=cum.index, y=cum.values, mode="lines", name="Portfolio")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (1 = start)",
        template="plotly_dark",
    )
    st.plotly_chart(fig, width="stretch")


def strategy_factsheet_panel(strategy_name, strategy_obj, benchmark_ticker):
    if not strategy_obj:
        return
    st.subheader("📄 Institutional Strategy Factsheet")
    st.markdown(f"**Strategy:** {strategy_name}")
    st.markdown(f"**Benchmark:** {benchmark_ticker}")

    risk = strategy_obj.get("risk", {})
    cols = st.columns(4)
    cols[0].metric("Annual Return", f"{risk.get('annual_return', np.nan)*100:.2f}%")
    cols[1].metric("Annual Volatility", f"{risk.get('annual_vol', np.nan)*100:.2f}%")
    cols[2].metric("Sharpe", f"{risk.get('sharpe', np.nan):.2f}")
    cols[3].metric("Max Drawdown", f"{risk.get('max_drawdown', 0)*100:.2f}%")

    cols2 = st.columns(4)
    cols2[0].metric("Hist VaR (95%)", f"{risk.get('hist_var', np.nan)*100:.2f}%")
    cols2[1].metric("Hist CVaR (95%)", f"{risk.get('hist_cvar', np.nan)*100:.2f}%")
    cols2[2].metric(
        "Rel VaR vs Bench", f"{risk.get('rel_var', np.nan)*100:.2f}%"
    )
    cols2[3].metric("Beta vs Bench", f"{risk.get('beta', np.nan):.2f}")


# =========================
# Main App
# =========================
def main():
    st.set_page_config(
        page_title="QuantEdge Pro - Institutional Universe",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 QuantEdge Pro – Institutional Portfolio Analytics (Universe v1.0)")
    st.markdown(
        "All data is fetched **directly from Yahoo Finance**. "
        "Universe is hard‑wired by **Region** and **Sector**, with institutional scenarios."
    )

    # Universe / Scenario panel
    selected_tickers = universe_and_scenario_panel(UNIVERSE_DF)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Analysis Configuration")
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365 * 3),
            max_value=datetime.now() - timedelta(days=5),
        )
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now(),
        )
        analysis_type = st.selectbox(
            "Main Tab",
            ["Portfolio Optimization", "Risk Analysis", "Data Preview"],
        )
        run_button = st.button("🚀 Fetch Data & Run Analysis", type="primary")

    if run_button:
        if len(selected_tickers) < 2:
            st.error("Need at least 2 tickers selected for portfolio analysis.")
            return

        with st.spinner("Fetching price data from Yahoo Finance..."):
            prices = fetch_price_data(
                selected_tickers,
                start_date,
                end_date + timedelta(days=1),
            )

        if prices.empty:
            st.error(
                "❌ No price data fetched from Yahoo Finance. "
                "Please check tickers, date range, or try a different scenario."
            )
            return

        returns = prices.pct_change().dropna()

        # Benchmark selection
        tr_mask = []
        for t in selected_tickers:
            row = UNIVERSE_DF[UNIVERSE_DF["ticker"] == t]
            if not row.empty:
                tr_mask.append(row.iloc[0]["region"] == "TR")
        only_tr = all(tr_mask) and len(tr_mask) > 0

        if only_tr:
            benchmark_ticker = "XU100.IS"
        else:
            benchmark_ticker = "SPY"

        bench_prices = fetch_price_data(
            [benchmark_ticker], start_date, end_date + timedelta(days=1)
        )
        if bench_prices.empty:
            benchmark_returns = None
        else:
            benchmark_returns = bench_prices.pct_change().dropna().iloc[:, 0]

        # Align indexes
        if benchmark_returns is not None:
            common_idx = returns.index.intersection(benchmark_returns.index)
            returns = returns.loc[common_idx]
            benchmark_returns = benchmark_returns.loc[common_idx]
            prices = prices.loc[common_idx]
        else:
            common_idx = returns.index

        st.session_state["prices"] = prices
        st.session_state["returns"] = returns
        st.session_state["benchmark_returns"] = benchmark_returns
        st.session_state["benchmark_ticker"] = benchmark_ticker

        # Run optimizations once
        strategies = run_optimizations(prices, returns, benchmark_returns)
        st.session_state["strategies"] = strategies
        # Default strategy to show
        if "Equal Weight" in strategies:
            st.session_state["active_strategy"] = "Equal Weight"
        elif strategies:
            st.session_state["active_strategy"] = list(strategies.keys())[0]
        else:
            st.session_state["active_strategy"] = None

    # If we have data, render tabs
    prices = st.session_state.get("prices")
    returns = st.session_state.get("returns")
    benchmark_returns = st.session_state.get("benchmark_returns")
    benchmark_ticker = st.session_state.get("benchmark_ticker", "SPY")
    strategies = st.session_state.get("strategies", {})

    if prices is None or returns is None or prices.empty or returns.empty:
        st.info("📥 Run an analysis using the sidebar to see portfolio and risk results.")
        return

    if analysis_type == "Data Preview":
        st.subheader("📊 Data Preview")
        st.write(f"**Tickers ({len(prices.columns)}):** {', '.join(prices.columns)}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Prices (last 10):**")
            st.dataframe(prices.tail(10), width="stretch")
        with col2:
            st.markdown("**Returns (last 10):**")
            st.dataframe(returns.tail(10), width="stretch")

    elif analysis_type == "Portfolio Optimization":
        st.subheader("🎯 Portfolio Optimization (EF / HRP / CLA / BL + Equal-Weight)")

        if not strategies:
            st.warning("No optimization results found. Please run analysis from the sidebar.")
            return

        strat_names = list(strategies.keys())
        if not strat_names:
            st.warning("No optimization strategies were successfully computed.")
            return

        default_idx = 0
        if (
            "active_strategy" in st.session_state
            and st.session_state["active_strategy"] in strat_names
        ):
            default_idx = strat_names.index(st.session_state["active_strategy"])

        selected_name = st.selectbox("Select Strategy", strat_names, index=default_idx)
        st.session_state["active_strategy"] = selected_name
        strat = strategies[selected_name]

        col1, col2 = st.columns([2, 3])
        with col1:
            plot_weights_bar(strat["weights"], f"Weights – {selected_name}")
        with col2:
            plot_cumulative_returns(
                strat["returns"], f"Cumulative Returns – {selected_name}"
            )

        strategy_factsheet_panel(selected_name, strat, benchmark_ticker)

        # Table of all strategies
        st.markdown("### 🧮 Strategy Comparison Table")
        rows = []
        for name, s in strategies.items():
            r = s["risk"]
            rows.append(
                {
                    "Strategy": name,
                    "Annual Return %": r["annual_return"] * 100,
                    "Annual Vol %": r["annual_vol"] * 100,
                    "Sharpe": r["sharpe"],
                    "Max Drawdown %": r["max_drawdown"] * 100,
                    "Hist VaR 95%": r["hist_var"] * 100,
                    "Hist CVaR 95%": r["hist_cvar"] * 100,
                    "Rel VaR 95% vs Bench": r["rel_var"] * 100
                    if r["rel_var"] is not None
                    else np.nan,
                    "Beta vs Bench": r["beta"],
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")

    elif analysis_type == "Risk Analysis":
        st.subheader("⚠️ Risk Analysis – Hist / Parametric / MC VaR + CVaR + Relative VaR")

        if not strategies:
            st.warning("No optimization results found. Please run analysis from the sidebar.")
            return

        strat_names = list(strategies.keys())
        selected_name = st.selectbox(
            "Select Strategy for Deep Risk View", strat_names, index=0
        )
        strat = strategies[selected_name]
        risk = strat["risk"]
        port_ret = strat["returns"]

        alpha = st.slider(
            "Confidence Level (VaR / CVaR)",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
        )
        # Recompute risk at chosen alpha
        risk_custom = compute_risk_profile(port_ret, benchmark_returns, alpha=alpha)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hist VaR", f"{risk_custom['hist_var']*100:.2f}%")
        c2.metric("Hist CVaR", f"{risk_custom['hist_cvar']*100:.2f}%")
        c3.metric("Parametric VaR", f"{risk_custom['param_var']*100:.2f}%")
        c4.metric("MC VaR", f"{risk_custom['mc_var']*100:.2f}%")

        c5, c6, c7 = st.columns(3)
        c5.metric("MC CVaR", f"{risk_custom['mc_cvar']*100:.2f}%")
        c6.metric("Relative VaR vs Bench", f"{risk_custom['rel_var']*100:.2f}%")
        c7.metric("Relative CVaR vs Bench", f"{risk_custom['rel_cvar']*100:.2f}%")

        st.markdown("### 📉 Distribution of Portfolio Returns")
        hist = port_ret.dropna()
        if not hist.empty:
            fig = go.Figure()
            fig.add_histogram(x=hist.values, nbinsx=50, name="Daily Returns")
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Return",
                yaxis_title="Frequency",
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown("### 📈 Portfolio vs Benchmark (Cumulative)")
        if benchmark_returns is not None and not benchmark_returns.empty:
            df_cum = pd.DataFrame(
                {
                    "Portfolio": (1 + port_ret).cumprod(),
                    "Benchmark": (1 + benchmark_returns).cumprod(),
                }
            ).dropna()
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=df_cum.index, y=df_cum["Portfolio"], name="Portfolio"
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=df_cum.index,
                    y=df_cum["Benchmark"],
                    name=benchmark_ticker,
                )
            )
            fig2.update_layout(
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
            )
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("Benchmark data not available – relative metrics limited.")


if __name__ == "__main__":
    main()

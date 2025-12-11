import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 1) Instrument Universe Definitions
# ============================================================================

REGIONAL_UNIVERSE = {
    "US": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "AVGO", "ADBE", "NFLX"
    ],
    "TR": [
        # --- 30 Major TR Blue-Chips ---
        "AKBNK.IS", "GARAN.IS", "YKBNK.IS", "ISCTR.IS", "HALKB.IS", "VAKBN.IS",
        "TUPRS.IS", "BIMAS.IS", "THYAO.IS", "ASELS.IS", "SAHOL.IS", "KCHOL.IS",
        "EREGL.IS", "PETKM.IS", "TOASO.IS", "TCELL.IS", "ARCLK.IS", "SISE.IS",
        "TKFEN.IS", "DOHOL.IS", "PGSUS.IS", "TAVHL.IS", "HEKTS.IS", "KOZAA.IS",
        "KOZAL.IS", "ENJSA.IS", "TTKOM.IS", "ALARK.IS", "MGROS.IS", "FROTO.IS",
        # --- 30 Extended TR: leasing / insurance / GYO / industrials ---
        "ISFIN.IS", "VAKFN.IS", "SEKFK.IS", "AGESA.IS", "ANHYT.IS", "ANSGR.IS",
        "AKGRT.IS", "RAYSG.IS", "HALKS.IS", "ISGYO.IS", "TSGYO.IS", "SNGYO.IS",
        "KLGYO.IS", "ALGYO.IS", "VKGYO.IS", "ENKAI.IS", "OTKAR.IS", "VESBE.IS",
        "TTRAK.IS", "AYGAZ.IS", "ZOREN.IS", "TRGYO.IS", "NUGYO.IS", "HLGYO.IS",
        "AKSEN.IS", "CLEBI.IS", "KRDMD.IS", "OYAKC.IS", "CCOLA.IS", "BRSAN.IS"
    ],
    "JP": [
        # 10 Major JP stocks
        "7203.T", "6758.T", "9984.T", "7267.T", "8035.T",
        "4063.T", "6954.T", "7974.T", "9983.T", "4502.T",
        # 10 Major JP banks
        "8306.T", "8411.T", "8308.T", "8309.T", "8355.T",
        "8331.T", "8354.T", "7182.T", "7167.T", "7327.T"
    ],
    "KR": [
        "005930.KS", "000660.KS", "035420.KS", "005380.KS", "051910.KS",
        "035720.KS", "105560.KS", "015760.KS", "066570.KS", "005490.KS"
    ],
    "SG": [
        "D05.SI", "U11.SI", "O39.SI", "C07.SI", "Z74.SI",
        "C09.SI", "Y92.SI", "M44U.SI", "C52.SI", "BN4.SI"
    ],
    "CN": [
        # US-listed Chinese megacaps
        "BABA", "TCEHY", "JD", "PDD", "BIDU",
        "NTES", "NIO", "XPEV", "LI", "BILI"
    ]
}

# TR banks subset for scenario
TR_BANKS_10 = [
    "AKBNK.IS", "GARAN.IS", "YKBNK.IS", "ISCTR.IS", "HALKB.IS",
    "VAKBN.IS", "TSKB.IS", "QNBFB.IS", "ALBRK.IS", "SKBNK.IS"
]

US_CORE_TECH_10 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AVGO", "ADBE", "NFLX"
]

SCENARIO_PRESETS = {
    "Custom (by region selection)": [],
    "Global Tech + TR Banks": US_CORE_TECH_10 + TR_BANKS_10,
    "All Regions (full universe)": sorted(
        {t for lst in REGIONAL_UNIVERSE.values() for t in lst}
    ),
}

# ============================================================================
# 2) PyPortfolioOpt Imports
# ============================================================================

try:
    from pypfopt import expected_returns, risk_models, EfficientFrontier, CLA, HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

TRADING_DAYS_PER_YEAR = 252

# ============================================================================
# 3) Helper Functions
# ============================================================================


def build_ticker_list(selected_regions, scenario_name, manual_tickers_str):
    """Combine regional universe, scenario preset, and manual tickers."""
    # Start from scenario if chosen
    if scenario_name != "Custom (by region selection)":
        base = set(SCENARIO_PRESETS[scenario_name])
    else:
        base = set()
        for region in selected_regions:
            base.update(REGIONAL_UNIVERSE.get(region, []))

    # Add manual tickers
    if manual_tickers_str:
        manual = [
            t.strip().upper()
            for t in manual_tickers_str.split(",")
            if t.strip()
        ]
        base.update(manual)

    return sorted(base)


def fetch_price_data(tickers, start_date, end_date):
    """Fetch adjusted close prices from Yahoo Finance."""
    if not tickers:
        raise ValueError("No tickers selected.")

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "No data returned from Yahoo Finance. "
            "Check tickers and date range."
        )

    # MultiIndex vs single index columns
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns.levels[0]:
            prices = data["Close"].copy()
        else:
            raise ValueError("Unexpected data format from Yahoo Finance.")
    else:
        # Single ticker case
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        prices = data[[col]].copy()
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="all")

    # Drop columns with too few observations
    valid_cols = [c for c in prices.columns if prices[c].dropna().shape[0] > 30]
    prices = prices[valid_cols]

    if prices.shape[1] < 2:
        raise ValueError("Need at least 2 assets with data for optimization.")

    return prices


def compute_portfolio_performance(weights, mu, cov):
    """Given weights, expected returns, and covariance, compute annualized metrics."""
    weights = np.array(weights)
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(weights.T @ cov @ weights))
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def optimize_portfolio(model_name, prices):
    """Run chosen optimization model and return weights & performance."""
    returns = prices.pct_change().dropna()
    mu = expected_returns.mean_historical_return(
        prices, frequency=TRADING_DAYS_PER_YEAR
    )
    cov = risk_models.sample_cov(prices, frequency=TRADING_DAYS_PER_YEAR)

    n = len(mu)
    tickers = list(mu.index)

    # If PyPortfolioOpt is missing, or Equal Weight chosen → fallback
    if model_name == "Equal Weight" or not PYPFOPT_AVAILABLE:
        weights = np.ones(n) / n
        label = "Equal Weight"

    elif model_name == "Mean-Variance (Max Sharpe)":
        ef = EfficientFrontier(mu, cov)
        ef.max_sharpe()
        weights = np.array(list(ef.clean_weights().values()))
        label = "Max Sharpe (MV)"

    elif model_name == "Min Volatility":
        ef = EfficientFrontier(mu, cov)
        ef.min_volatility()
        weights = np.array(list(ef.clean_weights().values()))
        label = "Min Volatility"

    elif model_name == "Critical Line Algorithm (CLA)":
        cla = CLA(mu, cov)
        cla.max_sharpe()
        weights = np.array(list(cla.clean_weights().values()))
        label = "CLA Max Sharpe"

    elif model_name == "Hierarchical Risk Parity (HRP)":
        hrp = HRPOpt(returns)
        hrp_weights = hrp.optimize()
        weights = np.array([hrp_weights[t] for t in tickers])
        label = "HRP"

    elif model_name == "Black-Litterman":
        # Simple BL: equal market caps, no explicit views
        market_caps = pd.Series(np.ones(n), index=tickers)
        prior = market_implied_prior_returns(
            market_caps,
            cov_matrix=cov,
        )
        bl = BlackLittermanModel(cov, pi=prior)
        mu_bl = bl.bl_returns()
        cov_bl = bl.bl_cov()
        ef = EfficientFrontier(mu_bl, cov_bl)
        ef.max_sharpe()
        weights = np.array(list(ef.clean_weights().values()))
        label = "Black-Litterman"

    else:
        weights = np.ones(n) / n
        label = "Equal Weight (fallback)"

    ret, vol, sharpe = compute_portfolio_performance(
        weights, mu.values, cov.values
    )
    return pd.Series(weights, index=tickers), ret, vol, sharpe, mu, cov, label


def generate_random_portfolios(mu, cov, n_portfolios=2000, seed=42):
    """Monte Carlo cloud of random portfolios for efficient frontier visualisation."""
    np.random.seed(seed)
    n_assets = len(mu)
    returns = []
    vols = []
    weights_list = []

    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, _ = compute_portfolio_performance(w, mu.values, cov.values)
        returns.append(r)
        vols.append(v)
        weights_list.append(w)

    return np.array(returns), np.array(vols), weights_list


# ============================================================================
# 4) Streamlit App
# ============================================================================

def main():
    st.set_page_config(
        page_title="QuantEdge Pro - Portfolio Optimization",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 QuantEdge Pro – Portfolio Optimization (Enhanced Universe)")
    st.markdown(
        """
        **This streamlined version focuses on one thing: working, visible optimization.**

        ✅ Expanded instrument universe (US / TR / JP / KR / SG / CN)  
        ✅ Scenario presets (e.g. *Global Tech + TR Banks*)  
        ✅ Models: Equal Weight, MV Max Sharpe, Min Vol, CLA, HRP, Black-Litterman  
        ✅ Monte-Carlo efficient frontier cloud + optimized vs equal-weight markers
        """
    )

    # ---------------- Sidebar configuration ----------------
    with st.sidebar:
        st.header("Universe & Scenario")

        selected_regions = st.multiselect(
            "Regions",
            options=list(REGIONAL_UNIVERSE.keys()),
            default=["US", "TR"],
        )

        scenario_name = st.selectbox(
            "Scenario preset",
            options=list(SCENARIO_PRESETS.keys()),
            index=0,
            help="Preset will override region selection (you can still add extra tickers below).",
        )

        manual_tickers_str = st.text_area(
            "Extra tickers (optional)",
            value="",
            help="Comma-separated list, e.g. NVDA, TUPRS.IS",
        )

        st.subheader("Date range")
        end_date = st.date_input("End date", value=datetime.today())
        start_date = st.date_input(
            "Start date",
            value=end_date - timedelta(days=365 * 3),
        )

        st.subheader("Optimization model")
        model_name = st.selectbox(
            "Optimization model",
            options=[
                "Equal Weight",
                "Mean-Variance (Max Sharpe)",
                "Min Volatility",
                "Critical Line Algorithm (CLA)",
                "Hierarchical Risk Parity (HRP)",
                "Black-Litterman",
            ],
        )

        run_btn = st.button("🚀 Fetch & Optimize", type="primary")

        if not PYPFOPT_AVAILABLE:
            st.warning(
                "PyPortfolioOpt not found – advanced models will fall back to Equal Weight.\n"
                "Install with `pip install PyPortfolioOpt` for full functionality."
            )

    # ---------------- Main logic ----------------
    if run_btn:
        try:
            tickers = build_ticker_list(
                selected_regions, scenario_name, manual_tickers_str
            )
            st.write(f"**Selected tickers ({len(tickers)}):** {', '.join(tickers)}")

            with st.spinner("Fetching data from Yahoo Finance..."):
                prices = fetch_price_data(tickers, start_date, end_date)

            st.success(
                f"✅ Data loaded: {prices.shape[1]} assets, {prices.shape[0]} rows."
            )

            returns = prices.pct_change().dropna()

            # ---------------- Portfolio Optimization ----------------
            st.subheader("🎯 Portfolio Optimization Results")

            (
                weights_opt,
                ret_opt,
                vol_opt,
                sharpe_opt,
                mu,
                cov,
                model_label,
            ) = optimize_portfolio(model_name, prices)

            # Equal-weight benchmark
            w_eq = np.ones(len(mu)) / len(mu)
            ret_eq, vol_eq, sharpe_eq = compute_portfolio_performance(
                w_eq, mu.values, cov.values
            )

            # KPI cards – Optimized
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Optimized Return (ann.)", f"{ret_opt * 100:.2f}%")
            with c2:
                st.metric("Optimized Volatility (ann.)", f"{vol_opt * 100:.2f}%")
            with c3:
                st.metric("Optimized Sharpe", f"{sharpe_opt:.2f}")

            # KPI cards – Equal Weight
            c4, c5, c6 = st.columns(3)
            with c4:
                st.metric("EQ Return (ann.)", f"{ret_eq * 100:.2f}%")
            with c5:
                st.metric("EQ Volatility (ann.)", f"{vol_eq * 100:.2f}%")
            with c6:
                st.metric("EQ Sharpe", f"{sharpe_eq:.2f}")

            # Weights table
            st.markdown("#### Portfolio Weights")
            weights_df = pd.DataFrame(
                {
                    "Ticker": weights_opt.index,
                    "Weight": weights_opt.values,
                }
            ).sort_values("Weight", ascending=False)
            st.dataframe(weights_df, use_container_width=True)

            # ---------------- Efficient frontier cloud ----------------
            st.markdown("#### Efficient Frontier (Monte-Carlo Cloud)")

            rand_rets, rand_vols, _ = generate_random_portfolios(mu, cov, n_portfolios=2000)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=rand_vols,
                    y=rand_rets,
                    mode="markers",
                    name="Random Portfolios",
                    opacity=0.4,
                    marker=dict(size=4),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[vol_eq],
                    y=[ret_eq],
                    mode="markers",
                    name="Equal Weight",
                    marker=dict(size=12, symbol="x"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[vol_opt],
                    y=[ret_opt],
                    mode="markers",
                    name=f"Optimized ({model_label})",
                    marker=dict(size=14, symbol="star"),
                )
            )
            fig.update_layout(
                xaxis_title="Volatility (ann.)",
                yaxis_title="Return (ann.)",
                template="plotly_dark",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---------------- Asset-level expected stats ----------------
            st.markdown("#### Asset-level Expected Stats (Annualised)")
            asset_stats = pd.DataFrame(
                {
                    "Expected Return %": mu * 100,
                    "Volatility %": np.sqrt(np.diag(cov)) * 100,
                    "Weight": weights_opt,
                }
            )
            st.dataframe(
                asset_stats.sort_values("Weight", ascending=False),
                use_container_width=True,
            )

            # Quick preview of price & return data
            with st.expander("📊 Data preview (prices & returns)"):
                t1, t2 = st.tabs(["Prices (tail)", "Returns (tail)"])
                with t1:
                    st.dataframe(prices.tail(), use_container_width=True)
                with t2:
                    st.dataframe(returns.tail(), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()

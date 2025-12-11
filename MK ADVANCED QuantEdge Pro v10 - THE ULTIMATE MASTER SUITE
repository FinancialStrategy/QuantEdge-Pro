# ==============================================================================
# QuantEdge Pro v10 - THE ULTIMATE MASTER SUITE
# ==============================================================================
# Description:
#   This is the fully-featured, institutional-grade Portfolio Management System.
#   It integrates:
#   1. Global Commodities (Futures) & BIST 30 Equities.
#   2. Deep Financials (Banks, Insurance, Factoring, Leasing).
#   3. Advanced Optimization (Max Sharpe, Min Volatility, HRP).
#   4. Comprehensive Risk Engine (VaR, CVaR - Historical, Parametric, Monte Carlo).
#   5. Stress Testing (Historical Scenarios).
#   6. Efficient Frontier & Monte Carlo Simulations.
#
# Libraries Required:
#   pip install streamlit pandas numpy yfinance plotly scipy PyPortfolioOpt
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
import io

# --- PyPortfolioOpt Imports (Critical for Optimization) ---
try:
    from pypfopt import efficient_frontier, risk_models, expected_returns
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt import objective_functions
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

# Suppress Warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION LAYER
# ==============================================================================
class Config:
    """Application-wide configuration settings."""
    APP_NAME = "QuantEdge Pro v10 - Ultimate Enterprise Edition"
    TRADING_DAYS = 252
    
    # Risk Free Rates (Annualized)
    RF_RATE_TRY = 0.35  # 35% Conservative Proxy for TR
    RF_RATE_USD = 0.045 # 4.5% US Treasury Proxy
    
    # Monte Carlo Settings
    MC_SIMS_VAR = 10000      # High precision for VaR
    MC_SIMS_FRONTIER = 2000  # For plotting the cloud
    
    # Data Settings
    CACHE_TTL = 3600 # Cache data for 1 hour to prevent API spam

# ==============================================================================
# 2. ASSET UNIVERSE LAYER (Complete List)
# ==============================================================================
class AssetUniverse:
    """Defines the complete list of available assets across all classes."""
    
    INSTRUMENTS = [
        # --- BENCHMARKS ---
        {"ticker": "XU100.IS", "name": "BIST 100 Index", "region": "TR", "sector": "Benchmark", "class": "Index"},
        {"ticker": "SPY", "name": "S&P 500 ETF", "region": "US", "sector": "Benchmark", "class": "Index"},

        # --- BIST 30: INDUSTRIAL GIANTS ---
        {"ticker": "THYAO.IS", "name": "Turkish Airlines", "region": "TR", "sector": "Transportation", "class": "Equity"},
        {"ticker": "TUPRS.IS", "name": "Tupras Refineries", "region": "TR", "sector": "Energy", "class": "Equity"},
        {"ticker": "EREGL.IS", "name": "Erdemir Steel", "region": "TR", "sector": "Materials", "class": "Equity"},
        {"ticker": "SISE.IS", "name": "Sisecam", "region": "TR", "sector": "Industrials", "class": "Equity"},
        {"ticker": "BIMAS.IS", "name": "BIM Retail", "region": "TR", "sector": "Consumer", "class": "Equity"},
        {"ticker": "ASELS.IS", "name": "Aselsan Defense", "region": "TR", "sector": "Defense", "class": "Equity"},
        {"ticker": "KCHOL.IS", "name": "Koc Holding", "region": "TR", "sector": "Conglomerate", "class": "Equity"},
        {"ticker": "SAHOL.IS", "name": "Sabanci Holding", "region": "TR", "sector": "Conglomerate", "class": "Equity"},
        {"ticker": "FROTO.IS", "name": "Ford Otosan", "region": "TR", "sector": "Automotive", "class": "Equity"},
        {"ticker": "TOASO.IS", "name": "Tofas Auto", "region": "TR", "sector": "Automotive", "class": "Equity"},
        {"ticker": "TCELL.IS", "name": "Turkcell", "region": "TR", "sector": "Telecom", "class": "Equity"},
        {"ticker": "ENKAI.IS", "name": "Enka Construction", "region": "TR", "sector": "Construction", "class": "Equity"},
        {"ticker": "KOZAL.IS", "name": "Koza Gold", "region": "TR", "sector": "Mining", "class": "Equity"},
        {"ticker": "ASTOR.IS", "name": "Astor Energy", "region": "TR", "sector": "Energy", "class": "Equity"},

        # --- BIST FINANCIALS: BANKS ---
        {"ticker": "AKBNK.IS", "name": "Akbank", "region": "TR", "sector": "Banking", "class": "Equity"},
        {"ticker": "GARAN.IS", "name": "Garanti BBVA", "region": "TR", "sector": "Banking", "class": "Equity"},
        {"ticker": "YKBNK.IS", "name": "Yapi Kredi", "region": "TR", "sector": "Banking", "class": "Equity"},
        {"ticker": "ISCTR.IS", "name": "Is Bankasi", "region": "TR", "sector": "Banking", "class": "Equity"},
        {"ticker": "HALKB.IS", "name": "Halkbank", "region": "TR", "sector": "Banking", "class": "Equity"},
        {"ticker": "VAKBN.IS", "name": "Vakifbank", "region": "TR", "sector": "Banking", "class": "Equity"},
        {"ticker": "TSKB.IS", "name": "TSKB", "region": "TR", "sector": "Banking", "class": "Equity"},

        # --- BIST FINANCIALS: NON-BANK (Insurance, Leasing, Factoring) ---
        {"ticker": "AKGRT.IS", "name": "Aksigorta", "region": "TR", "sector": "Insurance", "class": "Equity"},
        {"ticker": "TURSG.IS", "name": "Turkiye Sigorta", "region": "TR", "sector": "Insurance", "class": "Equity"},
        {"ticker": "ANSGR.IS", "name": "Anadolu Sigorta", "region": "TR", "sector": "Insurance", "class": "Equity"},
        {"ticker": "ISFIN.IS", "name": "Is Leasing", "region": "TR", "sector": "Leasing", "class": "Equity"},
        {"ticker": "LIDFA.IS", "name": "Lider Factoring", "region": "TR", "sector": "Factoring", "class": "Equity"},
        {"ticker": "ULUFA.IS", "name": "Ulusal Factoring", "region": "TR", "sector": "Factoring", "class": "Equity"},

        # --- GLOBAL COMMODITIES & METALS (Futures) ---
        {"ticker": "GC=F", "name": "Gold Futures", "region": "Global", "sector": "Precious Metals", "class": "Commodity"},
        {"ticker": "SI=F", "name": "Silver Futures", "region": "Global", "sector": "Precious Metals", "class": "Commodity"},
        {"ticker": "PL=F", "name": "Platinum Futures", "region": "Global", "sector": "Precious Metals", "class": "Commodity"},
        {"ticker": "HG=F", "name": "Copper Futures", "region": "Global", "sector": "Industrial Metals", "class": "Commodity"},
        {"ticker": "CL=F", "name": "Crude Oil (WTI)", "region": "Global", "sector": "Energy", "class": "Commodity"},
        {"ticker": "BZ=F", "name": "Brent Crude", "region": "Global", "sector": "Energy", "class": "Commodity"},
        {"ticker": "NG=F", "name": "Natural Gas", "region": "Global", "sector": "Energy", "class": "Commodity"},
        {"ticker": "ZC=F", "name": "Corn", "region": "Global", "sector": "Agriculture", "class": "Commodity"},
        {"ticker": "ZW=F", "name": "Wheat", "region": "Global", "sector": "Agriculture", "class": "Commodity"},
    ]

    @classmethod
    def get_df(cls):
        return pd.DataFrame(cls.INSTRUMENTS)

    @staticmethod
    def get_scenarios():
        """Returns filters for preset scenarios."""
        return {
            "BIST 30 (Core Industrial & Holding)": lambda df: df[
                (df["region"] == "TR") & 
                (df["sector"].isin(["Transportation", "Energy", "Materials", "Industrials", "Consumer", "Defense", "Conglomerate", "Automotive", "Telecom", "Mining"]))
            ],
            "TR Financial Complex (Bank/Ins/Leas/Fact)": lambda df: df[
                (df["region"] == "TR") & 
                (df["sector"].isin(["Banking", "Insurance", "Leasing", "Factoring"]))
            ],
            "Commodities Super-Cycle (Inflation Hedge)": lambda df: df[
                df["class"] == "Commodity"
            ],
            "Hybrid: TR Banks + Precious Metals": lambda df: df[
                ((df["sector"] == "Banking") & (df["region"] == "TR")) | 
                (df["sector"] == "Precious Metals")
            ],
            "Full Universe (Manual Selection)": lambda df: df
        }

# ==============================================================================
# 3. DATA LAYER
# ==============================================================================
class DataEngine:
    """Handles fetching and cleaning of financial data."""

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def fetch_data(tickers, start_date, end_date):
        if not tickers:
            return pd.DataFrame()
        
        # Buffer to ensure moving averages or returns are valid at start_date
        buffer = start_date - timedelta(days=60)
        
        try:
            # Download with group_by='column' for consistent MultiIndex handling
            raw = yf.download(tickers, start=buffer, end=end_date, auto_adjust=True, progress=False, group_by='column')
            
            if raw.empty:
                return pd.DataFrame()

            # Column Extraction Strategy
            if isinstance(raw.columns, pd.MultiIndex):
                if 'Adj Close' in raw.columns.levels[0]:
                    data = raw['Adj Close']
                elif 'Close' in raw.columns.levels[0]:
                    data = raw['Close']
                else:
                    # Fallback
                    data = raw.iloc[:, :len(tickers)]
            else:
                # Single ticker case
                data = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=tickers[0])

            # Data Cleaning:
            # 1. Remove columns that are entirely NaN (failed downloads)
            data = data.dropna(axis=1, how='all')
            # 2. Forward fill holidays
            data = data.ffill()
            # 3. Drop remaining NaNs
            data = data.dropna()
            
            # Filter to requested start date
            data = data[data.index >= pd.to_datetime(start_date)]
            
            return data

        except Exception as e:
            st.error(f"DataEngine Error: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_benchmark(start_date, end_date):
        return DataEngine.fetch_data(["XU100.IS"], start_date, end_date)

# ==============================================================================
# 4. OPTIMIZATION LAYER (PyPortfolioOpt)
# ==============================================================================
class OptimizationEngine:
    """Encapsulates the mathematical optimization logic."""

    @staticmethod
    def optimize(prices, strategy="Max Sharpe", risk_free_rate=0.0, constraints=(0.0, 1.0)):
        if not PYPFOPT_AVAILABLE:
            return None, None, "Library Missing"
        
        if prices.shape[1] < 2:
            return None, None, "Need at least 2 assets."

        try:
            # 1. Inputs: Ledoit-Wolf Covariance & Mean Historical Returns
            mu = expected_returns.mean_historical_return(prices, frequency=Config.TRADING_DAYS)
            S = risk_models.CovarianceShrinkage(prices, frequency=Config.TRADING_DAYS).ledoit_wolf()

            weights = {}
            perf = {}

            if strategy == "HRP (Hierarchical Risk Parity)":
                # HRP Logic (Clustering based)
                rets = prices.pct_change().dropna()
                hrp = HRPOpt(returns=rets, cov_matrix=S)
                weights = hrp.optimize()
                
                # Manual Perf Calc for HRP
                w_s = pd.Series(weights)
                ret = w_s.dot(mu)
                vol = np.sqrt(w_s.dot(S).dot(w_s))
                sharpe = (ret - risk_free_rate) / vol
                perf = (ret, vol, sharpe)
                
            else:
                # Mean-Variance Logic
                ef = efficient_frontier.EfficientFrontier(mu, S)
                
                # Apply Constraints
                ef.add_constraint(lambda w: w >= constraints[0])
                ef.add_constraint(lambda w: w <= constraints[1])
                
                if strategy == "Max Sharpe":
                    ef.max_sharpe(risk_free_rate=risk_free_rate)
                elif strategy == "Min Volatility":
                    ef.min_volatility()
                elif strategy == "Efficient Return (Target 40%)":
                    ef.efficient_return(target_return=0.40)
                
                weights = ef.clean_weights()
                perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

            return weights, perf, "Success"

        except Exception as e:
            return None, None, str(e)

# ==============================================================================
# 5. RISK ANALYTICS LAYER (VaR/CVaR)
# ==============================================================================
class RiskEngine:
    """
    Dedicated engine for calculating Value at Risk (VaR) and Conditional VaR (CVaR).
    Implements 3 Methodologies: Historical, Parametric, Monte Carlo.
    """
    
    @staticmethod
    def calculate_metrics(returns, confidence=0.95):
        """
        Computes VaR and CVaR using multiple methods.
        Returns a Pandas DataFrame suitable for display.
        """
        if returns.empty:
            return pd.DataFrame()
        
        mu = returns.mean()
        sigma = returns.std()
        
        data = []
        
        # --- Method 1: Historical Simulation ---
        # "What happened in the past?"
        var_hist = np.percentile(returns, (1 - confidence) * 100)
        cvar_hist = returns[returns <= var_hist].mean()
        data.append({
            "Methodology": "Historical Simulation",
            "Description": "Based on actual past distribution",
            "VaR (95%)": abs(var_hist),
            "CVaR (95%)": abs(cvar_hist)
        })
        
        # --- Method 2: Parametric (Variance-Covariance) ---
        # "Assuming Normal Distribution"
        var_param = norm.ppf(1 - confidence, mu, sigma)
        # Analytical CVaR for Normal Dist
        alpha = 1 - confidence
        z = norm.ppf(alpha)
        pdf = norm.pdf(z)
        cvar_param = mu - sigma * (pdf / alpha)
        data.append({
            "Methodology": "Parametric (Gaussian)",
            "Description": "Assumes Normal Distribution",
            "VaR (95%)": abs(var_param),
            "CVaR (95%)": abs(cvar_param)
        })
        
        # --- Method 3: Monte Carlo Simulation ---
        # "Simulating 10,000 alternative futures"
        sim_rets = np.random.normal(mu, sigma, Config.MC_SIMS_VAR)
        var_mc = np.percentile(sim_rets, (1 - confidence) * 100)
        cvar_mc = sim_rets[sim_rets <= var_mc].mean()
        data.append({
            "Methodology": "Monte Carlo",
            "Description": f"Simulated {Config.MC_SIMS_VAR} scenarios",
            "VaR (95%)": abs(var_mc),
            "CVaR (95%)": abs(cvar_mc)
        })
        
        return pd.DataFrame(data)

# ==============================================================================
# 6. STRESS TESTING LAYER
# ==============================================================================
class StressTestEngine:
    """Backtests portfolio against specific historical crisis periods."""
    
    EVENTS = {
        "2023 TR Elections Volatility": ("2023-04-01", "2023-06-15"),
        "2022 Russia-Ukraine War": ("2022-02-20", "2022-04-30"),
        "2021 TR Currency Crisis (Dec)": ("2021-11-01", "2021-12-31"),
        "2020 COVID-19 Market Crash": ("2020-02-15", "2020-03-31"),
        "2018 TR Brunson Crisis": ("2018-07-01", "2018-09-01"),
        "2008 Global Financial Crisis": ("2008-09-01", "2009-03-01")
    }
    
    @staticmethod
    def run_stress_test(weights, tickers):
        # Determine data requirements
        dates = [pd.to_datetime(d[0]) for d in StressTestEngine.EVENTS.values()]
        start_fetch = min(dates) - timedelta(days=30)
        
        full_data = DataEngine.fetch_data(tickers, start_fetch, datetime.now())
        if full_data.empty: return pd.DataFrame()
        
        full_rets = full_data.pct_change().dropna()
        results = []
        
        for event, (s_date, e_date) in StressTestEngine.EVENTS.items():
            try:
                # Slice period
                period = full_rets.loc[s_date:e_date]
                if period.empty: continue
                
                # Align assets (some might not exist in 2008)
                valid_assets = [t for t in tickers if t in period.columns]
                if not valid_assets: continue
                
                # Normalize weights
                w_vec = np.array([weights.get(t, 0) for t in valid_assets])
                if w_vec.sum() > 0:
                    w_vec = w_vec / w_vec.sum()
                
                # Calc Return
                port_ret = period[valid_assets].dot(w_vec)
                total_ret = (1 + port_ret).prod() - 1
                max_dd = (1 + port_ret).cumprod().div((1 + port_ret).cumprod().cummax()).sub(1).min()
                vol = port_ret.std() * np.sqrt(252)
                
                results.append({
                    "Scenario": event,
                    "Total Return": total_ret,
                    "Max Drawdown": max_dd,
                    "Ann. Volatility": vol
                })
            except: continue
            
        return pd.DataFrame(results).sort_values("Max Drawdown")

# ==============================================================================
# 7. VISUALIZATION LAYER
# ==============================================================================
class VizEngine:
    """Plotting library using Plotly."""
    
    @staticmethod
    def plot_frontier(mu, S, opt_coords=None):
        n_samples = Config.MC_SIMS_FRONTIER
        w = np.random.dirichlet(np.ones(len(mu)), n_samples)
        rets = w.dot(mu)
        vols = np.sqrt(np.diag(w @ S @ w.T))
        sharpes = rets / vols
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vols, y=rets, mode='markers',
            marker=dict(color=sharpes, colorscale='Viridis', size=5, showscale=True, colorbar=dict(title="Sharpe")),
            name="Simulated Portfolios"
        ))
        
        if opt_coords:
            fig.add_trace(go.Scatter(
                x=[opt_coords[1]], y=[opt_coords[0]], mode='markers+text',
                marker=dict(color='red', size=15, symbol='star'),
                name="Optimal Strategy",
                text=["Selected"], textposition="top left"
            ))
            
        fig.update_layout(title="Markowitz Efficient Frontier Simulation", xaxis_title="Risk (Vol)", yaxis_title="Return", template="plotly_white")
        return fig

    @staticmethod
    def plot_cumulative(port, bench=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port.index, y=port, name="Portfolio", line=dict(color='#00CC96', width=3)))
        if bench is not None:
            fig.add_trace(go.Scatter(x=bench.index, y=bench, name="Benchmark (XU100)", line=dict(color='gray', dash='dot')))
        fig.update_layout(title="Cumulative Performance (Rebased to 1.0)", template="plotly_white")
        return fig

    @staticmethod
    def plot_drawdown(returns):
        cum = (1 + returns).cumprod()
        dd = cum.div(cum.cummax()).sub(1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', line=dict(color='red'), name="Drawdown"))
        fig.update_layout(title="Underwater Plot (Drawdown)", yaxis_tickformat='.2%', template="plotly_white")
        return fig

# ==============================================================================
# 8. MAIN UI (STREAMLIT)
# ==============================================================================
def main():
    st.set_page_config(page_title="QuantEdge Pro v10", layout="wide", page_icon="🏛️")
    
    if not PYPFOPT_AVAILABLE:
        st.error("ERROR: Missing Library. Run `pip install PyPortfolioOpt`.")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🎛️ Control Center")
        
        st.header("1. Asset Universe")
        uni_df = AssetUniverse.get_df()
        scenarios = AssetUniverse.get_scenarios()
        
        sel_scen = st.selectbox("Portfolio Scenario", list(scenarios.keys()))
        
        if sel_scen == "Full Universe (Manual Selection)":
            sectors = st.multiselect("Filter Sector", uni_df["sector"].unique())
            subset = uni_df[uni_df["sector"].isin(sectors)] if sectors else uni_df
            sel_tickers = st.multiselect("Select Assets", subset["ticker"].tolist(), default=["THYAO.IS", "AKBNK.IS", "TUPRS.IS"])
            tickers = sel_tickers
        else:
            tgt_df = scenarios[sel_scen](uni_df)
            tickers = tgt_df["ticker"].tolist()
            st.success(f"{len(tickers)} Assets Loaded")
            with st.expander("View Assets"):
                st.dataframe(tgt_df[["ticker", "name", "sector"]], hide_index=True)

        st.header("2. Optimization")
        method = st.selectbox("Objective", ["Max Sharpe", "Min Volatility", "HRP (Hierarchical Risk Parity)", "Efficient Return (Target 40%)"])
        
        # Auto RF Rate
        is_tr = sum([1 for t in tickers if ".IS" in t]) > len(tickers)/2
        def_rf = Config.RF_RATE_TRY if is_tr else Config.RF_RATE_USD
        rf = st.number_input("Risk Free Rate (%)", value=def_rf*100) / 100
        
        alloc_cap = st.slider("Max Weight per Asset", 0.05, 1.0, 0.20)
        
        st.header("3. History")
        yrs = st.slider("Lookback Years", 1, 10, 3)
        
        run = st.button("🚀 EXECUTE ANALYSIS", type="primary")

    # --- MAIN CONTENT ---
    st.title(f"🏛️ {Config.APP_NAME}")
    st.markdown("### Advanced Institutional Analytics Engine")

    if run and tickers:
        if len(tickers) < 2:
            st.error("Select at least 2 assets.")
            st.stop()

        with st.status("Running Quantitative Engine...", expanded=True):
            st.write("Fetching Market Data...")
            start_dt = datetime.now() - timedelta(days=yrs*365)
            prices = DataEngine.fetch_data(tickers, start_dt, datetime.now())
            
            if prices.empty:
                st.error("Data Fetch Failed.")
                st.stop()
            
            st.write("Optimizing Portfolio Weights...")
            weights, perf, status = OptimizationEngine.optimize(prices, method, rf, (0.0, alloc_cap))
            
            if not weights:
                st.error(f"Optimization Failed: {status}")
                st.stop()

            st.write("Calculating Risk Metrics (VaR/CVaR)...")
            # Filter noise
            w_s = pd.Series(weights).sort_values(ascending=False)
            w_s = w_s[w_s > 0.001]
            
            # Backtest
            returns = prices.pct_change().dropna()
            w_align = w_s.reindex(returns.columns).fillna(0)
            port_ret = returns.dot(w_align)
            
            # Benchmark
            bench_df = DataEngine.get_benchmark(start_dt, datetime.now())
            bench_ret = bench_df.pct_change().dropna().iloc[:, 0] if not bench_df.empty else None
            
            # Engines
            risk_df = RiskEngine.calculate_metrics(port_ret)
            stress_df = StressTestEngine.run_stress_test(weights, tickers)
            
            st.write("Done!")

        # --- OUTPUT ---
        t1, t2, t3, t4, t5 = st.tabs(["Summary", "Risk (VaR/CVaR)", "Stress Test", "Efficient Frontier", "Data"])

        with t1:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Performance Estimates")
                if method != "HRP (Hierarchical Risk Parity)":
                    st.metric("Exp. Annual Return", f"{perf[0]:.2%}")
                    st.metric("Exp. Volatility", f"{perf[1]:.2%}")
                    st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                else:
                    st.metric("Hist. Volatility", f"{port_ret.std()*np.sqrt(252):.2%}")
                
                st.markdown("#### Allocation")
                st.dataframe(w_s.to_frame("Weight").style.format("{:.2%}"), height=400)
            
            with c2:
                cum_port = (1 + port_ret).cumprod()
                cum_bench = (1 + bench_ret).cumprod() if bench_ret is not None else None
                st.plotly_chart(VizEngine.plot_cumulative(cum_port, cum_bench), use_container_width=True)
                st.plotly_chart(px.pie(values=w_s.values, names=w_s.index, title="Asset Allocation"), use_container_width=True)

        with t2:
            st.subheader("Advanced Risk Analytics")
            st.markdown("Detailed breakdown of Value at Risk (VaR) and Conditional VaR (Expected Shortfall).")
            
            # VaR Table
            st.dataframe(risk_df.style.format({
                "VaR (95%)": "{:.2%}", "CVaR (95%)": "{:.2%}"
            }), use_container_width=True)
            
            # VaR Chart
            fig_risk = px.bar(
                risk_df.melt(id_vars="Methodology", value_vars=["VaR (95%)", "CVaR (95%)"]), 
                x="Methodology", y="value", color="variable", barmode="group",
                title="VaR vs CVaR Comparison",
                color_discrete_map={"VaR (95%)": "#FFA726", "CVaR (95%)": "#EF5350"}
            )
            fig_risk.update_layout(yaxis_tickformat='.2%')
            st.plotly_chart(fig_risk, use_container_width=True)
            
            st.subheader("Drawdown Analysis")
            st.plotly_chart(VizEngine.plot_drawdown(port_ret), use_container_width=True)

        with t3:
            st.subheader("Historical Crisis Simulations")
            if not stress_df.empty:
                st.dataframe(stress_df.style.format({
                    "Total Return": "{:+.2%}", "Max Drawdown": "{:.2%}", "Ann. Volatility": "{:.2%}"
                }).background_gradient(subset=["Max Drawdown"], cmap="Reds_r", vmin=-0.5, vmax=0), use_container_width=True)
                
                fig_stress = px.bar(stress_df, x="Scenario", y="Max Drawdown", color="Max Drawdown", color_continuous_scale="Reds_r")
                st.plotly_chart(fig_stress, use_container_width=True)
            else:
                st.warning("Insufficient data for stress testing.")

        with t4:
            st.subheader("Efficient Frontier Cloud")
            mu = expected_returns.mean_historical_return(prices)
            S = risk_models.sample_cov(prices)
            coords = (perf[0], perf[1]) if perf else None
            st.plotly_chart(VizEngine.plot_frontier(mu, S, coords), use_container_width=True)

        with t5:
            st.dataframe(prices)
            csv = prices.to_csv()
            st.download_button("Download CSV", csv, "market_data.csv", "text/csv")

if __name__ == "__main__":
    main()

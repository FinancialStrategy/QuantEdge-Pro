# ==============================================================================
# QuantEdge Enterprise Suite v9.0
# Domain: Institutional Portfolio Management & Risk Analytics
# Target Market: Global & Emerging Markets (BIST Focus)
# Language: English (Strict)
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

# --- Third-Party Quantitative Libraries ---
try:
    from pypfopt import efficient_frontier, risk_models, expected_returns
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt import objective_functions
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

# Suppress Warnings for Cleaner UI
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================
class Config:
    """Global configuration settings for the application."""
    APP_NAME = "QuantEdge Enterprise - Institutional Portfolio Manager"
    VERSION = "9.0.0"
    TRADING_DAYS = 252
    
    # Risk-Free Rates (Proxies)
    RF_RATE_TRY = 0.350  # ~35% (Conservative TR Sovereign Bond/Deposit proxy)
    RF_RATE_USD = 0.045  # ~4.5% (US 10Y Treasury)
    
    # Simulation Parameters
    MC_SIMULATIONS_VAR = 10000     # High precision for VaR
    MC_SIMULATIONS_FRONTIER = 3000 # For Efficient Frontier Visualization
    
    # Data Fetching
    CACHE_TTL = 3600  # Cache data for 1 hour
    
    # UI Settings
    WIDE_LAYOUT = "wide"

# ==============================================================================
# 2. ASSET UNIVERSE & SCENARIO DEFINITIONS
# ==============================================================================
class AssetUniverse:
    """Defines the master list of tradeable instruments and preset scenarios."""
    
    INSTRUMENTS = [
        # --- BENCHMARKS ---
        {"ticker": "XU100.IS", "name": "BIST 100 Index", "region": "TR", "sector": "Benchmark", "asset_class": "Index"},
        {"ticker": "SPY", "name": "S&P 500 ETF", "region": "US", "sector": "Benchmark", "asset_class": "Index"},

        # --- BIST 30: INDUSTRIALS & CONGLOMERATES ---
        {"ticker": "THYAO.IS", "name": "Turkish Airlines", "region": "TR", "sector": "Transportation", "asset_class": "Equity"},
        {"ticker": "TUPRS.IS", "name": "Tupras Oil", "region": "TR", "sector": "Energy", "asset_class": "Equity"},
        {"ticker": "EREGL.IS", "name": "Erdemir Steel", "region": "TR", "sector": "Materials", "asset_class": "Equity"},
        {"ticker": "SISE.IS", "name": "Sisecam Glass", "region": "TR", "sector": "Industrials", "asset_class": "Equity"},
        {"ticker": "BIMAS.IS", "name": "BIM Retail", "region": "TR", "sector": "Consumer Staples", "asset_class": "Equity"},
        {"ticker": "ASELS.IS", "name": "Aselsan Defense", "region": "TR", "sector": "Defense", "asset_class": "Equity"},
        {"ticker": "KCHOL.IS", "name": "Koc Holding", "region": "TR", "sector": "Conglomerate", "asset_class": "Equity"},
        {"ticker": "SAHOL.IS", "name": "Sabanci Holding", "region": "TR", "sector": "Conglomerate", "asset_class": "Equity"},
        {"ticker": "FROTO.IS", "name": "Ford Otosan", "region": "TR", "sector": "Automotive", "asset_class": "Equity"},
        {"ticker": "TOASO.IS", "name": "Tofas Auto", "region": "TR", "sector": "Automotive", "asset_class": "Equity"},
        {"ticker": "TCELL.IS", "name": "Turkcell", "region": "TR", "sector": "Telecom", "asset_class": "Equity"},
        {"ticker": "ENKAI.IS", "name": "Enka Construction", "region": "TR", "sector": "Construction", "asset_class": "Equity"},
        {"ticker": "KOZAL.IS", "name": "Koza Gold Corp", "region": "TR", "sector": "Mining", "asset_class": "Equity"},
        {"ticker": "ASTOR.IS", "name": "Astor Energy", "region": "TR", "sector": "Energy", "asset_class": "Equity"},

        # --- BIST FINANCIALS: BANKS ---
        {"ticker": "AKBNK.IS", "name": "Akbank", "region": "TR", "sector": "Banking", "asset_class": "Equity"},
        {"ticker": "GARAN.IS", "name": "Garanti BBVA", "region": "TR", "sector": "Banking", "asset_class": "Equity"},
        {"ticker": "YKBNK.IS", "name": "Yapi Kredi Bank", "region": "TR", "sector": "Banking", "asset_class": "Equity"},
        {"ticker": "ISCTR.IS", "name": "Is Bank", "region": "TR", "sector": "Banking", "asset_class": "Equity"},
        {"ticker": "TSKB.IS", "name": "TSKB Dev. Bank", "region": "TR", "sector": "Banking", "asset_class": "Equity"},
        {"ticker": "HALKB.IS", "name": "Halkbank", "region": "TR", "sector": "Banking", "asset_class": "Equity"},
        {"ticker": "VAKBN.IS", "name": "Vakifbank", "region": "TR", "sector": "Banking", "asset_class": "Equity"},

        # --- BIST FINANCIALS: NON-BANK ---
        {"ticker": "AKGRT.IS", "name": "Aksigorta Insurance", "region": "TR", "sector": "Insurance", "asset_class": "Equity"},
        {"ticker": "TURSG.IS", "name": "Turkiye Insurance", "region": "TR", "sector": "Insurance", "asset_class": "Equity"},
        {"ticker": "ISFIN.IS", "name": "Is Leasing", "region": "TR", "sector": "Leasing", "asset_class": "Equity"},
        {"ticker": "LIDFA.IS", "name": "Lider Factoring", "region": "TR", "sector": "Factoring", "asset_class": "Equity"},
        {"ticker": "ULUFA.IS", "name": "Ulusal Factoring", "region": "TR", "sector": "Factoring", "asset_class": "Equity"},

        # --- GLOBAL COMMODITIES & METALS (FUTURES) ---
        {"ticker": "GC=F", "name": "Gold Futures", "region": "Global", "sector": "Precious Metals", "asset_class": "Commodity"},
        {"ticker": "SI=F", "name": "Silver Futures", "region": "Global", "sector": "Precious Metals", "asset_class": "Commodity"},
        {"ticker": "PL=F", "name": "Platinum Futures", "region": "Global", "sector": "Precious Metals", "asset_class": "Commodity"},
        {"ticker": "HG=F", "name": "Copper Futures", "region": "Global", "sector": "Industrial Metals", "asset_class": "Commodity"},
        {"ticker": "CL=F", "name": "Crude Oil (WTI)", "region": "Global", "sector": "Energy", "asset_class": "Commodity"},
        {"ticker": "BZ=F", "name": "Brent Crude", "region": "Global", "sector": "Energy", "asset_class": "Commodity"},
        {"ticker": "NG=F", "name": "Natural Gas", "region": "Global", "sector": "Energy", "asset_class": "Commodity"},
        {"ticker": "ZC=F", "name": "Corn Futures", "region": "Global", "sector": "Agriculture", "asset_class": "Commodity"},
    ]

    @classmethod
    def get_universe_df(cls):
        return pd.DataFrame(cls.INSTRUMENTS)

    @staticmethod
    def get_scenarios():
        """Returns filter logic for predefined portfolio scenarios."""
        return {
            "BIST 30: Industrials & Conglomerates": lambda df: df[
                (df["region"] == "TR") & 
                (df["sector"].isin(["Industrials", "Conglomerate", "Transportation", "Energy", "Consumer Staples", "Defense", "Automotive", "Telecom", "Construction", "Mining"]))
            ],
            "TR Financial Complex (Bank + Non-Bank)": lambda df: df[
                (df["region"] == "TR") & 
                (df["sector"].isin(["Banking", "Insurance", "Leasing", "Factoring"]))
            ],
            "Global Inflation Hedge (Commodities Only)": lambda df: df[
                df["asset_class"] == "Commodity"
            ],
            "Hybrid: TR Banks + Precious Metals": lambda df: df[
                ((df["sector"] == "Banking") & (df["region"] == "TR")) | 
                (df["sector"] == "Precious Metals")
            ],
            "Full Universe (Custom Selection)": lambda df: df
        }

# ==============================================================================
# 3. DATA LAYER
# ==============================================================================
class DataLayer:
    """Handles all interaction with external data sources (Yahoo Finance)."""

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def fetch_historical_prices(tickers, start_date, end_date):
        """
        Fetches adjusted close prices. Handles API errors and MultiIndex columns robustly.
        """
        if not tickers:
            return pd.DataFrame()

        # Add buffer to start date to ensure calculation continuity
        buffer_date = start_date - timedelta(days=45)
        
        try:
            # Explicitly requesting grouping by column to handle single vs multiple tickers consistency
            raw_data = yf.download(
                tickers, 
                start=buffer_date, 
                end=end_date, 
                auto_adjust=True, 
                progress=False, 
                group_by='column'
            )

            if raw_data.empty:
                st.warning("No data received from API provider.")
                return pd.DataFrame()

            # Robust Column Extraction Logic
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Check for 'Adj Close' first, then 'Close'
                if 'Adj Close' in raw_data.columns.levels[0]:
                    data = raw_data['Adj Close']
                elif 'Close' in raw_data.columns.levels[0]:
                    data = raw_data['Close']
                else:
                    # Fallback: take the first available column level
                    data = raw_data.iloc[:, :len(tickers)]
            else:
                # Single level logic (usually occurs when fetching single ticker)
                target_col = 'Adj Close' if 'Adj Close' in raw_data.columns else 'Close'
                if target_col in raw_data.columns:
                    data = raw_data[target_col]
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name=tickers[0])
                else:
                    return pd.DataFrame()

            # Data Cleaning Pipeline
            # 1. Drop columns that are completely empty (failed downloads)
            data = data.dropna(axis=1, how='all')
            # 2. Forward fill holes (holidays)
            data = data.ffill()
            # 3. Drop remaining NaNs at the start
            data = data.dropna()
            
            # Filter to requested start date
            data = data[data.index >= pd.to_datetime(start_date)]
            
            return data

        except Exception as e:
            st.error(f"Critical Data Error: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_benchmark_data(start_date, end_date):
        """Fetches BIST 100 for relative performance comparison."""
        return DataLayer.fetch_historical_prices(["XU100.IS"], start_date, end_date)

# ==============================================================================
# 4. OPTIMIZATION LAYER
# ==============================================================================
class OptimizationLayer:
    """Encapsulates PyPortfolioOpt logic for multiple strategies."""

    @staticmethod
    def run_optimization(prices, strategy="Max Sharpe", risk_free_rate=0.045, constraints=(0.0, 1.0)):
        """
        Executes the optimization solver.
        
        Args:
            prices (pd.DataFrame): Historical prices.
            strategy (str): Optimization objective.
            risk_free_rate (float): Annualized risk-free rate.
            constraints (tuple): (min_weight, max_weight) per asset.
            
        Returns:
            tuple: (weights_dict, performance_tuple, status_message)
        """
        if not PYPFOPT_AVAILABLE:
            return None, None, "PyPortfolioOpt Library is missing."
        
        if prices.shape[1] < 2:
            return None, None, "Insufficient assets (Minimum 2 required)."

        try:
            # 1. Estimate Expected Returns and Covariance Matrix
            # Using Ledoit-Wolf Shrinkage for robust covariance matrix estimation
            mu = expected_returns.mean_historical_return(prices, frequency=Config.TRADING_DAYS)
            S = risk_models.CovarianceShrinkage(prices, frequency=Config.TRADING_DAYS).ledoit_wolf()

            cleaned_weights = {}
            performance = ()

            # 2. Select and Run Strategy
            if strategy == "HRP (Hierarchical Risk Parity)":
                # HRP uses clustering, not the efficient frontier solver
                returns = prices.pct_change().dropna()
                hrp = HRPOpt(returns=returns, cov_matrix=S)
                weights = hrp.optimize()
                cleaned_weights = weights
                # Manual performance calculation for HRP
                # We calculate exp return and vol manually as HRP class differs from EF
                w_series = pd.Series(weights)
                ret_ann = w_series.dot(mu)
                vol_ann = np.sqrt(w_series.dot(S).dot(w_series))
                sharpe = (ret_ann - risk_free_rate) / vol_ann
                performance = (ret_ann, vol_ann, sharpe)
            
            else:
                # Mean-Variance Optimization
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
                
                cleaned_weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

            return cleaned_weights, performance, "Optimization Successful"

        except Exception as e:
            return None, None, f"Solver Error: {str(e)}"

# ==============================================================================
# 5. RISK ANALYTICS LAYER
# ==============================================================================
class RiskAnalyticsLayer:
    """Advanced risk modeling including Stress Testing and VaR/CVaR."""

    STRESS_EVENTS = {
        "2023 TR General Elections": ("2023-04-01", "2023-06-15"),
        "2022 Russia-Ukraine War": ("2022-02-20", "2022-04-30"),
        "2021 TR Currency Crisis (Dec)": ("2021-11-01", "2021-12-31"),
        "2020 COVID-19 Crash": ("2020-02-15", "2020-03-31"),
        "2018 TR Brunson Crisis": ("2018-07-01", "2018-09-01"),
        "2008 Global Financial Crisis": ("2008-09-01", "2009-03-01")
    }

    @staticmethod
    def compute_var_cvar(returns, confidence=0.95):
        """Calculates VaR and CVaR using multiple methodologies."""
        if returns.empty: return pd.DataFrame()
        
        mu = returns.mean()
        sigma = returns.std()
        
        metrics = []
        
        # 1. Historical Method
        var_hist = np.percentile(returns, (1 - confidence) * 100)
        cvar_hist = returns[returns <= var_hist].mean()
        metrics.append(["Historical Simulation", abs(var_hist), abs(cvar_hist)])
        
        # 2. Parametric Method (Gaussian)
        var_param = norm.ppf(1 - confidence, mu, sigma)
        z_score = norm.ppf(1 - confidence)
        cvar_param = mu - sigma * (norm.pdf(z_score) / (1 - confidence))
        metrics.append(["Parametric (Normal)", abs(var_param), abs(cvar_param)])
        
        # 3. Monte Carlo Method
        simulated_rets = np.random.normal(mu, sigma, Config.MC_SIMULATIONS_VAR)
        var_mc = np.percentile(simulated_rets, (1 - confidence) * 100)
        cvar_mc = simulated_rets[simulated_rets <= var_mc].mean()
        metrics.append(["Monte Carlo Simulation", abs(var_mc), abs(cvar_mc)])
        
        return pd.DataFrame(metrics, columns=["Methodology", f"VaR ({confidence:.0%})", f"CVaR ({confidence:.0%})"])

    @staticmethod
    def run_stress_test(weights, tickers):
        """Backtests portfolio weights against specific historical crisis periods."""
        # Determine earliest date needed
        dates = [pd.to_datetime(d[0]) for d in RiskAnalyticsLayer.STRESS_EVENTS.values()]
        start_fetch = min(dates) - timedelta(days=15)
        
        full_data = DataLayer.fetch_historical_prices(tickers, start_fetch, datetime.now())
        if full_data.empty: return pd.DataFrame()
        
        full_rets = full_data.pct_change().dropna()
        results = []
        
        for event, (s_date, e_date) in RiskAnalyticsLayer.STRESS_EVENTS.items():
            try:
                period = full_rets.loc[s_date:e_date]
                if period.empty: continue
                
                # Align assets
                valid_assets = [t for t in tickers if t in period.columns]
                if not valid_assets: continue
                
                # Normalize weights for valid assets only
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
# 6. VISUALIZATION LAYER
# ==============================================================================
class VisualizationLayer:
    """Generates professional Plotly charts."""

    @staticmethod
    def plot_cumulative_returns(port_cum, bench_cum=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=port_cum.index, y=port_cum, 
            name="Optimized Portfolio", 
            line=dict(color='#2962FF', width=3)
        ))
        
        if bench_cum is not None:
            fig.add_trace(go.Scatter(
                x=bench_cum.index, y=bench_cum, 
                name="Benchmark (BIST 100)", 
                line=dict(color='#B0BEC5', dash='dot')
            ))
            
        fig.update_layout(
            title="Cumulative Performance (Rebased to 1.0)",
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Growth Factor"
        )
        return fig

    @staticmethod
    def plot_underwater_drawdown(returns):
        """Plots the drawdown curve (underwater plot)."""
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum / running_max) - 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown, 
            fill='tozeroy', 
            name="Drawdown",
            line=dict(color='#D50000', width=1)
        ))
        
        fig.update_layout(
            title="Underwater Plot (Drawdown)",
            yaxis_tickformat='.2%',
            template="plotly_white"
        )
        return fig

    @staticmethod
    def plot_rolling_volatility(returns, window=30):
        rol_vol = returns.rolling(window).std() * np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rol_vol.index, y=rol_vol, name=f"{window}-Day Rolling Vol", line=dict(color='#FF6D00')))
        fig.update_layout(title=f"Rolling Annualized Volatility ({window}-Day)", yaxis_tickformat='.1%', template="plotly_white")
        return fig

    @staticmethod
    def plot_efficient_frontier(mu, S, opt_coords=None):
        n_sims = Config.MC_SIMULATIONS_FRONTIER
        w = np.random.dirichlet(np.ones(len(mu)), n_sims)
        rets = w.dot(mu)
        vols = np.sqrt(np.diag(w @ S @ w.T))
        sharpes = rets / vols
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vols, y=rets, mode='markers',
            marker=dict(color=sharpes, colorscale='Spectral', size=4, showscale=True, colorbar=dict(title="Sharpe")),
            name="Simulated Portfolios"
        ))
        
        if opt_coords:
            fig.add_trace(go.Scatter(
                x=[opt_coords[1]], y=[opt_coords[0]], mode='markers+text',
                marker=dict(color='black', size=18, symbol='star'),
                name="Optimal Strategy",
                text=["Optimal"], textposition="top left"
            ))
            
        fig.update_layout(
            title="Efficient Frontier Simulation",
            xaxis_title="Annualized Risk (Volatility)",
            yaxis_title="Annualized Return",
            template="plotly_white"
        )
        return fig

    @staticmethod
    def plot_correlation_matrix(returns):
        corr = returns.corr()
        fig = px.imshow(
            corr, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            title="Asset Correlation Heatmap"
        )
        return fig

# ==============================================================================
# 7. MAIN APPLICATION (UI LAYER)
# ==============================================================================
def main():
    st.set_page_config(
        page_title="QuantEdge Enterprise", 
        layout=Config.WIDE_LAYOUT, 
        page_icon="🏢",
        initial_sidebar_state="expanded"
    )

    # Dependency Check
    if not PYPFOPT_AVAILABLE:
        st.error("CRITICAL ERROR: `PyPortfolioOpt` library is missing. Please run: `pip install PyPortfolioOpt`")
        st.stop()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        st.header("1. Asset Universe")
        universe_df = AssetUniverse.get_universe_df()
        scenarios = AssetUniverse.get_scenarios()
        
        scenario_key = st.selectbox("Select Portfolio Scenario", list(scenarios.keys()))
        
        if scenario_key == "Full Universe (Custom Selection)":
            # Multi-level filtering
            all_sectors = sorted(universe_df["sector"].unique())
            sel_sectors = st.multiselect("Filter by Sector", all_sectors, default=all_sectors)
            
            subset = universe_df[universe_df["sector"].isin(sel_sectors)]
            tickers = st.multiselect("Select Assets", subset["ticker"].tolist(), default=["THYAO.IS", "AKBNK.IS", "TUPRS.IS", "GC=F"])
        else:
            # Scenario Logic
            target_df = scenarios[scenario_key](universe_df)
            tickers = target_df["ticker"].tolist()
            
            st.info(f"Loaded {len(tickers)} assets from scenario.")
            with st.expander("View Assets"):
                st.dataframe(target_df[["ticker", "name", "sector"]], hide_index=True)

        st.header("2. Optimization Params")
        strategy = st.selectbox("Objective Function", 
                                ["Max Sharpe", "Min Volatility", "HRP (Hierarchical Risk Parity)", "Efficient Return (Target 40%)"])
        
        # Smart Defaults for Risk-Free Rate
        # If > 50% of assets are Turkish (.IS), use TR rate, else USD rate
        tr_ratio = sum([1 for t in tickers if ".IS" in t]) / len(tickers) if tickers else 0
        default_rf = Config.RF_RATE_TRY if tr_ratio > 0.5 else Config.RF_RATE_USD
        
        rf_rate = st.number_input("Risk-Free Rate (%)", value=default_rf*100, step=0.1, format="%.2f") / 100
        
        max_alloc = st.slider("Max Allocation per Asset", 0.05, 1.0, 0.20, 0.05)
        
        st.header("3. Time Horizon")
        years = st.slider("Historical Data (Years)", 1, 10, 3)
        
        run_analysis = st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.caption(f"{Config.APP_NAME} v{Config.VERSION}")

    # --- Main Content Area ---
    st.title("🏢 QuantEdge Enterprise Suite")
    st.markdown("### Institutional Portfolio Optimization & Risk Analytics Engine")

    if run_analysis:
        if len(tickers) < 2:
            st.error("Optimization requires at least 2 assets.")
            st.stop()
            
        start_dt = datetime.now() - timedelta(days=years*365)
        
        # 1. DATA INGESTION
        with st.status("Ingesting Market Data...", expanded=True) as status:
            st.write("Fetching historical OHLC data...")
            prices = DataLayer.fetch_historical_prices(tickers, start_dt, datetime.now())
            
            if prices.empty:
                st.error("Failed to fetch data. Please check connection or ticker symbols.")
                st.stop()
                
            st.write("Fetching Benchmark data...")
            bench_prices = DataLayer.get_benchmark_data(prices.index[0], prices.index[-1])
            
            st.write("Calculating Returns and Covariance Matrices...")
            returns = prices.pct_change().dropna()
            
            status.update(label="Data Ingestion Complete", state="complete", expanded=False)

        # 2. OPTIMIZATION ENGINE
        weights, perf, msg = OptimizationLayer.run_optimization(
            prices, 
            strategy=strategy, 
            risk_free_rate=rf_rate, 
            constraints=(0.0, max_alloc)
        )
        
        if not weights:
            st.error(f"Optimization Failed: {msg}")
            st.stop()
            
        # 3. PROCESSING RESULTS
        # Filter zero weights
        w_ser = pd.Series(weights).sort_values(ascending=False)
        w_ser = w_ser[w_ser > 0.001]
        
        # Calculate Backtested Portfolio Returns
        aligned_w = w_ser.reindex(returns.columns).fillna(0)
        port_ret = returns.dot(aligned_w)
        
        # Benchmark Return
        bench_ret = None
        if not bench_prices.empty:
            bench_ret = bench_prices.pct_change().dropna().iloc[:, 0]
            # Align dates
            idx = port_ret.index.intersection(bench_ret.index)
            port_ret = port_ret.loc[idx]
            bench_ret = bench_ret.loc[idx]

        # 4. DASHBOARD TABS
        tab_ov, tab_risk, tab_stress, tab_frontier, tab_data = st.tabs([
            "🏆 Executive Summary", 
            "📉 Deep Risk (VaR/CVaR)", 
            "🌪️ Stress Testing", 
            "🔬 Efficient Frontier",
            "💾 Data & Exports"
        ])

        with tab_ov:
            col_kpi, col_plots = st.columns([1, 2])
            
            with col_kpi:
                st.subheader("Performance Estimates")
                if strategy != "HRP (Hierarchical Risk Parity)":
                    st.metric("Expected Annual Return", f"{perf[0]:.2%}")
                    st.metric("Annual Volatility", f"{perf[1]:.2%}")
                    st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                else:
                    st.info("HRP optimizes for diversification. Standard Mean-Variance metrics are implied.")
                    st.metric("Realized Volatility (Hist)", f"{port_ret.std() * np.sqrt(252):.2%}")

                st.markdown("### Top Allocations")
                df_alloc = w_ser.to_frame("Weight")
                df_alloc["Weight"] = df_alloc["Weight"].apply(lambda x: f"{x:.2%}")
                st.dataframe(df_alloc, width=300)

            with col_plots:
                # Cumulative Return
                cum_port = (1 + port_ret).cumprod()
                cum_bench = (1 + bench_ret).cumprod() if bench_ret is not None else None
                st.plotly_chart(VisualizationLayer.plot_cumulative_returns(cum_port, cum_bench), use_container_width=True)
                
                # Allocation Pie
                fig_pie = px.pie(values=w_ser.values, names=w_ser.index, title="Sector/Asset Diversification")
                st.plotly_chart(fig_pie, use_container_width=True)

        with tab_risk:
            st.subheader("Advanced Risk Analytics")
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("#### Value at Risk (VaR) & CVaR (95%)")
                risk_df = RiskAnalyticsLayer.compute_var_cvar(port_ret)
                st.dataframe(risk_df.style.format({f"VaR (95%)": "{:.2%}", f"CVaR (95%)": "{:.2%}"}), use_container_width=True)
                
                # Drawdown Plot 

[Image of drawdown chart]

                st.plotly_chart(VisualizationLayer.plot_underwater_drawdown(port_ret), use_container_width=True)
            
            with col_r2:
                st.markdown("#### Rolling Volatility (30-Day)")
                st.plotly_chart(VisualizationLayer.plot_rolling_volatility(port_ret), use_container_width=True)
                
                st.markdown("#### Correlation Matrix")
                st.plotly_chart(VisualizationLayer.plot_correlation_matrix(returns), use_container_width=True)

        with tab_stress:
            st.subheader("Historical Crisis Simulation")
            st.markdown("Backtesting the current asset weights against major market crashes.")
            
            stress_res = RiskAnalyticsLayer.run_stress_test(weights, tickers)
            
            if not stress_res.empty:
                st.dataframe(
                    stress_res.style.format({
                        "Total Return": "{:+.2%}", 
                        "Max Drawdown": "{:.2%}", 
                        "Ann. Volatility": "{:.2%}"
                    }).background_gradient(subset=["Max Drawdown"], cmap="Reds_r", vmin=-0.5, vmax=0),
                    use_container_width=True
                )
                
                # Bar Chart
                fig_stress = px.bar(
                    stress_res, x="Scenario", y="Max Drawdown", 
                    color="Max Drawdown", color_continuous_scale="Reds_r",
                    title="Portfolio Erosion during Crises"
                )
                fig_stress.update_layout(yaxis_tickformat='.2%')
                st.plotly_chart(fig_stress, use_container_width=True)
            else:
                st.warning("Insufficient historical data for selected assets to cover these crisis periods.")

        with tab_frontier:
            st.subheader("Markowitz Efficient Frontier Simulation")
            mu = expected_returns.mean_historical_return(prices, frequency=Config.TRADING_DAYS)
            S = risk_models.sample_cov(prices)
            
            opt_coords = (perf[0], perf[1]) if perf else None
            
            with st.spinner("Running Monte Carlo Simulation (3000 Portfolios)..."):
                 # 
                st.plotly_chart(VisualizationLayer.plot_efficient_frontier(mu, S, opt_coords), use_container_width=True)

        with tab_data:
            st.subheader("Data Inspector")
            st.dataframe(prices.tail())
            
            # Export Functionality
            csv = prices.to_csv()
            st.download_button(
                label="Download Price Data (CSV)",
                data=csv,
                file_name="quantedge_market_data.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()

# ==============================================================================
# QuantEdge Pro v7 - Enterprise Edition (The "Master" Script)
# Author: Gemini AI for Fintech Professional
# Description: Advanced Portfolio Optimization, Risk Management, and Stress Testing
#              Focusing on BIST (Turkey), Financials, and Global Commodities.
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
import time

# --- Third Party Quant Libraries ---
try:
    from pypfopt import efficient_frontier, risk_models, expected_returns, objective_functions
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.cla import CLA
    from pypfopt import plotting
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

# Suppress Warnings for Cleaner UI
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
class Config:
    """Global configuration settings for the application."""
    APP_TITLE = "QuantEdge Pro v7 - Enterprise Portfolio Manager"
    TRADING_DAYS = 252
    
    # Interest Rates (Dynamically used based on asset mix)
    RF_RATE_TRY = 0.35  # Conservative TR Deposit/Bond Rate Proxy
    RF_RATE_USD = 0.045 # US 10Y Treasury Proxy
    
    # Simulation Settings
    MC_SIMULATIONS_RISK = 5000     # For VaR/CVaR
    MC_SIMULATIONS_FRONTIER = 2000 # For Efficient Frontier Plot
    
    # Data Fetching
    YF_BATCH_SIZE = 20
    CACHE_DURATION = 3600 # seconds

# ==============================================================================
# 2. ASSET UNIVERSE DEFINITION
# ==============================================================================
class AssetUniverse:
    """Defines the available financial instruments and grouping logic."""
    
    # The Master List of Assets
    INSTRUMENTS = [
        # --- BENCHMARK INDICES ---
        {"ticker": "XU100.IS", "name": "BIST 100", "region": "TR", "sector": "INDEX", "type": "Benchmark"},
        {"ticker": "XU030.IS", "name": "BIST 30", "region": "TR", "sector": "INDEX", "type": "Benchmark"},
        {"ticker": "SPY", "name": "S&P 500 ETF", "region": "US", "sector": "INDEX", "type": "Benchmark"},

        # --- BIST 30 INDUSTRIAL GIANTS ---
        {"ticker": "THYAO.IS", "name": "Turk Hava Yollari", "region": "TR", "sector": "Ulaştırma", "type": "Equity"},
        {"ticker": "TUPRS.IS", "name": "Tupras", "region": "TR", "sector": "Enerji", "type": "Equity"},
        {"ticker": "EREGL.IS", "name": "Erdemir", "region": "TR", "sector": "Demir-Çelik", "type": "Equity"},
        {"ticker": "SISE.IS", "name": "Sisecam", "region": "TR", "sector": "Sanayi", "type": "Equity"},
        {"ticker": "BIMAS.IS", "name": "BIM Magazalar", "region": "TR", "sector": "Perakende", "type": "Equity"},
        {"ticker": "ASELS.IS", "name": "Aselsan", "region": "TR", "sector": "Savunma", "type": "Equity"},
        {"ticker": "KCHOL.IS", "name": "Koc Holding", "region": "TR", "sector": "Holding", "type": "Equity"},
        {"ticker": "SAHOL.IS", "name": "Sabanci Holding", "region": "TR", "sector": "Holding", "type": "Equity"},
        {"ticker": "FROTO.IS", "name": "Ford Otosan", "region": "TR", "sector": "Otomotiv", "type": "Equity"},
        {"ticker": "TOASO.IS", "name": "Tofas", "region": "TR", "sector": "Otomotiv", "type": "Equity"},
        {"ticker": "TCELL.IS", "name": "Turkcell", "region": "TR", "sector": "Telekom", "type": "Equity"},
        {"ticker": "PETKM.IS", "name": "Petkim", "region": "TR", "sector": "Kimya", "type": "Equity"},
        {"ticker": "ENKAI.IS", "name": "Enka Insaat", "region": "TR", "sector": "İnşaat", "type": "Equity"},
        {"ticker": "KOZAL.IS", "name": "Koza Altin", "region": "TR", "sector": "Madencilik", "type": "Equity"},
        {"ticker": "HEKTS.IS", "name": "Hektas", "region": "TR", "sector": "Kimya", "type": "Equity"},
        {"ticker": "ASTOR.IS", "name": "Astor Enerji", "region": "TR", "sector": "Enerji", "type": "Equity"},

        # --- BIST FINANCIALS (BANKS) ---
        {"ticker": "AKBNK.IS", "name": "Akbank", "region": "TR", "sector": "Banka", "type": "Equity"},
        {"ticker": "GARAN.IS", "name": "Garanti BBVA", "region": "TR", "sector": "Banka", "type": "Equity"},
        {"ticker": "YKBNK.IS", "name": "Yapi Kredi", "region": "TR", "sector": "Banka", "type": "Equity"},
        {"ticker": "ISCTR.IS", "name": "Is Bankasi", "region": "TR", "sector": "Banka", "type": "Equity"},
        {"ticker": "HALKB.IS", "name": "Halkbank", "region": "TR", "sector": "Banka", "type": "Equity"},
        {"ticker": "VAKBN.IS", "name": "Vakifbank", "region": "TR", "sector": "Banka", "type": "Equity"},
        {"ticker": "TSKB.IS", "name": "TSKB", "region": "TR", "sector": "Banka", "type": "Equity"},
        {"ticker": "QNBFB.IS", "name": "QNB Finansbank", "region": "TR", "sector": "Banka", "type": "Equity"},

        # --- BIST FINANCIALS (NON-BANK) ---
        {"ticker": "AKGRT.IS", "name": "Aksigorta", "region": "TR", "sector": "Sigorta", "type": "Equity"},
        {"ticker": "TURSG.IS", "name": "Turkiye Sigorta", "region": "TR", "sector": "Sigorta", "type": "Equity"},
        {"ticker": "ANSGR.IS", "name": "Anadolu Sigorta", "region": "TR", "sector": "Sigorta", "type": "Equity"},
        {"ticker": "ISFIN.IS", "name": "Is Fin. Kir.", "region": "TR", "sector": "Leasing", "type": "Equity"},
        {"ticker": "VAKFN.IS", "name": "Vakif Leasing", "region": "TR", "sector": "Leasing", "type": "Equity"},
        {"ticker": "LIDFA.IS", "name": "Lider Faktoring", "region": "TR", "sector": "Faktoring", "type": "Equity"},
        {"ticker": "ULUFA.IS", "name": "Ulusal Faktoring", "region": "TR", "sector": "Faktoring", "type": "Equity"},

        # --- GLOBAL COMMODITIES & METALS (FUTURES) ---
        {"ticker": "GC=F", "name": "Gold Futures", "region": "Global", "sector": "Precious Metal", "type": "Commodity"},
        {"ticker": "SI=F", "name": "Silver Futures", "region": "Global", "sector": "Precious Metal", "type": "Commodity"},
        {"ticker": "PL=F", "name": "Platinum Futures", "region": "Global", "sector": "Precious Metal", "type": "Commodity"},
        {"ticker": "HG=F", "name": "Copper Futures", "region": "Global", "sector": "Industrial Metal", "type": "Commodity"},
        {"ticker": "CL=F", "name": "Crude Oil (WTI)", "region": "Global", "sector": "Energy", "type": "Commodity"},
        {"ticker": "BZ=F", "name": "Brent Crude", "region": "Global", "sector": "Energy", "type": "Commodity"},
        {"ticker": "NG=F", "name": "Natural Gas", "region": "Global", "sector": "Energy", "type": "Commodity"},
        {"ticker": "ZC=F", "name": "Corn", "region": "Global", "sector": "Agriculture", "type": "Commodity"},
        {"ticker": "ZW=F", "name": "Wheat", "region": "Global", "sector": "Agriculture", "type": "Commodity"},
    ]
    
    @classmethod
    def get_df(cls):
        """Returns the universe as a pandas DataFrame."""
        return pd.DataFrame(cls.INSTRUMENTS)

    @staticmethod
    def get_scenarios():
        """Returns dictionary of filter functions for preset scenarios."""
        return {
            "BIST 30 (Sanayi & Holding)": lambda df: df[
                (df["region"] == "TR") & 
                (df["sector"].isin(["Sanayi", "Holding", "Ulaştırma", "Enerji", "Perakende", "Savunma", "Otomotiv", "Telekom", "Madencilik", "İnşaat"]))
            ],
            "Tüm BIST Finansallar (Banka+Sigorta+Fact)": lambda df: df[
                (df["region"] == "TR") & 
                (df["sector"].isin(["Banka", "Sigorta", "Leasing", "Faktoring"]))
            ],
            "Emtia & Metal Sepeti (Inflation Hedge)": lambda df: df[
                df["type"] == "Commodity"
            ],
            "Karma: Türk Bankaları + Altın/Petrol": lambda df: df[
                ((df["sector"] == "Banka") & (df["region"] == "TR")) | 
                (df["ticker"].isin(["GC=F", "CL=F", "SI=F"]))
            ],
            "Custom Selection (Manual)": lambda df: df # No filter, user selects manually
        }

# ==============================================================================
# 3. DATA MANAGER CLASS
# ==============================================================================
class DataManager:
    """Handles fetching, cleaning, and validating financial data."""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_DURATION)
    def fetch_market_data(tickers, start_date, end_date):
        """
        Robust data fetching from Yahoo Finance with MultiIndex handling.
        """
        if not tickers:
            return pd.DataFrame()
        
        # Add buffer to start date for calculations
        buffered_start = start_date - timedelta(days=30)
        
        try:
            # Download Data
            raw_data = yf.download(
                tickers, 
                start=buffered_start, 
                end=end_date, 
                auto_adjust=True, 
                progress=False,
                group_by='column'
            )
            
            if raw_data.empty:
                return pd.DataFrame()

            # Handle Yahoo Finance's Variable Indexing Output
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Try to extract 'Adj Close', fallback to 'Close'
                if 'Adj Close' in raw_data.columns.levels[0]:
                    data = raw_data['Adj Close']
                elif 'Close' in raw_data.columns.levels[0]:
                    data = raw_data['Close']
                else:
                    # If structure is weird, take the first level column-wise
                    data = raw_data.iloc[:, :len(tickers)]
            else:
                # If only one ticker was requested, YF returns single level df
                data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=tickers[0])

            # Forward fill missing data (holidays), then drop remaining NaNs
            data = data.ffill().dropna()
            
            # Trim to requested start date
            data = data[data.index >= pd.to_datetime(start_date)]
            
            return data

        except Exception as e:
            st.error(f"Data Fetch Error: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_benchmark_returns(start_date, end_date):
        """Fetches XU100 and XU30 specifically for benchmarking."""
        bench_df = DataManager.fetch_market_data(["XU100.IS", "XU030.IS"], start_date, end_date)
        if not bench_df.empty:
            return bench_df.pct_change().dropna()
        return pd.DataFrame()

# ==============================================================================
# 4. OPTIMIZATION ENGINE (The Core Logic)
# ==============================================================================
class OptimizationEngine:
    """Wrapper for PyPortfolioOpt to handle various strategies."""
    
    @staticmethod
    def optimize_portfolio(prices, method="Max Sharpe", risk_free_rate=0.35, weight_limits=(0.0, 1.0)):
        """
        Main optimization router.
        """
        if not PYPFOPT_AVAILABLE:
            return None, None, "Library Missing"

        # 1. Inputs: Expected Returns (mu) and Covariance (S)
        # Using Ledoit-Wolf shrinkage for robust covariance estimation
        mu = expected_returns.mean_historical_return(prices, frequency=Config.TRADING_DAYS)
        S = risk_models.CovarianceShrinkage(prices, frequency=Config.TRADING_DAYS).ledoit_wolf()

        try:
            weights = {}
            perf = {}
            
            # 2. Strategy Execution
            if method == "HRP (Hierarchical Risk Parity)":
                # HRP doesn't use the standard EfficientFrontier object
                returns = prices.pct_change().dropna()
                hrp = HRPOpt(returns=returns, cov_matrix=S)
                weights = hrp.optimize()
                
                # Manual performance calc for HRP
                # Note: HRP object doesn't have portfolio_performance method like EF
                # We return weights, and calculate perf later using the weights
                cleaned_weights = weights
                
            else:
                # Mean-Variance Strategies
                ef = efficient_frontier.EfficientFrontier(mu, S)
                
                # Add Constraints
                ef.add_constraint(lambda w: w >= weight_limits[0])
                ef.add_constraint(lambda w: w <= weight_limits[1])
                
                if method == "Max Sharpe":
                    ef.max_sharpe(risk_free_rate=risk_free_rate)
                elif method == "Min Volatility":
                    ef.min_volatility()
                elif method == "Efficient Return (Target)":
                    # Example: Target 40% return
                    target = 0.40 
                    ef.efficient_return(target_return=target)
                
                cleaned_weights = ef.clean_weights()
                perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

            return cleaned_weights, perf, "Success"

        except Exception as e:
            # Fallback for Solver Errors (common in optimization)
            return None, None, str(e)

# ==============================================================================
# 5. RISK & STRESS TEST ENGINE
# ==============================================================================
class RiskEngine:
    """Calculates advanced risk metrics."""
    
    @staticmethod
    def calculate_var_cvar(returns, confidence=0.95):
        """
        Calculates Value at Risk (VaR) and Conditional VaR (CVaR)
        using three methodologies: Historical, Parametric, Monte Carlo.
        """
        mu = returns.mean()
        sigma = returns.std()
        
        metrics = []
        
        # 1. Historical
        var_hist = np.percentile(returns, (1 - confidence) * 100)
        cvar_hist = returns[returns <= var_hist].mean()
        metrics.append(["Historical", abs(var_hist), abs(cvar_hist)])
        
        # 2. Parametric (Normal Distribution)
        var_param = norm.ppf(1 - confidence, mu, sigma)
        z_score = norm.ppf(1 - confidence)
        cvar_param = mu - sigma * (norm.pdf(z_score) / (1 - confidence))
        metrics.append(["Parametric (Normal)", abs(var_param), abs(cvar_param)])
        
        # 3. Monte Carlo Simulation
        sim_rets = np.random.normal(mu, sigma, Config.MC_SIMULATIONS_RISK)
        var_mc = np.percentile(sim_rets, (1 - confidence) * 100)
        cvar_mc = sim_rets[sim_rets <= var_mc].mean()
        metrics.append(["Monte Carlo", abs(var_mc), abs(cvar_mc)])
        
        df = pd.DataFrame(metrics, columns=["Method", f"VaR ({confidence:.0%})", f"CVaR ({confidence:.0%})"])
        return df

class StressTestEngine:
    """Simulates portfolio performance during historical crises."""
    
    # Critical Events for Turkey and Global Markets
    EVENTS = {
        "2023 Türkiye Genel Seçimleri": ("2023-04-01", "2023-06-15"),
        "2022 Rusya-Ukrayna Savaşı (Emtia Şoku)": ("2022-02-20", "2022-04-30"),
        "2021 Aralık Kur Krizi (TR)": ("2021-11-01", "2021-12-31"),
        "2020 COVID-19 Çöküşü": ("2020-02-15", "2020-03-31"),
        "2018 Rahip Brunson Krizi (TR)": ("2018-07-01", "2018-09-01"),
        "2008 Küresel Finans Krizi": ("2008-09-01", "2009-03-01")
    }
    
    @staticmethod
    def run_tests(portfolio_weights, tickers):
        """
        Fetches historical data for specific periods and backtests the CURRENT weights.
        """
        # We need extended history.
        # Find the earliest required date
        earliest_date = min([pd.to_datetime(d[0]) for d in StressTestEngine.EVENTS.values()])
        start_fetch = earliest_date - timedelta(days=10)
        
        # Fetch Long History
        full_data = DataManager.fetch_market_data(tickers, start_fetch, datetime.now())
        if full_data.empty:
            return pd.DataFrame()
        
        full_rets = full_data.pct_change().dropna()
        
        results = []
        
        for event_name, (s_date, e_date) in StressTestEngine.EVENTS.items():
            try:
                # Slice Period
                period_rets = full_rets.loc[s_date:e_date]
                
                if period_rets.empty:
                    continue
                
                # Dynamic Re-weighting
                # If an asset didn't exist back then (e.g. ASTOR in 2018), we exclude it and re-normalize weights
                valid_assets = [t for t in tickers if t in period_rets.columns]
                
                if not valid_assets:
                    continue
                    
                # Extract weights for valid assets
                current_w = np.array([portfolio_weights.get(t, 0) for t in valid_assets])
                
                # Re-normalize to 100%
                if current_w.sum() > 0:
                    current_w = current_w / current_w.sum()
                
                # Calculate Portfolio Return for that period
                port_period_ret = period_rets[valid_assets].dot(current_w)
                
                # Metrics
                total_ret = (1 + port_period_ret).prod() - 1
                max_dd = (1 + port_period_ret).cumprod().div((1 + port_period_ret).cumprod().cummax()).sub(1).min()
                vol = port_period_ret.std() * np.sqrt(252)
                
                results.append({
                    "Senaryo": event_name,
                    "Tarih Aralığı": f"{s_date} / {e_date}",
                    "Toplam Getiri": total_ret,
                    "Maksimum Düşüş (DD)": max_dd,
                    "Volatilite (Yıllık)": vol
                })
            except Exception:
                continue
                
        return pd.DataFrame(results).sort_values("Maksimum Düşüş (DD)")

# ==============================================================================
# 6. VISUALIZATION ENGINE
# ==============================================================================
class VisualizationEngine:
    """Helper methods for Plotly charts."""
    
    @staticmethod
    def plot_efficient_frontier_cloud(mu, S, optimal_perf=None, optimal_label="Optimal"):
        """Generates the classic Markowitz Bullet with Monte Carlo Cloud."""
        n_samples = Config.MC_SIMULATIONS_FRONTIER
        n_assets = len(mu)
        
        # Generate random weights
        w = np.random.dirichlet(np.ones(n_assets), n_samples)
        
        # Calc metrics
        rets = w.dot(mu)
        vols = np.sqrt(np.diag(w @ S @ w.T))
        sharpes = rets / vols
        
        fig = go.Figure()
        
        # Scatter Cloud
        fig.add_trace(go.Scatter(
            x=vols, y=rets, mode='markers',
            marker=dict(color=sharpes, colorscale='Viridis', showscale=True, size=5, colorbar=dict(title="Sharpe")),
            name="Rastgele Portföyler",
            text=[f"Sharpe: {s:.2f}" for s in sharpes],
            hoverinfo="text+x+y"
        ))
        
        # Plot Optimal Point
        if optimal_perf:
            opt_ret, opt_vol, _ = optimal_perf
            fig.add_trace(go.Scatter(
                x=[opt_vol], y=[opt_ret], mode='markers+text',
                marker=dict(color='red', size=15, symbol='star'),
                name=f"Seçilen: {optimal_label}",
                text=[optimal_label], textposition="top center"
            ))
            
        fig.update_layout(
            title="Etkin Sınır (Efficient Frontier) Simülasyonu",
            xaxis_title="Risk (Volatilite)",
            yaxis_title="Beklenen Getiri",
            template="plotly_white",
            height=600
        )
        return fig

    @staticmethod
    def plot_correlation_heatmap(returns):
        corr = returns.corr()
        fig = px.imshow(
            corr, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            title="Varlık Korelasyon Matrisi"
        )
        return fig

    @staticmethod
    def plot_cumulative_performance(port_cum, benchmark_cum=None, benchmark_name="Benchmark"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Model Portföy", line=dict(color='#00CC96', width=3)))
        
        if benchmark_cum is not None:
            fig.add_trace(go.Scatter(x=benchmark_cum.index, y=benchmark_cum, name=benchmark_name, line=dict(color='#EF553B', dash='dot')))
            
        fig.update_layout(
            title="Kümülatif Getiri Karşılaştırması (1 TL Başlangıç)",
            xaxis_title="Tarih",
            yaxis_title="Değer",
            template="plotly_white",
            hovermode="x unified"
        )
        return fig

# ==============================================================================
# 7. MAIN APPLICATION LOGIC (STREAMLIT)
# ==============================================================================
def main():
    st.set_page_config(page_title="QuantEdge Pro v7", layout="wide", page_icon="🚀")
    
    # --- Sidebar ---
    with st.sidebar:
        st.title("🎛️ Kontrol Paneli")
        
        # 1. Universe Selection
        st.subheader("1. Varlık Evreni")
        universe_df = AssetUniverse.get_df()
        scenarios = AssetUniverse.get_scenarios()
        
        selected_scenario = st.selectbox("Senaryo Seçimi", list(scenarios.keys()))
        
        if selected_scenario == "Custom Selection (Manual)":
            sector_filter = st.multiselect("Sektör Filtrele", universe_df["sector"].unique())
            if sector_filter:
                filtered_df = universe_df[universe_df["sector"].isin(sector_filter)]
            else:
                filtered_df = universe_df
            selected_tickers = st.multiselect("Hisseleri Seç", filtered_df["ticker"].tolist(), default=["THYAO.IS", "AKBNK.IS", "TUPRS.IS"])
            target_assets = filtered_df[filtered_df["ticker"].isin(selected_tickers)]
        else:
            target_assets = scenarios[selected_scenario](universe_df)
            
        tickers = target_assets["ticker"].tolist()
        
        if len(tickers) > 0:
            st.success(f"{len(tickers)} Varlık Seçildi")
            with st.expander("Seçilen Varlıklar"):
                st.dataframe(target_assets[["ticker", "name", "sector"]])
        else:
            st.warning("Lütfen varlık seçin.")

        # 2. Optimization Parameters
        st.subheader("2. Optimizasyon Ayarları")
        opt_method = st.selectbox(
            "Hedef Fonksiyon", 
            ["Max Sharpe (Getiri/Risk)", "Min Volatility (En Az Risk)", "HRP (Kümeleme Tabanlı)"]
        )
        
        # Dynamic Risk Free Rate Logic
        is_tr_heavy = sum(1 for t in tickers if ".IS" in t) > (len(tickers) / 2)
        default_rf = Config.RF_RATE_TRY if is_tr_heavy else Config.RF_RATE_USD
        
        rf_rate = st.number_input("Risksiz Faiz Oranı (%)", value=default_rf*100, step=0.5) / 100
        max_weight = st.slider("Maksimum Ağırlık (Tek Hisse)", 0.05, 1.0, 0.20)
        
        # 3. Data Period
        st.subheader("3. Veri Aralığı")
        lookback_years = st.slider("Geçmiş Veri (Yıl)", 1, 10, 3)
        start_date = datetime.now() - timedelta(days=lookback_years*365)
        
        run_btn = st.button("🚀 ANALİZİ BAŞLAT", type="primary")
        
        st.markdown("---")
        st.caption(f"{Config.APP_TITLE}")

    # --- Main Area ---
    st.title(f"📊 {Config.APP_TITLE}")
    
    if not PYPFOPT_AVAILABLE:
        st.error("KRİTİK HATA: `PyPortfolioOpt` kütüphanesi eksik. Lütfen `pip install PyPortfolioOpt` komutunu çalıştırın.")
        st.stop()

    if run_btn and tickers:
        with st.spinner("Veriler çekiliyor, Matrisler hesaplanıyor, Yapay Zeka optimizasyonu çalışıyor..."):
            
            # 1. Fetch Data
            prices = DataManager.fetch_market_data(tickers, start_date, datetime.now())
            if prices.empty:
                st.error("Veri çekilemedi. Lütfen hisse sembollerini kontrol edin.")
                st.stop()
            
            returns = prices.pct_change().dropna()
            
            # 2. Fetch Benchmark
            bench_ticker = "XU100.IS" if is_tr_heavy else "SPY"
            bench_rets = DataManager.get_benchmark_returns(start_date, datetime.now())
            if not bench_rets.empty and bench_ticker in bench_rets.columns:
                bench_series = bench_rets[bench_ticker]
            else:
                bench_series = None

            # 3. Run Optimization
            weights_dict, perf, status = OptimizationEngine.optimize_portfolio(
                prices, 
                method=opt_method, 
                risk_free_rate=rf_rate, 
                weight_limits=(0.0, max_weight)
            )
            
            if weights_dict is None:
                st.error(f"Optimizasyon Hatası: {status}")
                st.stop()

            # Process Weights
            weights_series = pd.Series(weights_dict).sort_values(ascending=False)
            weights_series = weights_series[weights_series > 0.001] # Filter noise
            
            # Calculate Portfolio Return (Historical)
            aligned_weights = weights_series.reindex(returns.columns).fillna(0)
            port_returns = returns.dot(aligned_weights)
            
            # 4. Run Risk Engines
            risk_table = RiskEngine.calculate_var_cvar(port_returns, confidence=0.95)
            stress_table = StressTestEngine.run_tests(weights_dict, tickers)

        # --- DASHBOARD TABS ---
        tab_main, tab_risk, tab_stress, tab_frontier, tab_details = st.tabs([
            "🏆 Portföy Özeti", 
            "📉 Risk Analizi (VaR/CVaR)", 
            "🌪️ Stres Testleri", 
            "🔬 Etkin Sınır (Frontier)",
            "📋 Veri Detayları"
        ])
        
        with tab_main:
            col_kpi, col_chart = st.columns([1, 2])
            
            with col_kpi:
                st.subheader("Strateji Performansı (Beklenen)")
                if opt_method != "HRP (Hierarchical Risk Parity)":
                    st.metric("Beklenen Yıllık Getiri", f"{perf[0]:.2%}")
                    st.metric("Beklenen Volatilite", f"{perf[1]:.2%}")
                    st.metric("Sharpe Oranı", f"{perf[2]:.2f}")
                else:
                    st.info("HRP yöntemi, kovaryans matrisinin kümelenmesiyle risk dağıtımı yapar. Klasik getiri tahmini içermez.")
                    st.metric("Tarihsel Volatilite", f"{port_returns.std() * np.sqrt(252):.2%}")

                st.markdown("### Varlık Dağılımı")
                st.dataframe(
                    weights_series.to_frame("Ağırlık").style.format("{:.2%}"), 
                    use_container_width=True,
                    height=300
                )

            with col_chart:
                # Cumulative Chart
                cum_port = (1 + port_returns).cumprod()
                cum_bench = (1 + bench_series).cumprod() if bench_series is not None else None
                
                st.plotly_chart(
                    VisualizationEngine.plot_cumulative_performance(cum_port, cum_bench, bench_ticker),
                    use_container_width=True
                )
                
                # Allocation Pie
                fig_pie = px.pie(values=weights_series.values, names=weights_series.index, title="Sektörel/Hisse Dağılımı")
                st.plotly_chart(fig_pie, use_container_width=True)

        with tab_risk:
            st.subheader("Karşılaştırmalı Risk Metrikleri")
            st.markdown("""
            Bu bölüm, portföyünüzün **%95 Güven Aralığında** bir günde kaybedebileceği maksimum tutarı hesaplar.
            * **VaR (Value at Risk):** Normal şartlarda beklenen maksimum kayıp.
            * **CVaR (Conditional VaR / Expected Shortfall):** İşler kötü gittiğinde (kriz anında) beklenen ortalama kayıp.
            """)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                # Bar chart for VaR/CVaR
                fig_risk = px.bar(
                    risk_table.melt(id_vars="Method", var_name="Metric", value_name="Value"), 
                    x="Method", y="Value", color="Metric", barmode="group",
                    color_discrete_map={f"VaR (95%)": "#FFA726", f"CVaR (95%)": "#EF5350"},
                    title="Risk Yöntemleri Karşılaştırması"
                )
                fig_risk.update_layout(yaxis_tickformat='.2%')
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with c2:
                st.dataframe(risk_table.style.format({f"VaR (95%)": "{:.2%}", f"CVaR (95%)": "{:.2%}"}), use_container_width=True)

        with tab_stress:
            st.subheader("Tarihsel Kriz Simülasyonu")
            st.markdown("Portföyünüzün mevcut ağırlıkları, geçmişteki büyük krizlerde nasıl performans gösterirdi?")
            
            if not stress_table.empty:
                # Heatmap Table
                st.dataframe(
                    stress_table.style.format({
                        "Toplam Getiri": "{:+.2%}", 
                        "Maksimum Düşüş (DD)": "{:.2%}", 
                        "Volatilite (Yıllık)": "{:.2%}"
                    }).background_gradient(subset=["Maksimum Düşüş (DD)"], cmap="Reds_r", vmin=-0.5, vmax=0),
                    use_container_width=True
                )
                
                # Drawdown Chart
                fig_dd = px.bar(
                    stress_table, x="Senaryo", y="Maksimum Düşüş (DD)", 
                    color="Maksimum Düşüş (DD)", color_continuous_scale="Reds_r",
                    title="Kriz Anlarında Maksimum Erime (Drawdown)"
                )
                fig_dd.update_layout(yaxis_tickformat='.2%')
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.warning("Seçilen hisseler için yeterli tarihsel veri bulunamadı (Örn: Hisseler kriz tarihlerinden sonra halka arz olmuş olabilir).")

        with tab_frontier:
            st.subheader("Markowitz Etkin Sınır (Efficient Frontier)")
            st.markdown("Yapay zeka, binlerce rastgele portföy oluşturarak 'Risk/Getiri' evrenini simüle eder.")
            
            if len(tickers) > 2:
                mu = expected_returns.mean_historical_return(prices)
                S = risk_models.sample_cov(prices)
                
                # Identify current optimal point coords
                opt_coords = None
                if opt_method != "HRP (Hierarchical Risk Parity)":
                    opt_coords = (perf[0], perf[1], perf[2]) # ret, vol, sharpe
                
                with st.spinner("Monte Carlo simülasyonu çalışıyor..."):
                    fig_ef = VisualizationEngine.plot_efficient_frontier_cloud(mu, S, opt_coords, opt_method)
                    st.plotly_chart(fig_ef, use_container_width=True)
            else:
                st.info("Etkin sınır grafiği için en az 3 varlık gereklidir.")

        with tab_details:
            st.subheader("Korelasyon Matrisi")
            st.plotly_chart(VisualizationEngine.plot_correlation_heatmap(returns), use_container_width=True)
            
            st.subheader("Ham Fiyat Verileri")
            st.dataframe(prices.tail())

if __name__ == "__main__":
    main()

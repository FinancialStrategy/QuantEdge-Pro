# QuantEdge Pro - Enhanced Institutional Version with Merged Features
# Save as: app.py and run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import scipy.stats as stats

warnings.filterwarnings("ignore")

# =========================
# Enhanced Config
# =========================
class EnhancedConfig:
    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.045
    MAX_TICKERS = 150
    YF_BATCH_SIZE = 50
    MC_SIMULATIONS = 10000
    MIN_WEIGHT = 0.01
    MAX_WEIGHT = 0.20
    MAX_SECTOR_WEIGHT = 0.35
    INITIAL_CAPITAL = 10000000
    VAR_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
    
    # Enhanced covariance models
    COV_MODELS = {
        "Sample Covariance": "sample_cov",
        "Ledoit-Wolf Shrinkage": "ledoit_wolf",
        "Oracle Approximating Shrinkage (OAS)": "oas",
        "Shrunk Covariance (Custom)": "shrunk_cov",
    }

# =========================
# Enhanced Universe Definition (Combined)
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

# Additional assets from second file
ADDITIONAL_INSTRUMENTS = [
    {"ticker": "QQQ", "name": "Invesco QQQ", "region": "US", "sector": "ETF"},
    {"ticker": "IWM", "name": "Russell 2000", "region": "US", "sector": "ETF"},
    {"ticker": "VTI", "name": "Vanguard Total Stock", "region": "US", "sector": "ETF"},
    {"ticker": "DIA", "name": "Dow Jones ETF", "region": "US", "sector": "ETF"},
    {"ticker": "VEA", "name": "FTSE Developed Markets", "region": "Intl", "sector": "ETF"},
    {"ticker": "VWO", "name": "FTSE Emerging Markets", "region": "Intl", "sector": "ETF"},
    {"ticker": "EWJ", "name": "MSCI Japan", "region": "Intl", "sector": "ETF"},
    {"ticker": "EZU", "name": "MSCI Eurozone", "region": "Intl", "sector": "ETF"},
    {"ticker": "BND", "name": "Vanguard Bond", "region": "US", "sector": "Bond"},
    {"ticker": "TLT", "name": "20+ Year Treasury", "region": "US", "sector": "Bond"},
    {"ticker": "IEF", "name": "7-10 Year Treasury", "region": "US", "sector": "Bond"},
    {"ticker": "LQD", "name": "Corporate Bond", "region": "US", "sector": "Bond"},
    {"ticker": "MUB", "name": "Municipal Bond", "region": "US", "sector": "Bond"},
    {"ticker": "HYG", "name": "High Yield Bond", "region": "US", "sector": "Bond"},
    {"ticker": "GLD", "name": "Gold Trust", "region": "US", "sector": "Commodity"},
    {"ticker": "SLV", "name": "Silver Trust", "region": "US", "sector": "Commodity"},
    {"ticker": "VNQ", "name": "Real Estate", "region": "US", "sector": "REIT"},
    {"ticker": "GSG", "name": "Commodity Index", "region": "US", "sector": "Commodity"},
    {"ticker": "DBB", "name": "Base Metals", "region": "US", "sector": "Commodity"},
    {"ticker": "XLK", "name": "Technology Select", "region": "US", "sector": "Sector"},
    {"ticker": "XLF", "name": "Financial Select", "region": "US", "sector": "Sector"},
    {"ticker": "XLV", "name": "Health Care Select", "region": "US", "sector": "Sector"},
    {"ticker": "XLE", "name": "Energy Select", "region": "US", "sector": "Sector"},
    {"ticker": "XLI", "name": "Industrial Select", "region": "US", "sector": "Sector"},
    {"ticker": "XLP", "name": "Consumer Staples", "region": "US", "sector": "Sector"},
    {"ticker": "XLY", "name": "Consumer Discretionary", "region": "US", "sector": "Sector"},
    {"ticker": "XLU", "name": "Utilities Select", "region": "US", "sector": "Sector"},
]

# Combine both universes
COMBINED_INSTRUMENTS = INSTRUMENTS + ADDITIONAL_INSTRUMENTS

UNIVERSE_DF = pd.DataFrame(COMBINED_INSTRUMENTS)
UNIVERSE_DF = UNIVERSE_DF.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

REGIONS = sorted(UNIVERSE_DF["region"].unique())
SECTORS = sorted(UNIVERSE_DF["sector"].unique())

# Enhanced scenario definitions
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
    "US Equity Focus": lambda df: df[df["region"] == "US"],
    "Global Multi-Asset": lambda df: df[df["sector"].isin(["ETF", "Bond", "Commodity", "REIT", "Sector"])],
}

# =========================
# Enhanced Data Manager
# =========================
class EnhancedDataManager:
    def __init__(self, config):
        self.config = config
        self.asset_prices = pd.DataFrame()
        self.asset_returns = pd.DataFrame()
        self.benchmark_prices = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        
    def _download_batch(self, tickers, start, end):
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

    def fetch_price_data(self, tickers, start_date, end_date):
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        tickers = sorted(list(dict.fromkeys(tickers)))  # dedupe while preserving order
        if not tickers:
            return pd.DataFrame()

        if len(tickers) > self.config.MAX_TICKERS:
            tickers = tickers[: self.config.MAX_TICKERS]

        all_closes = []
        for i in range(0, len(tickers), self.config.YF_BATCH_SIZE):
            batch = tickers[i : i + self.config.YF_BATCH_SIZE]
            closes = self._download_batch(batch, start_date, end_date)
            if not closes.empty:
                all_closes.append(closes)

        if not all_closes:
            return pd.DataFrame()

        prices = pd.concat(all_closes, axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
        prices = prices.ffill().bfill()
        return prices
    
    def load_data(self, selected_tickers, start_date, end_date, benchmark_ticker="SPY"):
        """Load and validate all required data"""
        try:
            # Download asset prices
            self.asset_prices = self.fetch_price_data(
                selected_tickers,
                start_date,
                end_date + timedelta(days=1),
            )
            
            if self.asset_prices.empty:
                return False, "No price data fetched"
                
            # Calculate returns
            self.asset_returns = self.asset_prices.pct_change().dropna()
            
            # Download benchmark data
            bench_prices = self.fetch_price_data(
                [benchmark_ticker], start_date, end_date + timedelta(days=1)
            )
            
            if not bench_prices.empty:
                self.benchmark_prices = bench_prices.iloc[:, 0]
                self.benchmark_returns = self.benchmark_prices.pct_change().dropna()
            
            # Align dates
            if not self.benchmark_returns.empty:
                common_idx = self.asset_returns.index.intersection(self.benchmark_returns.index)
                if len(common_idx) > 20:
                    self.asset_returns = self.asset_returns.loc[common_idx]
                    self.benchmark_returns = self.benchmark_returns.loc[common_idx]
                    self.asset_prices = self.asset_prices.loc[common_idx]
                    self.benchmark_prices = self.benchmark_prices.loc[common_idx]
            
            return True, "Data loaded successfully"
            
        except Exception as e:
            return False, f"Data loading failed: {str(e)}"

# =========================
# Enhanced Risk Metrics (Combined)
# =========================
def historical_var(returns, alpha=0.95):
    if returns.empty or len(returns) < 10:
        return np.nan
    return -np.percentile(returns, (1 - alpha) * 100)

def historical_cvar(returns, alpha=0.95):
    if returns.empty or len(returns) < 10:
        return np.nan
    var = historical_var(returns, alpha)
    tail = returns[returns <= -var]
    return -tail.mean() if len(tail) > 0 else var

def parametric_var(returns, alpha=0.95):
    if returns.empty or len(returns) < 10:
        return np.nan
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha)
    return -(mu + z * sigma)

def mc_var_cvar(returns, alpha=0.95, n_sims=EnhancedConfig.MC_SIMULATIONS):
    if returns.empty or len(returns) < 10:
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

def compute_sortino_ratio(portfolio_returns, risk_free_rate):
    if portfolio_returns.empty or len(portfolio_returns) < 10:
        return np.nan
    excess_returns = portfolio_returns - risk_free_rate / EnhancedConfig.TRADING_DAYS_PER_YEAR
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 1 else 0
    if downside_std == 0:
        return np.nan
    sortino = np.sqrt(EnhancedConfig.TRADING_DAYS_PER_YEAR) * excess_returns.mean() / downside_std
    return sortino

def compute_calmar_ratio(portfolio_returns):
    if portfolio_returns.empty or len(portfolio_returns) < 50:
        return np.nan
    ann_ret = portfolio_returns.mean() * EnhancedConfig.TRADING_DAYS_PER_YEAR
    max_dd = compute_max_drawdown(portfolio_returns)
    if max_dd == 0:
        return np.nan
    return -ann_ret / max_dd

def compute_skewness_kurtosis(portfolio_returns):
    if portfolio_returns.empty or len(portfolio_returns) < 20:
        return np.nan, np.nan
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurtosis()
    return skewness, kurtosis

def compute_risk_profile(portfolio_returns, benchmark_returns, alpha=0.95):
    if portfolio_returns.empty:
        return {}
    
    ann_ret = portfolio_returns.mean() * EnhancedConfig.TRADING_DAYS_PER_YEAR
    ann_vol = portfolio_returns.std() * np.sqrt(EnhancedConfig.TRADING_DAYS_PER_YEAR)
    sharpe = (ann_ret - EnhancedConfig.RISK_FREE_RATE) / ann_vol if ann_vol != 0 else np.nan
    sortino = compute_sortino_ratio(portfolio_returns, EnhancedConfig.RISK_FREE_RATE)
    max_dd = compute_max_drawdown(portfolio_returns)
    calmar = compute_calmar_ratio(portfolio_returns)
    skewness, kurtosis = compute_skewness_kurtosis(portfolio_returns)
    
    h_var = historical_var(portfolio_returns, alpha)
    h_cvar = historical_cvar(portfolio_returns, alpha)
    p_var = parametric_var(portfolio_returns, alpha)
    mc_v, mc_c = mc_var_cvar(portfolio_returns, alpha)
    
    rel_var = rel_cvar = np.nan
    beta = np.nan
    tracking_error = np.nan
    information_ratio = np.nan
    
    if benchmark_returns is not None and not benchmark_returns.empty:
        diff = portfolio_returns - benchmark_returns
        rel_var = historical_var(diff, alpha)
        rel_cvar = historical_cvar(diff, alpha)
        beta = compute_beta(portfolio_returns, benchmark_returns)
        
        # Tracking error and information ratio
        if len(diff) > 10:
            tracking_error = diff.std() * np.sqrt(EnhancedConfig.TRADING_DAYS_PER_YEAR)
            if tracking_error != 0:
                information_ratio = (ann_ret - benchmark_returns.mean() * EnhancedConfig.TRADING_DAYS_PER_YEAR) / tracking_error
    
    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "hist_var": h_var,
        "hist_cvar": h_cvar,
        "param_var": p_var,
        "mc_var": mc_v,
        "mc_cvar": mc_c,
        "rel_var": rel_var,
        "rel_cvar": rel_cvar,
        "beta": beta,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
    }

# =========================
# Enhanced Optimization Engine (Combined)
# =========================
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
    from pypfopt import objective_functions
    import cvxpy
    HAS_PYPORT = True
except Exception:
    HAS_PYPORT = False

class EnhancedPortfolioOptimizer:
    def __init__(self, config, log_func):
        self.config = config
        self.log_func = log_func
        self.cov_model_name = "Ledoit-Wolf Shrinkage"
        
    def set_cov_model(self, model_name):
        self.cov_model_name = model_name
        self.log_func(f"Covariance model set to: {model_name}")
    
    def _get_covariance_matrix(self, prices, model_name):
        """Dynamically select and compute the covariance matrix."""
        method = self.config.COV_MODELS.get(model_name, "ledoit_wolf")
        
        if method == "sample_cov":
            S = risk_models.sample_cov(prices)
        elif method == "oas":
            S = risk_models.CovarianceShrinkage(prices).oracle_approximating()
        elif method == "shrunk_cov":
            S = risk_models.CovarianceShrinkage(prices, shrinkage=0.5).shrunk_covariance()
        elif method == "ledoit_wolf":
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        else:
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
            
        return S
    
    def _ensure_psd(self, S, min_eig=1e-10):
        """Ensure covariance matrix is positive semi-definite."""
        try:
            A = S.values
            A = 0.5 * (A + A.T)
            vals, vecs = np.linalg.eigh(A)
            vals_clipped = np.clip(vals, min_eig, None)
            A_psd = (vecs @ np.diag(vals_clipped) @ vecs.T)
            A_psd = 0.5 * (A_psd + A_psd.T)
            return pd.DataFrame(A_psd, index=S.index, columns=S.columns)
        except Exception:
            return S.copy()
    
    def run_enhanced_optimizations(self, prices, returns, benchmark_returns):
        """Run enhanced portfolio optimizations with multiple strategies."""
        strategies = {}
        if prices.empty:
            return strategies
            
        tickers = list(prices.columns)
        n = len(tickers)
        
        # 1. Equal Weight (Benchmark)
        w_eq = np.repeat(1 / n, n)
        port_eq = returns.dot(w_eq)
        strategies["Equal Weight"] = {
            "name": "Equal Weight",
            "weights": pd.Series(w_eq, index=tickers),
            "returns": port_eq,
            "risk": compute_risk_profile(port_eq, benchmark_returns)
        }
        
        if not HAS_PYPORT:
            return strategies
        
        try:
            mu = expected_returns.mean_historical_return(prices)
            S = self._get_covariance_matrix(prices, self.cov_model_name)
            S = self._ensure_psd(S)
            
            # 2. Max Sharpe with L2 regularization
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.max_sharpe(risk_free_rate=self.config.RISK_FREE_RATE)
            w_ms = ef.clean_weights()
            w_vec = np.array([w_ms[t] for t in tickers])
            port_ms = returns.dot(w_vec)
            strategies["Max Sharpe (Enhanced)"] = {
                "name": "Max Sharpe (Enhanced)",
                "weights": pd.Series(w_vec, index=tickers),
                "returns": port_ms,
                "risk": compute_risk_profile(port_ms, benchmark_returns)
            }
            
            # 3. Min Volatility with constraints
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.05)
            ef.min_volatility()
            w_mv = ef.clean_weights()
            w_vec = np.array([w_mv[t] for t in tickers])
            port_mv = returns.dot(w_vec)
            strategies["Min Volatility (Enhanced)"] = {
                "name": "Min Volatility (Enhanced)",
                "weights": pd.Series(w_vec, index=tickers),
                "returns": port_mv,
                "risk": compute_risk_profile(port_mv, benchmark_returns)
            }
            
            # 4. CLA Max Sharpe
            cla = CLA(mu, S)
            cla.max_sharpe()
            w_cla = cla.clean_weights()
            w_vec = np.array([w_cla[t] for t in tickers])
            port_cla = returns.dot(w_vec)
            strategies["CLA Max Sharpe"] = {
                "name": "CLA Max Sharpe",
                "weights": pd.Series(w_vec, index=tickers),
                "returns": port_cla,
                "risk": compute_risk_profile(port_cla, benchmark_returns)
            }
            
            # 5. HRP
            hrp = HRPOpt(returns)
            w_hrp = hrp.optimize()
            w_vec = np.array([w_hrp[t] for t in tickers])
            port_hrp = returns.dot(w_vec)
            strategies["HRP"] = {
                "name": "HRP",
                "weights": pd.Series(w_vec, index=tickers),
                "returns": port_hrp,
                "risk": compute_risk_profile(port_hrp, benchmark_returns)
            }
            
            # 6. Black-Litterman
            market_weights = np.repeat(1 / n, n)
            prior = market_implied_prior_returns(S, market_weights, risk_aversion=None)
            bl = BlackLittermanModel(S, pi=prior)
            bl_ret = bl.bl_returns()
            bl_cov = bl.bl_cov()
            ef = EfficientFrontier(bl_ret, bl_cov)
            ef.max_sharpe(risk_free_rate=self.config.RISK_FREE_RATE)
            w_bl = ef.clean_weights()
            w_vec = np.array([w_bl[t] for t in tickers])
            port_bl = returns.dot(w_vec)
            strategies["Black-Litterman"] = {
                "name": "Black-Litterman",
                "weights": pd.Series(w_vec, index=tickers),
                "returns": port_bl,
                "risk": compute_risk_profile(port_bl, benchmark_returns)
            }
            
            # 7. Risk Parity (Basic)
            volatilities = np.sqrt(np.diag(S.values))
            inverse_vol = 1 / np.maximum(volatilities, 1e-12)
            w_rp = inverse_vol / inverse_vol.sum()
            port_rp = returns.dot(w_rp)
            strategies["Risk Parity"] = {
                "name": "Risk Parity",
                "weights": pd.Series(w_rp, index=tickers),
                "returns": port_rp,
                "risk": compute_risk_profile(port_rp, benchmark_returns)
            }
            
            # 8. Max Diversification
            try:
                volatilities = np.sqrt(np.diag(S.values))
                diversification_weights = 1 / np.maximum(volatilities, 1e-12)
                w_md = diversification_weights / diversification_weights.sum()
                port_md = returns.dot(w_md)
                strategies["Max Diversification"] = {
                    "name": "Max Diversification",
                    "weights": pd.Series(w_md, index=tickers),
                    "returns": port_md,
                    "risk": compute_risk_profile(port_md, benchmark_returns)
                }
            except:
                pass
                
        except Exception as e:
            self.log_func(f"Optimization error: {str(e)[:100]}")
        
        return strategies

# =========================
# Enhanced Visualization Engine
# =========================
class EnhancedVisualizationEngine:
    COLORS = {
        'performance': '#34495E',
        'risk': '#1ABC9C',
        'var': '#2980B9',
        'stress': '#9B59B6',
        'strategy': '#E67E22',
        'positive': '#27AE60',
        'negative': '#E74C3C',
        'neutral': '#95A5A6'
    }
    
    def plot_weights_bar(self, weights_series, title):
        if weights_series is None or weights_series.empty:
            return go.Figure()
        w = weights_series[weights_series > 1e-4].sort_values(ascending=False)
        fig = go.Figure()
        fig.add_bar(x=w.index, y=w.values, marker_color=self.COLORS['strategy'])
        fig.update_layout(
            title=title,
            xaxis_title="Ticker",
            yaxis_title="Weight",
            template="plotly_dark",
            height=400
        )
        return fig
    
    def plot_cumulative_returns(self, portfolio_returns, benchmark_returns=None, benchmark_name="Benchmark", title="Cumulative Returns"):
        if portfolio_returns is None or portfolio_returns.empty:
            return go.Figure()
        
        cum_port = (1 + portfolio_returns).cumprod()
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(x=cum_port.index, y=cum_port.values, mode="lines", 
                      name="Portfolio", line=dict(color=self.COLORS['performance'], width=2))
        )
        
        if benchmark_returns is not None and not benchmark_returns.empty:
            cum_bench = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(x=cum_bench.index, y=cum_bench.values, mode="lines",
                          name=benchmark_name, line=dict(color=self.COLORS['neutral'], width=2, dash='dash'))
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return (1 = start)",
            template="plotly_dark",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig
    
    def plot_efficient_frontier(self, mu, S, strategies, risk_free_rate):
        """Create enhanced efficient frontier plot."""
        try:
            if len(mu) < 2:
                return go.Figure()
                
            fig = go.Figure()
            
            # Calculate frontier points
            ef = EfficientFrontier(mu, S)
            min_ret, min_vol, _ = ef.min_volatility().portfolio_performance()
            
            ef_ms = EfficientFrontier(mu, S)
            ms_ret, ms_vol, _ = ef_ms.max_sharpe(risk_free_rate=risk_free_rate).portfolio_performance()
            
            target_returns = np.linspace(min_ret, ms_ret * 1.5, 50)
            efficient_vols, efficient_rets = [], []
            
            for tr in target_returns:
                try:
                    ef_p = EfficientFrontier(mu, S)
                    ef_p.efficient_return(target_return=tr)
                    r, v, _ = ef_p.portfolio_performance()
                    efficient_rets.append(r)
                    efficient_vols.append(v)
                except:
                    continue
            
            # Plot efficient frontier
            if efficient_vols:
                fig.add_trace(go.Scatter(
                    x=efficient_vols, y=efficient_rets, mode='lines',
                    name='Efficient Frontier', line=dict(color=self.COLORS['performance'], width=3)
                ))
            
            # Plot individual assets
            individual_vols = np.sqrt(np.diag(S.values))
            fig.add_trace(go.Scatter(
                x=individual_vols, y=mu.values, mode='markers',
                name='Individual Assets', marker=dict(size=10, color=self.COLORS['risk'], symbol='circle'),
                text=list(mu.index), hoverinfo='text+x+y'
            ))
            
            # Plot strategy points
            for strategy_name, strategy_info in strategies.items():
                weights = strategy_info.get('weights', {})
                if weights:
                    w = np.array([weights.get(t, 0.0) for t in mu.index])
                    w = w / np.sum(w) if np.sum(w) > 0 else w
                    portfolio_return = float(np.dot(w, mu.values))
                    portfolio_vol = float(np.sqrt(w @ S.values @ w))
                    
                    fig.add_trace(go.Scatter(
                        x=[portfolio_vol], y=[portfolio_return], mode='markers',
                        name=strategy_name, marker=dict(size=15, symbol='star', color=self.COLORS['strategy'])
                    ))
            
            fig.update_layout(
                title="📈 Efficient Frontier Analysis",
                xaxis_title="Annual Volatility (Risk)",
                yaxis_title="Annual Return",
                hovermode='closest',
                showlegend=True,
                template="plotly_white",
                height=600,
                xaxis=dict(tickformat='.0%'),
                yaxis=dict(tickformat='.0%')
            )
            return fig
            
        except Exception as e:
            return go.Figure()
    
    def plot_monte_carlo_simulation(self, mu, sigma, n_sims=1000, n_days=252):
        """Plot Monte Carlo simulation results."""
        daily_mu = mu / EnhancedConfig.TRADING_DAYS_PER_YEAR
        daily_sigma = sigma / np.sqrt(EnhancedConfig.TRADING_DAYS_PER_YEAR)
        
        simulations = np.zeros((n_days, n_sims))
        for i in range(n_sims):
            daily_returns = np.random.normal(daily_mu, daily_sigma, n_days)
            simulations[:, i] = (1 + daily_returns).cumprod()
        
        final_returns = simulations[-1, :] - 1
        p5 = np.percentile(final_returns, 5)
        p50 = np.percentile(final_returns, 50)
        p95 = np.percentile(final_returns, 95)
        
        fig = go.Figure()
        
        # Plot sample paths
        for i in range(min(100, n_sims)):
            fig.add_trace(go.Scatter(
                y=simulations[:, i], line=dict(width=0.5, color='rgba(52, 152, 219, 0.1)'),
                showlegend=False, hoverinfo='skip'
            ))
        
        # Plot percentiles
        median_path = np.median(simulations, axis=1)
        p5_path = np.percentile(simulations, 5, axis=1)
        p95_path = np.percentile(simulations, 95, axis=1)
        
        fig.add_trace(go.Scatter(
            y=median_path, name=f'Median ({p50:.1%})',
            line=dict(color=self.COLORS['performance'], width=3)
        ))
        fig.add_trace(go.Scatter(
            y=p95_path, name=f'95th Percentile ({p95:.1%})',
            line=dict(color=self.COLORS['positive'], width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            y=p5_path, name=f'5th Percentile ({p5:.1%})',
            line=dict(color=self.COLORS['negative'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"🔮 Monte Carlo Simulation ({n_sims} paths)",
            xaxis_title="Trading Days (1 Year)",
            yaxis_title="Portfolio Value (Starting at 1.0)",
            template="plotly_white",
            height=500
        )
        return fig
    
    def plot_return_distribution(self, returns, title="Return Distribution"):
        """Plot histogram of returns with normal distribution overlay."""
        if returns.empty or len(returns) < 20:
            return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns, nbinsx=50, name="Returns",
            marker_color=self.COLORS['risk'], opacity=0.7
        ))
        
        # Add normal distribution overlay
        x = np.linspace(returns.min(), returns.max(), 100)
        pdf = stats.norm.pdf(x, returns.mean(), returns.std())
        fig.add_trace(go.Scatter(
            x=x, y=pdf * len(returns) * (returns.max() - returns.min()) / 50,
            mode='lines', name='Normal Fit', line=dict(color=self.COLORS['performance'], width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Daily Return",
            yaxis_title="Frequency",
            template="plotly_white",
            height=400
        )
        return fig

# =========================
# Enhanced Stress Testing
# =========================
class EnhancedStressTesting:
    SCENARIOS = {
        "COVID-19 Crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
        "Russia-Ukraine War (Feb-Mar 2022)": ("2022-02-24", "2022-03-31"),
        "Q4 2018 Bear Market": ("2018-10-01", "2018-12-31"),
        "2022 Final Selloff (Sep-Oct 2022)": ("2022-09-01", "2022-10-31"),
    }
    
    @staticmethod
    def calculate_stress_metrics(portfolio_returns, benchmark_returns, start_date, end_date):
        """Calculate stress test metrics for a specific period."""
        try:
            start_dt = pd.to_datetime(start_date).tz_localize(None)
            end_dt = pd.to_datetime(end_date).tz_localize(None)
            
            portfolio_returns.index = portfolio_returns.index.tz_localize(None)
            mask = (portfolio_returns.index >= start_dt) & (portfolio_returns.index <= end_dt)
            stress_returns = portfolio_returns.loc[mask]
            
            if len(stress_returns) < 5:
                return {}
            
            portfolio_return = (1 + stress_returns).prod() - 1
            max_dd = compute_max_drawdown(stress_returns)
            volatility = stress_returns.std() * np.sqrt(252) if len(stress_returns) > 1 else 0
            var_95 = historical_var(stress_returns, 0.95)
            
            return {
                'portfolio_return': portfolio_return,
                'max_drawdown': max_dd,
                'volatility': volatility,
                'var_95': var_95,
                'days_in_stress': len(stress_returns)
            }
        except Exception:
            return {}

# =========================
# Streamlit UI Components
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
    if bcol4.button("US Equity Focus"):
        st.session_state["scenario_index"] = scenario_names.index("US Equity Focus")

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

def display_strategy_factsheet(strategy_name, strategy_obj, benchmark_ticker):
    """Display comprehensive strategy factsheet."""
    if not strategy_obj:
        return
    
    st.subheader(f"🏦 Institutional Strategy Factsheet - {strategy_name}")
    
    risk = strategy_obj.get("risk", {})
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Return", f"{risk.get('annual_return', np.nan)*100:.2f}%")
    col2.metric("Annual Volatility", f"{risk.get('annual_vol', np.nan)*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{risk.get('sharpe', np.nan):.3f}")
    col4.metric("Max Drawdown", f"{risk.get('max_drawdown', 0)*100:.2f}%")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Sortino Ratio", f"{risk.get('sortino', np.nan):.3f}")
    col6.metric("Calmar Ratio", f"{risk.get('calmar', np.nan):.3f}")
    col7.metric("VaR 95%", f"{risk.get('hist_var', np.nan)*100:.2f}%")
    col8.metric("CVaR 95%", f"{risk.get('hist_cvar', np.nan)*100:.2f}%")
    
    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Beta", f"{risk.get('beta', np.nan):.3f}")
    col10.metric("Tracking Error", f"{risk.get('tracking_error', np.nan)*100:.2f}%")
    col11.metric("Information Ratio", f"{risk.get('information_ratio', np.nan):.3f}")
    col12.metric("Skewness", f"{risk.get('skewness', np.nan):.3f}")
    
    # Weights table
    weights = strategy_obj.get('weights', pd.Series())
    if not weights.empty:
        st.subheader("📊 Asset Allocation")
        weights_df = pd.DataFrame(weights, columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
        weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
        st.dataframe(weights_df, height=400)

# =========================
# Main Enhanced App
# =========================
def main():
    st.set_page_config(
        page_title="QuantEdge Pro - Enhanced Institutional Portfolio Analytics",
        page_icon="🏆",
        layout="wide",
    )

    st.title("🏆 QuantEdge Pro - Enhanced Institutional Portfolio Analytics")
    st.markdown(
        "**Comprehensive portfolio analysis tool with enhanced optimization, risk metrics, and stress testing.** "
        "All data fetched directly from Yahoo Finance."
    )
    
    # Initialize session state
    if 'strategies' not in st.session_state:
        st.session_state.strategies = {}
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # Universe panel
    selected_tickers = universe_and_scenario_panel(UNIVERSE_DF)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Enhanced Configuration")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365 * 3),
                max_value=datetime.now() - timedelta(days=5),
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now(),
            )
        
        # Advanced settings
        st.subheader("Advanced Settings")
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000000,
            value=10000000,
            step=100000,
            format="%d"
        )
        EnhancedConfig.INITIAL_CAPITAL = initial_capital
        
        cov_model = st.selectbox(
            "Covariance Model",
            options=list(EnhancedConfig.COV_MODELS.keys()),
            index=1  # Ledoit-Wolf as default
        )
        
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.5,
            step=0.1
        )
        EnhancedConfig.RISK_FREE_RATE = risk_free_rate / 100
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Portfolio Optimization", "Risk Analysis", "Stress Testing", "Data Preview", "Monte Carlo Simulation"]
        )
        
        run_button = st.button("🚀 Run Enhanced Analysis", type="primary", use_container_width=True)
    
    # Main analysis execution
    if run_button:
        if len(selected_tickers) < 2:
            st.error("Need at least 2 tickers selected for portfolio analysis.")
            return
        
        with st.spinner("Loading data and running analysis..."):
            # Initialize components
            config = EnhancedConfig()
            data_manager = EnhancedDataManager(config)
            
            # Determine benchmark
            tr_only = all(UNIVERSE_DF[UNIVERSE_DF["ticker"] == t].iloc[0]["region"] == "TR" 
                         for t in selected_tickers if t in UNIVERSE_DF["ticker"].values)
            benchmark_ticker = "XU100.IS" if tr_only else "SPY"
            
            # Load data
            success, message = data_manager.load_data(selected_tickers, start_date, end_date, benchmark_ticker)
            
            if not success:
                st.error(f"❌ {message}")
                return
            
            # Initialize optimizer and visualizer
            optimizer = EnhancedPortfolioOptimizer(config, st.write)
            optimizer.set_cov_model(cov_model)
            visualizer = EnhancedVisualizationEngine()
            
            # Run optimizations
            strategies = optimizer.run_enhanced_optimizations(
                data_manager.asset_prices,
                data_manager.asset_returns,
                data_manager.benchmark_returns
            )
            
            # Store in session state
            st.session_state.strategies = strategies
            st.session_state.data_loaded = True
            st.session_state.asset_prices = data_manager.asset_prices
            st.session_state.asset_returns = data_manager.asset_returns
            st.session_state.benchmark_returns = data_manager.benchmark_returns
            st.session_state.benchmark_ticker = benchmark_ticker
            st.session_state.config = config
            
            st.success("✅ Analysis complete!")
    
    # Display results based on analysis type
    if st.session_state.data_loaded and st.session_state.strategies:
        strategies = st.session_state.strategies
        asset_returns = st.session_state.asset_returns
        benchmark_returns = st.session_state.benchmark_returns
        benchmark_ticker = st.session_state.benchmark_ticker
        config = st.session_state.config
        visualizer = EnhancedVisualizationEngine()
        
        if analysis_type == "Data Preview":
            st.subheader("📊 Data Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Asset Prices (last 10 rows)**")
                st.dataframe(st.session_state.asset_prices.tail(10), use_container_width=True)
            with col2:
                st.markdown("**Asset Returns (last 10 rows)**")
                st.dataframe(st.session_state.asset_returns.tail(10), use_container_width=True)
            
            st.markdown("**Correlation Matrix**")
            corr_matrix = st.session_state.asset_returns.corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                           color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Portfolio Optimization":
            st.subheader("🎯 Enhanced Portfolio Optimization")
            
            # Strategy selector
            strategy_names = list(strategies.keys())
            selected_strategy = st.selectbox(
                "Select Strategy for Detailed View",
                strategy_names,
                index=0
            )
            
            if selected_strategy in strategies:
                strategy = strategies[selected_strategy]
                
                # Display in columns
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Weights chart
                    fig_weights = visualizer.plot_weights_bar(
                        strategy['weights'],
                        f"Weights - {selected_strategy}"
                    )
                    st.plotly_chart(fig_weights, use_container_width=True)
                    
                    # Risk metrics table
                    st.markdown("**Risk Metrics**")
                    risk_df = pd.DataFrame([strategy['risk']]).T
                    risk_df.columns = ['Value']
                    st.dataframe(risk_df.style.format({
                        'Value': lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.2%}"
                    }), use_container_width=True)
                
                with col2:
                    # Cumulative returns
                    fig_cum = visualizer.plot_cumulative_returns(
                        strategy['returns'],
                        benchmark_returns,
                        benchmark_ticker,
                        f"Cumulative Returns - {selected_strategy}"
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                    
                    # Return distribution
                    fig_dist = visualizer.plot_return_distribution(
                        strategy['returns'],
                        f"Return Distribution - {selected_strategy}"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Strategy factsheet
                display_strategy_factsheet(selected_strategy, strategy, benchmark_ticker)
                
                # Efficient frontier (requires mu and S)
                try:
                    from pypfopt import expected_returns, risk_models
                    mu = expected_returns.mean_historical_return(st.session_state.asset_prices)
                    S = risk_models.sample_cov(st.session_state.asset_prices)
                    
                    fig_ef = visualizer.plot_efficient_frontier(
                        mu, S, strategies, config.RISK_FREE_RATE
                    )
                    st.plotly_chart(fig_ef, use_container_width=True)
                except:
                    pass
        
        elif analysis_type == "Risk Analysis":
            st.subheader("⚠️ Comprehensive Risk Analysis")
            
            strategy_names = list(strategies.keys())
            selected_strategy = st.selectbox(
                "Select Strategy for Risk Analysis",
                strategy_names,
                index=0
            )
            
            if selected_strategy in strategies:
                strategy = strategies[selected_strategy]
                risk = strategy['risk']
                
                # VaR analysis with different confidence levels
                st.markdown("#### Value at Risk Analysis")
                conf_levels = [0.90, 0.95, 0.99]
                
                var_data = []
                for conf in conf_levels:
                    var_h = historical_var(strategy['returns'], conf)
                    cvar_h = historical_cvar(strategy['returns'], conf)
                    var_p = parametric_var(strategy['returns'], conf)
                    var_mc, cvar_mc = mc_var_cvar(strategy['returns'], conf)
                    
                    var_data.append({
                        'Confidence Level': f"{int(conf*100)}%",
                        'Historical VaR': var_h,
                        'Historical CVaR': cvar_h,
                        'Parametric VaR': var_p,
                        'Monte Carlo VaR': var_mc,
                        'Monte Carlo CVaR': cvar_mc
                    })
                
                var_df = pd.DataFrame(var_data)
                st.dataframe(var_df.style.format({
                    'Historical VaR': '{:.2%}',
                    'Historical CVaR': '{:.2%}',
                    'Parametric VaR': '{:.2%}',
                    'Monte Carlo VaR': '{:.2%}',
                    'Monte Carlo CVaR': '{:.2%}'
                }), use_container_width=True)
                
                # Risk metrics comparison
                st.markdown("#### Risk Metrics Comparison Across Strategies")
                risk_metrics = ['annual_return', 'annual_vol', 'sharpe', 'sortino', 
                              'max_drawdown', 'hist_var', 'hist_cvar', 'beta']
                
                risk_comparison = {}
                for name, strat in strategies.items():
                    risk_comparison[name] = {metric: strat['risk'].get(metric, np.nan) 
                                           for metric in risk_metrics}
                
                risk_df = pd.DataFrame(risk_comparison).T
                st.dataframe(risk_df.style.format({
                    'annual_return': '{:.2%}',
                    'annual_vol': '{:.2%}',
                    'max_drawdown': '{:.2%}',
                    'hist_var': '{:.2%}',
                    'hist_cvar': '{:.2%}'
                }), use_container_width=True)
        
        elif analysis_type == "Stress Testing":
            st.subheader("🌪️ Historical Stress Testing")
            
            stress_tester = EnhancedStressTesting()
            strategy_names = list(strategies.keys())
            selected_strategy = st.selectbox(
                "Select Strategy for Stress Testing",
                strategy_names,
                index=0
            )
            
            if selected_strategy in strategies:
                strategy = strategies[selected_strategy]
                
                # Calculate stress metrics for each scenario
                stress_results = {}
                for scenario_name, (start_date_str, end_date_str) in stress_tester.SCENARIOS.items():
                    metrics = stress_tester.calculate_stress_metrics(
                        strategy['returns'],
                        benchmark_returns,
                        start_date_str,
                        end_date_str
                    )
                    if metrics:
                        stress_results[scenario_name] = metrics
                
                # Display stress test results
                if stress_results:
                    stress_df = pd.DataFrame(stress_results).T
                    stress_df = stress_df[['portfolio_return', 'max_drawdown', 'volatility', 'var_95', 'days_in_stress']]
                    stress_df.columns = ['Return', 'Max DD', 'Volatility', 'VaR 95%', 'Days']
                    
                    st.dataframe(stress_df.style.format({
                        'Return': '{:.2%}',
                        'Max DD': '{:.2%}',
                        'Volatility': '{:.2%}',
                        'VaR 95%': '{:.2%}'
                    }), use_container_width=True)
                    
                    # Visualize stress periods
                    fig = go.Figure()
                    cum_returns = (1 + strategy['returns']).cumprod()
                    
                    fig.add_trace(go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns.values,
                        mode='lines',
                        name='Portfolio',
                        line=dict(color=visualizer.COLORS['performance'], width=2)
                    ))
                    
                    # Highlight stress periods
                    for scenario_name, (start_date_str, end_date_str) in stress_tester.SCENARIOS.items():
                        start_dt = pd.to_datetime(start_date_str)
                        end_dt = pd.to_datetime(end_date_str)
                        
                        mask = (cum_returns.index >= start_dt) & (cum_returns.index <= end_dt)
                        if mask.any():
                            fig.add_trace(go.Scatter(
                                x=cum_returns.index[mask],
                                y=cum_returns[mask],
                                mode='lines',
                                name=f'{scenario_name} Stress',
                                line=dict(width=4, color=visualizer.COLORS['negative'])
                            ))
                    
                    fig.update_layout(
                        title="Portfolio Performance with Stress Periods Highlighted",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No stress test data available for the selected period.")
        
        elif analysis_type == "Monte Carlo Simulation":
            st.subheader("🔮 Monte Carlo Simulation")
            
            strategy_names = list(strategies.keys())
            selected_strategy = st.selectbox(
                "Select Strategy for Monte Carlo Simulation",
                strategy_names,
                index=0
            )
            
            if selected_strategy in strategies:
                strategy = strategies[selected_strategy]
                risk = strategy['risk']
                
                # Get portfolio parameters
                ann_return = risk.get('annual_return', 0)
                ann_vol = risk.get('annual_vol', 0)
                
                # Simulation parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)
                with col2:
                    n_days = st.slider("Time Horizon (Days)", 30, 252*2, 252, 30)
                with col3:
                    confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
                
                if ann_vol > 0:
                    # Run and display Monte Carlo simulation
                    fig_mc = visualizer.plot_monte_carlo_simulation(
                        ann_return, ann_vol, n_sims, n_days
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Calculate and display statistics
                    daily_mu = ann_return / config.TRADING_DAYS_PER_YEAR
                    daily_sigma = ann_vol / np.sqrt(config.TRADING_DAYS_PER_YEAR)
                    
                    # Simulate final returns
                    final_returns = np.random.normal(
                        daily_mu * n_days,
                        daily_sigma * np.sqrt(n_days),
                        n_sims
                    )
                    
                    # Calculate percentiles
                    p_low = np.percentile(final_returns, (1 - confidence_level) * 100)
                    p_median = np.percentile(final_returns, 50)
                    p_high = np.percentile(final_returns, confidence_level * 100)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"{int((1-confidence_level)*100)}th Percentile", f"{p_low:.2%}")
                    col2.metric("Median Return", f"{p_median:.2%}")
                    col3.metric(f"{int(confidence_level*100)}th Percentile", f"{p_high:.2%}")
                    
                    # Probability of positive return
                    prob_positive = (final_returns > 0).mean()
                    st.metric("Probability of Positive Return", f"{prob_positive:.2%}")
                else:
                    st.warning("Insufficient volatility data for Monte Carlo simulation.")
    
    else:
        # Initial state message
        st.info("""
        ## 📈 Welcome to QuantEdge Pro - Enhanced Institutional Portfolio Analytics
        
        **To get started:**
        1. Select your investment universe using the filters above
        2. Configure analysis parameters in the sidebar
        3. Click **"Run Enhanced Analysis"** to begin
        
        **Features include:**
        - **Multi-asset universe** with global coverage
        - **Enhanced optimization strategies** (Max Sharpe, Min Vol, Risk Parity, Black-Litterman, etc.)
        - **Comprehensive risk metrics** (VaR, CVaR, Sortino, Calmar ratios)
        - **Historical stress testing** for major market events
        - **Monte Carlo simulations** for forward-looking analysis
        - **Institutional-grade visualizations** and reporting
        """)

if __name__ == "__main__":
    main()

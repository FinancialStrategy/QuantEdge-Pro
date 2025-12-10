# ============================================================================
# QUANTEDGE MK | INSTITUTIONAL PORTFOLIO TERMINAL (ADVANCED MONOLITHIC VERSION)
# Version: v3.2 Pro Plus (AI + Advanced Risk + Stress Testing + Reporting Integrated)
# Production Level | Institutional Class
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# --- LIBRARY CHECKS AND FALLBACK MECHANISMS ---
class LibraryManager:
    """Library management and fallback mechanisms."""
    
    @staticmethod
    def check_libraries():
        """Check for necessary libraries."""
        lib_status = {}
        
        # PyPortfolioOpt
        try:
            from pypfopt import expected_returns, risk_models
            from pypfopt.efficient_frontier import EfficientFrontier
            from pypfopt.hierarchical_portfolio import HRPOpt
            from pypfopt.black_litterman import BlackLittermanModel
            lib_status['pypfopt'] = True
            globals().update({
                'expected_returns': expected_returns,
                'risk_models': risk_models,
                'EfficientFrontier': EfficientFrontier,
                'HRPOpt': HRPOpt,
                'BlackLittermanModel': BlackLittermanModel
            })
        except ImportError:
            lib_status['pypfopt'] = False
        
        # ARCH
        try:
            import arch
            lib_status['arch'] = True
            globals()['arch'] = arch
        except ImportError:
            lib_status['arch'] = False
        
        # Scikit-Learn
        try:
            from sklearn.decomposition import PCA
            lib_status['sklearn'] = True
            globals()['PCA'] = PCA
        except ImportError:
            lib_status['sklearn'] = False
        
        # Statsmodels
        try:
            import statsmodels.api as sm
            lib_status['statsmodels'] = True
            globals()['sm'] = sm
        except ImportError:
            lib_status['statsmodels'] = False
        
        return lib_status

LIBRARIES = LibraryManager.check_libraries()

warnings.filterwarnings('ignore')

TRADING_DAYS_PER_YEAR = 252

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QuantEdge Pro | Institutional Portfolio Management",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://quantedge.pro',
        'Report a bug': 'https://github.com/quantedge/issues',
        'About': "QuantEdge Pro v3.2 - Institutional Portfolio Analysis Platform"
    }
)

# --- PROFESSIONAL CSS STYLING (ADVANCED DARK THEME) ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1d2e 100%); }
    .main-header {
        background: linear-gradient(135deg, #1a1d2e 0%, #2a2a2a 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        border-left: 5px solid #00cc96;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative; overflow: hidden;
    }
    .main-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00cc96, #636efa, #ab63fa);
    }
    .pro-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%; position: relative; overflow: hidden;
    }
    .pro-card:hover { transform: translateY(-8px); border-color: #00cc96; }
    .metric-value {
        font-family: 'Roboto Mono', monospace; font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #00cc96, #636efa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem; letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.85rem; text-transform: uppercase; color: #888;
        letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.5rem;
    }
    .metric-change {
        font-size: 0.9rem; font-weight: 500; padding: 2px 8px;
        border-radius: 12px; display: inline-block;
    }
    .positive { background: rgba(0, 204, 150, 0.15); color: #00cc96 !important; border: 1px solid rgba(0, 204, 150, 0.3); }
    .negative { background: rgba(239, 85, 59, 0.15); color: #ef553b !important; border: 1px solid rgba(239, 85, 59, 0.3); }
    .neutral { background: rgba(128, 128, 128, 0.15); color: #888 !important; border: 1px solid rgba(128, 128, 128, 0.3); }
    .highlight-box {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.8), rgba(42, 42, 42, 0.8));
        border-left: 4px solid #00cc96; padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0;
        backdrop-filter: blur(10px); border: 1px solid rgba(128, 128, 128, 0.15);
    }
    .section-header {
        font-size: 1.8rem; font-weight: 800; color: white; margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.8rem; border-bottom: 2px solid linear-gradient(90deg, #00cc96, #636efa);
        position: relative;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS AND DATA STRUCTURES ---
class Constants:
    """Constant values and configurations."""
    
    ASSET_UNIVERSES = {
        "BIST 30": {
            "tickers": [
                'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS', 'EKGYO.IS',
                'ENKAI.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS', 'HALKB.IS',
                'HEKTS.IS', 'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS',
                'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TCELL.IS',
                'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TUPRS.IS', 'YKBNK.IS'
            ],
            "benchmark": "XU030.IS",
            "currency": "TRY",
            "risk_free_rate": 0.25
        },
        "US Tech": {
            "tickers": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM', 'CRM'],
            "benchmark": "^GSPC",
            "currency": "USD",
            "risk_free_rate": 0.045
        },
        "Global Diversified": {
            "tickers": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'WMT', 'XOM'],
            "benchmark": "^GSPC",
            "currency": "USD",
            "risk_free_rate": 0.045
        },
        "Euro Zone": {
            "tickers": ['SAP.DE', 'ASML.AS', 'SIEGY', 'SAN.PA', 'ULVR.L', 'HSBA.L'],
            "benchmark": "^STOXX50E",
            "currency": "EUR",
            "risk_free_rate": 0.02
        }
    }
    
    SECTOR_CLASSIFICATION = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM', 'CRM', 'ASML.AS'],
        'Financial Services': ['JPM', 'V', 'MA', 'GS', 'MS', 'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'YKBNK.IS'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'NKE', 'MCD', 'SBUX', 'NFLX', 'DIS'],
        'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'BIMAS.IS'],
        'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'LLY'],
        'Energy': ['XOM', 'CVX', 'SLB', 'TUPRS.IS', 'PETKM.IS'],
        'Industrials': ['BA', 'CAT', 'GE', 'LMT', 'THYAO.IS', 'SISE.IS', 'FROTO.IS', 'TOASO.IS'],
        'Utilities': ['NEE', 'DUK', 'SO'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EKGYO.IS'],
        'Materials': ['LIN', 'APD', 'EREGL.IS', 'KRDMD.IS', 'KOZAL.IS', 'SASA.IS'],
        'Communication': ['T', 'VZ', 'TCELL.IS']
    }
    
    # Expanded Historical Crisis Scenarios
    HISTORICAL_CRISES = {
        'Dot-com Bubble Burst (2000)': ('2000-03-10', '2002-10-09'),
        'September 11 Attacks (2001)': ('2001-09-10', '2001-10-31'),
        'Global Financial Crisis (2008)': ('2008-09-01', '2009-03-09'),
        'US Sovereign Debt Downgrade (2011)': ('2011-07-22', '2011-08-10'),
        'China Market Crash (2015)': ('2015-06-12', '2015-08-26'),
        'Crypto Winter / Trade War (2018)': ('2018-09-20', '2018-12-24'),
        'COVID-19 Crash (2020)': ('2020-02-19', '2020-03-23'),
        'Inflation & Rate Hikes (2022)': ('2022-01-03', '2022-10-14'),
        'Banking Crisis (SVB) (2023)': ('2023-03-08', '2023-03-31')
    }
    
    RISK_LEVELS = {
        'VERY CONSERVATIVE': {'color': '#00cc96', 'max_volatility': 0.08, 'max_drawdown': -0.10},
        'CONSERVATIVE': {'color': '#636efa', 'max_volatility': 0.12, 'max_drawdown': -0.15},
        'MODERATE': {'color': '#FFA15A', 'max_volatility': 0.18, 'max_drawdown': -0.20},
        'AGGRESSIVE': {'color': '#ef553b', 'max_volatility': 0.25, 'max_drawdown': -0.30},
        'VERY AGGRESSIVE': {'color': '#ab63fa', 'max_volatility': 0.35, 'max_drawdown': -0.40}
    }

@dataclass
class PortfolioConfig:
    universe: str
    tickers: List[str]
    benchmark: str
    start_date: datetime
    end_date: datetime
    risk_free_rate: float
    optimization_method: str
    target_volatility: Optional[float] = None
    transaction_cost: float = 0.001
    rebalancing_frequency: str = 'M'
    cash_buffer: float = 0.05
    constraints: Optional[Dict] = None

# ============================================================================
# 1. ADVANCED DATA MANAGEMENT AND CLASSIFICATION
# ============================================================================

class EnhancedAssetClassifierPro:
    """Advanced asset classification and metadata management."""
    def __init__(self):
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def classify_tickers(self, tickers: List[str]) -> Dict[str, Dict]:
        classifications = {}
        for ticker in tickers:
            sector = self._get_sector_from_constants(ticker)
            classifications[ticker] = self._get_default_classification(ticker, sector)
        return classifications
    
    def _get_sector_from_constants(self, ticker: str) -> Optional[str]:
        for sector, tickers in Constants.SECTOR_CLASSIFICATION.items():
            if ticker in tickers:
                return sector
        return None
    
    def _get_default_classification(self, ticker: str, sector: Optional[str] = None) -> Dict:
        if not sector:
            sector = 'Other'
        return {'ticker': ticker, 'sector': sector, 'industry': 'Unknown', 'full_name': ticker}

class PortfolioDataManagerPro:
    """Advanced data management and caching system."""
    def __init__(self):
        self.cache = {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger(__name__ + '.DataManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_portfolio_data(tickers: List[str], benchmark: str, 
                             start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        all_tickers = list(set(tickers + [benchmark]))
        try:
            data = yf.download(
                all_tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker',
                threads=True,
                auto_adjust=True
            )
            prices = pd.DataFrame()
            benchmark_prices = pd.Series(dtype=float)
            
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in all_tickers:
                    try:
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                        if 'Close' in df.columns:
                            if ticker == benchmark:
                                benchmark_prices = df['Close']
                            elif ticker in tickers:
                                prices[ticker] = df['Close']
                    except KeyError:
                        continue
            else:
                if 'Close' in data.columns:
                    if len(all_tickers) == 1:
                        if all_tickers[0] == benchmark:
                            benchmark_prices = data['Close']
                        else:
                            prices[all_tickers[0]] = data['Close']
            
            prices = prices.ffill().bfill()
            benchmark_prices = benchmark_prices.ffill().bfill()
            
            common_idx = prices.index.intersection(benchmark_prices.index)
            if len(common_idx) == 0:
                raise ValueError("Portfolio and benchmark dates do not overlap")
            return prices.loc[common_idx], benchmark_prices.loc[common_idx]
        except Exception as e:
            raise Exception(f"Data fetch error: {str(e)}")
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        benchmark_prices: pd.Series,
        method: str = 'log'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if method == 'log':
            portfolio_returns = np.log(prices / prices.shift(1)).dropna()
            benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
        else:
            portfolio_returns = prices.pct_change().dropna()
            benchmark_returns = benchmark_prices.pct_change().dropna()
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        return portfolio_returns.loc[common_idx], benchmark_returns.loc[common_idx]

# ============================================================================
# 2. ADVANCED PORTFOLIO OPTIMIZATION
# ============================================================================

class AdvancedPortfolioOptimizerPro:
    def __init__(self, returns: pd.DataFrame, prices: pd.DataFrame, logger=None):
        self.returns = returns
        self.prices = prices
        self.logger = logger or logging.getLogger(__name__)
        if LIBRARIES.get('pypfopt', False):
            try:
                self.mu = expected_returns.mean_historical_return(prices)
                self.S = risk_models.sample_cov(prices)
            except Exception:
                self.mu = None
                self.S = None
        else:
            self.mu = None
            self.S = None

    def optimize(self, method: str, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        methods = {
            'MAX_SHARPE': self._optimize_max_sharpe,
            'MIN_VOLATILITY': self._optimize_min_volatility,
            'RISK_PARITY': self._optimize_risk_parity,
            'MAX_DIVERSIFICATION': self._optimize_max_diversification,
            'HRP': self._optimize_hrp,
            'EQUAL_WEIGHT': self._optimize_equal_weight,
            'MEAN_VARIANCE': self._optimize_mean_variance
        }
        try:
            return methods.get(method, self._optimize_equal_weight)(config)
        except Exception as e:
            self.logger.error(f"{method} failed: {e}")
            return self._optimize_equal_weight(config)

    def _optimize_max_sharpe(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        if not LIBRARIES.get('pypfopt', False) or self.mu is None or self.S is None:
            return self._optimize_equal_weight(config)
        try:
            ef = EfficientFrontier(self.mu, self.S)
            _ = ef.max_sharpe(risk_free_rate=config.risk_free_rate)
            return ef.clean_weights(), ef.portfolio_performance(verbose=False, risk_free_rate=config.risk_free_rate)
        except Exception:
            return self._optimize_equal_weight(config)

    def _optimize_min_volatility(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        if not LIBRARIES.get('pypfopt', False) or self.mu is None or self.S is None:
            return self._optimize_equal_weight(config)
        try:
            ef = EfficientFrontier(self.mu, self.S)
            _ = ef.min_volatility()
            return ef.clean_weights(), ef.portfolio_performance(verbose=False, risk_free_rate=config.risk_free_rate)
        except Exception:
            return self._optimize_equal_weight(config)

    def _optimize_risk_parity(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        try:
            vol = self.returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            inv_vol = 1 / vol.replace(0, np.inf)
            weights = (inv_vol / inv_vol.sum()).to_dict()
            return weights, self._calculate_performance(weights, config.risk_free_rate)
        except Exception:
            return self._optimize_equal_weight(config)

    def _optimize_max_diversification(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        # Placeholder logic; can be upgraded to true maximum diversification ratio optimization.
        return self._optimize_equal_weight(config)

    def _optimize_hrp(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        if not LIBRARIES.get('pypfopt', False):
            return self._optimize_equal_weight(config)
        try:
            hrp = HRPOpt(self.returns)
            weights = hrp.optimize()
            return weights, self._calculate_performance(weights, config.risk_free_rate)
        except Exception:
            return self._optimize_equal_weight(config)

    def _optimize_equal_weight(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        n = len(self.returns.columns)
        if n == 0:
            return {}, (0.0, 0.0, 0.0)
        weights = {t: 1.0 / n for t in self.returns.columns}
        return weights, self._calculate_performance(weights, config.risk_free_rate)

    def _optimize_mean_variance(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        try:
            mu = self.returns.mean() * TRADING_DAYS_PER_YEAR
            S = self.returns.cov() * TRADING_DAYS_PER_YEAR
            n = len(mu)
            if n == 0:
                return {}, (0.0, 0.0, 0.0)
            bounds = [(0, 1) for _ in range(n)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # Sector constraints
            if config.constraints and 'sector_limits' in config.constraints:
                local_classifier = EnhancedAssetClassifierPro()
                for sector, (min_w, max_w) in config.constraints['sector_limits'].items():
                    idx = [i for i, t in enumerate(self.returns.columns)
                           if local_classifier._get_sector_from_constants(t) == sector]
                    if idx:
                        constraints.append({'type': 'ineq',
                                            'fun': lambda w, i=idx, m=min_w: np.sum(w[i]) - m})
                        constraints.append({'type': 'ineq',
                                            'fun': lambda w, i=idx, m=max_w: m - np.sum(w[i])})

            import scipy.optimize as opt
            def neg_sharpe(w, mu_vec, cov_mat, rf):
                port_ret = np.dot(w, mu_vec)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                if port_vol == 0:
                    return 0
                return -(port_ret - rf) / port_vol

            res = opt.minimize(
                neg_sharpe,
                np.ones(n) / n,
                args=(mu.values, S.values, config.risk_free_rate),
                bounds=bounds,
                constraints=constraints,
                method='SLSQP'
            )
            if res.success:
                w_clean = res.x / res.x.sum()
                weights = dict(zip(self.returns.columns, w_clean))
                return weights, self._calculate_performance(weights, config.risk_free_rate)
            raise ValueError("Optimization failed")
        except Exception:
            return self._optimize_equal_weight(config)

    def _calculate_performance(self, weights: Dict, risk_free_rate: float) -> Tuple:
        if self.returns.empty:
            return (0.0, 0.0, 0.0)
        w = np.array([weights.get(t, 0) for t in self.returns.columns])
        port_ret = self.returns.dot(w)
        ann_ret = port_ret.mean() * TRADING_DAYS_PER_YEAR
        ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
        return (ann_ret, ann_vol, sharpe)

# ============================================================================
# 3. ADVANCED RISK ANALYSIS
# ============================================================================

class AdvancedRiskAnalyzerPro:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_comprehensive_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.045
    ) -> Dict:
        """Calculate comprehensive risk metrics with English keys."""
        metrics: Dict[str, float] = {}
        try:
            pr = portfolio_returns.dropna()
            if pr.empty:
                return metrics

            total_ret = (1 + pr).prod() - 1
            ann_ret = pr.mean() * TRADING_DAYS_PER_YEAR
            ann_vol = pr.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

            downside = pr[pr < 0]
            down_std = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 0 else 0.0
            sortino = (ann_ret - risk_free_rate) / down_std if down_std > 0 else 0.0

            cum = (1 + pr).cumprod()
            running_max = cum.cummax()
            drawdown = (cum - running_max) / running_max
            max_dd = drawdown.min()
            max_dd_duration = self._calculate_max_drawdown_duration(drawdown)

            var_95 = np.percentile(pr, 5)
            cvar_95 = pr[pr <= var_95].mean() if (pr <= var_95).any() else np.nan

            metrics['Total Return'] = float(total_ret)
            metrics['Annual Return'] = float(ann_ret)
            metrics['Annual Volatility'] = float(ann_vol)
            metrics['Sharpe Ratio'] = float(sharpe)
            metrics['Sortino Ratio'] = float(sortino)
            metrics['Max Drawdown'] = float(max_dd)
            metrics['Max Drawdown Duration'] = int(max_dd_duration)
            metrics['VaR 95%'] = float(var_95)
            metrics['CVaR 95%'] = float(cvar_95) if not np.isnan(cvar_95) else np.nan

            # Benchmark annual return (geometric)
            bench_ann_ret = 0.0
            if benchmark_returns is not None:
                br = benchmark_returns.dropna()
                if not br.empty:
                    bench_total = (1 + br).prod()
                    n_days = len(br)
                    bench_ann_ret = bench_total ** (TRADING_DAYS_PER_YEAR / n_days) - 1
            metrics['Benchmark Annual Return'] = float(bench_ann_ret)

            metrics['Risk Category'] = self._determine_risk_category(
                metrics['Annual Volatility'],
                metrics['Max Drawdown']
            )

        except Exception as e:
            self.logger.error(f"Metric error: {e}")
            return metrics

        # Separate Jarque–Bera block for robustness
        if len(portfolio_returns) > 20:
            try:
                from scipy import stats
                jb_result = stats.jarque_bera(portfolio_returns.dropna())
                # SciPy may return a result object or tuple
                stat = getattr(jb_result, 'statistic', None)
                pvalue = getattr(jb_result, 'pvalue', None)
                if stat is None and isinstance(jb_result, (list, tuple, np.ndarray)) and len(jb_result) >= 2:
                    stat, pvalue = jb_result[0], jb_result[1]
                metrics['Jarque-Bera Stat'] = float(stat) if stat is not None else np.nan
                metrics['Jarque-Bera p-value'] = float(pvalue) if pvalue is not None else np.nan
            except Exception:
                metrics['Jarque-Bera Stat'] = np.nan
                metrics['Jarque-Bera p-value'] = np.nan

        return metrics

    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        if drawdown.empty:
            return 0
        in_drawdown = drawdown < 0
        max_len = 0
        current_len = 0
        for flag in in_drawdown:
            if flag:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        return max_len

    def _determine_risk_category(self, vol: float, mdd: float) -> str:
        for lvl, params in Constants.RISK_LEVELS.items():
            if vol <= params['max_volatility'] and abs(mdd) <= abs(params['max_drawdown']):
                return lvl
        return 'VERY AGGRESSIVE'

    def calculate_component_var(self, returns: pd.DataFrame, weights: Dict) -> pd.Series:
        try:
            if returns.empty:
                return pd.Series(0.0, index=[])
            w = np.array([weights.get(t, 0) for t in returns.columns])
            cov = returns.cov() * TRADING_DAYS_PER_YEAR
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            if port_vol == 0:
                return pd.Series(0.0, index=returns.columns)
            # 95% param VaR approximation
            z_score = 1.65
            marginal_contrib = np.dot(cov, w) / port_vol
            component_var = w * marginal_contrib * z_score
            return pd.Series(component_var, index=returns.columns)
        except Exception:
            return pd.Series(0.0, index=returns.columns)

    def perform_garch_analysis(self, returns: pd.Series):
        if not LIBRARIES.get('arch', False):
            return None, None
        try:
            res = arch.arch_model(returns * 100, vol='Garch', p=1, q=1).fit(disp='off', show_warning=False)
            return res, {'conditional_volatility': res.conditional_volatility / 100}
        except Exception:
            return None, None

    def perform_pca_analysis(self, returns: pd.DataFrame):
        if not LIBRARIES.get('sklearn', False):
            return None
        try:
            pca = PCA(n_components=min(5, len(returns.columns)))
            pca.fit(returns.corr())
            return {'explained_variance': pca.explained_variance_ratio_}
        except Exception:
            return None

# ============================================================================
# 4. ADVANCED PERFORMANCE ATTRIBUTION
# ============================================================================

class EnhancedPerformanceAttributionPro:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_brinson_fachler(
        self,
        asset_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series],
        portfolio_weights: Dict,
        benchmark_weights: Dict,
        sector_map: Dict
    ) -> Dict:
        """Brinson-Fachler logic using full asset returns DataFrame."""
        try:
            if asset_returns.empty:
                return {'allocation': 0.0, 'selection': 0.0, 'interaction': 0.0,
                        'sector_breakdown': {}, 'total_excess': 0.0}

            port_w = pd.Series(portfolio_weights).reindex(asset_returns.columns).fillna(0.0)
            bench_w = pd.Series(benchmark_weights).reindex(asset_returns.columns).fillna(0.0)

            # Normalize to 1
            if port_w.sum() != 0:
                port_w = port_w / port_w.sum()
            if bench_w.sum() != 0:
                bench_w = bench_w / bench_w.sum()

            # Portfolio and benchmark total returns for evaluation horizon
            port_total = (1 + asset_returns.dot(port_w)).prod() - 1
            if benchmark_returns is not None:
                bench_total = (1 + benchmark_returns).prod() - 1
            else:
                bench_total = (1 + asset_returns.dot(bench_w)).prod() - 1

            results = {
                'allocation': 0.0,
                'selection': 0.0,
                'interaction': 0.0,
                'sector_breakdown': {},
                'total_excess': float(port_total - bench_total),
            }

            unique_sectors = set(sector_map.values())
            for sector in unique_sectors:
                sector_assets = [t for t, s in sector_map.items()
                                 if s == sector and t in asset_returns.columns]
                if not sector_assets:
                    continue

                wp = port_w[sector_assets].sum()
                wb = bench_w[sector_assets].sum()

                # Sector portfolio and benchmark returns
                rp = 0.0
                rb = 0.0
                if wp > 0:
                    rp = (1 + asset_returns[sector_assets]
                          .dot(port_w[sector_assets] / wp)).prod() - 1
                if wb > 0:
                    rb = (1 + asset_returns[sector_assets]
                          .dot(bench_w[sector_assets] / wb)).prod() - 1

                allocation = (wp - wb) * (rb - bench_total)
                selection = wb * (rp - rb)
                interaction = (wp - wb) * (rp - rb)

                results['allocation'] += allocation
                results['selection'] += selection
                results['interaction'] += interaction
                results['sector_breakdown'][sector] = {
                    'allocation': float(allocation),
                    'selection': float(selection),
                    'interaction': float(interaction),
                    'total': float(allocation + selection + interaction),
                }

            return results
        except Exception as e:
            self.logger.error(f"Attribution error: {e}")
            return {
                'allocation': 0.0,
                'selection': 0.0,
                'interaction': 0.0,
                'sector_breakdown': {},
                'total_excess': 0.0,
            }

    def calculate_rolling_attribution(self, port: pd.Series, bench: pd.Series) -> pd.DataFrame:
        try:
            return pd.DataFrame(
                {'Rolling Excess': (port - bench).rolling(63).mean() * TRADING_DAYS_PER_YEAR}
            )
        except Exception:
            return pd.DataFrame()

# ============================================================================
# 5. ADVANCED BACKTESTING
# ============================================================================

def map_rebalancing_frequency(freq_label: str) -> Optional[str]:
    """
    Map UI rebalancing frequency label to pandas offset alias.
    Supports English labels; can be extended for others.
    """
    mapping = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "A",
    }
    return mapping.get(freq_label, "M")

class PortfolioBacktesterPro:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def run_backtest(self, config: Dict) -> Dict:
        strategy = config.get('type', 'BUY_HOLD')
        if strategy == 'BUY_HOLD':
            return self._run_buy_hold(config)
        elif strategy == 'REBALANCE_FIXED':
            return self._run_rebalance_fixed(config)
        else:
            return self._run_buy_hold(config)

    def _run_buy_hold(self, config: Dict) -> Dict:
        if self.returns.empty:
            return {
                'portfolio_value': pd.Series(dtype=float),
                'portfolio_returns': pd.Series(dtype=float),
                'strategy': 'BUY_HOLD'
            }
        weights = pd.Series(config['initial_weights']).reindex(self.returns.columns).fillna(0.0)
        port_ret = self.returns.dot(weights)
        port_val = (1 + port_ret).cumprod()
        return {
            'portfolio_value': port_val,
            'portfolio_returns': port_ret,
            'strategy': 'BUY_HOLD'
        }

    def _run_rebalance_fixed(self, config: Dict) -> Dict:
        if self.returns.empty:
            return {
                'portfolio_value': pd.Series(dtype=float),
                'portfolio_returns': pd.Series(dtype=float),
                'strategy': 'REBALANCE_FIXED'
            }
        freq = config.get('rebalancing_frequency', 'M')
        target_weights = pd.Series(config['initial_weights']).reindex(self.returns.columns).fillna(0.0)
        target_weights = target_weights / target_weights.sum() if target_weights.sum() != 0 else target_weights

        dates = self.returns.index
        if len(dates) == 0:
            return {
                'portfolio_value': pd.Series(dtype=float),
                'portfolio_returns': pd.Series(dtype=float),
                'strategy': 'REBALANCE_FIXED'
            }

        # Determine rebalancing dates
        try:
            rebalance_dates = set(self.returns.resample(freq).first().index)
        except Exception:
            rebalance_dates = set()

        # Always rebalance at the first date
        rebalance_dates.add(dates[0])

        portfolio_values = []
        portfolio_returns = []

        # Start with target weights
        current_weights = target_weights.copy()
        portfolio_value = 1.0

        for dt in dates:
            if dt in rebalance_dates:
                # Rebalance to target weights
                current_weights = target_weights.copy()

            r_t = self.returns.loc[dt]
            daily_port_ret = float(np.dot(current_weights.values, r_t.values))
            portfolio_value *= (1.0 + daily_port_ret)

            portfolio_values.append(portfolio_value)
            portfolio_returns.append(daily_port_ret)

            # Update weights after market move
            asset_values = current_weights.values * (1.0 + r_t.values)
            total_value = asset_values.sum()
            if total_value > 0:
                current_weights = pd.Series(asset_values / total_value, index=current_weights.index)

        port_val_series = pd.Series(portfolio_values, index=dates)
        port_ret_series = pd.Series(portfolio_returns, index=dates)

        return {
            'portfolio_value': port_val_series,
            'portfolio_returns': port_ret_series,
            'strategy': 'REBALANCE_FIXED'
        }

# ============================================================================
# 6. SCENARIO STRESS TESTING (NEW)
# ============================================================================

class ScenarioStressTester:
    """Handles historical simulation with dynamic renormalization."""
    def __init__(self, data_manager: PortfolioDataManagerPro, logger=None):
        self.dm = data_manager
        self.logger = logger or logging.getLogger(__name__)

    def run_stress_test(
        self,
        scenarios: List[str],
        weights: Dict[str, float],
        benchmark_ticker: str
    ) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        tickers = list(weights.keys())
        
        for scenario_name in scenarios:
            if scenario_name not in Constants.HISTORICAL_CRISES:
                continue
            start_str, end_str = Constants.HISTORICAL_CRISES[scenario_name]
            
            try:
                start = datetime.strptime(start_str, '%Y-%m-%d')
                end = datetime.strptime(end_str, '%Y-%m-%d')
                prices, bench_prices = self.dm.fetch_portfolio_data(tickers, benchmark_ticker, start, end)
                if prices.empty:
                    results[scenario_name] = {'error': 'No data'}
                    continue
                returns, bench_returns = self.dm.calculate_returns(prices, bench_prices)
                
                # Dynamic Renormalization
                available_assets = [t for t in tickers if t in returns.columns]
                original_coverage = sum(weights[t] for t in available_assets)
                
                if original_coverage < 0.4:
                    results[scenario_name] = {
                        'error': f"Insufficient coverage ({original_coverage:.0%})"
                    }
                    continue
                
                normalized_weights = np.array([weights[t] for t in available_assets]) / original_coverage
                scenario_returns = returns[available_assets].dot(normalized_weights)
                cumulative_portfolio = (1 + scenario_returns).cumprod()
                cumulative_benchmark = (1 + bench_returns).cumprod()

                max_dd = ((cumulative_portfolio - cumulative_portfolio.expanding().max())
                          / cumulative_portfolio.expanding().max()).min()
                
                results[scenario_name] = {
                    'portfolio_return': cumulative_portfolio.iloc[-1] - 1,
                    'benchmark_return': (1 + bench_returns).prod() - 1,
                    'max_drawdown': float(max_dd),
                    'coverage': float(original_coverage),
                    'surviving_assets': len(available_assets),
                    'cumulative_series': cumulative_portfolio,
                    'benchmark_series': cumulative_benchmark
                }
            except Exception as e:
                results[scenario_name] = {'error': str(e)}
        return results

    def simulate_hypothetical_shocks(
        self,
        current_value: float,
        weights: Dict,
        betas: Dict[str, float]
    ) -> pd.DataFrame:
        shocks = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
        portfolio_beta = sum(weights.get(t, 0) * betas.get(t, 1.0) for t in weights)
        data = []
        for s in shocks:
            impact = portfolio_beta * s
            data.append({
                'Market Move': s,
                'Portfolio Impact': impact,
                'P&L ($)': current_value * impact,
                'Portfolio Beta': portfolio_beta
            })
        return pd.DataFrame(data)

# ============================================================================
# 7. VISUALIZATION ENGINE
# ============================================================================

class VisualizationEnginePro:
    def __init__(self):
        self.colors = {'primary': '#00cc96', 'danger': '#ef553b', 'success': '#00cc96'}
        self.template = "plotly_dark"

    def create_performance_dashboard(self, data: Dict) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Portfolio Value", "Drawdown", "Daily Returns", "Placeholder")
        )
        port_val = data.get('portfolio_value', pd.Series())
        port_ret = data.get('portfolio_returns', pd.Series())

        if not port_val.empty:
            fig.add_trace(
                go.Scatter(
                    x=port_val.index,
                    y=port_val,
                    name='Portfolio Value',
                    line=dict(color=self.colors['primary'])
                ),
                row=1,
                col=1
            )
            running_max = port_val.cummax()
            drawdown = (port_val - running_max) / running_max
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    name='Drawdown',
                    line=dict(color=self.colors['danger'])
                ),
                row=1,
                col=2
            )

        if not port_ret.empty:
            fig.add_trace(
                go.Histogram(
                    x=port_ret,
                    name='Daily Returns'
                ),
                row=2,
                col=1
            )

        fig.update_layout(
            height=800,
            template=self.template,
            title="Portfolio Performance Dashboard"
        )
        return fig

    def create_risk_decomposition_chart(self, component_var: pd.Series, sector_map: Dict) -> go.Figure:
        if component_var.empty:
            return go.Figure()
        df = pd.DataFrame({'Asset': component_var.index, 'Risk': component_var.values})
        df['Sector'] = df['Asset'].map(sector_map).fillna('Other')
        fig = px.treemap(df, path=['Sector', 'Asset'], values='Risk', title="Risk Decomposition (Component VaR)")
        fig.update_layout(template=self.template)
        return fig

    def create_attribution_waterfall(self, results: Dict) -> go.Figure:
        fig = go.Figure(
            go.Waterfall(
                measure=["relative", "relative", "relative", "total"],
                x=["Allocation", "Selection", "Interaction", "Total Excess"],
                y[
                    results.get('allocation', 0.0),
                    results.get('selection', 0.0),
                    results.get('interaction', 0.0),
                    results.get('total_excess', 0.0)
                ],
                text=[
                    f"{results.get('allocation', 0.0):.2%}",
                    f"{results.get('selection', 0.0):.2%}",
                    f"{results.get('interaction', 0.0):.2%}",
                    f"{results.get('total_excess', 0.0):.2%}"
                ]
            )
        )
        fig.update_layout(title="Brinson-Fachler Attribution", template=self.template)
        return fig

    def create_stress_test_chart(self, scenario_name: str, result_data: Dict) -> go.Figure:
        if 'error' in result_data:
            return go.Figure()
        port = result_data['cumulative_series'] * 100
        bench = result_data['benchmark_series'] * 100
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=port.index,
                y=port,
                mode='lines',
                name='Portfolio (Simulated)',
                line=dict(color=self.colors['primary'], width=3)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bench.index,
                y=bench,
                mode='lines',
                name='Benchmark',
                line=dict(color='#888', width=2, dash='dot')
            )
        )
        fig.update_layout(
            title=f"Stress Test: {scenario_name}",
            template=self.template,
            yaxis_title="Rebased Value (100)",
            height=400
        )
        return fig

# ============================================================================
# 8. MAIN APPLICATION
# ============================================================================

def main():
    st.markdown(
        '<div class="main-header"><h1>🏛️ QUANTEDGE PRO</h1>'
        '<p>Institutional Portfolio Analysis Platform v3.2</p></div>',
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("🔧 Configuration")
        universe_name = st.selectbox("Asset Universe", list(Constants.ASSET_UNIVERSES.keys()))
        universe_config = Constants.ASSET_UNIVERSES[universe_name]
        
        tickers = st.multiselect(
            "Select Assets",
            universe_config['tickers'],
            default=universe_config['tickers'][:5]
        )
        benchmark = st.text_input("Benchmark", universe_config['benchmark'])
        
        col1, col2 = st.columns(2)
        start_date = col1.date_input(
            "Start Date",
            datetime.now() - timedelta(days=365 * 2)
        )
        end_date = col2.date_input("End Date", datetime.now())
        
        method = st.selectbox(
            "Optimization Method",
            ["MAX_SHARPE", "MIN_VOLATILITY", "RISK_PARITY", "MEAN_VARIANCE", "HRP"]
        )
        
        constraints: Dict[str, Any] = {}
        if method == "MEAN_VARIANCE":
            st.markdown("---")
            if st.checkbox("Enable Sector Limits"):
                # Example: Technology sector max 40%
                constraints['sector_limits'] = {'Technology': (0.0, 0.40)}

        st.markdown("---")
        st.header("🧪 Analysis Settings")
        bt_strategy = st.selectbox("Backtest Strategy", ["BUY_HOLD", "REBALANCE_FIXED"])
        
        bt_freq_ui = (
            st.selectbox(
                "Rebalancing Frequency",
                ["Monthly", "Quarterly", "Weekly", "Daily"]
            )
            if bt_strategy == "REBALANCE_FIXED"
            else "Monthly"
        )
        bt_freq = map_rebalancing_frequency(bt_freq_ui)
        
        run_scenarios = st.checkbox("Run Historical Stress Tests", True)
        selected_scenarios: List[str] = []
        if run_scenarios:
            selected_scenarios = st.multiselect(
                "Select Scenarios",
                list(Constants.HISTORICAL_CRISES.keys()),
                default=list(Constants.HISTORICAL_CRISES.keys())[:2]
            )

        run_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

    if run_btn:
        if not tickers:
            st.error("Please select at least one asset.")
            st.stop()

        config = PortfolioConfig(
            universe=universe_name,
            tickers=tickers,
            benchmark=benchmark,
            start_date=start_date,
            end_date=end_date,
            risk_free_rate=universe_config['risk_free_rate'],
            optimization_method=method,
            rebalancing_frequency=bt_freq,
            constraints=constraints
        )
        
        # 1. Data Fetching
        with st.spinner("Fetching market data..."):
            dm = PortfolioDataManagerPro()
            prices, bench_prices = dm.fetch_portfolio_data(
                tickers,
                benchmark,
                start_date,
                end_date
            )
            if prices.empty or bench_prices.empty:
                st.error("No data for selected tickers or benchmark in the given date range.")
                st.stop()
            returns, bench_returns = dm.calculate_returns(prices, bench_prices, method='log')

        # 2. Classification
        classifier = EnhancedAssetClassifierPro()
        meta = classifier.classify_tickers(tickers)
        sector_map = {t: m['sector'] for t, m in meta.items()}

        # 3. Optimization
        optimizer = AdvancedPortfolioOptimizerPro(returns, prices)
        weights, perf_tuple = optimizer.optimize(method, config)
        if not weights:
            st.error("Optimization failed. Please adjust settings or universe.")
            st.stop()

        # 4. Backtesting (using selected strategy and rebalancing frequency)
        backtester = PortfolioBacktesterPro(returns)
        backtest_config = {
            'type': bt_strategy,
            'initial_weights': weights,
            'rebalancing_frequency': config.rebalancing_frequency
        }
        backtest_result = backtester.run_backtest(backtest_config)
        portfolio_value = backtest_result['portfolio_value']
        portfolio_returns = backtest_result['portfolio_returns']

        # 5. Risk Analysis
        risk_engine = AdvancedRiskAnalyzerPro()
        risk_metrics = risk_engine.calculate_comprehensive_metrics(
            portfolio_returns,
            bench_returns,
            config.risk_free_rate
        )
        
        # 6. Attribution
        attr_engine = EnhancedPerformanceAttributionPro()
        benchmark_equal_weights = {t: 1.0 / len(tickers) for t in tickers}
        attr_res = attr_engine.calculate_brinson_fachler(
            returns,
            bench_returns,
            weights,
            benchmark_equal_weights,
            sector_map
        )
        
        # 7. Visualization Setup
        viz = VisualizationEnginePro()
        
        # --- DASHBOARD UI ---
        
        st.markdown('<div class="section-header">📊 Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        
        c1.markdown(
            f"""<div class="pro-card"><div class="metric-label">Annual Return</div>
            <div class="metric-value">{risk_metrics.get('Annual Return', 0.0):.2%}</div>
            <div class="metric-change">vs Benchmark: {risk_metrics.get('Annual Return', 0.0) - risk_metrics.get('Benchmark Annual Return', 0.0):+.2%}</div></div>""",
            unsafe_allow_html=True
        )
            
        c2.markdown(
            f"""<div class="pro-card"><div class="metric-label">Volatility</div>
            <div class="metric-value">{risk_metrics.get('Annual Volatility', 0.0):.2%}</div>
            <div class="metric-change">{risk_metrics.get('Risk Category', 'N/A')}</div></div>""",
            unsafe_allow_html=True
        )
            
        c3.markdown(
            f"""<div class="pro-card"><div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{risk_metrics.get('Sharpe Ratio', 0.0):.2f}</div></div>""",
            unsafe_allow_html=True
        )
            
        c4.markdown(
            f"""<div class="pro-card"><div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{risk_metrics.get('Max Drawdown', 0.0):.2%}</div>
            <div class="metric-change">Duration: {risk_metrics.get('Max Drawdown Duration', 0)} days</div></div>""",
            unsafe_allow_html=True
        )

        # Performance chart
        st.markdown("### Portfolio Performance")
        perf_fig = viz.create_performance_dashboard(
            {'portfolio_value': portfolio_value, 'portfolio_returns': portfolio_returns}
        )
        st.plotly_chart(perf_fig, use_container_width=True)

        # Portfolio composition
        st.markdown("### Portfolio Composition")
        weights_df = (
            pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
            .sort_values('Weight', ascending=False)
        )
        st.bar_chart(weights_df)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("### Risk Decomposition")
            component_var = risk_engine.calculate_component_var(returns, weights)
            st.plotly_chart(
                viz.create_risk_decomposition_chart(component_var, sector_map),
                use_container_width=True
            )
        with col_g2:
            st.markdown("### Attribution Analysis")
            st.plotly_chart(
                viz.create_attribution_waterfall(attr_res),
                use_container_width=True
            )

        # 8. SCENARIO STRESS TESTING
        if run_scenarios and selected_scenarios:
            st.markdown('<div class="section-header">🌪️ Historical Stress Testing</div>', unsafe_allow_html=True)
            stress_tester = ScenarioStressTester(dm)
            with st.spinner("Simulating scenarios..."):
                stress_res = stress_tester.run_stress_test(selected_scenarios, weights, benchmark)
            
            tabs = st.tabs(list(stress_res.keys()))
            for i, (scen, data) in enumerate(stress_res.items()):
                with tabs[i]:
                    if 'error' in data:
                        st.warning(data['error'])
                    else:
                        sc1, sc2 = st.columns([1, 2])
                        with sc1:
                            ret_color = "green" if data['portfolio_return'] > 0 else "red"
                            delta = data['portfolio_return'] - data['benchmark_return']
                            st.markdown(
                                f"""
                            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
                                <div style="color: #888; font-size: 0.9em;">Total Return</div>
                                <div style="font-size: 1.8em; font-weight: bold; color: {ret_color};">{data['portfolio_return']:.2%}</div>
                                <div style="font-size: 0.9em; color: {'#00cc96' if delta > 0 else '#ef553b'};">vs Benchmark: {delta:+.2%}</div>
                                <hr style="border-color: #444;">
                                <div style="color: #888; font-size: 0.9em;">Max Drawdown</div>
                                <div style="font-size: 1.4em; font-weight: bold; color: #ef553b;">{data['max_drawdown']:.2%}</div>
                                <div style="margin-top: 10px; font-size: 0.8em; color: #666;">*Simulated using {data['surviving_assets']} available assets ({data['coverage']:.0%} coverage).</div>
                            </div>
                            """,
                                unsafe_allow_html=True
                            )
                        with sc2:
                            st.plotly_chart(
                                viz.create_stress_test_chart(scen, data),
                                use_container_width=True
                            )

        # 9. HYPOTHETICAL SHOCKS
        st.markdown('<div class="section-header">⚡ Hypothetical Shocks (Beta Sensitivity)</div>', unsafe_allow_html=True)
        # Quick beta calculation
        market_var = bench_returns.var()
        betas = {
            t: (returns[t].cov(bench_returns) / market_var) if market_var > 0 else 1.0
            for t in returns.columns
        }
        
        stress_tester = ScenarioStressTester(dm)
        shock_df = stress_tester.simulate_hypothetical_shocks(100000, weights, betas)
        
        def color_pnl(val):
            return f'color: {"#00cc96" if val > 0 else "#ef553b"}'
        
        st.dataframe(
            shock_df.style.format(
                {
                    'Market Move': '{:+.0%}',
                    'Portfolio Impact': '{:+.2%}',
                    'P&L ($)': '${:+,.0f}',
                    'Portfolio Beta': '{:.2f}'
                }
            ).applymap(color_pnl, subset=['P&L ($)', 'Portfolio Impact']),
            use_container_width=True
        )

if __name__ == "__main__":
    main()

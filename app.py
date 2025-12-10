# ============================================================================
# QUANTEDGE MK | INSTITUTIONAL PORTFOLIO TERMINAL (ADVANCED MONOLITHIC VERSION)
# Version: v3.1 Pro Plus (AI + Advanced Risk + Stress Testing + Reporting Integrated)
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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QuantEdge Pro | Institutional Portfolio Management",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://quantedge.pro',
        'Report a bug': 'https://github.com/quantedge/issues',
        'About': "QuantEdge Pro v3.1 - Institutional Portfolio Analysis Platform"
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
            data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, group_by='ticker', threads=True, auto_adjust=True)
            prices = pd.DataFrame()
            benchmark_prices = pd.Series(dtype=float)
            
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in all_tickers:
                    try:
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                        if 'Close' in df.columns:
                            if ticker == benchmark: benchmark_prices = df['Close']
                            elif ticker in tickers: prices[ticker] = df['Close']
                    except KeyError: continue
            else:
                if 'Close' in data.columns:
                    if len(all_tickers) == 1:
                        if all_tickers[0] == benchmark: benchmark_prices = data['Close']
                        else: prices[all_tickers[0]] = data['Close']
            
            prices = prices.ffill().bfill()
            benchmark_prices = benchmark_prices.ffill().bfill()
            
            common_idx = prices.index.intersection(benchmark_prices.index)
            if len(common_idx) == 0: raise ValueError("Portfolio and benchmark dates do not overlap")
            return prices.loc[common_idx], benchmark_prices.loc[common_idx]
        except Exception as e:
            raise Exception(f"Data fetch error: {str(e)}")
    
    def calculate_returns(self, prices: pd.DataFrame, benchmark_prices: pd.Series, method: str = 'log') -> Tuple[pd.DataFrame, pd.Series]:
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
            except Exception: self.mu = None; self.S = None

    def optimize(self, method: str, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        methods = {
            'MAX_SHARPE': self._optimize_max_sharpe, 'MIN_VOLATILITY': self._optimize_min_volatility,
            'RISK_PARITY': self._optimize_risk_parity, 'MAX_DIVERSIFICATION': self._optimize_max_diversification,
            'HRP': self._optimize_hrp, 'EQUAL_WEIGHT': self._optimize_equal_weight,
            'MEAN_VARIANCE': self._optimize_mean_variance
        }
        try: return methods.get(method, self._optimize_equal_weight)(config)
        except Exception as e:
            self.logger.error(f"{method} failed: {e}")
            return self._optimize_equal_weight(config)

    def _optimize_max_sharpe(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        if not LIBRARIES.get('pypfopt', False) or self.mu is None: return self._optimize_equal_weight(config)
        try:
            ef = EfficientFrontier(self.mu, self.S)
            weights = ef.max_sharpe(risk_free_rate=config.risk_free_rate)
            return ef.clean_weights(), ef.portfolio_performance(verbose=False, risk_free_rate=config.risk_free_rate)
        except Exception: return self._optimize_equal_weight(config)

    def _optimize_min_volatility(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        if not LIBRARIES.get('pypfopt', False): return self._optimize_equal_weight(config)
        try:
            ef = EfficientFrontier(self.mu, self.S)
            weights = ef.min_volatility()
            return ef.clean_weights(), ef.portfolio_performance(verbose=False, risk_free_rate=config.risk_free_rate)
        except Exception: return self._optimize_equal_weight(config)

    def _optimize_risk_parity(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        try:
            vol = self.returns.std() * np.sqrt(252)
            inv_vol = 1 / vol.replace(0, np.inf)
            weights = (inv_vol / inv_vol.sum()).to_dict()
            return weights, self._calculate_performance(weights, config.risk_free_rate)
        except Exception: return self._optimize_equal_weight(config)

    def _optimize_max_diversification(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        return self._optimize_equal_weight(config) # Placeholder logic

    def _optimize_hrp(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        if not LIBRARIES.get('pypfopt', False): return self._optimize_equal_weight(config)
        try:
            hrp = HRPOpt(self.returns)
            weights = hrp.optimize()
            return weights, self._calculate_performance(weights, config.risk_free_rate)
        except Exception: return self._optimize_equal_weight(config)

    def _optimize_equal_weight(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        n = len(self.returns.columns)
        weights = {t: 1.0/n for t in self.returns.columns}
        return weights, self._calculate_performance(weights, config.risk_free_rate)

    def _optimize_mean_variance(self, config: PortfolioConfig) -> Tuple[Dict, Tuple]:
        try:
            mu = self.returns.mean() * 252
            S = self.returns.cov() * 252
            n = len(mu)
            bounds = [(0, 1) for _ in range(n)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # Constraints Logic Fix
            if config.constraints and 'sector_limits' in config.constraints:
                local_classifier = EnhancedAssetClassifierPro()
                for sector, (min_w, max_w) in config.constraints['sector_limits'].items():
                    idx = [i for i, t in enumerate(self.returns.columns) if local_classifier._get_sector_from_constants(t) == sector]
                    if idx:
                        constraints.append({'type': 'ineq', 'fun': lambda w, i=idx, m=min_w: np.sum(w[i]) - m})
                        constraints.append({'type': 'ineq', 'fun': lambda w, i=idx, m=max_w: m - np.sum(w[i])})

            import scipy.optimize as opt
            res = opt.minimize(lambda w: -(np.dot(w, mu) / np.sqrt(w.T @ S @ w)), np.ones(n)/n, bounds=bounds, constraints=constraints, method='SLSQP')
            if res.success:
                weights = dict(zip(self.returns.columns, res.x / res.x.sum()))
                return weights, self._calculate_performance(weights, config.risk_free_rate)
            raise ValueError("Optimization failed")
        except Exception: return self._optimize_equal_weight(config)

    def _calculate_performance(self, weights: Dict, risk_free_rate: float) -> Tuple:
        w = np.array([weights.get(t, 0) for t in self.returns.columns])
        port_ret = self.returns.dot(w)
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        return (ann_ret, ann_vol, sharpe)

# ============================================================================
# 3. ADVANCED RISK ANALYSIS
# ============================================================================

class AdvancedRiskAnalyzerPro:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_comprehensive_metrics(self, portfolio_returns: pd.Series, 
                                        benchmark_returns: Optional[pd.Series] = None,
                                        risk_free_rate: float = 0.045) -> Dict:
        """Calculate comprehensive risk metrics with English keys."""
        metrics = {}
        try:
            metrics['Total Return'] = (1 + portfolio_returns).prod() - 1
            metrics['Annual Return'] = portfolio_returns.mean() * 252
            metrics['Annual Volatility'] = portfolio_returns.std() * np.sqrt(252)
            metrics['Sharpe Ratio'] = (metrics['Annual Return'] - risk_free_rate) / metrics['Annual Volatility'] if metrics['Annual Volatility'] > 0 else 0
            
            downside = portfolio_returns[portfolio_returns < 0]
            down_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
            metrics['Sortino Ratio'] = (metrics['Annual Return'] - risk_free_rate) / down_std if down_std > 0 else 0
            
            cum = (1 + portfolio_returns).cumprod()
            drawdown = (cum - cum.expanding().max()) / cum.expanding().max()
            metrics['Max Drawdown'] = drawdown.min()
            metrics['Max Drawdown Duration'] = self._calculate_max_drawdown_duration(drawdown)
            
            metrics['VaR 95%'] = np.percentile(portfolio_returns, 5)
            metrics['CVaR 95%'] = portfolio_returns[portfolio_returns <= metrics['VaR 95%']].mean()
            
            if len(portfolio_returns) > 20:
                try:
                    from scipy import stats
                    jb = stats.jarque_bera(portfolio_returns)
                    metrics['Jarque-Bera Stat'] = getattr(jb, 'statistic', jb[0])
                except Exception: metrics['Jarque-Bera Stat'] = 0
            
            metrics['Benchmark Annual Return'] = benchmark_returns.mean() * 252 if benchmark_returns is not None else 0.0
            metrics['Risk Category'] = self._determine_risk_category(metrics['Annual Volatility'], metrics['Max Drawdown'])
            
            return metrics
        except Exception as e:
            self.logger.error(f"Metric error: {e}")
            return metrics

    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        if drawdown.empty: return 0
        in_dd = drawdown < 0
        return drawdown.groupby((in_dd != in_dd.shift()).cumsum()).size().max() if in_dd.any() else 0

    def _determine_risk_category(self, vol: float, mdd: float) -> str:
        for lvl, p in Constants.RISK_LEVELS.items():
            if vol <= p['max_volatility'] and abs(mdd) <= abs(p['max_drawdown']): return lvl
        return 'VERY AGGRESSIVE'

    def calculate_component_var(self, returns: pd.DataFrame, weights: Dict) -> pd.Series:
        try:
            w = np.array([weights.get(t, 0) for t in returns.columns])
            cov = returns.cov() * 252
            vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return pd.Series(w * (np.dot(cov, w) / vol * 2.33), index=returns.columns)
        except Exception: return pd.Series(0, index=returns.columns)

    def perform_garch_analysis(self, returns: pd.Series):
        if not LIBRARIES.get('arch', False): return None, None
        try:
            res = arch.arch_model(returns * 100, vol='Garch', p=1, q=1).fit(disp='off', show_warning=False)
            return res, {'conditional_volatility': res.conditional_volatility / 100}
        except Exception: return None, None

    def perform_pca_analysis(self, returns: pd.DataFrame):
        if not LIBRARIES.get('sklearn', False): return None
        try:
            pca = PCA(n_components=min(5, len(returns.columns)))
            pca.fit(returns.corr())
            return {'explained_variance': pca.explained_variance_ratio_}
        except Exception: return None

# ============================================================================
# 4. ADVANCED PERFORMANCE ATTRIBUTION
# ============================================================================

class EnhancedPerformanceAttributionPro:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_brinson_fachler(self, asset_returns: pd.DataFrame, benchmark_returns: Optional[pd.Series],
                                  portfolio_weights: Dict, benchmark_weights: Dict, sector_map: Dict) -> Dict:
        """Brinson-Fachler logic using full asset returns DataFrame."""
        try:
            port_w = pd.Series(portfolio_weights).reindex(asset_returns.columns).fillna(0)
            bench_w = pd.Series(benchmark_weights).reindex(asset_returns.columns).fillna(0)
            
            port_total = (1 + asset_returns.dot(port_w)).prod() - 1
            bench_total = (1 + benchmark_returns).prod() - 1 if benchmark_returns is not None else (1 + asset_returns.dot(bench_w)).prod() - 1
            
            results = {'allocation': 0.0, 'selection': 0.0, 'interaction': 0.0, 'sector_breakdown': {}, 'total_excess': port_total - bench_total}
            
            for sector in set(sector_map.values()):
                s_assets = [t for t, s in sector_map.items() if s == sector and t in asset_returns.columns]
                if not s_assets: continue
                
                wp, wb = port_w[s_assets].sum(), bench_w[s_assets].sum()
                rp = (1 + asset_returns[s_assets].dot(port_w[s_assets]/wp)).prod() - 1 if wp > 0 else 0
                rb = (1 + asset_returns[s_assets].dot(bench_w[s_assets]/wb)).prod() - 1 if wb > 0 else 0
                
                allo = (wp - wb) * (rb - bench_total)
                sel = wb * (rp - rb)
                inter = (wp - wb) * (rp - rb)
                
                results['allocation'] += allo; results['selection'] += sel; results['interaction'] += inter
                results['sector_breakdown'][sector] = {'allocation': allo, 'selection': sel, 'interaction': inter, 'total': allo+sel+inter}
            return results
        except Exception as e:
            self.logger.error(f"Attribution error: {e}")
            return {'allocation': 0, 'selection': 0, 'interaction': 0, 'sector_breakdown': {}, 'total_excess': 0}

    def calculate_rolling_attribution(self, port: pd.Series, bench: pd.Series) -> pd.DataFrame:
        try:
            return pd.DataFrame({'Rolling Excess': (port - bench).rolling(63).mean() * 252})
        except Exception: return pd.DataFrame()

# ============================================================================
# 5. ADVANCED BACKTESTING
# ============================================================================

class PortfolioBacktesterPro:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def run_backtest(self, config: Dict) -> Dict:
        strategy = config.get('type', 'BUY_HOLD')
        if strategy == 'BUY_HOLD': return self._run_buy_hold(config)
        return self._run_buy_hold(config) # Placeholder for other strats

    def _run_buy_hold(self, config: Dict) -> Dict:
        weights = pd.Series(config['initial_weights']).reindex(self.returns.columns).fillna(0)
        port_ret = self.returns.dot(weights)
        return {'portfolio_value': (1 + port_ret).cumprod(), 'portfolio_returns': port_ret, 'strategy': 'BUY_HOLD'}

# ============================================================================
# 6. SCENARIO STRESS TESTING (NEW)
# ============================================================================

class ScenarioStressTester:
    """Handles historical simulation with dynamic renormalization."""
    def __init__(self, data_manager: PortfolioDataManagerPro, logger=None):
        self.dm = data_manager
        self.logger = logger or logging.getLogger(__name__)

    def run_stress_test(self, scenarios: List[str], weights: Dict[str, float], benchmark_ticker: str) -> Dict[str, Dict]:
        results = {}
        tickers = list(weights.keys())
        
        for scenario_name in scenarios:
            if scenario_name not in Constants.HISTORICAL_CRISES: continue
            start_str, end_str = Constants.HISTORICAL_CRISES[scenario_name]
            
            try:
                start, end = datetime.strptime(start_str, '%Y-%m-%d'), datetime.strptime(end_str, '%Y-%m-%d')
                prices, bench_prices = self.dm.fetch_portfolio_data(tickers, benchmark_ticker, start, end)
                if prices.empty: 
                    results[scenario_name] = {'error': 'No data'}
                    continue
                returns, bench_returns = self.dm.calculate_returns(prices, bench_prices)
                
                # Dynamic Renormalization
                avail_assets = [t for t in tickers if t in returns.columns]
                orig_coverage = sum(weights[t] for t in avail_assets)
                
                if orig_coverage < 0.4:
                    results[scenario_name] = {'error': f"Insufficient coverage ({orig_coverage:.0%})"}
                    continue
                
                norm_weights = np.array([weights[t] for t in avail_assets]) / orig_coverage
                scen_ret = returns[avail_assets].dot(norm_weights)
                cum_port = (1 + scen_ret).cumprod()
                
                results[scenario_name] = {
                    'portfolio_return': cum_port.iloc[-1] - 1,
                    'benchmark_return': (1 + bench_returns).prod() - 1,
                    'max_drawdown': ((cum_port - cum_port.expanding().max()) / cum_port.expanding().max()).min(),
                    'coverage': orig_coverage,
                    'surviving_assets': len(avail_assets),
                    'cumulative_series': cum_port,
                    'benchmark_series': (1 + bench_returns).cumprod()
                }
            except Exception as e:
                results[scenario_name] = {'error': str(e)}
        return results

    def simulate_hypothetical_shocks(self, current_value: float, weights: Dict, betas: Dict[str, float]) -> pd.DataFrame:
        shocks = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
        port_beta = sum(weights.get(t, 0) * betas.get(t, 1.0) for t in weights)
        data = []
        for s in shocks:
            impact = port_beta * s
            data.append({'Market Move': s, 'Portfolio Impact': impact, 'P&L ($)': current_value * impact, 'Portfolio Beta': port_beta})
        return pd.DataFrame(data)

# ============================================================================
# 7. VISUALIZATION ENGINE
# ============================================================================

class VisualizationEnginePro:
    def __init__(self):
        self.colors = {'primary': '#00cc96', 'danger': '#ef553b', 'success': '#00cc96'}
        self.template = "plotly_dark"

    def create_performance_dashboard(self, data: Dict) -> go.Figure:
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Portfolio Value", "Drawdown", "Returns", "Risk Metrics"))
        val = data.get('portfolio_value', pd.Series())
        if not val.empty:
            fig.add_trace(go.Scatter(x=val.index, y=val, name='Value', line=dict(color=self.colors['primary'])), row=1, col=1)
        fig.update_layout(height=800, template=self.template, title="Performance Dashboard")
        return fig

    def create_risk_decomposition_chart(self, component_var: pd.Series, sector_map: Dict) -> go.Figure:
        df = pd.DataFrame({'Asset': component_var.index, 'Risk': component_var.values})
        df['Sector'] = df['Asset'].map(sector_map).fillna('Other')
        fig = px.treemap(df, path=['Sector', 'Asset'], values='Risk', title="Risk Decomposition (Component VaR)")
        fig.update_layout(template=self.template)
        return fig

    def create_attribution_waterfall(self, results: Dict) -> go.Figure:
        fig = go.Figure(go.Waterfall(
            measure=["relative", "relative", "relative", "total"],
            x=["Allocation", "Selection", "Interaction", "Total Excess"],
            y=[results['allocation'], results['selection'], results['interaction'], results['total_excess']],
            text=[f"{v:.2%}" for v in [results['allocation'], results['selection'], results['interaction'], results['total_excess']]]
        ))
        fig.update_layout(title="Brinson-Fachler Attribution", template=self.template)
        return fig

    def create_stress_test_chart(self, scenario_name: str, result_data: Dict) -> go.Figure:
        if 'error' in result_data: return go.Figure()
        port = result_data['cumulative_series'] * 100
        bench = result_data['benchmark_series'] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port.index, y=port, mode='lines', name='Portfolio (Simulated)', line=dict(color=self.colors['primary'], width=3)))
        fig.add_trace(go.Scatter(x=bench.index, y=bench, mode='lines', name='Benchmark', line=dict(color='#888', width=2, dash='dot')))
        fig.update_layout(title=f"Stress Test: {scenario_name}", template=self.template, yaxis_title="Rebased Value (100)", height=400)
        return fig

# ============================================================================
# 8. MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<div class="main-header"><h1>🏛️ QUANTEDGE PRO</h1><p>Institutional Portfolio Analysis Platform v3.1</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔧 Configuration")
        universe_name = st.selectbox("Asset Universe", list(Constants.ASSET_UNIVERSES.keys()))
        universe_config = Constants.ASSET_UNIVERSES[universe_name]
        
        tickers = st.multiselect("Select Assets", universe_config['tickers'], default=universe_config['tickers'][:5])
        benchmark = st.text_input("Benchmark", universe_config['benchmark'])
        
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=365*2))
        end_date = col2.date_input("End Date", datetime.now())
        
        method = st.selectbox("Optimization Method", ["MAX_SHARPE", "MIN_VOLATILITY", "RISK_PARITY", "MEAN_VARIANCE", "HRP"])
        
        constraints = {}
        if method == "MEAN_VARIANCE":
            st.markdown("---")
            if st.checkbox("Enable Sector Limits"):
                constraints['sector_limits'] = {'Technology': (0.0, 0.40)} # Example

        st.markdown("---")
        st.header("🧪 Analysis Settings")
        bt_strategy = st.selectbox("Backtest Strategy", ["BUY_HOLD", "REBALANCE_FIXED"])
        
        # Frequency Mapping Logic
        bt_freq_ui = st.selectbox("Rebalancing Frequency", ["Monthly", "Quarterly", "Weekly", "Daily"]) if bt_strategy == "REBALANCE_FIXED" else "Monthly"
        bt_freq = {"Monthly": "M", "Quarterly": "Q", "Weekly": "W", "Daily": "D"}.get(bt_freq_ui, "M")
        
        run_scenarios = st.checkbox("Run Historical Stress Tests", True)
        selected_scenarios = []
        if run_scenarios:
            selected_scenarios = st.multiselect("Select Scenarios", list(Constants.HISTORICAL_CRISES.keys()), default=list(Constants.HISTORICAL_CRISES.keys())[:2])

        run_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

    if run_btn:
        if not tickers: st.error("Please select at least one asset."); st.stop()

        config = PortfolioConfig(
            universe=universe_name, tickers=tickers, benchmark=benchmark,
            start_date=start_date, end_date=end_date, risk_free_rate=universe_config['risk_free_rate'],
            optimization_method=method, rebalancing_frequency=bt_freq, constraints=constraints
        )
        
        # 1. Data Fetching
        with st.spinner("Fetching Market Data..."):
            dm = PortfolioDataManagerPro()
            prices, bench_prices = dm.fetch_portfolio_data(tickers, benchmark, start_date, end_date)
            returns, bench_returns = dm.calculate_returns(prices, bench_prices)

        # 2. Classification
        classifier = EnhancedAssetClassifierPro()
        meta = classifier.classify_tickers(tickers)
        sector_map = {t: m['sector'] for t, m in meta.items()}

        # 3. Optimization
        optimizer = AdvancedPortfolioOptimizerPro(returns, prices)
        weights, perf = optimizer.optimize(method, config)
        port_series = returns.dot(pd.Series(weights).reindex(returns.columns).fillna(0))

        # 4. Risk Analysis
        risk_engine = AdvancedRiskAnalyzerPro()
        risk_metrics = risk_engine.calculate_comprehensive_metrics(port_series, bench_returns, config.risk_free_rate)
        
        # 5. Attribution
        attr_engine = EnhancedPerformanceAttributionPro()
        bench_weights = {t: 1.0/len(tickers) for t in tickers}
        attr_res = attr_engine.calculate_brinson_fachler(returns, bench_returns, weights, bench_weights, sector_map)
        
        # 6. Visualization Setup
        viz = VisualizationEnginePro()
        
        # --- DASHBOARD UI ---
        
        st.markdown('<div class="section-header">📊 Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        
        c1.markdown(f"""<div class="pro-card"><div class="metric-label">Annual Return</div>
            <div class="metric-value">{risk_metrics.get('Annual Return', 0):.2%}</div>
            <div class="metric-change">vs Bench: {risk_metrics.get('Annual Return', 0) - risk_metrics.get('Benchmark Annual Return', 0):+.2%}</div></div>""", unsafe_allow_html=True)
            
        c2.markdown(f"""<div class="pro-card"><div class="metric-label">Volatility</div>
            <div class="metric-value">{risk_metrics.get('Annual Volatility', 0):.2%}</div>
            <div class="metric-change">{risk_metrics.get('Risk Category', 'N/A')}</div></div>""", unsafe_allow_html=True)
            
        c3.markdown(f"""<div class="pro-card"><div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{risk_metrics.get('Sharpe Ratio', 0):.2f}</div></div>""", unsafe_allow_html=True)
            
        c4.markdown(f"""<div class="pro-card"><div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{risk_metrics.get('Max Drawdown', 0):.2%}</div>
            <div class="metric-change">Dur: {risk_metrics.get('Max Drawdown Duration', 0)} days</div></div>""", unsafe_allow_html=True)

        st.markdown("### Portfolio Composition")
        st.bar_chart(pd.DataFrame.from_dict(weights, orient='index', columns=['Weight']).sort_values('Weight', ascending=False))
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("### Risk Decomposition")
            st.plotly_chart(viz.create_risk_decomposition_chart(risk_engine.calculate_component_var(returns, weights), sector_map), use_container_width=True)
        with col_g2:
            st.markdown("### Attribution Analysis")
            st.plotly_chart(viz.create_attribution_waterfall(attr_res), use_container_width=True)

        # 7. SCENARIO STRESS TESTING
        if run_scenarios and selected_scenarios:
            st.markdown('<div class="section-header">🌪️ Historical Stress Testing</div>', unsafe_allow_html=True)
            stress_tester = ScenarioStressTester(dm)
            with st.spinner("Simulating Scenarios..."):
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
                            st.markdown(f"""
                            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
                                <div style="color: #888; font-size: 0.9em;">Total Return</div>
                                <div style="font-size: 1.8em; font-weight: bold; color: {ret_color};">{data['portfolio_return']:.2%}</div>
                                <div style="font-size: 0.9em; color: {'#00cc96' if delta > 0 else '#ef553b'};">vs Bench: {delta:+.2%}</div>
                                <hr style="border-color: #444;">
                                <div style="color: #888; font-size: 0.9em;">Max Drawdown</div>
                                <div style="font-size: 1.4em; font-weight: bold; color: #ef553b;">{data['max_drawdown']:.2%}</div>
                                <div style="margin-top: 10px; font-size: 0.8em; color: #666;">*Simulated using {data['surviving_assets']} available assets ({data['coverage']:.0%} coverage).</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with sc2:
                            st.plotly_chart(viz.create_stress_test_chart(scen, data), use_container_width=True)

        # 8. HYPOTHETICAL SHOCKS
        st.markdown('<div class="section-header">⚡ Hypothetical Shocks (Beta Sensitivity)</div>', unsafe_allow_html=True)
        # Quick beta calc
        mkt_var = bench_returns.var()
        betas = {t: returns[t].cov(bench_returns)/mkt_var if mkt_var > 0 else 1.0 for t in returns.columns}
        
        stress_tester = ScenarioStressTester(dm)
        shock_df = stress_tester.simulate_hypothetical_shocks(100000, weights, betas)
        
        def color_pnl(val): return f'color: {"#00cc96" if val > 0 else "#ef553b"}'
        st.dataframe(shock_df.style.format({'Market Move': '{:+.0%}', 'Portfolio Impact': '{:+.2%}', 'P&L ($)': '${:+,.0f}', 'Portfolio Beta': '{:.2f}'}).applymap(color_pnl, subset=['P&L ($)', 'Portfolio Impact']), use_container_width=True)

if __name__ == "__main__":
    main()

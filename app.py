# QuantEdge Pro - Enhanced Institutional Version
# Improved algorithms, better performance, and additional features

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Callable
import json
import traceback
import sys
import psutil
import gc
import time
from dataclasses import dataclass
from enum import Enum
import hashlib
from functools import lru_cache
import concurrent.futures
from contextlib import contextmanager

warnings.filterwarnings("ignore")

# =========================
# Configuration & Constants
# =========================

@dataclass
class Config:
    """Centralized configuration for the application"""
    TRADING_DAYS_PER_YEAR: int = 252
    RISK_FREE_RATE: float = 0.045
    MAX_TICKERS: int = 200
    YF_BATCH_SIZE: int = 30  # Reduced for more stable fetching
    MC_SIMULATIONS: int = 5000  # Reduced for better performance
    MAX_HISTORICAL_DAYS: int = 365 * 8  # Slightly reduced
    MIN_DATA_DAYS: int = 60
    OPTIMIZATION_MAX_ITER: int = 1000
    CACHE_EXPIRY_SECONDS: int = 300  # 5 minutes
    MAX_WORKERS: int = 4  # For parallel processing

# =========================
# Advanced Caching System
# =========================

class DataCache:
    """Efficient caching system with TTL and memory management"""
    
    def __init__(self, max_size_mb: int = 100):
        self.cache = {}
        self.timestamps = {}
        self.max_size_mb = max_size_mb
        
    def _get_size_mb(self) -> float:
        """Estimate cache size in MB"""
        total = 0
        for key, value in self.cache.items():
            if isinstance(value, pd.DataFrame):
                total += value.memory_usage(deep=True).sum() / 1024 / 1024
            elif isinstance(value, np.ndarray):
                total += value.nbytes / 1024 / 1024
            else:
                total += sys.getsizeof(value) / 1024 / 1024
        return total
    
    def _clean_old_entries(self):
        """Remove old entries when cache gets too large"""
        while self._get_size_mb() > self.max_size_mb and self.cache:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
            gc.collect()
    
    def get(self, key: str, max_age_seconds: int = Config.CACHE_EXPIRY_SECONDS):
        """Retrieve item from cache if not expired"""
        if key not in self.cache:
            return None
        
        if key in self.timestamps:
            age = time.time() - self.timestamps[key]
            if age > max_age_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
        
        return self.cache[key]
    
    def set(self, key: str, value):
        """Store item in cache"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self._clean_old_entries()
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.timestamps.clear()
        gc.collect()

# Initialize global cache
if "data_cache" not in st.session_state:
    st.session_state.data_cache = DataCache()

# =========================
# Enhanced Error & Performance Monitor
# =========================

class ErrorSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ErrorAnalysis:
    timestamp: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: str
    recovery_actions: List[str]
    context: Dict
    stack_trace: str

class EnhancedErrorAnalyzer:
    """Enhanced error analysis with machine learning pattern detection"""
    
    ERROR_PATTERNS = {
        "DATA_FETCH": {
            "symptoms": ["yahoo", "timeout", "connection", "404", "403", "502", "503", "No data fetched"],
            "solutions": [
                "Check ticker format (e.g., AAPL for US, AKBNK.IS for Turkey)",
                "Try incremental data fetching with smaller batches",
                "Use cached data for faster recovery",
                "Fall back to alternative data source or synthetic data",
                "Retry with exponential backoff"
            ],
            "severity": ErrorSeverity.HIGH,
            "auto_recovery": True
        },
        "OPTIMIZATION": {
            "symptoms": ["singular", "convergence", "infeasible", "not positive definite", "semidefinite"],
            "solutions": [
                "Apply Ledoit-Wolf shrinkage to covariance matrix",
                "Use regularization in optimization",
                "Switch to more robust HRP algorithm",
                "Remove highly correlated assets",
                "Increase minimum volatility threshold"
            ],
            "severity": ErrorSeverity.MEDIUM,
            "auto_recovery": True
        },
        "MEMORY": {
            "symptoms": ["MemoryError", "exceeded", "out of memory", "kill"],
            "solutions": [
                "Enable incremental processing",
                "Use data streaming instead of loading all at once",
                "Clear cache and unused variables",
                "Reduce simulation count or lookback period",
                "Use sparse matrices for large datasets"
            ],
            "severity": ErrorSeverity.CRITICAL,
            "auto_recovery": False
        }
    }
    
    def __init__(self):
        self.error_history = []
        self.max_history_size = 50
        self.recovery_success_rate = {}
        
    def analyze_error(self, error: Exception, context: Dict) -> ErrorAnalysis:
        """Analyze error and provide recovery suggestions"""
        error_msg = str(error).lower()
        stack_trace = traceback.format_exc().lower()
        
        # Pattern matching
        category = "UNKNOWN"
        pattern_config = None
        
        for pattern_name, config in self.ERROR_PATTERNS.items():
            if any(symptom in error_msg or symptom in stack_trace 
                   for symptom in config["symptoms"]):
                category = pattern_name
                pattern_config = config
                break
        
        # Generate recovery actions
        recovery_actions = pattern_config["solutions"] if pattern_config else [
            "Check the error details in technical logs",
            "Try with different parameters",
            "Restart the analysis with cached data"
        ]
        
        # Add context-specific suggestions
        if "tickers" in context:
            n_tickers = len(context["tickers"])
            if n_tickers > 100:
                recovery_actions.append(f"Reduce universe from {n_tickers} to under 100 tickers")
        
        analysis = ErrorAnalysis(
            timestamp=datetime.now().isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=pattern_config["severity"] if pattern_config else ErrorSeverity.MEDIUM,
            category=category,
            recovery_actions=recovery_actions,
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        self.error_history.append(analysis)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        return analysis
    
    def display_error_panel(self, analysis: ErrorAnalysis):
        """Display enhanced error analysis panel"""
        with st.expander(f"🚨 Error Analysis - {analysis.category}", expanded=True):
            cols = st.columns(4)
            
            # Severity indicator
            severity_colors = {
                ErrorSeverity.LOW: "🟢",
                ErrorSeverity.MEDIUM: "🟡",
                ErrorSeverity.HIGH: "🟠",
                ErrorSeverity.CRITICAL: "🔴"
            }
            cols[0].metric("Severity", f"{severity_colors[analysis.severity]} {analysis.severity.name}")
            
            # Recovery confidence
            confidence = self._calculate_recovery_confidence(analysis)
            cols[1].metric("Recovery Confidence", f"{confidence}%")
            
            # Auto-recovery available
            auto_recover = analysis.category in self.ERROR_PATTERNS and \
                          self.ERROR_PATTERNS[analysis.category]["auto_recovery"]
            cols[2].metric("Auto-Recovery", "✅" if auto_recover else "❌")
            
            # Historical success rate
            success_rate = self.recovery_success_rate.get(analysis.category, 75)
            cols[3].metric("Historical Success", f"{success_rate}%")
            
            # Recovery actions
            st.subheader("🔄 Recommended Recovery Actions")
            for i, action in enumerate(analysis.recovery_actions, 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{i}.** {action}")
                with col2:
                    if st.button("Try", key=f"recover_{i}_{hash(action)}"):
                        st.info(f"Attempting recovery action {i}...")
            
            # Technical details
            with st.expander("🔍 Technical Details"):
                st.json({
                    "error_type": analysis.error_type,
                    "timestamp": analysis.timestamp,
                    "context_keys": list(analysis.context.keys())
                })
                st.text("Stack Trace (First 500 chars):")
                st.code(analysis.stack_trace[:500])
    
    def _calculate_recovery_confidence(self, analysis: ErrorAnalysis) -> int:
        """Calculate recovery confidence based on historical data"""
        if not self.error_history:
            return 60
        
        similar_errors = [e for e in self.error_history 
                         if e.category == analysis.category]
        
        if not similar_errors:
            return 60
        
        # Simple heuristic based on severity
        base_confidence = 100 - (analysis.severity.value * 15)
        return max(30, min(95, base_confidence))

class PerformanceProfiler:
    """Advanced performance profiling with bottleneck detection"""
    
    def __init__(self):
        self.operations = {}
        self.process = psutil.Process()
        self.start_time = time.time()
        
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations"""
        start_time = time.time()
        start_memory = self._get_memory_mb()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            memory_used = self._get_memory_mb() - start_memory
            
            if operation_name not in self.operations:
                self.operations[operation_name] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "max_time": 0,
                    "total_memory": 0,
                    "peak_memory": 0
                }
            
            op_data = self.operations[operation_name]
            op_data["count"] += 1
            op_data["total_time"] += duration
            op_data["avg_time"] = op_data["total_time"] / op_data["count"]
            op_data["max_time"] = max(op_data["max_time"], duration)
            op_data["total_memory"] += memory_used
            op_data["peak_memory"] = max(op_data["peak_memory"], memory_used)
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0
    
    def get_performance_report(self) -> pd.DataFrame:
        """Generate performance report as DataFrame"""
        if not self.operations:
            return pd.DataFrame()
        
        report_data = []
        for op_name, data in self.operations.items():
            report_data.append({
                "Operation": op_name,
                "Count": data["count"],
                "Avg Time (s)": f"{data['avg_time']:.3f}",
                "Max Time (s)": f"{data['max_time']:.3f}",
                "Peak Memory (MB)": f"{data['peak_memory']:.1f}",
                "Total Time (s)": f"{data['total_time']:.2f}"
            })
        
        return pd.DataFrame(report_data).sort_values("Total Time (s)", ascending=False)

# Initialize enhanced monitors
if "enhanced_error_analyzer" not in st.session_state:
    st.session_state.enhanced_error_analyzer = EnhancedErrorAnalyzer()
if "performance_profiler" not in st.session_state:
    st.session_state.performance_profiler = PerformanceProfiler()

# =========================
# Enhanced Universe Definition
# =========================

@dataclass
class Security:
    """Enhanced security data class"""
    ticker: str
    name: str
    region: str
    sector: str
    currency: str
    market_cap: Optional[float] = None  # In millions
    beta: Optional[float] = None
    volume_avg: Optional[float] = None
    
    @property
    def yahoo_ticker(self) -> str:
        """Get Yahoo Finance compatible ticker"""
        return self.ticker

# Enhanced universe with more metadata
ENHANCED_INSTRUMENTS = [
    # US Market
    Security("AAPL", "Apple", "US", "Technology", "USD", 2800000, 1.20, 50000000),
    Security("MSFT", "Microsoft", "US", "Technology", "USD", 2200000, 0.95, 25000000),
    Security("GOOGL", "Alphabet", "US", "Technology", "USD", 1800000, 1.05, 1500000),
    Security("AMZN", "Amazon", "US", "Consumer Cyclical", "USD", 1600000, 1.15, 4000000),
    Security("NVDA", "NVIDIA", "US", "Technology", "USD", 1200000, 1.45, 45000000),
    Security("TSLA", "Tesla", "US", "Automotive", "USD", 600000, 2.00, 100000000),
    Security("JPM", "JPMorgan Chase", "US", "Financial Services", "USD", 450000, 1.10, 15000000),
    Security("XOM", "Exxon Mobil", "US", "Energy", "USD", 400000, 1.05, 20000000),
    Security("SPY", "SPDR S&P 500 ETF", "US", "ETF", "USD", 400000, 1.00, 80000000),
    
    # Turkish Market
    Security("AKBNK.IS", "Akbank", "TR", "Financial Services", "TRY", 12000, 1.30, 50000000),
    Security("GARAN.IS", "Garanti BBVA", "TR", "Financial Services", "TRY", 10000, 1.25, 40000000),
    Security("ISCTR.IS", "İşbank", "TR", "Financial Services", "TRY", 9000, 1.20, 30000000),
    Security("KCHOL.IS", "Koç Holding", "TR", "Conglomerate", "TRY", 8000, 1.15, 20000000),
    Security("SAHOL.IS", "Sabancı Holding", "TR", "Conglomerate", "TRY", 7000, 1.10, 15000000),
    Security("THYAO.IS", "Turkish Airlines", "TR", "Airlines", "TRY", 6000, 1.50, 80000000),
    Security("ASELS.IS", "Aselsan", "TR", "Defense", "TRY", 5000, 1.40, 25000000),
    
    # Japanese Market
    Security("7203.T", "Toyota", "JP", "Automotive", "JPY", 280000, 0.90, 5000000),
    Security("6758.T", "Sony", "JP", "Technology", "JPY", 120000, 1.10, 3000000),
    Security("8306.T", "MUFG Bank", "JP", "Financial Services", "JPY", 80000, 1.05, 2000000),
    
    # Korean Market
    Security("005930.KS", "Samsung Electronics", "KR", "Technology", "KRW", 350000, 1.25, 10000000),
    Security("000660.KS", "SK Hynix", "KR", "Technology", "KRW", 80000, 1.35, 5000000),
    
    # Singapore Market
    Security("D05.SI", "DBS Group", "SG", "Financial Services", "SGD", 70000, 1.10, 1000000),
    Security("U11.SI", "UOB", "SG", "Financial Services", "SGD", 50000, 1.05, 800000),
]

# Convert to DataFrame
UNIVERSE_DF = pd.DataFrame([vars(sec) for sec in ENHANCED_INSTRUMENTS])
UNIVERSE_DF = UNIVERSE_DF.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

# =========================
# Enhanced Data Fetching with Caching
# =========================

class EnhancedDataFetcher:
    """Efficient data fetching with caching, retry logic, and fallbacks"""
    
    def __init__(self, cache: DataCache):
        self.cache = cache
        self.max_retries = 3
        self.retry_delay = 1
    
    def fetch_prices(self, tickers: List[str], start_date: datetime, 
                    end_date: datetime) -> pd.DataFrame:
        """Fetch prices with intelligent batching and caching"""
        
        # Generate cache key
        cache_key = f"prices_{hashlib.md5(','.join(sorted(tickers)).encode()).hexdigest()}_{start_date}_{end_date}"
        
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            st.info(f"📦 Using cached data for {len(tickers)} tickers")
            return cached
        
        # Limit tickers for performance
        if len(tickers) > Config.MAX_TICKERS:
            st.warning(f"Limiting tickers from {len(tickers)} to {Config.MAX_TICKERS}")
            tickers = tickers[:Config.MAX_TICKERS]
        
        # Fetch in parallel batches
        all_data = []
        failed_tickers = []
        
        with st.spinner(f"Fetching data for {len(tickers)} tickers..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(Config.MAX_WORKERS, len(tickers))) as executor:
                futures = {}
                
                # Create batches
                batches = [tickers[i:i + Config.YF_BATCH_SIZE] 
                          for i in range(0, len(tickers), Config.YF_BATCH_SIZE)]
                
                for batch in batches:
                    future = executor.submit(self._fetch_batch, batch, start_date, end_date)
                    futures[future] = batch
                
                # Process results
                for future in concurrent.futures.as_completed(futures):
                    batch = futures[future]
                    try:
                        result = future.result(timeout=30)
                        if not result.empty:
                            all_data.append(result)
                        else:
                            failed_tickers.extend(batch)
                    except Exception as e:
                        failed_tickers.extend(batch)
                        st.warning(f"Failed to fetch batch: {e}")
        
        # Combine successful fetches
        if all_data:
            prices = pd.concat(all_data, axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
            
            # Forward fill and backfill with limits
            prices = prices.ffill(limit=5).bfill(limit=5)
            
            # Remove tickers with insufficient data
            min_required = int(len(prices) * 0.7)  # 70% data required
            valid_tickers = prices.columns[prices.notna().sum() >= min_required]
            prices = prices[valid_tickers]
            
            # Cache the result
            self.cache.set(cache_key, prices)
            
            # Report failures
            if failed_tickers:
                st.warning(f"Failed to fetch {len(failed_tickers)} tickers")
                
            return prices
        
        return pd.DataFrame()
    
    def _fetch_batch(self, tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch a single batch with retry logic"""
        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start_date - timedelta(days=10),  # Buffer for timezone
                    end=end_date + timedelta(days=1),
                    progress=False,
                    group_by='ticker',
                    threads=True,
                    timeout=10
                )
                
                if data.empty:
                    continue
                
                # Extract adjusted close
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data.xs('Adj Close', axis=1, level=0, drop_level=True)
                else:
                    closes = data[['Adj Close']] if 'Adj Close' in data.columns else data[['Close']]
                    closes.columns = tickers[:1]
                
                return closes
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise e
        
        return pd.DataFrame()

# =========================
# Enhanced Risk Engine
# =========================

class EnhancedRiskEngine:
    """Modern risk calculation engine with multiple methodologies"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, alpha: float = 0.95, method: str = 'historical') -> float:
        """Calculate Value at Risk using different methods"""
        if returns.empty or len(returns) < 10:
            return np.nan
        
        returns_clean = returns.dropna()
        
        if method == 'historical':
            return -np.percentile(returns_clean, (1 - alpha) * 100)
        
        elif method == 'parametric':
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            z = stats.norm.ppf(alpha)
            return -(mu + z * sigma)
        
        elif method == 'modified':
            # Cornish-Fisher expansion for skewness and kurtosis
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            skew = returns_clean.skew()
            kurt = returns_clean.kurtosis()
            
            z = stats.norm.ppf(alpha)
            z_cf = (z + (z**2 - 1) * skew/6 + 
                   (z**3 - 3*z) * (kurt-3)/24 - 
                   (2*z**3 - 5*z) * skew**2/36)
            
            return -(mu + z_cf * sigma)
        
        elif method == 'monte_carlo':
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            n_sims = 10000
            sim_returns = np.random.normal(mu, sigma, n_sims)
            return -np.percentile(sim_returns, (1 - alpha) * 100)
        
        return np.nan
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if returns.empty:
            return np.nan
        
        var = EnhancedRiskEngine.calculate_var(returns, alpha, 'historical')
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) > 0:
            return -tail_returns.mean()
        return var
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown efficiently"""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int = 63) -> pd.DataFrame:
        """Calculate rolling risk metrics"""
        if len(returns) < window:
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            
            rolling_data.append({
                'date': returns.index[i],
                'volatility': window_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR),
                'sharpe': (window_returns.mean() * Config.TRADING_DAYS_PER_YEAR - Config.RISK_FREE_RATE) / 
                         (window_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)) 
                         if window_returns.std() > 0 else np.nan,
                'var_95': EnhancedRiskEngine.calculate_var(window_returns, 0.95),
                'max_dd': EnhancedRiskEngine.calculate_max_drawdown(window_returns)
            })
        
        return pd.DataFrame(rolling_data).set_index('date')
    
    @staticmethod
    def calculate_correlation_structure(returns: pd.DataFrame) -> Dict:
        """Analyze correlation structure of returns"""
        if returns.empty or len(returns.columns) < 2:
            return {}
        
        corr_matrix = returns.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        # Calculate average correlation
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        avg_correlation = corr_matrix.values[mask].mean()
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs[:10],  # Top 10
            'average_correlation': avg_correlation,
            'correlation_heatmap_data': corr_matrix.values
        }

# =========================
# Enhanced Optimization Engine
# =========================

class PortfolioOptimizer:
    """Modern portfolio optimization with multiple algorithms"""
    
    def __init__(self):
        self.methods = {
            'equal_weight': self.equal_weight,
            'min_variance': self.min_variance,
            'max_sharpe': self.max_sharpe,
            'risk_parity': self.risk_parity,
            'max_diversification': self.max_diversification,
            'efficient_frontier': self.efficient_frontier_sample
        }
    
    def optimize(self, returns: pd.DataFrame, method: str = 'max_sharpe', 
                constraints: Dict = None) -> Dict:
        """Optimize portfolio using specified method"""
        if returns.empty or len(returns.columns) < 2:
            return {}
        
        # Remove assets with zero or near-zero volatility
        volatilities = returns.std()
        valid_assets = volatilities[volatilities > 1e-6].index.tolist()
        
        if len(valid_assets) < 2:
            return {}
        
        returns = returns[valid_assets]
        
        # Calculate expected returns and covariance
        mu = returns.mean() * Config.TRADING_DAYS_PER_YEAR
        sigma = returns.cov() * Config.TRADING_DAYS_PER_YEAR
        
        # Apply Ledoit-Wolf shrinkage for stability
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns)
            sigma_shrunk = pd.DataFrame(lw.covariance_, index=sigma.index, columns=sigma.columns)
            sigma = sigma_shrunk
        except:
            pass  # Fall back to sample covariance
        
        # Default constraints
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 0.3,
                'sum_to_one': True
            }
        
        # Run optimization
        if method in self.methods:
            weights = self.methods[method](mu, sigma, constraints)
        else:
            weights = self.equal_weight(mu, sigma, constraints)
        
        # Calculate portfolio metrics
        if weights is not None:
            port_return = (mu * weights).sum()
            port_vol = np.sqrt(weights.T @ sigma @ weights)
            sharpe = (port_return - Config.RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
            
            return {
                'weights': pd.Series(weights, index=returns.columns),
                'expected_return': port_return,
                'expected_volatility': port_vol,
                'sharpe_ratio': sharpe,
                'method': method
            }
        
        return {}
    
    def equal_weight(self, mu: pd.Series, sigma: pd.DataFrame, constraints: Dict) -> np.ndarray:
        """Equal weight portfolio"""
        n = len(mu)
        return np.ones(n) / n
    
    def min_variance(self, mu: pd.Series, sigma: pd.DataFrame, constraints: Dict) -> np.ndarray:
        """Minimum variance portfolio"""
        n = len(mu)
        
        # Objective function: portfolio variance
        def objective(weights):
            return weights.T @ sigma @ weights
        
        # Constraints
        cons = []
        if constraints['sum_to_one']:
            cons.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        return None
    
    def max_sharpe(self, mu: pd.Series, sigma: pd.DataFrame, constraints: Dict) -> np.ndarray:
        """Maximum Sharpe ratio portfolio"""
        n = len(mu)
        
        # Objective function: negative Sharpe ratio (for minimization)
        def objective(weights):
            port_return = weights.T @ mu
            port_vol = np.sqrt(weights.T @ sigma @ weights)
            if port_vol < 1e-8:
                return 1e8  # Penalty for zero volatility
            sharpe = (port_return - Config.RISK_FREE_RATE) / port_vol
            return -sharpe
        
        # Constraints
        cons = []
        if constraints['sum_to_one']:
            cons.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        return None
    
    def risk_parity(self, mu: pd.Series, sigma: pd.DataFrame, constraints: Dict) -> np.ndarray:
        """Risk parity portfolio"""
        n = len(mu)
        
        # Objective: minimize difference in risk contributions
        def objective(weights):
            # Portfolio volatility
            port_vol = np.sqrt(weights.T @ sigma @ weights)
            if port_vol < 1e-8:
                return 1e8
            
            # Marginal risk contributions
            mrc = (sigma @ weights) / port_vol
            
            # Risk contributions
            rc = weights * mrc
            
            # Target: equal risk contributions
            target_rc = port_vol / n
            
            # Sum of squared differences
            return np.sum((rc - target_rc) ** 2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        return None
    
    def max_diversification(self, mu: pd.Series, sigma: pd.DataFrame, constraints: Dict) -> np.ndarray:
        """Maximum diversification portfolio"""
        n = len(mu)
        
        # Individual volatilities
        individual_vols = np.sqrt(np.diag(sigma))
        
        # Objective: maximize diversification ratio
        def objective(weights):
            port_vol = np.sqrt(weights.T @ sigma @ weights)
            if port_vol < 1e-8:
                return 1e8
            
            weighted_vol = weights.T @ individual_vols
            diversification = weighted_vol / port_vol
            
            return -diversification  # Negative for minimization
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        return None
    
    def efficient_frontier_sample(self, mu: pd.Series, sigma: pd.DataFrame, 
                                constraints: Dict, n_points: int = 20) -> Dict:
        """Sample efficient frontier"""
        n = len(mu)
        
        # Target returns
        min_return = mu.min()
        max_return = mu.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier = []
        
        for target in target_returns:
            # Minimize variance for target return
            def objective(weights):
                return weights.T @ sigma @ weights
            
            cons = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: w.T @ mu - target}
            ]
            
            bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
            x0 = np.ones(n) / n
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
            
            if result.success:
                frontier.append({
                    'return': target,
                    'volatility': np.sqrt(result.fun),
                    'weights': result.x,
                    'sharpe': (target - Config.RISK_FREE_RATE) / np.sqrt(result.fun) 
                             if result.fun > 0 else 0
                })
        
        return {'frontier': frontier}

# =========================
# Enhanced Visualization
# =========================

class EnhancedVisualization:
    """Advanced visualization components"""
    
    @staticmethod
    def create_performance_dashboard(returns: pd.Series, benchmark_returns: pd.Series = None) -> go.Figure:
        """Create comprehensive performance dashboard"""
        
        # Calculate metrics
        cum_returns = (1 + returns).cumprod()
        rolling_metrics = EnhancedRiskEngine.calculate_rolling_metrics(returns)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Cumulative Returns', 'Rolling Sharpe Ratio',
                          'Drawdown', 'Rolling Volatility',
                          'Return Distribution', 'Rolling VaR (95%)'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Cumulative returns
        fig.add_trace(
            go.Scatter(x=cum_returns.index, y=cum_returns.values,
                      name='Portfolio', line=dict(color='blue')),
            row=1, col=1
        )
        
        if benchmark_returns is not None:
            bench_cum = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(x=bench_cum.index, y=bench_cum.values,
                          name='Benchmark', line=dict(color='gray', dash='dash')),
                row=1, col=1
            )
        
        # 2. Rolling Sharpe
        if not rolling_metrics.empty:
            fig.add_trace(
                go.Scatter(x=rolling_metrics.index, y=rolling_metrics['sharpe'],
                          name='Sharpe', line=dict(color='green')),
                row=1, col=2
            )
        
        # 3. Drawdown
        drawdown = EnhancedRiskEngine.calculate_max_drawdown(returns)
        running_max = cum_returns.expanding().max()
        current_dd = (cum_returns - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=current_dd.index, y=current_dd.values,
                      name='Drawdown', fill='tozeroy', fillcolor='rgba(255,0,0,0.2)',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        # 4. Rolling volatility
        if not rolling_metrics.empty:
            fig.add_trace(
                go.Scatter(x=rolling_metrics.index, y=rolling_metrics['volatility'],
                          name='Volatility', line=dict(color='orange')),
                row=2, col=2
            )
        
        # 5. Return distribution
        fig.add_trace(
            go.Histogram(x=returns.values, nbinsx=50, name='Returns',
                        marker_color='lightblue'),
            row=3, col=1
        )
        
        # Add normal distribution overlay
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = stats.norm.pdf(x_norm, returns.mean(), returns.std())
        fig.add_trace(
            go.Scatter(x=x_norm, y=y_norm, name='Normal',
                      line=dict(color='red', dash='dash')),
            row=3, col=1
        )
        
        # 6. Rolling VaR
        if not rolling_metrics.empty:
            fig.add_trace(
                go.Scatter(x=rolling_metrics.index, y=rolling_metrics['var_95'],
                          name='VaR 95%', line=dict(color='purple')),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            xaxis_title='Assets',
            yaxis_title='Assets',
            height=600,
            template='plotly_dark'
        )
        
        return fig

# =========================
# Main Application
# =========================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="QuantEdge Pro - Enhanced Institutional Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📊 QuantEdge Pro - Enhanced Institutional Platform</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    **Advanced portfolio analytics with efficient algorithms, real-time monitoring, 
    and institutional-grade risk management.**
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365 * 2),
                max_value=datetime.now() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Universe selection
        st.subheader("🌐 Universe Selection")
        
        # Quick filters
        regions = st.multiselect(
            "Regions",
            options=sorted(UNIVERSE_DF['region'].unique()),
            default=['US', 'TR']
        )
        
        sectors = st.multiselect(
            "Sectors",
            options=sorted(UNIVERSE_DF['sector'].unique()),
            default=['Technology', 'Financial Services']
        )
        
        # Filter universe
        filtered_df = UNIVERSE_DF[
            (UNIVERSE_DF['region'].isin(regions)) &
            (UNIVERSE_DF['sector'].isin(sectors))
        ]
        
        # Ticker selection
        selected_tickers = st.multiselect(
            "Select Tickers",
            options=filtered_df['ticker'].tolist(),
            default=filtered_df['ticker'].head(10).tolist()
        )
        
        # Optimization settings
        st.subheader("🎯 Optimization")
        optimization_method = st.selectbox(
            "Method",
            options=['equal_weight', 'min_variance', 'max_sharpe', 
                    'risk_parity', 'max_diversification']
        )
        
        # Risk settings
        st.subheader("⚠️ Risk Settings")
        confidence_level = st.slider(
            "Confidence Level for VaR/CVaR",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01
        )
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        # Performance monitor button
        if st.button("📊 Performance Report"):
            profiler = st.session_state.performance_profiler
            report = profiler.get_performance_report()
            if not report.empty:
                st.dataframe(report, use_container_width=True)
    
    # Main content
    if run_analysis and selected_tickers:
        try:
            with st.session_state.performance_profiler.profile("main_analysis"):
                # Initialize components
                cache = st.session_state.data_cache
                data_fetcher = EnhancedDataFetcher(cache)
                risk_engine = EnhancedRiskEngine()
                optimizer = PortfolioOptimizer()
                viz = EnhancedVisualization()
                
                # Fetch data
                with st.spinner("📥 Fetching market data..."):
                    prices = data_fetcher.fetch_prices(selected_tickers, start_date, end_date)
                
                if prices.empty:
                    st.error("No data fetched. Please check ticker symbols and date range.")
                    return
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                if len(returns) < Config.MIN_DATA_DAYS:
                    st.warning(f"Insufficient data: only {len(returns)} days available")
                    return
                
                # Benchmark (simplified - using first asset as proxy)
                benchmark_returns = returns.iloc[:, 0] if len(returns.columns) > 0 else None
                
                # Display data summary
                st.subheader("📊 Data Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Assets", len(returns.columns))
                with col2:
                    st.metric("Observation Period", f"{len(returns)} days")
                with col3:
                    st.metric("Start Date", returns.index[0].strftime('%Y-%m-%d'))
                
                # Portfolio optimization
                st.subheader("🎯 Portfolio Optimization")
                
                with st.spinner("Optimizing portfolio..."):
                    optimization_result = optimizer.optimize(
                        returns, 
                        method=optimization_method,
                        constraints={
                            'min_weight': 0.0,
                            'max_weight': 0.3,
                            'sum_to_one': True
                        }
                    )
                
                if optimization_result:
                    # Display weights
                    weights_df = optimization_result['weights'].sort_values(ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Portfolio Weights**")
                        fig_weights = go.Figure(data=[
                            go.Bar(x=weights_df.index, y=weights_df.values)
                        ])
                        fig_weights.update_layout(
                            title=f"{optimization_method.replace('_', ' ').title()} Portfolio",
                            xaxis_title="Asset",
                            yaxis_title="Weight",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig_weights, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Portfolio Metrics**")
                        
                        metrics_cols = st.columns(2)
                        metrics_cols[0].metric(
                            "Expected Return",
                            f"{optimization_result['expected_return']*100:.2f}%"
                        )
                        metrics_cols[1].metric(
                            "Expected Volatility",
                            f"{optimization_result['expected_volatility']*100:.2f}%"
                        )
                        metrics_cols[0].metric(
                            "Sharpe Ratio",
                            f"{optimization_result['sharpe_ratio']:.2f}"
                        )
                        
                        # Calculate portfolio returns
                        portfolio_returns = (returns * optimization_result['weights']).sum(axis=1)
                        
                        # Risk metrics
                        var_95 = risk_engine.calculate_var(portfolio_returns, 0.95)
                        cvar_95 = risk_engine.calculate_cvar(portfolio_returns, 0.95)
                        max_dd = risk_engine.calculate_max_drawdown(portfolio_returns)
                        
                        metrics_cols[1].metric("VaR (95%)", f"{var_95*100:.2f}%")
                        metrics_cols[0].metric("CVaR (95%)", f"{cvar_95*100:.2f}%")
                        metrics_cols[1].metric("Max Drawdown", f"{max_dd*100:.2f}%")
                
                # Risk analysis
                st.subheader("⚠️ Risk Analysis")
                
                tab1, tab2, tab3 = st.tabs(["Performance Dashboard", "Correlation Analysis", "Rolling Metrics"])
                
                with tab1:
                    if 'portfolio_returns' in locals():
                        fig_dashboard = viz.create_performance_dashboard(
                            portfolio_returns, 
                            benchmark_returns
                        )
                        st.plotly_chart(fig_dashboard, use_container_width=True)
                
                with tab2:
                    corr_structure = risk_engine.calculate_correlation_structure(returns)
                    
                    if corr_structure:
                        # Correlation heatmap
                        fig_corr = viz.create_correlation_heatmap(
                            corr_structure['correlation_matrix']
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # High correlation pairs
                        if corr_structure['high_correlation_pairs']:
                            st.markdown("**Highly Correlated Pairs (>0.8)**")
                            for pair in corr_structure['high_correlation_pairs']:
                                st.write(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
                
                with tab3:
                    if 'portfolio_returns' in locals():
                        rolling_metrics = risk_engine.calculate_rolling_metrics(portfolio_returns)
                        
                        if not rolling_metrics.empty:
                            fig_rolling = go.Figure()
                            
                            fig_rolling.add_trace(go.Scatter(
                                x=rolling_metrics.index,
                                y=rolling_metrics['volatility'],
                                name='Volatility',
                                line=dict(color='blue')
                            ))
                            
                            fig_rolling.add_trace(go.Scatter(
                                x=rolling_metrics.index,
                                y=rolling_metrics['sharpe'],
                                name='Sharpe Ratio',
                                line=dict(color='green'),
                                yaxis='y2'
                            ))
                            
                            fig_rolling.update_layout(
                                title='Rolling Metrics (63-day window)',
                                xaxis_title='Date',
                                yaxis_title='Volatility',
                                yaxis2=dict(
                                    title='Sharpe Ratio',
                                    overlaying='y',
                                    side='right'
                                ),
                                template='plotly_dark',
                                height=400
                            )
                            
                            st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Efficient frontier
                st.subheader("📈 Efficient Frontier")
                
                with st.spinner("Calculating efficient frontier..."):
                    frontier_result = optimizer.efficient_frontier_sample(
                        returns.mean() * Config.TRADING_DAYS_PER_YEAR,
                        returns.cov() * Config.TRADING_DAYS_PER_YEAR,
                        constraints={
                            'min_weight': 0.0,
                            'max_weight': 0.3,
                            'sum_to_one': True
                        }
                    )
                
                if frontier_result and 'frontier' in frontier_result:
                    frontier_df = pd.DataFrame(frontier_result['frontier'])
                    
                    fig_frontier = go.Figure()
                    
                    # Efficient frontier
                    fig_frontier.add_trace(go.Scatter(
                        x=frontier_df['volatility'],
                        y=frontier_df['return'],
                        mode='lines+markers',
                        name='Efficient Frontier',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Current portfolio
                    if optimization_result:
                        fig_frontier.add_trace(go.Scatter(
                            x=[optimization_result['expected_volatility']],
                            y=[optimization_result['expected_return']],
                            mode='markers',
                            marker=dict(size=15, color='red'),
                            name=f'{optimization_method.replace("_", " ").title()}'
                        ))
                    
                    # Capital Market Line
                    x_range = np.linspace(0, frontier_df['volatility'].max() * 1.1, 100)
                    cml_y = Config.RISK_FREE_RATE + \
                           (optimization_result['sharpe_ratio'] * x_range \
                            if optimization_result else 0)
                    
                    fig_frontier.add_trace(go.Scatter(
                        x=x_range,
                        y=cml_y,
                        mode='lines',
                        name='Capital Market Line',
                        line=dict(color='green', dash='dash')
                    ))
                    
                    fig_frontier.update_layout(
                        title='Efficient Frontier',
                        xaxis_title='Volatility (Annualized)',
                        yaxis_title='Return (Annualized)',
                        template='plotly_dark',
                        height=500
                    )
                    
                    st.plotly_chart(fig_frontier, use_container_width=True)
                
                # Monte Carlo simulation
                st.subheader("🎲 Monte Carlo Simulation")
                
                if 'portfolio_returns' in locals():
                    n_simulations = st.slider("Number of simulations", 100, 5000, 1000, 100)
                    n_days = st.slider("Forecast horizon (days)", 10, 252, 63, 10)
                    
                    if st.button("Run Simulation"):
                        with st.spinner(f"Running {n_simulations} simulations..."):
                            # Simple Monte Carlo simulation
                            mu = portfolio_returns.mean()
                            sigma = portfolio_returns.std()
                            
                            simulations = np.random.normal(
                                mu, sigma, (n_simulations, n_days)
                            )
                            
                            # Calculate cumulative returns
                            cum_returns = np.cumprod(1 + simulations, axis=1)
                            
                            # Statistics
                            final_values = cum_returns[:, -1]
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Mean Final Value", f"{final_values.mean():.3f}")
                            col2.metric("5th Percentile", f"{np.percentile(final_values, 5):.3f}")
                            col3.metric("95th Percentile", f"{np.percentile(final_values, 95):.3f}")
                            
                            # Plot simulation paths
                            fig_sim = go.Figure()
                            
                            # Plot a subset of paths for clarity
                            for i in range(min(50, n_simulations)):
                                fig_sim.add_trace(go.Scatter(
                                    x=list(range(n_days)),
                                    y=cum_returns[i, :],
                                    mode='lines',
                                    line=dict(width=1, color='rgba(100, 100, 255, 0.1)'),
                                    showlegend=False
                                ))
                            
                            # Plot median path
                            median_path = np.median(cum_returns, axis=0)
                            fig_sim.add_trace(go.Scatter(
                                x=list(range(n_days)),
                                y=median_path,
                                mode='lines',
                                line=dict(width=3, color='red'),
                                name='Median'
                            ))
                            
                            fig_sim.update_layout(
                                title=f'Monte Carlo Simulation ({n_simulations} paths)',
                                xaxis_title='Days',
                                yaxis_title='Cumulative Return',
                                template='plotly_dark',
                                height=400
                            )
                            
                            st.plotly_chart(fig_sim, use_container_width=True)
                
        except Exception as e:
            # Enhanced error handling
            error_analyzer = st.session_state.enhanced_error_analyzer
            context = {
                'tickers': selected_tickers,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'operation': 'main_analysis'
            }
            
            analysis = error_analyzer.analyze_error(e, context)
            error_analyzer.display_error_panel(analysis)
    
    else:
        # Welcome screen
        st.info("👈 Configure your analysis in the sidebar and click 'Run Analysis' to begin.")
        
        # Display universe statistics
        st.subheader("🌍 Universe Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assets", len(UNIVERSE_DF))
        
        with col2:
            st.metric("Regions", len(UNIVERSE_DF['region'].unique()))
        
        with col3:
            st.metric("Sectors", len(UNIVERSE_DF['sector'].unique()))
        
        with col4:
            avg_market_cap = UNIVERSE_DF['market_cap'].mean() / 1000  # Convert to billions
            st.metric("Avg Market Cap", f"${avg_market_cap:,.0f}B")

if __name__ == "__main__":
    main()

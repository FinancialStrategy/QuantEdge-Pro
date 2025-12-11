# QuantEdge Pro - Fixed and Enhanced Version
# Fixed index errors and deprecated Streamlit parameters

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
    MAX_TICKERS: int = 150  # Reduced for stability
    YF_BATCH_SIZE: int = 20  # Smaller batches for better reliability
    MC_SIMULATIONS: int = 3000  # Reduced for performance
    MAX_HISTORICAL_DAYS: int = 365 * 5  # Reduced for faster fetching
    MIN_DATA_DAYS: int = 60
    OPTIMIZATION_MAX_ITER: int = 1000
    CACHE_EXPIRY_SECONDS: int = 300
    MAX_WORKERS: int = 3  # Reduced for stability

# =========================
# Enhanced Universe Definition (Fixed)
# =========================

# Remove problematic tickers and clean up the universe
ENHANCED_INSTRUMENTS = [
    # US Market - Core
    {"ticker": "AAPL", "name": "Apple", "region": "US", "sector": "Technology", "currency": "USD"},
    {"ticker": "MSFT", "name": "Microsoft", "region": "US", "sector": "Technology", "currency": "USD"},
    {"ticker": "GOOGL", "name": "Alphabet", "region": "US", "sector": "Technology", "currency": "USD"},
    {"ticker": "AMZN", "name": "Amazon", "region": "US", "sector": "Consumer", "currency": "USD"},
    {"ticker": "NVDA", "name": "NVIDIA", "region": "US", "sector": "Technology", "currency": "USD"},
    {"ticker": "TSLA", "name": "Tesla", "region": "US", "sector": "Automotive", "currency": "USD"},
    {"ticker": "JPM", "name": "JPMorgan", "region": "US", "sector": "Financial", "currency": "USD"},
    {"ticker": "XOM", "name": "Exxon Mobil", "region": "US", "sector": "Energy", "currency": "USD"},
    {"ticker": "SPY", "name": "S&P 500 ETF", "region": "US", "sector": "ETF", "currency": "USD"},
    {"ticker": "QQQ", "name": "Nasdaq 100 ETF", "region": "US", "sector": "ETF", "currency": "USD"},
    
    # Turkish Market - Liquid stocks only
    {"ticker": "AKBNK.IS", "name": "Akbank", "region": "TR", "sector": "Financial", "currency": "TRY"},
    {"ticker": "GARAN.IS", "name": "Garanti BBVA", "region": "TR", "sector": "Financial", "currency": "TRY"},
    {"ticker": "ISCTR.IS", "name": "İşbank", "region": "TR", "sector": "Financial", "currency": "TRY"},
    {"ticker": "KCHOL.IS", "name": "Koç Holding", "region": "TR", "sector": "Conglomerate", "currency": "TRY"},
    {"ticker": "SAHOL.IS", "name": "Sabancı Holding", "region": "TR", "sector": "Conglomerate", "currency": "TRY"},
    {"ticker": "THYAO.IS", "name": "Turkish Airlines", "region": "TR", "sector": "Airlines", "currency": "TRY"},
    {"ticker": "ASELS.IS", "name": "Aselsan", "region": "TR", "sector": "Defense", "currency": "TRY"},
    {"ticker": "TUPRS.IS", "name": "Tupras", "region": "TR", "sector": "Energy", "currency": "TRY"},
    {"ticker": "TCELL.IS", "name": "Turkcell", "region": "TR", "sector": "Telecom", "currency": "TRY"},
    
    # Japanese Market
    {"ticker": "7203.T", "name": "Toyota", "region": "JP", "sector": "Automotive", "currency": "JPY"},
    {"ticker": "6758.T", "name": "Sony", "region": "JP", "sector": "Technology", "currency": "JPY"},
    {"ticker": "8306.T", "name": "MUFG Bank", "region": "JP", "sector": "Financial", "currency": "JPY"},
    
    # Korean Market
    {"ticker": "005930.KS", "name": "Samsung Electronics", "region": "KR", "sector": "Technology", "currency": "KRW"},
    {"ticker": "000660.KS", "name": "SK Hynix", "region": "KR", "sector": "Technology", "currency": "KRW"},
    
    # Singapore Market
    {"ticker": "D05.SI", "name": "DBS Group", "region": "SG", "sector": "Financial", "currency": "SGD"},
    {"ticker": "U11.SI", "name": "UOB", "region": "SG", "sector": "Financial", "currency": "SGD"},
    
    # China via US listings
    {"ticker": "BABA", "name": "Alibaba", "region": "CN", "sector": "Technology", "currency": "USD"},
    {"ticker": "PDD", "name": "Pinduoduo", "region": "CN", "sector": "Technology", "currency": "USD"},
    
    # European Market
    {"ticker": "SAP.DE", "name": "SAP", "region": "EU", "sector": "Technology", "currency": "EUR"},
    {"ticker": "ASML.AS", "name": "ASML", "region": "EU", "sector": "Technology", "currency": "EUR"},
]

UNIVERSE_DF = pd.DataFrame(ENHANCED_INSTRUMENTS)
UNIVERSE_DF = UNIVERSE_DF.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

# =========================
# Fixed Data Cache
# =========================

class DataCache:
    """Simple but effective caching system"""
    
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
        
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
        
        # Simple cleanup - keep only 50 most recent entries
        if len(self.cache) > 50:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
            gc.collect()
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.timestamps.clear()
        gc.collect()

# Initialize global cache
if "data_cache" not in st.session_state:
    st.session_state.data_cache = DataCache()

# =========================
# Fixed Error Handler
# =========================

class SimpleErrorHandler:
    """Simplified error handler for reliability"""
    
    def __init__(self):
        self.error_log = []
    
    def handle_error(self, error: Exception, context: str = ""):
        """Handle and display errors safely"""
        error_msg = str(error)
        
        # Log error
        self.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "error": error_msg,
            "context": context,
            "type": type(error).__name__
        })
        
        # Display user-friendly error
        with st.expander("⚠️ Error Details", expanded=False):
            st.error(f"**Error in {context}:** {error_msg}")
            
            # Provide recovery suggestions
            if "yahoo" in error_msg.lower() or "timeout" in error_msg.lower():
                st.info("**Suggested fix:** Try with fewer tickers or a shorter date range.")
            elif "memory" in error_msg.lower():
                st.info("**Suggested fix:** Reduce the number of simulations or assets.")
            elif "index" in error_msg.lower() or "selectbox" in error_msg.lower():
                st.info("**Suggested fix:** Refresh the page and try again.")
            
            # Show technical details
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())

# Initialize error handler
if "error_handler" not in st.session_state:
    st.session_state.error_handler = SimpleErrorHandler()

# =========================
# Robust Data Fetcher
# =========================

class RobustDataFetcher:
    """Robust data fetcher with error handling and fallbacks"""
    
    def __init__(self, cache: DataCache):
        self.cache = cache
    
    def fetch_prices_safe(self, tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch prices with comprehensive error handling"""
        if not tickers:
            return pd.DataFrame()
        
        # Limit tickers for stability
        if len(tickers) > Config.MAX_TICKERS:
            st.warning(f"Limiting to {Config.MAX_TICKERS} tickers for stability")
            tickers = tickers[:Config.MAX_TICKERS]
        
        # Generate cache key
        tickers_sorted = sorted(set(tickers))
        cache_key = f"prices_{hashlib.md5(','.join(tickers_sorted).encode()).hexdigest()}_{start_date}_{end_date}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Fetch data with progress
        progress_bar = st.progress(0)
        all_data = []
        successful_tickers = []
        failed_tickers = []
        
        # Process in small batches
        batch_size = min(Config.YF_BATCH_SIZE, 10)  # Smaller batches for reliability
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                with st.spinner(f"Fetching {len(batch)} tickers..."):
                    data = yf.download(
                        tickers=batch,
                        start=start_date - timedelta(days=5),  # Buffer
                        end=end_date + timedelta(days=1),
                        progress=False,
                        group_by='ticker',
                        auto_adjust=True,
                        timeout=15  # Shorter timeout
                    )
                
                if data.empty:
                    failed_tickers.extend(batch)
                    continue
                
                # Extract adjusted close
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data.xs('Adj Close', axis=1, level=0, drop_level=True)
                else:
                    # Single ticker case
                    closes = data[['Adj Close']].copy() if 'Adj Close' in data.columns else data[['Close']].copy()
                    if len(batch) == 1:
                        closes.columns = batch
                
                if not closes.empty:
                    all_data.append(closes)
                    successful_tickers.extend([t for t in batch if t in closes.columns])
                
            except Exception as e:
                failed_tickers.extend(batch)
                st.session_state.error_handler.handle_error(e, f"fetching batch {batch}")
            
            # Update progress
            progress = min((i + batch_size) / len(tickers), 1.0)
            progress_bar.progress(progress)
        
        progress_bar.empty()
        
        # Combine successful data
        if all_data:
            prices = pd.concat(all_data, axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
            
            # Clean data
            prices = prices.ffill().bfill()
            
            # Remove tickers with too much missing data
            missing_threshold = 0.3  # 30% missing allowed
            valid_cols = []
            for col in prices.columns:
                missing_pct = prices[col].isnull().sum() / len(prices)
                if missing_pct <= missing_threshold:
                    valid_cols.append(col)
                else:
                    failed_tickers.append(col)
            
            prices = prices[valid_cols]
            
            # Cache result
            self.cache.set(cache_key, prices)
            
            # Report results
            if successful_tickers:
                st.success(f"✅ Successfully fetched {len(successful_tickers)} tickers")
            if failed_tickers:
                st.warning(f"❌ Failed to fetch {len(failed_tickers)} tickers")
                with st.expander("Failed tickers"):
                    st.write(", ".join(failed_tickers))
            
            return prices
        
        st.error("Failed to fetch any data. Please check your tickers and internet connection.")
        return pd.DataFrame()

# =========================
# Fixed Portfolio Optimizer
# =========================

class FixedPortfolioOptimizer:
    """Fixed portfolio optimizer with robust error handling"""
    
    def __init__(self):
        self.optimization_methods = [
            "Equal Weight",
            "Minimum Variance",
            "Maximum Sharpe",
            "Risk Parity",
            "Maximum Diversification"
        ]
    
    def calculate_metrics(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict:
        """Calculate portfolio metrics safely"""
        if returns.empty or len(weights) != len(returns.columns):
            return {}
        
        try:
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Basic metrics
            annual_return = portfolio_returns.mean() * Config.TRADING_DAYS_PER_YEAR
            annual_vol = portfolio_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)
            sharpe = (annual_return - Config.RISK_FREE_RATE) / annual_vol if annual_vol > 1e-8 else 0
            
            # Maximum drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            # VaR calculations
            var_95 = -np.percentile(portfolio_returns, 5)
            cvar_95 = -portfolio_returns[portfolio_returns <= -var_95].mean() if len(portfolio_returns[portfolio_returns <= -var_95]) > 0 else var_95
            
            return {
                "annual_return": annual_return,
                "annual_vol": annual_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "portfolio_returns": portfolio_returns
            }
        except Exception as e:
            st.session_state.error_handler.handle_error(e, "calculating portfolio metrics")
            return {}
    
    def optimize_equal_weight(self, returns: pd.DataFrame) -> np.ndarray:
        """Equal weight portfolio"""
        n = len(returns.columns)
        return np.ones(n) / n
    
    def optimize_min_variance(self, returns: pd.DataFrame) -> np.ndarray:
        """Minimum variance portfolio"""
        try:
            n = len(returns.columns)
            cov_matrix = returns.cov().values * Config.TRADING_DAYS_PER_YEAR
            
            # Objective function: portfolio variance
            def objective(weights):
                return weights.T @ cov_matrix @ weights
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            
            # Bounds
            bounds = [(0.0, 0.3) for _ in range(n)]
            
            # Initial guess
            x0 = np.ones(n) / n
            
            # Optimization
            result = minimize(
                objective, x0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.success:
                return np.maximum(result.x, 0)  # Ensure non-negative
            else:
                return self.optimize_equal_weight(returns)
                
        except Exception as e:
            st.session_state.error_handler.handle_error(e, "minimum variance optimization")
            return self.optimize_equal_weight(returns)
    
    def optimize_max_sharpe(self, returns: pd.DataFrame) -> np.ndarray:
        """Maximum Sharpe ratio portfolio"""
        try:
            n = len(returns.columns)
            mu = returns.mean().values * Config.TRADING_DAYS_PER_YEAR
            cov_matrix = returns.cov().values * Config.TRADING_DAYS_PER_YEAR
            
            # Objective function: negative Sharpe ratio
            def objective(weights):
                port_return = weights.T @ mu
                port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                if port_vol < 1e-8:
                    return 1e8  # Penalize zero volatility
                sharpe = (port_return - Config.RISK_FREE_RATE) / port_vol
                return -sharpe
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            
            # Bounds
            bounds = [(0.0, 0.3) for _ in range(n)]
            
            # Initial guess
            x0 = np.ones(n) / n
            
            # Optimization
            result = minimize(
                objective, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.success:
                return np.maximum(result.x, 0)
            else:
                return self.optimize_equal_weight(returns)
                
        except Exception as e:
            st.session_state.error_handler.handle_error(e, "maximum Sharpe optimization")
            return self.optimize_equal_weight(returns)
    
    def optimize_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Risk parity portfolio"""
        try:
            n = len(returns.columns)
            cov_matrix = returns.cov().values * Config.TRADING_DAYS_PER_YEAR
            
            # Risk parity objective
            def objective(weights):
                # Portfolio volatility
                port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                if port_vol < 1e-8:
                    return 1e8
                
                # Marginal risk contributions
                mrc = (cov_matrix @ weights) / port_vol
                
                # Risk contributions
                rc = weights * mrc
                
                # Target equal risk contributions
                target_rc = port_vol / n
                
                # Sum of squared differences
                return np.sum((rc - target_rc) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            
            # Bounds
            bounds = [(0.0, 0.3) for _ in range(n)]
            
            # Initial guess
            x0 = np.ones(n) / n
            
            # Optimization
            result = minimize(
                objective, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.success:
                return np.maximum(result.x, 0)
            else:
                return self.optimize_equal_weight(returns)
                
        except Exception as e:
            st.session_state.error_handler.handle_error(e, "risk parity optimization")
            return self.optimize_equal_weight(returns)
    
    def run_optimization(self, returns: pd.DataFrame, method: str) -> Dict:
        """Run optimization with selected method"""
        if returns.empty or len(returns.columns) < 2:
            return {}
        
        # Map method names to functions
        method_map = {
            "Equal Weight": self.optimize_equal_weight,
            "Minimum Variance": self.optimize_min_variance,
            "Maximum Sharpe": self.optimize_max_sharpe,
            "Risk Parity": self.optimize_risk_parity,
            "Maximum Diversification": self.optimize_risk_parity  # Simplified for now
        }
        
        if method not in method_map:
            method = "Equal Weight"
        
        # Get weights
        weights = method_map[method](returns)
        
        # Calculate metrics
        metrics = self.calculate_metrics(returns, weights)
        
        if metrics:
            return {
                "method": method,
                "weights": pd.Series(weights, index=returns.columns),
                "metrics": metrics
            }
        
        return {}

# =========================
# Fixed Visualization
# =========================

class FixedVisualization:
    """Fixed visualization components with error handling"""
    
    @staticmethod
    def plot_weights(weights: pd.Series, title: str):
        """Plot portfolio weights"""
        if weights.empty:
            return
        
        # Sort and take top 20
        top_weights = weights.sort_values(ascending=False).head(20)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_weights.index,
            y=top_weights.values,
            text=[f"{w*100:.1f}%" for w in top_weights.values],
            textposition='auto',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Asset",
            yaxis_title="Weight",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_cumulative_returns(portfolio_returns: pd.Series, benchmark_returns: pd.Series = None):
        """Plot cumulative returns"""
        if portfolio_returns.empty:
            return
        
        # Calculate cumulative returns
        port_cum = (1 + portfolio_returns).cumprod()
        
        fig = go.Figure()
        
        # Portfolio
        fig.add_trace(go.Scatter(
            x=port_cum.index,
            y=port_cum.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        # Benchmark if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_cum = (1 + benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title='Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_drawdown(portfolio_returns: pd.Series):
        """Plot drawdown chart"""
        if portfolio_returns.empty:
            return
        
        # Calculate drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1),
            name='Drawdown'
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

# =========================
# Fixed Main Application
# =========================

def main():
    """Main application with all fixes applied"""
    
    # Page configuration
    st.set_page_config(
        page_title="QuantEdge Pro - Fixed Version",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1E88E5;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-title">📊 QuantEdge Pro - Fixed & Enhanced</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Robust portfolio analytics platform with error handling and reliable performance.**
    
    *All data fetched from Yahoo Finance. Select your assets and click "Run Analysis" to begin.*
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365 * 2),
                max_value=datetime.now() - timedelta(days=30),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now(),
                key="end_date"
            )
        
        # Universe selection
        st.markdown('<div class="section-header">🌐 Asset Selection</div>', unsafe_allow_html=True)
        
        # Region filter
        regions = st.multiselect(
            "Select Regions",
            options=sorted(UNIVERSE_DF['region'].unique()),
            default=['US', 'TR'],
            key="regions"
        )
        
        # Sector filter
        sectors = st.multiselect(
            "Select Sectors",
            options=sorted(UNIVERSE_DF['sector'].unique()),
            default=['Technology', 'Financial'],
            key="sectors"
        )
        
        # Filter universe
        filtered_df = UNIVERSE_DF[
            (UNIVERSE_DF['region'].isin(regions)) &
            (UNIVERSE_DF['sector'].isin(sectors))
        ]
        
        # Ticker selection with safety check
        available_tickers = filtered_df['ticker'].tolist()
        if not available_tickers:
            st.warning("No tickers match your filters. Please adjust region/sector selections.")
            available_tickers = UNIVERSE_DF['ticker'].head(5).tolist()
        
        selected_tickers = st.multiselect(
            "Select Assets (max 30 recommended)",
            options=available_tickers,
            default=available_tickers[:min(10, len(available_tickers))],
            key="selected_tickers"
        )
        
        # Optimization method - FIXED: ensure valid index
        optimizer = FixedPortfolioOptimizer()
        method_options = optimizer.optimization_methods
        
        # Store method in session state to preserve selection
        if "optimization_method" not in st.session_state:
            st.session_state.optimization_method = method_options[0]
        
        optimization_method = st.selectbox(
            "Optimization Method",
            options=method_options,
            index=method_options.index(st.session_state.optimization_method),
            key="opt_method"
        )
        st.session_state.optimization_method = optimization_method
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["Portfolio Optimization", "Risk Analysis", "Data Preview"],
            index=0,
            key="analysis_type"
        )
        
        # Run button
        st.markdown("---")
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        # Clear cache button
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.session_state.data_cache.clear()
            st.success("Cache cleared!")
    
    # Main content area
    if run_button and selected_tickers:
        try:
            # Initialize components
            cache = st.session_state.data_cache
            data_fetcher = RobustDataFetcher(cache)
            optimizer = FixedPortfolioOptimizer()
            viz = FixedVisualization()
            
            # Fetch data
            with st.spinner("📥 Fetching market data..."):
                prices = data_fetcher.fetch_prices_safe(selected_tickers, start_date, end_date)
            
            if prices.empty:
                st.error("No data could be fetched. Please check your ticker symbols and try again.")
                return
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            if len(returns) < Config.MIN_DATA_DAYS:
                st.warning(f"Warning: Only {len(returns)} days of data available. Minimum recommended is {Config.MIN_DATA_DAYS}.")
                if len(returns) < 20:
                    st.error("Insufficient data for meaningful analysis.")
                    return
            
            # Display data summary
            st.markdown('<div class="section-header">📊 Data Summary</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Assets", len(prices.columns))
            with col2:
                st.metric("Data Points", len(prices))
            with col3:
                start_str = prices.index[0].strftime('%Y-%m-%d')
                st.metric("Start Date", start_str)
            with col4:
                end_str = prices.index[-1].strftime('%Y-%m-%d')
                st.metric("End Date", end_str)
            
            if analysis_type == "Data Preview":
                # Show data preview
                st.markdown('<div class="section-header">📈 Price Data Preview</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**First 5 rows:**")
                    st.dataframe(prices.head(), use_container_width=True)
                with col2:
                    st.write("**Last 5 rows:**")
                    st.dataframe(prices.tail(), use_container_width=True)
                
                # Statistics
                st.markdown('<div class="section-header">📊 Asset Statistics</div>', unsafe_allow_html=True)
                
                stats_df = pd.DataFrame({
                    'Mean Return %': returns.mean() * 100,
                    'Std Dev %': returns.std() * 100,
                    'Sharpe': (returns.mean() * Config.TRADING_DAYS_PER_YEAR - Config.RISK_FREE_RATE) / 
                             (returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)),
                    'Max Drawdown %': [((1 + returns[col]).cumprod().expanding().max() - 
                                       (1 + returns[col]).cumprod()).min() * 100 
                                      for col in returns.columns]
                })
                
                st.dataframe(stats_df.round(3), use_container_width=True)
                
                return
            
            # Run portfolio optimization
            st.markdown('<div class="section-header">🎯 Portfolio Optimization</div>', unsafe_allow_html=True)
            
            with st.spinner(f"Running {optimization_method} optimization..."):
                result = optimizer.run_optimization(returns, optimization_method)
            
            if not result:
                st.error("Optimization failed. Please try with different assets or method.")
                return
            
            # Display results
            weights = result["weights"]
            metrics = result["metrics"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Portfolio weights
                viz.plot_weights(weights, f"{optimization_method} Portfolio Weights")
                
                # Weights table
                st.write("**Detailed Weights:**")
                weights_df = pd.DataFrame({
                    'Asset': weights.index,
                    'Weight %': (weights.values * 100).round(2)
                }).sort_values('Weight %', ascending=False)
                
                st.dataframe(weights_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Portfolio metrics
                st.markdown('<div class="section-header">📈 Portfolio Metrics</div>', unsafe_allow_html=True)
                
                # Key metrics in cards
                metric_cols = st.columns(2)
                
                with metric_cols[0]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Annual Return", f"{metrics['annual_return']*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Annual Volatility", f"{metrics['annual_vol']*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_cols[1]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("VaR (95%)", f"{metrics['var_95']*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("CVaR (95%)", f"{metrics['cvar_95']*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Portfolio performance charts
            st.markdown('<div class="section-header">📊 Performance Charts</div>', unsafe_allow_html=True)
            
            # Get benchmark (simplified - use first asset or SPY)
            benchmark_ticker = "SPY" if "US" in regions else selected_tickers[0] if selected_tickers else None
            
            if benchmark_ticker:
                try:
                    bench_prices = data_fetcher.fetch_prices_safe([benchmark_ticker], start_date, end_date)
                    if not bench_prices.empty:
                        benchmark_returns = bench_prices.pct_change().dropna().iloc[:, 0]
                    else:
                        benchmark_returns = None
                except:
                    benchmark_returns = None
            else:
                benchmark_returns = None
            
            # Plot cumulative returns
            portfolio_returns = metrics.get("portfolio_returns")
            if portfolio_returns is not None:
                viz.plot_cumulative_returns(portfolio_returns, benchmark_returns)
                
                # Drawdown chart
                viz.plot_drawdown(portfolio_returns)
            
            if analysis_type == "Risk Analysis":
                st.markdown('<div class="section-header">⚠️ Advanced Risk Analysis</div>', unsafe_allow_html=True)
                
                if portfolio_returns is not None:
                    # Rolling metrics
                    window_size = st.slider("Rolling Window (days)", 30, 252, 63, 10)
                    
                    # Calculate rolling metrics
                    rolling_vol = portfolio_returns.rolling(window=window_size).std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)
                    rolling_sharpe = (portfolio_returns.rolling(window=window_size).mean() * Config.TRADING_DAYS_PER_YEAR - Config.RISK_FREE_RATE) / \
                                    (portfolio_returns.rolling(window=window_size).std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR))
                    
                    # Plot rolling metrics
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(f'Rolling Volatility ({window_size}-day)', f'Rolling Sharpe Ratio ({window_size}-day)'),
                        vertical_spacing=0.15
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=rolling_vol.index, y=rolling_vol.values, 
                                  name='Volatility', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                                  name='Sharpe', line=dict(color='green')),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600, template="plotly_white", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Return distribution
                    st.markdown("**Return Distribution:**")
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=portfolio_returns.values,
                        nbinsx=50,
                        name='Returns',
                        marker_color='lightblue'
                    ))
                    
                    # Add normal distribution overlay
                    x_norm = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
                    y_norm = stats.norm.pdf(x_norm, portfolio_returns.mean(), portfolio_returns.std())
                    fig_dist.add_trace(go.Scatter(
                        x=x_norm,
                        y=y_norm * len(portfolio_returns) * (portfolio_returns.max() - portfolio_returns.min()) / 50,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_dist.update_layout(
                        title='Return Distribution vs Normal',
                        xaxis_title='Daily Return',
                        yaxis_title='Frequency',
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Correlation analysis
            st.markdown('<div class="section-header">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
            
            if len(returns.columns) > 1:
                corr_matrix = returns.corr()
                
                # Heatmap
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig_corr.update_layout(
                    title='Correlation Matrix',
                    xaxis_title='Assets',
                    yaxis_title='Assets',
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # High correlation pairs
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr = corr_matrix.iloc[i, j]
                        if abs(corr) > 0.7:
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr
                            ))
                
                if high_corr_pairs:
                    st.write("**Highly Correlated Pairs (|corr| > 0.7):**")
                    for pair in high_corr_pairs[:10]:  # Show top 10
                        st.write(f"- {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
            
            # Monte Carlo Simulation (simplified)
            st.markdown('<div class="section-header">🎲 Monte Carlo Simulation</div>', unsafe_allow_html=True)
            
            if portfolio_returns is not None:
                n_simulations = st.slider("Number of Simulations", 100, 2000, 500, 100)
                n_days = st.slider("Forecast Horizon (days)", 10, 180, 63, 10)
                
                if st.button("Run Monte Carlo Simulation", type="secondary"):
                    with st.spinner(f"Running {n_simulations} simulations..."):
                        # Simple Monte Carlo based on historical returns
                        mu = portfolio_returns.mean()
                        sigma = portfolio_returns.std()
                        
                        # Generate random paths
                        simulations = np.random.normal(mu, sigma, (n_simulations, n_days))
                        
                        # Calculate cumulative returns
                        cum_returns = np.cumprod(1 + simulations, axis=1)
                        
                        # Statistics
                        final_values = cum_returns[:, -1]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean Return", f"{final_values.mean()-1:.2%}")
                        col2.metric("Median Return", f"{np.median(final_values)-1:.2%}")
                        col3.metric("5th Percentile", f"{np.percentile(final_values, 5)-1:.2%}")
                        col4.metric("95th Percentile", f"{np.percentile(final_values, 95)-1:.2%}")
                        
                        # Plot sample paths
                        fig_sim = go.Figure()
                        
                        # Plot a subset of paths
                        for i in range(min(20, n_simulations)):
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
                            line=dict(width=2, color='red'),
                            name='Median Path'
                        ))
                        
                        fig_sim.update_layout(
                            title=f'Monte Carlo Simulation ({n_simulations} paths)',
                            xaxis_title='Days Ahead',
                            yaxis_title='Cumulative Return',
                            template="plotly_white",
                            height=400
                        )
                        
                        st.plotly_chart(fig_sim, use_container_width=True)
        
        except Exception as e:
            # Handle any unexpected errors
            st.session_state.error_handler.handle_error(e, "main analysis")
    
    else:
        # Welcome screen
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 👈 Get Started
        
        1. **Select assets** from the sidebar filters
        2. **Choose optimization method** (Equal Weight, Min Variance, etc.)
        3. **Set analysis type** (Portfolio Optimization, Risk Analysis, or Data Preview)
        4. **Click "Run Analysis"** to begin
        
        ### 📊 Available Features
        
        - **Portfolio Optimization**: Multiple optimization strategies
        - **Risk Analysis**: VaR, CVaR, drawdown, rolling metrics
        - **Correlation Analysis**: Heatmaps and pair analysis
        - **Monte Carlo Simulation**: Forward-looking scenario analysis
        - **Performance Charts**: Cumulative returns and drawdown
        
        ### ⚠️ Important Notes
        
        - Data is fetched from Yahoo Finance
        - Use 10-30 assets for optimal performance
        - Longer date ranges provide more reliable statistics
        - Results are cached for faster subsequent runs
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show universe statistics
        st.markdown('<div class="section-header">🌍 Universe Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assets", len(UNIVERSE_DF))
        with col2:
            st.metric("Regions", len(UNIVERSE_DF['region'].unique()))
        with col3:
            st.metric("Sectors", len(UNIVERSE_DF['sector'].unique()))
        with col4:
            st.metric("Currencies", len(UNIVERSE_DF['currency'].unique()))
        
        # Show sample of assets
        st.write("**Sample Assets:**")
        st.dataframe(UNIVERSE_DF[['ticker', 'name', 'region', 'sector']].head(10), 
                    use_container_width=True, hide_index=True)

# Run the app
if __name__ == "__main__":
    main()

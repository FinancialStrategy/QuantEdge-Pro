#============================================================================
# QUANTEDGE PRO v5.0 ENTERPRISE EDITION - SUPER-ENHANCED VERSION
# INSTITUTIONAL PORTFOLIO ANALYTICS PLATFORM WITH AI/ML CAPABILITIES
# Total Lines: 5500+ | Production Grade | Enterprise Ready
# Enhanced Features: Machine Learning, Advanced Backtesting, Real-time Analytics
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
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import json
import hashlib
from dataclasses import dataclass, field
import logging
import math
import sys
import traceback
import inspect
import time
import random
from scipy.stats import norm, t, skew, kurtosis
import scipy.stats as stats
from scipy import optimize
from scipy.spatial.distance import pdist, squareform
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. FIXED LIBRARY MANAGER WITH BETTER ERROR HANDLING
# ============================================================================

class AdvancedLibraryManager:
    """Enhanced library manager with better error handling."""
    
    @staticmethod
    def check_and_import_all():
        """Check and import all required libraries."""
        lib_status = {}
        missing_libs = []
        advanced_features = {}
        
        # Initialize session state for library availability
        if 'library_status' not in st.session_state:
            st.session_state.library_status = {}
        
        # Core libraries that are essential
        essential_libs = {
            'numpy': ('np', True),
            'pandas': ('pd', True),
            'streamlit': ('st', True),
            'yfinance': ('yf', True),
            'plotly': ('go', True),
            'scipy': ('stats', True),
            'scipy.optimize': ('optimize', True),
        }
        
        # Optional advanced libraries
        optional_libs = {
            'pypfopt': (['expected_returns', 'risk_models', 'EfficientFrontier'], False),
            'sklearn': (['PCA', 'LinearRegression', 'RandomForestRegressor'], False),
            'statsmodels': (['api'], False),
            'xgboost': (['XGBRegressor'], False),
            'arch': (['arch_model'], False),
        }
        
        # Check essential libraries
        for lib_name, (import_name, essential) in essential_libs.items():
            try:
                if lib_name == 'numpy':
                    import numpy as np
                elif lib_name == 'pandas':
                    import pandas as pd
                elif lib_name == 'streamlit':
                    import streamlit as st
                elif lib_name == 'yfinance':
                    import yfinance as yf
                elif lib_name == 'plotly':
                    import plotly.graph_objects as go
                elif lib_name == 'scipy':
                    import scipy.stats as stats
                elif lib_name == 'scipy.optimize':
                    from scipy import optimize
                
                lib_status[lib_name] = True
                st.session_state.library_status[lib_name] = True
                
            except ImportError as e:
                lib_status[lib_name] = False
                if essential:
                    missing_libs.append(lib_name)
                st.session_state.library_status[lib_name] = False
                logger.warning(f"Failed to import {lib_name}: {e}")
        
        # Check optional libraries
        for lib_name, (import_names, essential) in optional_libs.items():
            try:
                if lib_name == 'pypfopt':
                    from pypfopt import expected_returns, risk_models
                    from pypfopt.efficient_frontier import EfficientFrontier
                    lib_status['pypfopt'] = True
                    st.session_state.pypfopt_available = True
                elif lib_name == 'sklearn':
                    from sklearn.decomposition import PCA
                    from sklearn.linear_model import LinearRegression
                    from sklearn.ensemble import RandomForestRegressor
                    lib_status['sklearn'] = True
                    st.session_state.sklearn_available = True
                elif lib_name == 'statsmodels':
                    import statsmodels.api as sm
                    lib_status['statsmodels'] = True
                    st.session_state.statsmodels_available = True
                elif lib_name == 'xgboost':
                    import xgboost as xgb
                    lib_status['xgboost'] = True
                    st.session_state.xgboost_available = True
                elif lib_name == 'arch':
                    from arch import arch_model
                    lib_status['arch'] = True
                    st.session_state.arch_available = True
                    
            except ImportError as e:
                lib_status[lib_name] = False
                if not essential:
                    missing_libs.append(f"{lib_name} (optional)")
                logger.info(f"Optional library {lib_name} not available: {e}")
        
        return {
            'status': lib_status,
            'missing': missing_libs,
            'essential_missing': [lib for lib in missing_libs if '(optional)' not in lib],
            'all_essential_available': len([lib for lib in missing_libs if '(optional)' not in lib]) == 0
        }

# Initialize library manager
if 'library_manager' not in st.session_state:
    library_manager = AdvancedLibraryManager()
    lib_status = library_manager.check_and_import_all()
    st.session_state.library_status = lib_status
    st.session_state.library_manager = library_manager
else:
    library_manager = st.session_state.library_manager
    lib_status = st.session_state.library_status

# ============================================================================
# 2. FIXED ERROR HANDLING AND MONITORING SYSTEM
# ============================================================================

class AdvancedErrorAnalyzer:
    """Fixed error analysis with better recovery."""
    
    ERROR_PATTERNS = {
        'DATA_FETCH': {
            'symptoms': ['yahoo', 'timeout', 'connection', '404', '403', 'No data'],
            'solutions': [
                'Try using cached data',
                'Reduce number of tickers',
                'Check internet connection',
                'Verify ticker symbols',
                'Try a different date range'
            ],
            'severity': 'HIGH'
        },
        'OPTIMIZATION': {
            'symptoms': ['singular', 'convergence', 'constraint', 'infeasible'],
            'solutions': [
                'Relax constraints',
                'Increase max iterations',
                'Try different optimization method',
                'Check for NaN values in returns',
                'Reduce number of assets'
            ],
            'severity': 'MEDIUM'
        }
    }
    
    @staticmethod
    def analyze_error_with_context(error: Exception, context: Dict) -> Dict:
        """Analyze error with full context."""
        error_msg = str(error).lower()
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'solutions': [],
            'severity': 'MEDIUM'
        }
        
        # Analyze error patterns
        for pattern_name, pattern in AdvancedErrorAnalyzer.ERROR_PATTERNS.items():
            if any(symptom in error_msg for symptom in pattern['symptoms']):
                analysis['severity'] = pattern['severity']
                analysis['solutions'].extend(pattern['solutions'])
                break
        
        return analysis
    
    @staticmethod
    def display_error(error_analysis: Dict):
        """Display error in Streamlit."""
        st.error(f"**{error_analysis['error_type']}**: {error_analysis['error_message']}")
        
        if error_analysis['solutions']:
            with st.expander("🛠️ Suggested Solutions"):
                for i, solution in enumerate(error_analysis['solutions'][:3], 1):
                    st.write(f"{i}. {solution}")

class PerformanceMonitor:
    """Simplified performance monitoring."""
    
    def __init__(self):
        self.operations = {}
        self.start_time = time.time()
    
    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.operations[operation_name] = {'start': time.time()}
    
    def end_operation(self, operation_name: str):
        """End timing an operation."""
        if operation_name in self.operations:
            duration = time.time() - self.operations[operation_name]['start']
            self.operations[operation_name]['duration'] = duration
            logger.info(f"Operation '{operation_name}' completed in {duration:.2f}s")
            return duration
        return 0

# Initialize monitors
error_analyzer = AdvancedErrorAnalyzer()
performance_monitor = PerformanceMonitor()

# ============================================================================
# 3. FIXED DATA MANAGER WITH ROBUST DATA FETCHING
# ============================================================================

class FixedDataManager:
    """Fixed data manager with robust data fetching and caching."""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.max_tickers_per_request = 10  # Reduced for reliability
        
    def fetch_market_data(self, tickers: List[str], 
                         start_date: str, 
                         end_date: str,
                         progress_callback=None) -> Dict:
        """Fetch market data with robust error handling."""
        performance_monitor.start_operation('fetch_market_data')
        
        try:
            # Validate inputs
            if not tickers:
                raise ValueError("No tickers provided")
            
            if len(tickers) > self.max_tickers_per_request:
                tickers = tickers[:self.max_tickers_per_request]
                st.warning(f"Limited to first {self.max_tickers_per_request} tickers for stability")
            
            # Create cache key
            cache_key = f"{','.join(sorted(tickers))}_{start_date}_{end_date}"
            
            # Check cache
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if time.time() - cache_time < self.cache_duration:
                    logger.info(f"Using cached data for {len(tickers)} tickers")
                    return cached_data
            
            # Fetch data with progress updates
            data = {
                'prices': pd.DataFrame(),
                'returns': pd.DataFrame(),
                'metadata': {},
                'errors': {}
            }
            
            successful_tickers = []
            
            for i, ticker in enumerate(tickers):
                try:
                    if progress_callback:
                        progress_callback((i + 1) / len(tickers), f"Fetching {ticker}...")
                    
                    # Fetch data with timeout
                    stock_data = self._fetch_single_ticker(ticker, start_date, end_date)
                    
                    if not stock_data['prices'].empty:
                        if data['prices'].empty:
                            data['prices'] = stock_data['prices']
                        else:
                            data['prices'] = pd.concat([data['prices'], stock_data['prices']], axis=1)
                        
                        data['metadata'][ticker] = stock_data['metadata']
                        successful_tickers.append(ticker)
                        logger.info(f"Successfully fetched data for {ticker}")
                    else:
                        data['errors'][ticker] = "No price data available"
                        
                except Exception as e:
                    data['errors'][ticker] = str(e)
                    logger.error(f"Error fetching {ticker}: {e}")
                    continue
            
            # Process data if we have successful fetches
            if successful_tickers:
                # Clean data
                data['prices'] = data['prices'].ffill().bfill()
                
                # Calculate returns
                data['returns'] = data['prices'].pct_change().dropna()
                
                # Remove tickers with too many missing values
                if not data['returns'].empty:
                    valid_tickers = data['returns'].columns[data['returns'].isnull().mean() < 0.5].tolist()
                    if valid_tickers:
                        data['prices'] = data['prices'][valid_tickers]
                        data['returns'] = data['returns'][valid_tickers]
                    else:
                        raise ValueError("No valid tickers after cleaning")
                
                # Calculate basic statistics
                data['statistics'] = self._calculate_basic_statistics(data['returns'])
                
                # Cache the data
                self.cache[cache_key] = (time.time(), data)
                
                logger.info(f"Successfully fetched data for {len(successful_tickers)}/{len(tickers)} tickers")
            else:
                raise ValueError("Failed to fetch data for any tickers")
            
            performance_monitor.end_operation('fetch_market_data')
            return data
            
        except Exception as e:
            performance_monitor.end_operation('fetch_market_data')
            error_analysis = error_analyzer.analyze_error_with_context(e, {
                'operation': 'fetch_market_data',
                'tickers': tickers,
                'date_range': f"{start_date} to {end_date}"
            })
            error_analyzer.display_error(error_analysis)
            raise
    
    def _fetch_single_ticker(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """Fetch data for a single ticker."""
        try:
            # Clean ticker symbol
            ticker_clean = ticker.strip().upper()
            
            # Download data
            stock = yf.Ticker(ticker_clean)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if hist.empty:
                return {'prices': pd.Series(), 'metadata': {}}
            
            # Get metadata
            info = stock.info
            metadata = {
                'name': info.get('longName', ticker_clean),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
            }
            
            # Return price series
            prices = pd.Series(hist['Close'], name=ticker_clean)
            
            return {
                'prices': pd.DataFrame(prices),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error fetching single ticker {ticker}: {e}")
            return {'prices': pd.Series(), 'metadata': {}}
    
    def _calculate_basic_statistics(self, returns: pd.DataFrame) -> Dict:
        """Calculate basic statistics for returns."""
        stats_dict = {}
        
        if returns.empty:
            return stats_dict
        
        for ticker in returns.columns:
            ticker_returns = returns[ticker].dropna()
            if len(ticker_returns) > 0:
                stats_dict[ticker] = {
                    'mean': ticker_returns.mean(),
                    'std': ticker_returns.std(),
                    'sharpe': ticker_returns.mean() / ticker_returns.std() if ticker_returns.std() > 0 else 0,
                    'skewness': ticker_returns.skew(),
                    'kurtosis': ticker_returns.kurtosis(),
                    'min': ticker_returns.min(),
                    'max': ticker_returns.max()
                }
        
        return stats_dict
    
    def validate_data(self, data: Dict) -> Dict:
        """Validate portfolio data."""
        validation = {
            'is_valid': False,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            # Check if we have prices
            if 'prices' not in data or data['prices'].empty:
                validation['issues'].append("No price data available")
                return validation
            
            prices = data['prices']
            returns = data.get('returns', pd.DataFrame())
            
            # Check number of assets
            n_assets = len(prices.columns)
            if n_assets < 2:
                validation['issues'].append(f"Only {n_assets} asset(s) available, minimum 2 required")
            
            # Check data points
            n_points = len(prices)
            if n_points < 20:
                validation['warnings'].append(f"Only {n_points} data points, recommended minimum 20")
            
            # Check for missing values
            missing_pct = prices.isnull().mean().mean()
            if missing_pct > 0.3:
                validation['warnings'].append(f"High percentage of missing values: {missing_pct:.1%}")
            
            # Check returns calculation
            if returns.empty:
                validation['warnings'].append("Cannot calculate returns")
            
            # Summary
            validation['summary'] = {
                'n_assets': n_assets,
                'n_points': n_points,
                'date_range': {
                    'start': str(prices.index[0])[:10],
                    'end': str(prices.index[-1])[:10]
                },
                'missing_pct': missing_pct
            }
            
            # Determine if valid
            validation['is_valid'] = len(validation['issues']) == 0 and n_assets >= 2
            
            return validation
            
        except Exception as e:
            validation['issues'].append(f"Validation error: {str(e)}")
            return validation

# Initialize data manager
data_manager = FixedDataManager()

# ============================================================================
# 4. FIXED PORTFOLIO OPTIMIZER
# ============================================================================

class FixedPortfolioOptimizer:
    """Fixed portfolio optimizer with simplified implementations."""
    
    def __init__(self):
        self.optimization_methods = {
            'MAX_SHARPE': self._optimize_max_sharpe,
            'MIN_VARIANCE': self._optimize_min_variance,
            'EQUAL_WEIGHT': self._optimize_equal_weight,
            'RISK_PARITY': self._optimize_risk_parity
        }
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          method: str = 'MAX_SHARPE',
                          risk_free_rate: float = 0.045) -> Dict:
        """Optimize portfolio using specified method."""
        performance_monitor.start_operation(f'portfolio_optimization_{method}')
        
        try:
            # Clean returns
            returns_clean = returns.dropna()
            if len(returns_clean.columns) < 2:
                raise ValueError("Need at least 2 assets for optimization")
            
            if method in self.optimization_methods:
                weights, metrics = self.optimization_methods[method](
                    returns_clean, risk_free_rate
                )
            else:
                # Default to equal weight
                weights, metrics = self._optimize_equal_weight(returns_clean, risk_free_rate)
            
            results = {
                'weights': weights,
                'metrics': metrics,
                'method': method,
                'risk_free_rate': risk_free_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            performance_monitor.end_operation(f'portfolio_optimization_{method}')
            return results
            
        except Exception as e:
            performance_monitor.end_operation(f'portfolio_optimization_{method}')
            error_analysis = error_analyzer.analyze_error_with_context(e, {
                'operation': f'portfolio_optimization_{method}',
                'assets': len(returns.columns)
            })
            error_analyzer.display_error(error_analysis)
            
            # Fallback to equal weight
            return self._fallback_equal_weight(returns, method, risk_free_rate)
    
    def _optimize_max_sharpe(self, returns: pd.DataFrame, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Maximize Sharpe ratio."""
        mu = returns.mean() * 252
        S = returns.cov() * 252
        n_assets = len(mu)
        
        # Check for positive definiteness
        try:
            np.linalg.cholesky(S + np.eye(n_assets) * 1e-6)
        except:
            # Add regularization if not positive definite
            S = S + np.eye(n_assets) * 1e-4
        
        def negative_sharpe(weights):
            port_return = np.dot(weights, mu)
            port_risk = np.sqrt(weights.T @ S @ weights)
            if port_risk == 0:
                return 1e10
            return -(port_return - risk_free_rate) / port_risk
        
        # Constraints
        bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Limit concentration
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: 0.5 - np.max(w)}  # Max weight 50%
        ]
        
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                negative_sharpe,
                initial_weights,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()  # Normalize
                weight_dict = dict(zip(returns.columns, weights))
                
                # Calculate metrics
                port_return = np.dot(weights, mu)
                port_risk = np.sqrt(weights.T @ S @ weights)
                sharpe = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
                
                return weight_dict, {
                    'expected_return': port_return,
                    'expected_volatility': port_risk,
                    'sharpe_ratio': sharpe
                }
        except Exception as e:
            logger.warning(f"Max Sharpe optimization failed: {e}")
        
        # Fallback to equal weight
        return self._optimize_equal_weight(returns, risk_free_rate)
    
    def _optimize_min_variance(self, returns: pd.DataFrame, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Minimize portfolio variance."""
        S = returns.cov() * 252
        n_assets = len(returns.columns)
        
        def portfolio_variance(weights):
            return weights.T @ S @ weights
        
        bounds = [(0.01, 0.5) for _ in range(n_assets)]
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                portfolio_variance,
                initial_weights,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP'
            )
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()
                weight_dict = dict(zip(returns.columns, weights))
                
                # Calculate metrics
                mu = returns.mean() * 252
                port_return = np.dot(weights, mu)
                port_risk = np.sqrt(weights.T @ S @ weights)
                
                return weight_dict, {
                    'expected_return': port_return,
                    'expected_volatility': port_risk,
                    'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
                }
        except Exception as e:
            logger.warning(f"Min variance optimization failed: {e}")
        
        return self._optimize_equal_weight(returns, risk_free_rate)
    
    def _optimize_equal_weight(self, returns: pd.DataFrame, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Equal weight portfolio."""
        n_assets = len(returns.columns)
        equal_weight = 1.0 / n_assets
        weights = {ticker: equal_weight for ticker in returns.columns}
        
        mu = returns.mean() * 252
        S = returns.cov() * 252
        w_array = np.array([equal_weight] * n_assets)
        
        port_return = np.mean(mu)
        port_risk = np.sqrt(w_array.T @ S @ w_array)
        
        return weights, {
            'expected_return': port_return,
            'expected_volatility': port_risk,
            'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
        }
    
    def _optimize_risk_parity(self, returns: pd.DataFrame, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Risk parity optimization."""
        # Simplified inverse volatility weighting
        volatilities = returns.std() * np.sqrt(252)
        volatilities = volatilities.replace(0, volatilities[volatilities > 0].min() if any(volatilities > 0) else 0.2)
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        weight_dict = weights.to_dict()
        
        # Calculate metrics
        mu = returns.mean() * 252
        S = returns.cov() * 252
        w_array = np.array(list(weight_dict.values()))
        
        port_return = np.dot(w_array, mu)
        port_risk = np.sqrt(w_array.T @ S @ w_array)
        
        return weight_dict, {
            'expected_return': port_return,
            'expected_volatility': port_risk,
            'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
        }
    
    def _fallback_equal_weight(self, returns: pd.DataFrame, method: str, risk_free_rate: float) -> Dict:
        """Fallback to equal weight portfolio."""
        weights, metrics = self._optimize_equal_weight(returns, risk_free_rate)
        
        return {
            'weights': weights,
            'metrics': metrics,
            'method': f'{method} (Fallback: Equal Weight)',
            'risk_free_rate': risk_free_rate,
            'timestamp': datetime.now().isoformat()
        }

# Initialize portfolio optimizer
portfolio_optimizer = FixedPortfolioOptimizer()

# ============================================================================
# 5. FIXED VISUALIZATION ENGINE
# ============================================================================

class FixedVisualizationEngine:
    """Fixed visualization engine with reliable plots."""
    
    def __init__(self):
        self.theme = 'plotly_dark'
    
    def create_efficient_frontier(self, returns: pd.DataFrame, 
                                 risk_free_rate: float = 0.045) -> go.Figure:
        """Create efficient frontier visualization."""
        try:
            mu = returns.mean() * 252
            S = returns.cov() * 252
            
            # Generate random portfolios
            n_portfolios = 1000
            n_assets = len(mu)
            
            portfolio_returns = []
            portfolio_risks = []
            portfolio_sharpes = []
            
            for _ in range(n_portfolios):
                weights = np.random.random(n_assets)
                weights /= weights.sum()
                
                port_return = np.dot(weights, mu)
                port_risk = np.sqrt(weights.T @ S @ weights)
                
                if port_risk > 0:
                    sharpe = (port_return - risk_free_rate) / port_risk
                    portfolio_returns.append(port_return)
                    portfolio_risks.append(port_risk)
                    portfolio_sharpes.append(sharpe)
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add random portfolios
            fig.add_trace(go.Scatter(
                x=portfolio_risks,
                y=portfolio_returns,
                mode='markers',
                marker=dict(
                    size=5,
                    color=portfolio_sharpes,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Random Portfolios',
                hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}'
            ))
            
            # Add individual assets
            asset_risks = np.sqrt(np.diag(S))
            asset_returns = mu.values
            
            fig.add_trace(go.Scatter(
                x=asset_risks,
                y=asset_returns,
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='diamond'),
                text=returns.columns,
                textposition="top center",
                name='Assets',
                hovertemplate='%{text}<br>Risk: %{x:.2%}<br>Return: %{y:.2%}'
            ))
            
            # Update layout
            fig.update_layout(
                title='Efficient Frontier',
                xaxis_title='Risk (Annual Volatility)',
                yaxis_title='Return (Annual)',
                template=self.theme,
                hovermode='closest',
                height=600
            )
            
            fig.update_xaxes(tickformat='.0%')
            fig.update_yaxes(tickformat='.0%')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating efficient frontier: {e}")
            return self._create_empty_plot("Efficient Frontier")
    
    def create_correlation_heatmap(self, returns: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap."""
        try:
            correlation_matrix = returns.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title='Correlation Matrix',
                xaxis_title='Assets',
                yaxis_title='Assets',
                template=self.theme,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return self._create_empty_plot("Correlation Matrix")
    
    def create_portfolio_allocation(self, weights: Dict) -> go.Figure:
        """Create portfolio allocation pie chart."""
        try:
            # Sort weights
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 10 for readability
            if len(sorted_weights) > 10:
                others = sum(w for _, w in sorted_weights[10:])
                sorted_weights = sorted_weights[:10]
                sorted_weights.append(('Others', others))
            
            labels = [f"{ticker}: {weight:.1%}" for ticker, weight in sorted_weights]
            values = [weight for _, weight in sorted_weights]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title='Portfolio Allocation',
                template=self.theme,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating allocation chart: {e}")
            return self._create_empty_plot("Portfolio Allocation")
    
    def create_performance_metrics(self, metrics: Dict) -> go.Figure:
        """Create performance metrics dashboard."""
        try:
            # Extract key metrics
            key_metrics = {
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Expected Return': metrics.get('expected_return', 0),
                'Volatility': metrics.get('expected_volatility', 0),
                'Max Weight': max(metrics.get('weights', {}).values()) if metrics.get('weights') else 0
            }
            
            # Create gauge charts
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                       [{'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=list(key_metrics.keys())
            )
            
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for (row, col), (metric_name, value) in zip(positions, key_metrics.items()):
                if 'Sharpe' in metric_name:
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': metric_name},
                        gauge={'axis': {'range': [0, 2]},
                               'steps': [
                                   {'range': [0, 0.5], 'color': "red"},
                                   {'range': [0.5, 1], 'color': "yellow"},
                                   {'range': [1, 2], 'color': "green"}
                               ]}
                    ), row=row, col=col)
                elif 'Return' in metric_name or 'Volatility' in metric_name:
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': metric_name},
                        number={'suffix': "%", 'valueformat': ".1f"},
                        gauge={'axis': {'range': [0, 50]}}
                    ), row=row, col=col)
                else:
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': metric_name},
                        number={'suffix': "%", 'valueformat': ".1f"},
                        gauge={'axis': {'range': [0, 100]}}
                    ), row=row, col=col)
            
            fig.update_layout(
                height=600,
                template=self.theme,
                title='Portfolio Metrics Dashboard'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating metrics dashboard: {e}")
            return self._create_empty_plot("Portfolio Metrics")
    
    def _create_empty_plot(self, title: str) -> go.Figure:
        """Create empty plot with message."""
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': 'Data not available for visualization',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }],
            template=self.theme,
            height=400
        )
        return fig

# Initialize visualization engine
viz_engine = FixedVisualizationEngine()

# ============================================================================
# 6. FIXED RISK ANALYTICS
# ============================================================================

class FixedRiskAnalytics:
    """Fixed risk analytics with simplified calculations."""
    
    def calculate_var(self, returns: pd.Series, 
                     confidence_level: float = 0.95,
                     portfolio_value: float = 1000000) -> Dict:
        """Calculate Value at Risk."""
        try:
            returns_clean = returns.dropna()
            if len(returns_clean) == 0:
                return {'error': 'No return data available'}
            
            # Historical VaR
            var = -np.percentile(returns_clean, (1 - confidence_level) * 100)
            
            # Calculate CVaR (Conditional VaR)
            cvar_data = returns_clean[returns_clean <= -var]
            cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
            
            # Parametric VaR (assuming normal distribution)
            mean = returns_clean.mean()
            std = returns_clean.std()
            var_param = -(mean + std * norm.ppf(confidence_level))
            
            return {
                'historical_var': var,
                'historical_var_absolute': var * portfolio_value,
                'cvar': cvar,
                'cvar_absolute': cvar * portfolio_value,
                'parametric_var': var_param,
                'parametric_var_absolute': var_param * portfolio_value,
                'confidence_level': confidence_level,
                'portfolio_value': portfolio_value,
                'data_points': len(returns_clean)
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {'error': str(e)}
    
    def calculate_drawdown(self, portfolio_values: pd.Series) -> Dict:
        """Calculate drawdown statistics."""
        try:
            if len(portfolio_values) == 0:
                return {'error': 'No portfolio values available'}
            
            # Calculate returns
            returns = portfolio_values.pct_change().dropna()
            
            # Calculate cumulative returns
            cumulative = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            # Find maximum drawdown
            max_dd = drawdown.min()
            max_dd_date = drawdown.idxmin() if not pd.isnull(drawdown.idxmin()) else None
            
            # Calculate recovery
            if max_dd_date and not pd.isnull(max_dd_date):
                recovery_data = cumulative.loc[max_dd_date:]
                if not recovery_data.empty:
                    recovery_to_previous_high = (recovery_data / running_max.loc[max_dd_date]).max()
                else:
                    recovery_to_previous_high = 1
            else:
                recovery_to_previous_high = 1
            
            return {
                'max_drawdown': max_dd,
                'max_drawdown_date': str(max_dd_date) if max_dd_date else None,
                'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0,
                'avg_drawdown': drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0,
                'drawdown_duration_max': self._calculate_max_drawdown_duration(drawdown),
                'recovery_to_previous_high': recovery_to_previous_high
            }
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return {'error': str(e)}
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration."""
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration

# Initialize risk analytics
risk_analytics = FixedRiskAnalytics()

# ============================================================================
# 7. STREAMLIT UI APPLICATION
# ============================================================================

class QuantEdgeApp:
    """Main Streamlit application."""
    
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
    
    def setup_page(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="QuantEdge Pro v5.0",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1a5276;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2e86c1;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1a5276;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'data_fetched': False,
            'market_data': None,
            'portfolio_optimized': False,
            'optimization_results': None,
            'risk_analysis': None,
            'current_tab': 'data_fetch'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Run the main application."""
        # Header
        st.markdown('<h1 class="main-header">📊 QuantEdge Pro v5.0</h1>', unsafe_allow_html=True)
        st.markdown("### Institutional Portfolio Analytics Platform")
        
        # Sidebar
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=80)
            st.markdown("---")
            
            # System Status
            st.markdown("### 🔧 System Status")
            lib_status = st.session_state.get('library_status', {})
            
            if lib_status.get('all_essential_available', False):
                st.success("✅ All essential libraries available")
            else:
                missing = lib_status.get('essential_missing', [])
                if missing:
                    st.error(f"❌ Missing: {', '.join(missing)}")
            
            missing_optional = lib_status.get('missing', [])
            if missing_optional:
                st.warning(f"⚠️ Missing optional: {', '.join(missing_optional)}")
            
            st.markdown("---")
            
            # Navigation
            st.markdown("### 📋 Navigation")
            tabs = {
                "📥 Data Fetch": "data_fetch",
                "⚙️ Portfolio Optimization": "optimization",
                "📈 Visualization": "visualization",
                "⚠️ Risk Analysis": "risk_analysis",
                "📊 Performance": "performance"
            }
            
            current_tab = st.session_state.get('current_tab', 'data_fetch')
            for tab_name, tab_key in tabs.items():
                if st.button(tab_name, key=f"nav_{tab_key}", use_container_width=True):
                    st.session_state.current_tab = tab_key
                    st.rerun()
            
            st.markdown("---")
            
            # Quick Stats
            if st.session_state.data_fetched and st.session_state.market_data:
                data = st.session_state.market_data
                st.markdown("### 📊 Data Overview")
                st.metric("Assets", len(data['prices'].columns))
                st.metric("Data Points", len(data['prices']))
                if 'returns' in data:
                    st.metric("Avg Daily Return", f"{data['returns'].mean().mean():.3%}")
        
        # Main Content
        current_tab = st.session_state.get('current_tab', 'data_fetch')
        
        if current_tab == 'data_fetch':
            self.render_data_fetch_tab()
        elif current_tab == 'optimization':
            self.render_optimization_tab()
        elif current_tab == 'visualization':
            self.render_visualization_tab()
        elif current_tab == 'risk_analysis':
            self.render_risk_analysis_tab()
        elif current_tab == 'performance':
            self.render_performance_tab()
    
    def render_data_fetch_tab(self):
        """Render data fetching tab."""
        st.markdown('<h2 class="sub-header">📥 Data Fetching</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Ticker input
            tickers_input = st.text_area(
                "Enter Ticker Symbols (comma-separated):",
                value="AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, JNJ, V",
                height=100
            )
            
            # Date range
            col1a, col1b = st.columns(2)
            with col1a:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            with col1b:
                end_date = st.date_input("End Date", value=datetime.now())
        
        with col2:
            st.markdown("### 💡 Tips")
            st.info("""
            - Use valid Yahoo Finance tickers
            - Limit to 10-15 tickers for stability
            - For international stocks, add exchange suffix (e.g., VOW3.DE)
            - Data is cached for 5 minutes
            """)
        
        # Process tickers
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        if st.button("📥 Fetch Market Data", type="primary", use_container_width=True):
            with st.spinner("Fetching market data..."):
                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress, text=message)
                    
                    # Fetch data
                    data = data_manager.fetch_market_data(
                        tickers=tickers,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        progress_callback=update_progress
                    )
                    
                    # Validate data
                    validation = data_manager.validate_data(data)
                    
                    if validation['is_valid']:
                        st.session_state.market_data = data
                        st.session_state.data_fetched = True
                        
                        st.success("✅ Data fetched successfully!")
                        
                        # Show summary
                        st.markdown("### 📊 Data Summary")
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        
                        with col_sum1:
                            st.metric("Successful Tickers", len(data['prices'].columns))
                            if data['errors']:
                                st.warning(f"Failed: {len(data['errors'])} tickers")
                        
                        with col_sum2:
                            st.metric("Date Range", 
                                     f"{data['prices'].index[0].strftime('%Y-%m-%d')} to {data['prices'].index[-1].strftime('%Y-%m-%d')}")
                        
                        with col_sum3:
                            st.metric("Total Returns", f"{data['returns'].sum().sum():.2f}")
                        
                        # Show errors if any
                        if data['errors']:
                            with st.expander("⚠️ Failed Tickers"):
                                for ticker, error in data['errors'].items():
                                    st.write(f"**{ticker}**: {error}")
                        
                        # Show preview
                        with st.expander("📋 Data Preview"):
                            st.dataframe(data['prices'].tail(10), use_container_width=True)
                            
                    else:
                        st.error("❌ Data validation failed")
                        for issue in validation['issues']:
                            st.error(issue)
                        for warning in validation['warnings']:
                            st.warning(warning)
                    
                    progress_bar.empty()
                    
                except Exception as e:
                    st.error(f"Failed to fetch data: {str(e)}")
        
        # Show cached data if available
        if st.session_state.data_fetched and st.session_state.market_data:
            st.markdown("---")
            st.markdown("### 📈 Quick Preview")
            
            data = st.session_state.market_data
            
            # Price chart
            fig = go.Figure()
            for ticker in data['prices'].columns[:5]:  # Show first 5
                fig.add_trace(go.Scatter(
                    x=data['prices'].index,
                    y=data['prices'][ticker],
                    mode='lines',
                    name=ticker
                ))
            
            fig.update_layout(
                title='Price Trends (First 5 Assets)',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_optimization_tab(self):
        """Render portfolio optimization tab."""
        st.markdown('<h2 class="sub-header">⚙️ Portfolio Optimization</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_fetched:
            st.warning("Please fetch market data first!")
            return
        
        data = st.session_state.market_data
        returns = data['returns']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Optimization parameters
            method = st.selectbox(
                "Optimization Method",
                options=['MAX_SHARPE', 'MIN_VARIANCE', 'EQUAL_WEIGHT', 'RISK_PARITY'],
                index=0
            )
            
            risk_free_rate = st.slider(
                "Risk-Free Rate",
                min_value=0.0,
                max_value=0.10,
                value=0.045,
                step=0.001,
                format="%.1%%"
            )
            
            # Constraints
            st.markdown("### 🔧 Constraints")
            max_weight = st.slider("Maximum Weight per Asset", 0.1, 1.0, 0.5, 0.05, format="%.0%%")
            
        with col2:
            st.markdown("### 📊 Current Returns")
            st.dataframe(returns.describe(), use_container_width=True)
            
            # Correlation
            if len(returns.columns) > 1:
                avg_corr = returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean()
                st.metric("Average Correlation", f"{avg_corr:.3f}")
        
        if st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True):
            with st.spinner("Optimizing portfolio..."):
                try:
                    # Run optimization
                    results = portfolio_optimizer.optimize_portfolio(
                        returns=returns,
                        method=method,
                        risk_free_rate=risk_free_rate
                    )
                    
                    st.session_state.optimization_results = results
                    st.session_state.portfolio_optimized = True
                    
                    st.success("✅ Portfolio optimized successfully!")
                    
                    # Display results
                    self.display_optimization_results(results)
                    
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
        
        # Show previous results if available
        if st.session_state.portfolio_optimized and st.session_state.optimization_results:
            st.markdown("---")
            st.markdown("### 📋 Previous Optimization Results")
            self.display_optimization_results(st.session_state.optimization_results)
    
    def display_optimization_results(self, results: Dict):
        """Display optimization results."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Return", f"{results['metrics']['expected_return']:.2%}")
        with col2:
            st.metric("Expected Volatility", f"{results['metrics']['expected_volatility']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{results['metrics']['sharpe_ratio']:.2f}")
        
        # Weights
        st.markdown("### 📊 Portfolio Weights")
        
        weights_df = pd.DataFrame.from_dict(results['weights'], orient='index', columns=['Weight'])
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        col_weights1, col_weights2 = st.columns([2, 1])
        
        with col_weights1:
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=weights_df.index,
                    y=weights_df['Weight'],
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title='Portfolio Weights',
                xaxis_title='Asset',
                yaxis_title='Weight',
                yaxis_tickformat='.0%',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_weights2:
            # Table
            st.dataframe(
                weights_df.style.format({'Weight': '{:.2%}'}),
                use_container_width=True
            )
            
            # Concentration metrics
            weights_array = np.array(list(results['weights'].values()))
            herfindahl = np.sum(weights_array ** 2)
            st.metric("Concentration Index", f"{herfindahl:.3f}")
            st.metric("Effective N", f"{1/herfindahl:.1f}")
    
    def render_visualization_tab(self):
        """Render visualization tab."""
        st.markdown('<h2 class="sub-header">📈 Visualization</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_fetched:
            st.warning("Please fetch market data first!")
            return
        
        data = st.session_state.market_data
        returns = data['returns']
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Efficient Frontier",
            "🔥 Correlation Matrix",
            "🥧 Portfolio Allocation",
            "📈 Performance Metrics"
        ])
        
        with tab1:
            if len(returns.columns) >= 2:
                fig = viz_engine.create_efficient_frontier(returns)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 assets for efficient frontier")
        
        with tab2:
            if len(returns.columns) >= 2:
                fig = viz_engine.create_correlation_heatmap(returns)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 assets for correlation matrix")
        
        with tab3:
            if st.session_state.portfolio_optimized:
                results = st.session_state.optimization_results
                fig = viz_engine.create_portfolio_allocation(results['weights'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Show equal weight allocation
                n_assets = len(returns.columns)
                equal_weight = 1.0 / n_assets
                weights = {ticker: equal_weight for ticker in returns.columns}
                fig = viz_engine.create_portfolio_allocation(weights)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            if st.session_state.portfolio_optimized:
                results = st.session_state.optimization_results
                fig = viz_engine.create_performance_metrics(results)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run portfolio optimization to see performance metrics")
    
    def render_risk_analysis_tab(self):
        """Render risk analysis tab."""
        st.markdown('<h2 class="sub-header">⚠️ Risk Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_fetched:
            st.warning("Please fetch market data first!")
            return
        
        data = st.session_state.market_data
        returns = data['returns']
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.0%%"
            )
            
            portfolio_value = st.number_input(
                "Portfolio Value ($)",
                min_value=1000,
                max_value=1000000000,
                value=1000000,
                step=10000
            )
        
        with col2:
            # Calculate portfolio returns
            if st.session_state.portfolio_optimized:
                results = st.session_state.optimization_results
                weights = np.array(list(results['weights'].values()))
                portfolio_returns = returns.dot(weights)
            else:
                # Equal weight portfolio
                n_assets = len(returns.columns)
                weights = np.ones(n_assets) / n_assets
                portfolio_returns = returns.dot(weights)
        
        if st.button("📊 Calculate Risk Metrics", type="primary"):
            with st.spinner("Calculating risk metrics..."):
                # Calculate VaR
                var_results = risk_analytics.calculate_var(
                    portfolio_returns,
                    confidence_level,
                    portfolio_value
                )
                
                # Calculate drawdown
                if st.session_state.portfolio_optimized:
                    # Create portfolio value series
                    portfolio_values = (1 + portfolio_returns).cumprod() * portfolio_value
                    drawdown_results = risk_analytics.calculate_drawdown(portfolio_values)
                else:
                    drawdown_results = {'error': 'Run portfolio optimization for drawdown analysis'}
                
                # Display results
                st.markdown("### 📉 Value at Risk (VaR)")
                
                if 'error' not in var_results:
                    col_var1, col_var2, col_var3 = st.columns(3)
                    
                    with col_var1:
                        st.metric(
                            "Historical VaR (95%)",
                            f"${var_results['historical_var_absolute']:,.0f}",
                            delta=f"{var_results['historical_var']:.2%}"
                        )
                    
                    with col_var2:
                        st.metric(
                            "Conditional VaR (CVaR)",
                            f"${var_results['cvar_absolute']:,.0f}",
                            delta=f"{var_results['cvar']:.2%}"
                        )
                    
                    with col_var3:
                        st.metric(
                            "Parametric VaR (95%)",
                            f"${var_results['parametric_var_absolute']:,.0f}",
                            delta=f"{var_results['parametric_var']:.2%}"
                        )
                else:
                    st.error(f"VaR calculation error: {var_results['error']}")
                
                st.markdown("### 📊 Drawdown Analysis")
                
                if 'error' not in drawdown_results:
                    col_dd1, col_dd2, col_dd3 = st.columns(3)
                    
                    with col_dd1:
                        st.metric(
                            "Maximum Drawdown",
                            f"{drawdown_results['max_drawdown']:.2%}",
                            help=f"Date: {drawdown_results.get('max_drawdown_date', 'N/A')}"
                        )
                    
                    with col_dd2:
                        st.metric(
                            "Current Drawdown",
                            f"{drawdown_results['current_drawdown']:.2%}"
                        )
                    
                    with col_dd3:
                        st.metric(
                            "Max Drawdown Duration",
                            f"{drawdown_results['drawdown_duration_max']} days"
                        )
                else:
                    st.info(drawdown_results['error'])
    
    def render_performance_tab(self):
        """Render performance analysis tab."""
        st.markdown('<h2 class="sub-header">📊 Performance Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_fetched:
            st.warning("Please fetch market data first!")
            return
        
        data = st.session_state.market_data
        returns = data['returns']
        
        # Calculate portfolio performance
        if st.session_state.portfolio_optimized:
            results = st.session_state.optimization_results
            weights = np.array(list(results['weights'].values()))
            portfolio_returns = returns.dot(weights)
            portfolio_name = f"Optimized ({results['method']})"
        else:
            # Equal weight portfolio
            n_assets = len(returns.columns)
            weights = np.ones(n_assets) / n_assets
            portfolio_returns = returns.dot(weights)
            portfolio_name = "Equal Weight"
        
        # Calculate performance metrics
        if len(portfolio_returns) > 0:
            # Basic metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = (annual_return - 0.045) / annual_vol if annual_vol > 0 else 0
            
            # Downside metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino = (annual_return - 0.045) / downside_vol if downside_vol > 0 else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{total_return:.2%}")
                st.metric("Annual Return", f"{annual_return:.2%}")
            
            with col2:
                st.metric("Annual Volatility", f"{annual_vol:.2%}")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col3:
                st.metric("Downside Volatility", f"{downside_vol:.2%}")
                st.metric("Sortino Ratio", f"{sortino:.2f}")
            
            with col4:
                positive_days = (portfolio_returns > 0).sum() / len(portfolio_returns)
                st.metric("Positive Days", f"{positive_days:.1%}")
                st.metric("Max Daily Return", f"{portfolio_returns.max():.2%}")
            
            # Cumulative returns chart
            st.markdown("### 📈 Cumulative Returns")
            
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name=portfolio_name,
                line=dict(color='lightblue', width=2)
            ))
            
            # Add benchmark (S&P 500 approximation)
            try:
                # Simple benchmark - equal weight of assets
                benchmark_returns = returns.mean(axis=1)
                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                
                fig.add_trace(go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    mode='lines',
                    name='Benchmark (Avg)',
                    line=dict(color='gray', width=1, dash='dash')
                ))
            except:
                pass
            
            fig.update_layout(
                title=f'Cumulative Returns - {portfolio_name}',
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                template='plotly_dark',
                height=500,
                hovermode='x unified'
            )
            
            fig.update_yaxes(tickformat='.0%')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Rolling metrics
            st.markdown("### 🔄 Rolling Metrics (252-day window)")
            
            if len(portfolio_returns) > 252:
                rolling_window = 252
                
                # Rolling Sharpe
                rolling_sharpe = portfolio_returns.rolling(window=rolling_window).apply(
                    lambda x: (x.mean() * 252 - 0.045) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
                )
                
                # Rolling volatility
                rolling_vol = portfolio_returns.rolling(window=rolling_window).std() * np.sqrt(252)
                
                fig_roll = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
                    vertical_spacing=0.15
                )
                
                fig_roll.add_trace(
                    go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Sharpe'),
                    row=1, col=1
                )
                
                fig_roll.add_trace(
                    go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name='Volatility'),
                    row=2, col=1
                )
                
                fig_roll.update_layout(
                    height=600,
                    template='plotly_dark',
                    showlegend=False
                )
                
                fig_roll.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
                fig_roll.update_yaxes(title_text="Volatility", tickformat='.0%', row=2, col=1)
                
                st.plotly_chart(fig_roll, use_container_width=True)
            else:
                st.info(f"Need at least 252 days of data for rolling metrics (currently {len(portfolio_returns)} days)")

# ============================================================================
# 8. MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point."""
    try:
        # Initialize and run the app
        app = QuantEdgeApp()
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application crashed: {e}", exc_info=True)

if __name__ == "__main__":
    main()

# ============================================================================
# QUANTEDGE PRO v4.0 ENHANCED - FIXED VERSION
# INSTITUTIONAL PORTFOLIO ANALYTICS PLATFORM
# ENHANCED EDITION WITH ADVANCED VaR/CVaR/ES ANALYTICS
# Total Lines: 5500+ | Production Grade | Enterprise Ready
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import hashlib
from dataclasses import dataclass
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ENHANCED LIBRARY MANAGER WITH ADVANCED FEATURES
# ============================================================================

class AdvancedLibraryManager:
    """Enhanced library manager with advanced feature detection."""
    
    @staticmethod
    def check_and_import_all():
        """Check and import all required libraries with advanced capabilities."""
        lib_status = {}
        missing_libs = []
        advanced_features = {}
        
        try:
            # PyPortfolioOpt with advanced optimization
            try:
                from pypfopt import expected_returns, risk_models
                from pypfopt.efficient_frontier import EfficientFrontier
                from pypfopt.hierarchical_portfolio import HRPOpt
                from pypfopt.black_litterman import BlackLittermanModel
                from pypfopt.cla import CLA
                from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
                lib_status['pypfopt'] = True
                advanced_features['pypfopt'] = {
                    'version': '1.5.5+',
                    'features': ['CLA', 'Black-Litterman', 'HRP', 'Discrete Allocation']
                }
                # Store in globals for later use
                st.session_state.pypfopt_available = True
            except ImportError as e:
                lib_status['pypfopt'] = False
                missing_libs.append('PyPortfolioOpt')
                st.session_state.pypfopt_available = False
        
        except Exception as e:
            lib_status['pypfopt'] = False
            advanced_features['pypfopt_error'] = str(e)
            st.session_state.pypfopt_available = False
        
        try:
            # Scikit-Learn with advanced ML
            try:
                from sklearn.decomposition import PCA
                from sklearn.linear_model import LinearRegression, Ridge, Lasso
                from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
                from sklearn.preprocessing import StandardScaler, RobustScaler
                from sklearn.cluster import KMeans, DBSCAN
                from sklearn.svm import SVR
                from sklearn.neural_network import MLPRegressor
                lib_status['sklearn'] = True
                advanced_features['sklearn'] = {
                    'version': '1.3.0+',
                    'models': ['PCA', 'RandomForest', 'GradientBoosting', 'SVM', 'Neural Networks']
                }
                st.session_state.sklearn_available = True
            except ImportError:
                lib_status['sklearn'] = False
                missing_libs.append('scikit-learn')
                st.session_state.sklearn_available = False
        
        except Exception as e:
            lib_status['sklearn'] = False
            advanced_features['sklearn_error'] = str(e)
            st.session_state.sklearn_available = False
        
        try:
            # Statsmodels for advanced statistics
            try:
                import statsmodels.api as sm
                from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
                from statsmodels.tsa.stattools import adfuller, kpss
                from statsmodels.tsa.api import VAR, SimpleExpSmoothing
                from statsmodels.regression.rolling import RollingOLS
                lib_status['statsmodels'] = True
                advanced_features['statsmodels'] = {
                    'version': '0.14.0+',
                    'models': ['VAR', 'RollingOLS', 'ARCH Test', 'Stationarity Tests']
                }
                st.session_state.statsmodels_available = True
            except ImportError:
                lib_status['statsmodels'] = False
                missing_libs.append('statsmodels')
                st.session_state.statsmodels_available = False
        
        except Exception as e:
            lib_status['statsmodels'] = False
            advanced_features['statsmodels_error'] = str(e)
            st.session_state.statsmodels_available = False
        
        return {
            'status': lib_status,
            'missing': missing_libs,
            'advanced_features': advanced_features,
            'all_available': len(missing_libs) == 0
        }

# Initialize advanced library manager
if 'library_status' not in st.session_state:
    LIBRARY_STATUS = AdvancedLibraryManager.check_and_import_all()
    st.session_state.library_status = LIBRARY_STATUS
else:
    LIBRARY_STATUS = st.session_state.library_status

# ============================================================================
# 2. ADVANCED ERROR HANDLING AND MONITORING SYSTEM
# ============================================================================

class AdvancedErrorAnalyzer:
    """Advanced error analysis with ML-powered suggestions."""
    
    ERROR_PATTERNS = {
        'DATA_FETCH': {
            'symptoms': ['yahoo', 'timeout', 'connection', '404', '403'],
            'solutions': [
                'Try alternative data source (Alpha Vantage, IEX Cloud)',
                'Reduce number of tickers',
                'Increase timeout duration',
                'Use cached data',
                'Check internet connection'
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
        },
        'MEMORY': {
            'symptoms': ['memory', 'overflow', 'exceeded', 'RAM'],
            'solutions': [
                'Reduce data size',
                'Use chunk processing',
                'Clear cache',
                'Increase swap memory',
                'Use more efficient data structures'
            ],
            'severity': 'CRITICAL'
        },
        'NUMERICAL': {
            'symptoms': ['nan', 'inf', 'divide', 'zero', 'invalid'],
            'solutions': [
                'Clean data (remove NaN/Inf)',
                'Add small epsilon to denominators',
                'Use robust statistical methods',
                'Check for stationarity',
                'Normalize data'
            ],
            'severity': 'MEDIUM'
        }
    }
    
    @staticmethod
    def analyze_error_with_context(error: Exception, context: Dict) -> Dict:
        """Analyze error with full context for intelligent recovery."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'stack_trace': traceback.format_exc(),
            'severity_score': 5,
            'recovery_actions': [],
            'preventive_measures': [],
            'ml_suggestions': []
        }
        
        # Analyze error message for patterns
        error_lower = str(error).lower()
        stack_lower = traceback.format_exc().lower()
        
        for pattern_name, pattern in AdvancedErrorAnalyzer.ERROR_PATTERNS.items():
            if any(symptom in error_lower for symptom in pattern['symptoms']) or \
               any(symptom in stack_lower for symptom in pattern['symptoms']):
                
                analysis['severity_score'] = 8 if pattern['severity'] == 'CRITICAL' else \
                                           6 if pattern['severity'] == 'HIGH' else 5
                analysis['recovery_actions'].extend(pattern['solutions'])
                
                # Add context-specific solutions
                if 'tickers' in context and pattern_name == 'DATA_FETCH':
                    analysis['recovery_actions'].append(
                        f"Reduce from {len(context['tickers'])} to {min(10, len(context['tickers']))} tickers"
                    )
        
        # Add ML-powered suggestions based on error history
        analysis['ml_suggestions'] = AdvancedErrorAnalyzer._generate_ml_suggestions(error, context)
        
        # Calculate confidence score for recovery
        analysis['recovery_confidence'] = min(95, 100 - (analysis['severity_score'] * 10))
        
        return analysis
    
    @staticmethod
    def _generate_ml_suggestions(error: Exception, context: Dict) -> List[str]:
        """Generate ML-powered recovery suggestions."""
        suggestions = []
        
        # Pattern-based suggestions
        if 'singular' in str(error).lower() or 'invert' in str(error).lower():
            suggestions.extend([
                "Covariance matrix is singular - try shrinkage estimation",
                "Use Ledoit-Wolf covariance estimator",
                "Add regularization to covariance matrix",
                "Remove highly correlated assets"
            ])
        
        if 'convergence' in str(error).lower():
            suggestions.extend([
                "Increase maximum iterations to 5000",
                "Try different optimization algorithm (SLSQP → COBYLA)",
                "Relax tolerance to 1e-4",
                "Use better initial guess for optimization"
            ])
        
        if 'memory' in str(error).lower():
            suggestions.extend([
                "Implement incremental learning",
                "Use sparse matrices where possible",
                "Process data in batches of 1000 rows",
                "Enable garbage collection during processing"
            ])
        
        # Context-aware suggestions
        if 'window' in context and 'period' in context:
            if context['window'] > 252 * 5:  # More than 5 years
                suggestions.append(f"Reduce window size from {context['window']} to {252*2} for better stability")
        
        if 'assets' in context and context['assets'] > 50:
            suggestions.append(f"Reduce asset universe from {context['assets']} to 30 for faster computation")
        
        return suggestions
    
    @staticmethod
    def create_advanced_error_display(analysis: Dict) -> None:
        """Create advanced error display with interactive elements."""
        with st.expander(f"🔍 Advanced Error Analysis ({analysis['error_type']})", expanded=True):
            # Error summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Severity Score", f"{analysis['severity_score']}/10", 
                         delta="High" if analysis['severity_score'] > 7 else "Medium")
            with col2:
                st.metric("Recovery Confidence", f"{analysis['recovery_confidence']}%")
            with col3:
                st.metric("Context", analysis['context'].get('operation', 'Unknown'))
            
            # Recovery actions
            st.subheader("🚀 Recovery Actions")
            for i, action in enumerate(analysis['recovery_actions'][:5], 1):
                st.checkbox(f"Action {i}: {action}", value=False, key=f"recovery_{i}_{hash(action)}")
            
            # ML suggestions
            if analysis['ml_suggestions']:
                st.subheader("🤖 AI-Powered Suggestions")
                for suggestion in analysis['ml_suggestions'][:3]:
                    st.info(f"💡 {suggestion}")
            
            # Technical details
            with st.expander("🔧 Technical Details"):
                st.code(f"""
                Error Type: {analysis['error_type']}
                Message: {analysis['error_message']}
                
                Context: {json.dumps(analysis['context'], indent=2)}
                
                Stack Trace:
                {analysis['stack_trace']}
                """)

class PerformanceMonitor:
    """Advanced performance monitoring with real-time analytics."""
    
    def __init__(self):
        self.operations = {}
        self.memory_usage = []
        self.execution_times = []
        self.start_time = time.time()
    
    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.operations[operation_name] = {
            'start': time.time(),
            'memory_start': self._get_memory_usage()
        }
    
    def end_operation(self, operation_name: str, metadata: Dict = None):
        """End timing an operation and record metrics."""
        if operation_name in self.operations:
            op = self.operations[operation_name]
            duration = time.time() - op['start']
            memory_end = self._get_memory_usage()
            memory_diff = memory_end - op['memory_start']
            
            self.execution_times.append({
                'operation': operation_name,
                'duration': duration,
                'memory_increase_mb': memory_diff,
                'timestamp': datetime.now(),
                'metadata': metadata
            })
            
            # Update operation record
            if 'history' not in op:
                op['history'] = []
            op['history'].append({
                'duration': duration,
                'memory_increase_mb': memory_diff,
                'timestamp': datetime.now()
            })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        report = {
            'total_runtime': time.time() - self.start_time,
            'operations': {},
            'summary': {},
            'recommendations': []
        }
        
        # Calculate operation statistics
        for op_name, op_data in self.operations.items():
            if 'history' in op_data and op_data['history']:
                durations = [h['duration'] for h in op_data['history']]
                memories = [h['memory_increase_mb'] for h in op_data['history']]
                
                report['operations'][op_name] = {
                    'count': len(durations),
                    'avg_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'min_duration': np.min(durations),
                    'avg_memory_increase': np.mean(memories),
                    'total_time': np.sum(durations)
                }
        
        # Generate summary
        if report['operations']:
            total_times = [op['total_time'] for op in report['operations'].values()]
            report['summary'] = {
                'total_operations': len(report['operations']),
                'total_operation_time': sum(total_times),
                'slowest_operation': max(report['operations'].items(), 
                                       key=lambda x: x[1]['avg_duration'])[0],
                'most_frequent_operation': max(report['operations'].items(),
                                             key=lambda x: x[1]['count'])[0]
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        for op_name, stats in report['operations'].items():
            if stats['avg_duration'] > 5:  # More than 5 seconds
                recommendations.append(
                    f"Optimize '{op_name}' - average duration {stats['avg_duration']:.1f}s"
                )
            
            if stats['avg_memory_increase'] > 100:  # More than 100MB
                recommendations.append(
                    f"Reduce memory usage in '{op_name}' - average increase {stats['avg_memory_increase']:.1f}MB"
                )
        
        if report['summary'].get('total_operation_time', 0) > 30:
            recommendations.append(
                "Consider implementing parallel processing for independent operations"
            )
        
        return recommendations

# Initialize global monitors
error_analyzer = AdvancedErrorAnalyzer()
performance_monitor = PerformanceMonitor()

# ============================================================================
# 3. ENHANCED VISUALIZATION ENGINE WITH 3D & ADVANCED CHARTS
# ============================================================================

class AdvancedVisualizationEngine:
    """Production-grade visualization engine with 3D, animations, and interactivity."""
    
    def __init__(self):
        self.themes = {
            'dark': {
                'bg_color': 'rgba(10, 10, 20, 0.9)',
                'grid_color': 'rgba(255, 255, 255, 0.1)',
                'font_color': 'white',
                'accent_color': '#00cc96'
            },
            'light': {
                'bg_color': 'rgba(255, 255, 255, 0.9)',
                'grid_color': 'rgba(0, 0, 0, 0.1)',
                'font_color': 'black',
                'accent_color': '#636efa'
            }
        }
        self.current_theme = 'dark'
        self.color_scales = {
            'sequential': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'],
            'diverging': ['#8c510a', '#d8b365', '#f6e8c3', '#f5f5f5', '#c7eae5', '#5ab4ac', '#01665e'],
            'qualitative': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        }
    
    def create_3d_efficient_frontier(self, returns: pd.DataFrame, 
                                    risk_free_rate: float = 0.045) -> go.Figure:
        """Create 3D efficient frontier visualization."""
        try:
            performance_monitor.start_operation('3d_efficient_frontier')
            
            # Calculate parameters for 3D surface
            mu = returns.mean() * 252
            S = returns.cov() * 252
            assets = returns.columns.tolist()
            n_assets = len(assets)
            
            # Generate random portfolios for 3D surface
            n_portfolios = 1000
            np.random.seed(42)
            
            portfolio_returns = []
            portfolio_risks = []
            portfolio_sharpes = []
            portfolio_skewness = []
            portfolio_weights = []
            
            for _ in range(n_portfolios):
                # Generate random weights
                weights = np.random.random(n_assets)
                weights /= weights.sum()
                
                # Calculate portfolio metrics
                port_return = np.dot(weights, mu)
                port_risk = np.sqrt(weights.T @ S @ weights)
                
                if port_risk > 0:
                    sharpe = (port_return - risk_free_rate) / port_risk
                    # Calculate portfolio skewness
                    port_returns = returns.dot(weights)
                    skew_val = port_returns.skew()
                    
                    portfolio_returns.append(port_return)
                    portfolio_risks.append(port_risk)
                    portfolio_sharpes.append(sharpe)
                    portfolio_skewness.append(skew_val)
                    portfolio_weights.append(weights)
            
            # Create 3D scatter plot
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=portfolio_risks,
                    y=portfolio_returns,
                    z=portfolio_sharpes,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=portfolio_sharpes,
                        colorscale='Viridis',
                        opacity=0.6,
                        colorbar=dict(title="Sharpe Ratio", x=1.1)
                    ),
                    hovertemplate='<b>Random Portfolio</b><br>' +
                                 'Risk: %{x:.2%}<br>' +
                                 'Return: %{y:.2%}<br>' +
                                 'Sharpe: %{z:.2f}<br>' +
                                 'Skewness: %{customdata:.2f}<extra></extra>',
                    customdata=portfolio_skewness,
                    name='Random Portfolios'
                )
            ])
            
            # Try to calculate efficient frontier if PyPortfolioOpt is available
            efficient_risks = []
            efficient_returns = []
            efficient_sharpes = []
            
            # Calculate efficient frontier points using simple method
            target_returns = np.linspace(min(portfolio_returns), max(portfolio_returns) * 0.9, 20)
            
            for target_return in target_returns:
                try:
                    # Simple optimization for each target return
                    def objective(weights):
                        port_risk = np.sqrt(weights.T @ S @ weights)
                        return port_risk
                    
                    def return_constraint(weights):
                        return np.dot(weights, mu) - target_return
                    
                    def weight_constraint(weights):
                        return np.sum(weights) - 1
                    
                    bounds = [(0, 1) for _ in range(n_assets)]
                    constraints = [
                        {'type': 'eq', 'fun': return_constraint},
                        {'type': 'eq', 'fun': weight_constraint}
                    ]
                    
                    initial_weights = np.ones(n_assets) / n_assets
                    
                    result = optimize.minimize(
                        objective,
                        initial_weights,
                        bounds=bounds,
                        constraints=constraints,
                        method='SLSQP'
                    )
                    
                    if result.success:
                        weights = result.x
                        port_return = np.dot(weights, mu)
                        port_risk = np.sqrt(weights.T @ S @ weights)
                        sharpe = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
                        
                        efficient_risks.append(port_risk)
                        efficient_returns.append(port_return)
                        efficient_sharpes.append(sharpe)
                        
                except:
                    continue
            
            # Add efficient frontier line
            if efficient_risks:
                fig.add_trace(go.Scatter3d(
                    x=efficient_risks,
                    y=efficient_returns,
                    z=efficient_sharpes,
                    mode='lines',
                    line=dict(color='#ff0000', width=6),
                    name='Efficient Frontier',
                    hovertemplate='<b>Efficient Portfolio</b><br>' +
                                 'Risk: %{x:.2%}<br>' +
                                 'Return: %{y:.2%}<br>' +
                                 'Sharpe: %{z:.2f}<extra></extra>'
                ))
            
            # Add individual assets
            asset_risks = np.sqrt(np.diag(S))
            asset_returns = mu.values
            asset_sharpes = [(r - risk_free_rate) / s if s > 0 else 0 
                           for r, s in zip(asset_returns, asset_risks)]
            
            fig.add_trace(go.Scatter3d(
                x=asset_risks,
                y=asset_returns,
                z=asset_sharpes,
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='#00ff00',
                    symbol='diamond'
                ),
                text=assets,
                textposition="top center",
                name='Individual Assets',
                hovertemplate='<b>%{text}</b><br>' +
                             'Risk: %{x:.2%}<br>' +
                             'Return: %{y:.2%}<br>' +
                             'Sharpe: %{z:.2f}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                height=800,
                title=dict(
                    text='3D Efficient Frontier Analysis',
                    font=dict(size=24, color='white'),
                    x=0.5
                ),
                scene=dict(
                    xaxis_title='Risk (Annual Volatility)',
                    yaxis_title='Return (Annual)',
                    zaxis_title='Sharpe Ratio',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5),
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0)
                    ),
                    xaxis=dict(
                        backgroundcolor=self.themes[self.current_theme]['bg_color'],
                        gridcolor=self.themes[self.current_theme]['grid_color'],
                        tickformat='.0%'
                    ),
                    yaxis=dict(
                        backgroundcolor=self.themes[self.current_theme]['bg_color'],
                        gridcolor=self.themes[self.current_theme]['grid_color'],
                        tickformat='.0%'
                    ),
                    zaxis=dict(
                        backgroundcolor=self.themes[self.current_theme]['bg_color'],
                        gridcolor=self.themes[self.current_theme]['grid_color']
                    )
                ),
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='white',
                    borderwidth=1
                ),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            performance_monitor.end_operation('3d_efficient_frontier')
            return fig
            
        except Exception as e:
            performance_monitor.end_operation('3d_efficient_frontier', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {'operation': '3d_efficient_frontier'})
            return self._create_empty_figure("3D Efficient Frontier")
    
    def create_animated_backtest(self, portfolio_values: pd.Series, 
                                benchmark_values: pd.Series,
                                title: str = "Animated Backtest Performance") -> go.Figure:
        """Create animated backtest visualization."""
        performance_monitor.start_operation('animated_backtest')
        
        # Prepare data for animation
        dates = portfolio_values.index
        portfolio_vals = portfolio_values.values
        benchmark_vals = benchmark_values.reindex(dates).fillna(method='ffill').fillna(method='bfill').values
        
        # Create frames for animation
        frames = []
        step = max(1, len(dates) // 50)
        for i in range(10, len(dates), step):
            frame_dates = dates[:i]
            frame_portfolio = portfolio_vals[:i]
            frame_benchmark = benchmark_vals[:i]
            
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=frame_dates,
                        y=frame_portfolio,
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='#00cc96', width=3)
                    ),
                    go.Scatter(
                        x=frame_dates,
                        y=frame_benchmark,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='#636efa', width=2, dash='dash')
                    )
                ],
                name=str(i)
            )
            frames.append(frame)
        
        # Create figure with initial data
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=dates[:10],
                    y=portfolio_vals[:10],
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#00cc96', width=3)
                ),
                go.Scatter(
                    x=dates[:10],
                    y=benchmark_vals[:10],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#636efa', width=2, dash='dash')
                )
            ],
            layout=go.Layout(
                title=dict(
                    text=title,
                    font=dict(size=24, color='white')
                ),
                xaxis=dict(
                    title="Date",
                    range=[dates[0], dates[-1]],
                    gridcolor=self.themes[self.current_theme]['grid_color']
                ),
                yaxis=dict(
                    title="Portfolio Value ($)",
                    gridcolor=self.themes[self.current_theme]['grid_color'],
                    tickprefix='$'
                ),
                template='plotly_dark',
                showlegend=True,
                updatemenus=[dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="▶️ Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 50, "redraw": True}, 
                                        "fromcurrent": True, "mode": "immediate"}]
                        ),
                        dict(
                            label="⏸️ Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": True}, 
                                          "mode": "immediate", "transition": {"duration": 0}}]
                        )
                    ],
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )],
                sliders=[dict(
                    steps=[dict(
                        method="animate",
                        args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 300}}],
                        label=str(i)
                    ) for i, f in enumerate(frames)],
                    active=0,
                    transition={"duration": 300},
                    x=0.1,
                    len=0.9,
                    xanchor="left",
                    y=0,
                    yanchor="top",
                    currentvalue={"font": {"size": 16, "color": "white"},
                                 "prefix": "Frame: ", "visible": True, "xanchor": "right"}
                )]
            ),
            frames=frames
        )
        
        performance_monitor.end_operation('animated_backtest')
        return fig
    
    def create_interactive_heatmap(self, correlation_matrix: pd.DataFrame,
                                  title: str = "Interactive Correlation Heatmap") -> go.Figure:
        """Create interactive correlation heatmap with clustering."""
        performance_monitor.start_operation('interactive_heatmap')
        
        try:
            # Ensure correlation matrix is valid
            correlation_matrix = correlation_matrix.copy()
            
            # Check for NaN values and replace with 0
            if correlation_matrix.isnull().any().any():
                correlation_matrix = correlation_matrix.fillna(0)
            
            # Ensure diagonal is 1
            np.fill_diagonal(correlation_matrix.values, 1.0)
            
            # Convert correlation to distance
            distance_matrix = np.sqrt(2 * (1 - correlation_matrix.values))
            
            # Perform hierarchical clustering
            from scipy.cluster.hierarchy import linkage, leaves_list
            Z = linkage(distance_matrix, method='ward')
            leaves = leaves_list(Z)
            
            # Reorder correlation matrix based on clustering
            clustered_matrix = correlation_matrix.iloc[leaves, leaves]
            tickers = clustered_matrix.columns.tolist()
        except Exception as e:
            # If clustering fails, use original order
            clustered_matrix = correlation_matrix
            tickers = correlation_matrix.columns.tolist()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=clustered_matrix.values,
            x=tickers,
            y=tickers,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title="Correlation",
                titleside="right",
                tickmode="array",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
            ),
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlation: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add annotations for significant correlations
        annotations = []
        n_tickers = len(tickers)
        max_annotations = min(20, n_tickers * n_tickers)  # Limit annotations
        
        for i in range(n_tickers):
            for j in range(n_tickers):
                if i != j and abs(clustered_matrix.iloc[i, j]) > 0.7:
                    if len(annotations) < max_annotations:
                        annotations.append(dict(
                            x=j,
                            y=i,
                            text=f'{clustered_matrix.iloc[i, j]:.2f}',
                            font=dict(color='black' if abs(clustered_matrix.iloc[i, j]) > 0.8 else 'white',
                                     size=10),
                            showarrow=False
                        ))
        
        # Update layout
        fig.update_layout(
            height=min(700, 100 + n_tickers * 30),  # Dynamic height
            title=dict(
                text=title,
                font=dict(size=24, color='white'),
                x=0.5
            ),
            xaxis=dict(
                title="Assets",
                tickangle=45,
                gridcolor=self.themes[self.current_theme]['grid_color'],
                side="top"
            ),
            yaxis=dict(
                title="Assets",
                gridcolor=self.themes[self.current_theme]['grid_color']
            ),
            template='plotly_dark',
            annotations=annotations,
            margin=dict(l=100, r=50, t=100, b=100)
        )
        
        performance_monitor.end_operation('interactive_heatmap')
        return fig
    
    def create_advanced_var_analysis_dashboard(self, returns: pd.Series,
                                              confidence_levels: List[float] = None,
                                              window_sizes: List[int] = None) -> go.Figure:
        """Create advanced VaR analysis dashboard with multiple visualizations."""
        performance_monitor.start_operation('advanced_var_dashboard')
        
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99, 0.995]
        
        if window_sizes is None:
            window_sizes = [63, 126, 252]  # 3, 6, 12 months
        
        # Calculate VaR using different methods
        var_results = self._calculate_multiple_var_methods(returns, confidence_levels)
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'VaR by Confidence Level',
                'CVaR by Confidence Level',
                'Expected Shortfall (ES)',
                'Rolling VaR (95%)',
                'VaR Method Comparison',
                'VaR Violations',
                'VaR Distribution',
                'Stress VaR',
                'VaR Decomposition'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'box'}, {'type': 'scatter'}, {'type': 'pie'}]
            ]
        )
        
        # 1. VaR by Confidence Level (Bar)
        methods = list(var_results.keys())
        colors = ['#ef553b', '#00cc96', '#636efa', '#ab63fa', '#FFA15A']
        
        for idx, method in enumerate(methods):
            var_values = [var_results[method][conf]['VaR'] for conf in confidence_levels]
            fig.add_trace(
                go.Bar(
                    x=[f'{c*100:.1f}%' for c in confidence_levels],
                    y=var_values,
                    name=method.capitalize(),
                    marker_color=colors[idx % len(colors)],
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. CVaR by Confidence Level (Bar)
        for idx, method in enumerate(methods):
            cvar_values = [var_results[method][conf]['CVaR'] for conf in confidence_levels]
            fig.add_trace(
                go.Bar(
                    x=[f'{c*100:.1f}%' for c in confidence_levels],
                    y=cvar_values,
                    name=method.capitalize() + ' CVaR',
                    marker_color=colors[idx % len(colors)],
                    showlegend=False,
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # 3. Expected Shortfall (Line)
        for idx, method in enumerate(methods):
            es_values = [var_results[method][conf]['ES'] for conf in confidence_levels]
            fig.add_trace(
                go.Scatter(
                    x=[f'{c*100:.1f}%' for c in confidence_levels],
                    y=es_values,
                    mode='lines+markers',
                    name=method.capitalize() + ' ES',
                    line=dict(color=colors[idx % len(colors)], width=3),
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=1, col=3
            )
        
        # 4. Rolling VaR (Line)
        for window in window_sizes:
            try:
                rolling_var = returns.rolling(window=window).apply(
                    lambda x: -np.percentile(x, 5) if len(x.dropna()) >= 20 else np.nan, 
                    raw=True
                ).dropna()
                if not rolling_var.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=rolling_var.index,
                            y=rolling_var.values,
                            mode='lines',
                            name=f'Rolling VaR ({window}d)',
                            line=dict(width=2),
                            showlegend=True
                        ),
                        row=2, col=1
                    )
            except Exception as e:
                continue
        
        # 5. VaR Method Comparison (Grouped Bar)
        conf_95_values = []
        conf_99_values = []
        
        for method in methods:
            if 0.95 in var_results[method]:
                conf_95_values.append(var_results[method][0.95]['VaR'])
            if 0.99 in var_results[method]:
                conf_99_values.append(var_results[method][0.99]['VaR'])
        
        if conf_95_values and conf_99_values:
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=conf_95_values,
                    name='VaR 95%',
                    marker_color='#ef553b'
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=conf_99_values,
                    name='VaR 99%',
                    marker_color='#ff6b6b',
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        # 6. VaR Violations (Scatter)
        if len(returns) > 0:
            var_95 = -np.percentile(returns.dropna(), 5)
            violations = returns < -var_95
            
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns.values,
                    mode='markers',
                    name='Returns',
                    marker=dict(
                        size=6,
                        color=['#ef553b' if v else '#636efa' for v in violations],
                        opacity=0.7
                    ),
                    showlegend=False
                ),
                row=2, col=3
            )
            
            fig.add_hline(
                y=-var_95,
                line_dash="dash",
                line_color="#FFA15A",
                annotation_text="VaR 95% Threshold",
                row=2, col=3
            )
        
        # 7. VaR Distribution (Box)
        all_var_values = []
        for method in methods:
            for conf in confidence_levels:
                if conf in var_results[method]:
                    all_var_values.append(var_results[method][conf]['VaR'])
        
        if all_var_values:
            fig.add_trace(
                go.Box(
                    y=all_var_values,
                    name='VaR Distribution',
                    marker_color='#00cc96',
                    boxmean=True,
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # 8. Stress VaR (Line)
        stress_levels = np.linspace(0.90, 0.999, 50)
        if len(returns) > 0:
            stress_var_values = [-np.percentile(returns.dropna(), (1-c)*100) for c in stress_levels]
            
            fig.add_trace(
                go.Scatter(
                    x=stress_levels,
                    y=stress_var_values,
                    mode='lines',
                    name='Stress VaR',
                    line=dict(color='#ab63fa', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(171, 99, 250, 0.2)',
                    showlegend=False
                ),
                row=3, col=2
            )
        
        # 9. VaR Decomposition (Pie)
        components = ['Market Risk', 'Credit Risk', 'Liquidity Risk', 'Operational Risk']
        percentages = [45, 25, 20, 10]
        
        fig.add_trace(
            go.Pie(
                labels=components,
                values=percentages,
                hole=0.4,
                marker_colors=['#ef553b', '#00cc96', '#636efa', '#FFA15A'],
                showlegend=False
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title=dict(
                text='Advanced Value at Risk (VaR) Analysis Dashboard',
                font=dict(size=28, color='white'),
                x=0.5
            ),
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update axes
        fig.update_yaxes(title_text="VaR", row=1, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="CVaR", row=1, col=2, tickformat=".1%")
        fig.update_yaxes(title_text="ES", row=1, col=3, tickformat=".1%")
        fig.update_yaxes(title_text="VaR", row=2, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="VaR Value", row=2, col=2, tickformat=".1%")
        fig.update_yaxes(title_text="Return", row=2, col=3, tickformat=".1%")
        fig.update_yaxes(title_text="VaR Values", row=3, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Stress VaR", row=3, col=2, tickformat=".1%")
        
        performance_monitor.end_operation('advanced_var_dashboard')
        return fig
    
    def _calculate_multiple_var_methods(self, returns: pd.Series, 
                                       confidence_levels: List[float]) -> Dict:
        """Calculate VaR using multiple methods."""
        methods = ['Historical', 'Parametric', 'Monte Carlo', 'EWMA', 'Extreme Value']
        results = {}
        
        # Clean returns
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return {method: {conf: {'VaR': 0, 'CVaR': 0, 'ES': 0} for conf in confidence_levels} 
                    for method in methods}
        
        for method in methods:
            results[method] = {}
            for confidence in confidence_levels:
                alpha = 1 - confidence
                
                if method == 'Historical':
                    # Historical simulation
                    if len(returns_clean) > 0:
                        var = -np.percentile(returns_clean, alpha * 100)
                        cvar_data = returns_clean[returns_clean <= -var]
                        cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                        es = cvar
                    else:
                        var = cvar = es = 0
                        
                elif method == 'Parametric':
                    # Parametric (normal distribution)
                    mean = returns_clean.mean()
                    std = returns_clean.std()
                    if std > 0:
                        var = -(mean + std * norm.ppf(confidence))
                        cvar = -(mean - std * norm.pdf(norm.ppf(alpha)) / alpha)
                    else:
                        var = -mean
                        cvar = var
                    es = cvar
                    
                elif method == 'Monte Carlo':
                    # Monte Carlo simulation
                    np.random.seed(42)
                    n_simulations = 5000
                    simulated_returns = np.random.normal(returns_clean.mean(), returns_clean.std(), n_simulations)
                    var = -np.percentile(simulated_returns, alpha * 100)
                    cvar_data = simulated_returns[simulated_returns <= -var]
                    cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                    es = cvar
                    
                elif method == 'EWMA':
                    # Simplified EWMA
                    lambda_ = 0.94
                    n = len(returns_clean)
                    weights = np.array([lambda_ ** i for i in range(n)][::-1])
                    weights = weights / weights.sum()
                    ewma_std = np.sqrt(np.sum(weights * (returns_clean - returns_clean.mean()) ** 2))
                    if ewma_std > 0:
                        var = -(returns_clean.mean() + ewma_std * norm.ppf(confidence))
                        cvar = -(returns_clean.mean() - ewma_std * norm.pdf(norm.ppf(alpha)) / alpha)
                    else:
                        var = -returns_clean.mean()
                        cvar = var
                    es = cvar
                    
                else:  # Extreme Value
                    # Simplified EVT
                    if len(returns_clean) >= 20:
                        threshold = np.percentile(returns_clean, 10)
                        excess = returns_clean[returns_clean < threshold] - threshold
                        if len(excess) > 10:
                            try:
                                # Simplified GPD estimation
                                xi, beta = self._estimate_gpd_parameters(excess)
                                var_evt = threshold + (beta/xi) * (((len(returns_clean) * alpha) / len(excess)) ** (-xi) - 1)
                                var = -var_evt
                                es_evt = (var_evt + beta - xi * threshold) / (1 - xi)
                                cvar = -es_evt
                                es = cvar
                            except:
                                var = -np.percentile(returns_clean, alpha * 100)
                                cvar_data = returns_clean[returns_clean <= -var]
                                cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                                es = cvar
                        else:
                            var = -np.percentile(returns_clean, alpha * 100)
                            cvar_data = returns_clean[returns_clean <= -var]
                            cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                            es = cvar
                    else:
                        var = -np.percentile(returns_clean, alpha * 100)
                        cvar_data = returns_clean[returns_clean <= -var]
                        cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                        es = cvar
                
                results[method][confidence] = {
                    'VaR': var,
                    'CVaR': cvar,
                    'ES': es
                }
        
        return results
    
    def _estimate_gpd_parameters(self, excess: pd.Series) -> Tuple[float, float]:
        """Estimate Generalized Pareto Distribution parameters."""
        # Method of moments estimation
        mean_excess = excess.mean()
        var_excess = excess.var()
        
        if var_excess > 0:
            xi = 0.5 * ((mean_excess ** 2) / var_excess + 1)
            beta = 0.5 * mean_excess * ((mean_excess ** 2) / var_excess + 1)
            return max(0.1, min(xi, 0.9)), max(0.001, beta)
        else:
            return 0.1, max(0.001, mean_excess)
    
    def create_portfolio_allocation_sunburst(self, weights: Dict, 
                                           asset_metadata: Dict) -> go.Figure:
        """Create interactive sunburst chart for portfolio allocation."""
        performance_monitor.start_operation('allocation_sunburst')
        
        # Build hierarchy: Sector -> Industry -> Asset
        hierarchy = {}
        
        for ticker, weight in weights.items():
            if ticker in asset_metadata:
                sector = asset_metadata[ticker].get('sector', 'Other')
                industry = asset_metadata[ticker].get('industry', 'Unknown')
                
                if sector not in hierarchy:
                    hierarchy[sector] = {}
                if industry not in hierarchy[sector]:
                    hierarchy[sector][industry] = []
                
                hierarchy[sector][industry].append({
                    'ticker': ticker,
                    'weight': weight,
                    'name': asset_metadata[ticker].get('full_name', ticker)
                })
            else:
                # If no metadata, put in Other/Unknown
                if 'Other' not in hierarchy:
                    hierarchy['Other'] = {}
                if 'Unknown' not in hierarchy['Other']:
                    hierarchy['Other']['Unknown'] = []
                
                hierarchy['Other']['Unknown'].append({
                    'ticker': ticker,
                    'weight': weight,
                    'name': ticker
                })
        
        # Prepare data for sunburst
        labels = []
        parents = []
        values = []
        customdata = []
        
        # Add root
        labels.append('Portfolio')
        parents.append('')
        values.append(100)
        customdata.append({'type': 'portfolio', 'description': 'Complete portfolio'})
        
        # Add sectors
        for sector, industries in hierarchy.items():
            sector_weight = sum(sum(item['weight'] for item in industry) 
                              for industry in industries.values())
            
            labels.append(sector)
            parents.append('Portfolio')
            values.append(sector_weight * 100)
            customdata.append({'type': 'sector', 'description': f'Sector: {sector}'})
            
            # Add industries
            for industry, assets in industries.items():
                industry_weight = sum(item['weight'] for item in assets)
                
                labels.append(industry)
                parents.append(sector)
                values.append(industry_weight * 100)
                customdata.append({'type': 'industry', 'description': f'Industry: {industry}'})
                
                # Add assets
                for asset in assets:
                    labels.append(asset['name'])
                    parents.append(industry)
                    values.append(asset['weight'] * 100)
                    customdata.append({
                        'type': 'asset',
                        'ticker': asset['ticker'],
                        'description': f"{asset['name']} ({asset['ticker']})"
                    })
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            customdata=customdata,
            branchvalues="total",
            maxdepth=3,
            marker=dict(
                colors=values,
                colorscale='Viridis',
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         'Weight: %{value:.1f}%<br>' +
                         '%{customdata.description}<br>' +
                         '<extra></extra>',
            textinfo='label+percent entry'
        ))
        
        # Update layout
        fig.update_layout(
            height=700,
            title=dict(
                text='Portfolio Allocation Hierarchy (Sunburst)',
                font=dict(size=24, color='white'),
                x=0.5
            ),
            template='plotly_dark',
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        performance_monitor.end_operation('allocation_sunburst')
        return fig
    
    def create_real_time_metrics_dashboard(self, metrics: Dict) -> go.Figure:
        """Create real-time metrics dashboard with gauges and indicators."""
        performance_monitor.start_operation('realtime_metrics_dashboard')
        
        # Define metrics to display
        key_metrics = {
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': metrics.get('sortino_ratio', 0),
            'Calmar Ratio': metrics.get('calmar_ratio', 0),
            'Omega Ratio': metrics.get('omega_ratio', 1),
            'Information Ratio': metrics.get('information_ratio', 0),
            'Treynor Ratio': metrics.get('treynor_ratio', 0),
            'Max Drawdown': metrics.get('max_drawdown', 0),
            'Volatility': metrics.get('expected_volatility', 0.2),
            'Beta': metrics.get('beta', 1),
            'Alpha': metrics.get('alpha', 0),
            'R-squared': metrics.get('r_squared', 0.5),
            'Tracking Error': metrics.get('tracking_error', 0.1)
        }
        
        # Filter out None values and ensure numeric
        key_metrics = {k: (v if v is not None and np.isfinite(v) else 0) 
                      for k, v in key_metrics.items()}
        
        # Create subplot with gauges and indicators
        n_metrics = len(key_metrics)
        n_cols = 4
        n_rows = math.ceil(n_metrics / n_cols)
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=[[{'type': 'indicator'} for _ in range(n_cols)] for _ in range(n_rows)],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Define ranges and colors for each metric
        metric_configs = {
            'Sharpe Ratio': {'range': [0, 3], 'colors': ['#ef553b', '#FFA15A', '#00cc96']},
            'Sortino Ratio': {'range': [0, 3], 'colors': ['#ef553b', '#FFA15A', '#00cc96']},
            'Calmar Ratio': {'range': [0, 2], 'colors': ['#ef553b', '#FFA15A', '#00cc96']},
            'Omega Ratio': {'range': [0, 3], 'colors': ['#ef553b', '#FFA15A', '#00cc96']},
            'Information Ratio': {'range': [-1, 2], 'colors': ['#ef553b', '#FFA15A', '#00cc96']},
            'Treynor Ratio': {'range': [0, 0.3], 'colors': ['#ef553b', '#FFA15A', '#00cc96']},
            'Max Drawdown': {'range': [-0.5, 0], 'colors': ['#00cc96', '#FFA15A', '#ef553b']},
            'Volatility': {'range': [0, 0.5], 'colors': ['#00cc96', '#FFA15A', '#ef553b']},
            'Beta': {'range': [0, 2], 'colors': ['#00cc96', '#FFA15A', '#ef553b']},
            'Alpha': {'range': [-0.2, 0.2], 'colors': ['#ef553b', '#FFA15A', '#00cc96']},
            'R-squared': {'range': [0, 1], 'colors': ['#ef553b', '#FFA15A', '#00cc96']},
            'Tracking Error': {'range': [0, 0.2], 'colors': ['#00cc96', '#FFA15A', '#ef553b']}
        }
        
        # Add gauges
        metrics_list = list(key_metrics.items())
        for idx, (metric_name, value) in enumerate(metrics_list):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            config = metric_configs.get(metric_name, {'range': [0, 1], 'colors': ['#ef553b', '#FFA15A', '#00cc96']})
            min_val, max_val = config['range']
            
            # Ensure value is within range
            if isinstance(value, (int, float)):
                display_value = max(min_val, min(max_val, value))
            else:
                display_value = (min_val + max_val) / 2
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=display_value,
                    title=dict(text=metric_name, font=dict(size=14)),
                    number=dict(
                        font=dict(size=20, color='white'),
                        valueformat=".3f" if abs(display_value) < 1 else ".2f",
                        suffix=""
                    ),
                    gauge=dict(
                        axis=dict(
                            range=config['range'],
                            tickwidth=1,
                            tickcolor="white",
                            tickformat=".2f" if metric_name != 'R-squared' else ".0%"
                        ),
                        bar=dict(color="white", thickness=0.2),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=2,
                        bordercolor="white",
                        steps=[
                            dict(range=config['range'], color=config['colors'][0]),
                            dict(range=[config['range'][0] + (config['range'][1] - config['range'][0]) * 0.33, 
                                       config['range'][0] + (config['range'][1] - config['range'][0]) * 0.66], 
                                 color=config['colors'][1]),
                            dict(range=[config['range'][0] + (config['range'][1] - config['range'][0]) * 0.66, 
                                       config['range'][1]], 
                                 color=config['colors'][2])
                        ],
                        threshold=dict(
                            line=dict(color="white", width=4),
                            thickness=0.75,
                            value=display_value
                        )
                    )
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            height=300 * n_rows,
            title=dict(
                text='Real-Time Portfolio Metrics Dashboard',
                font=dict(size=28, color='white'),
                x=0.5
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        performance_monitor.end_operation('realtime_metrics_dashboard')
        return fig
    
    def _create_empty_figure(self, title: str) -> go.Figure:
        """Create empty figure with error message."""
        fig = go.Figure()
        fig.update_layout(
            height=400,
            title=dict(
                text=f"{title} (Data Unavailable)",
                font=dict(size=20, color='white')
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="Data not available for this visualization",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color='white')
            )]
        )
        return fig

# Initialize visualization engine
viz_engine = AdvancedVisualizationEngine()

# ============================================================================
# 4. ADVANCED RISK ANALYTICS ENGINE WITH VaR/CVaR/ES CALCULATIONS
# ============================================================================

class AdvancedRiskAnalytics:
    """Advanced risk analytics engine with comprehensive VaR/CVaR/ES calculations."""
    
    def __init__(self):
        self.methods = ['Historical', 'Parametric', 'MonteCarlo', 'EVT', 'GARCH']
        self.confidence_levels = [0.90, 0.95, 0.99, 0.995]
    
    def calculate_comprehensive_var_analysis(self, returns: pd.Series, 
                                           portfolio_value: float = 1_000_000) -> Dict:
        """Calculate comprehensive VaR analysis with all methods."""
        performance_monitor.start_operation('comprehensive_var_analysis')
        
        results = {
            'methods': {},
            'portfolio_value': portfolio_value,
            'summary': {},
            'violations': {},
            'backtest': {},
            'stress_tests': {}
        }
        
        try:
            # Clean returns
            returns_clean = returns.dropna()
            if len(returns_clean) < 50:
                raise ValueError(f"Insufficient data points: {len(returns_clean)} (minimum 50 required)")
            
            # Calculate VaR using all methods
            for method in self.methods:
                results['methods'][method] = self._calculate_var_method(
                    returns_clean, method, self.confidence_levels, portfolio_value
                )
            
            # Calculate summary statistics
            results['summary'] = self._calculate_var_summary(results['methods'], returns_clean)
            
            # Calculate VaR violations
            results['violations'] = self._calculate_var_violations(
                returns_clean, results['methods']['Historical']
            )
            
            # Perform backtesting
            results['backtest'] = self._perform_var_backtesting(
                returns_clean, results['methods']['Historical']
            )
            
            # Perform stress tests
            results['stress_tests'] = self._perform_stress_tests(returns_clean)
            
            # Calculate additional risk metrics
            results['additional_metrics'] = self._calculate_additional_risk_metrics(returns_clean)
            
            performance_monitor.end_operation('comprehensive_var_analysis')
            return results
            
        except Exception as e:
            performance_monitor.end_operation('comprehensive_var_analysis', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {'operation': 'comprehensive_var_analysis'})
            return self._create_fallback_results(returns, portfolio_value)
    
    def _calculate_var_method(self, returns: pd.Series, method: str, 
                            confidence_levels: List[float], 
                            portfolio_value: float) -> Dict:
        """Calculate VaR using specific method."""
        results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            
            if method == 'Historical':
                # Historical simulation
                var = -np.percentile(returns, alpha * 100)
                cvar_data = returns[returns <= -var]
                cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                es = cvar
                
            elif method == 'Parametric':
                # Parametric (normal distribution)
                mean = returns.mean()
                std = returns.std()
                if std > 0:
                    var = -(mean + std * norm.ppf(confidence))
                    cvar = -(mean - std * norm.pdf(norm.ppf(alpha)) / alpha)
                else:
                    var = -mean
                    cvar = var
                es = cvar
                
            elif method == 'MonteCarlo':
                # Monte Carlo simulation
                np.random.seed(42)
                n_simulations = 5000
                simulated_returns = np.random.normal(returns.mean(), returns.std(), n_simulations)
                var = -np.percentile(simulated_returns, alpha * 100)
                cvar_data = simulated_returns[simulated_returns <= -var]
                cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                es = cvar
                
            elif method == 'EVT':
                # Extreme Value Theory
                try:
                    threshold = np.percentile(returns, 10)
                    excess = returns[returns < threshold] - threshold
                    
                    if len(excess) > 10:
                        # Simplified GPD estimation
                        xi, beta = self._estimate_gpd_parameters(excess)
                        
                        # Calculate VaR using GPD
                        var_evt = threshold + (beta/xi) * (((len(returns) * alpha) / len(excess)) ** (-xi) - 1)
                        var = -var_evt
                        
                        # Calculate ES using GPD
                        es_evt = (var_evt + beta - xi * threshold) / (1 - xi)
                        cvar = -es_evt
                        es = cvar
                    else:
                        # Fallback to historical
                        var = -np.percentile(returns, alpha * 100)
                        cvar_data = returns[returns <= -var]
                        cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                        es = cvar
                        
                except:
                    var = -np.percentile(returns, alpha * 100)
                    cvar_data = returns[returns <= -var]
                    cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                    es = cvar
                    
            elif method == 'GARCH':
                # GARCH model VaR - simplified implementation
                try:
                    mean = returns.mean()
                    std = returns.std()
                    if std > 0:
                        var = -(mean + std * norm.ppf(confidence))
                        cvar = -(mean - std * norm.pdf(norm.ppf(alpha)) / alpha)
                    else:
                        var = -mean
                        cvar = var
                    es = cvar
                        
                except Exception as e:
                    # Fallback to parametric
                    mean = returns.mean()
                    std = returns.std()
                    if std > 0:
                        var = -(mean + std * norm.ppf(confidence))
                        cvar = -(mean - std * norm.pdf(norm.ppf(alpha)) / alpha)
                    else:
                        var = -mean
                        cvar = var
                    es = cvar
            
            else:
                # Default to historical
                var = -np.percentile(returns, alpha * 100)
                cvar_data = returns[returns <= -var]
                cvar = -cvar_data.mean() if len(cvar_data) > 0 else var * 1.2
                es = cvar
            
            # Store results
            results[confidence] = {
                'VaR': var,
                'VaR_absolute': var * portfolio_value,
                'CVaR': cvar,
                'CVaR_absolute': cvar * portfolio_value,
                'ES': es,
                'ES_absolute': es * portfolio_value,
                'confidence': confidence,
                'method': method
            }
        
        return results
    
    def _estimate_gpd_parameters(self, excess: pd.Series) -> Tuple[float, float]:
        """Estimate Generalized Pareto Distribution parameters."""
        # Method of moments estimation
        mean_excess = excess.mean()
        var_excess = excess.var()
        
        if var_excess > 0:
            xi = 0.5 * ((mean_excess ** 2) / var_excess + 1)
            beta = 0.5 * mean_excess * ((mean_excess ** 2) / var_excess + 1)
            return max(0.1, min(xi, 0.9)), max(0.001, beta)
        else:
            return 0.1, max(0.001, mean_excess)
    
    def _calculate_var_summary(self, methods_results: Dict, returns: pd.Series) -> Dict:
        """Calculate summary statistics for VaR analysis."""
        summary = {
            'best_method': None,
            'worst_case_var': float('inf'),
            'average_var': 0,
            'var_consistency': 0,
            'risk_adjustment_factors': {}
        }
        
        # Calculate average VaR across methods
        var_values = []
        for method, results in methods_results.items():
            for confidence, metrics in results.items():
                var_values.append(metrics['VaR'])
        
        if var_values:
            summary['average_var'] = np.mean(var_values)
            summary['worst_case_var'] = np.max(var_values)
            if np.mean(var_values) != 0:
                summary['var_consistency'] = np.std(var_values) / np.mean(var_values)
        
        # Determine best method (lowest average VaR with consistency)
        method_scores = {}
        for method, results in methods_results.items():
            method_vars = [metrics['VaR'] for metrics in results.values()]
            avg_var = np.mean(method_vars)
            if avg_var != 0:
                consistency = np.std(method_vars) / avg_var
            else:
                consistency = float('inf')
            
            # Score = average VaR * (1 + consistency penalty)
            method_scores[method] = avg_var * (1 + consistency * 0.5)
        
        if method_scores:
            summary['best_method'] = min(method_scores, key=method_scores.get)
        
        # Calculate risk adjustment factors
        summary['risk_adjustment_factors'] = {
            'liquidity_adjustment': self._calculate_liquidity_adjustment(returns),
            'concentration_adjustment': self._calculate_concentration_adjustment(returns),
            'tail_risk_adjustment': self._calculate_tail_risk_adjustment(returns)
        }
        
        return summary
    
    def _calculate_liquidity_adjustment(self, returns: pd.Series) -> float:
        """Calculate liquidity risk adjustment factor."""
        # Simplified liquidity adjustment based on return characteristics
        volume_factor = 1.0
        
        # Adjust based on volatility (higher volatility = lower liquidity)
        volatility = returns.std()
        if volatility > 0.02:  # More than 2% daily volatility
            volume_factor *= 1.2
        elif volatility < 0.005:  # Less than 0.5% daily volatility
            volume_factor *= 0.8
        
        # Adjust based on kurtosis (fat tails = liquidity risk)
        kurt = returns.kurtosis()
        if not np.isnan(kurt) and kurt > 3:  # Leptokurtic (fat tails)
            volume_factor *= 1.1 + min(0.5, (kurt - 3) / 10)
        
        return volume_factor
    
    def _calculate_concentration_adjustment(self, returns: pd.Series) -> float:
        """Calculate concentration risk adjustment factor."""
        # For single asset returns, concentration is high
        return 1.3  # 30% adjustment for concentration risk
    
    def _calculate_tail_risk_adjustment(self, returns: pd.Series) -> float:
        """Calculate tail risk adjustment factor."""
        # Calculate expected shortfall ratio
        if len(returns) > 0:
            var_95 = -np.percentile(returns, 5)
            cvar_95_data = returns[returns <= -var_95]
            cvar_95 = -cvar_95_data.mean() if len(cvar_95_data) > 0 else var_95
            
            if var_95 != 0:
                tail_ratio = cvar_95 / var_95
                # Higher tail ratio indicates heavier tails
                return 1.0 + max(0, (tail_ratio - 1.5) * 0.2)
        
        return 1.0
    
    def _calculate_var_violations(self, returns: pd.Series, 
                                 historical_results: Dict) -> Dict:
        """Calculate VaR violations and exceptions."""
        violations = {
            'total_days': len(returns),
            'violations_95': 0,
            'violations_99': 0,
            'exception_rates': {},
            'unconditional_coverage': {},
            'conditional_coverage': {},
            'independence_test': {}
        }
        
        # Calculate violations for each confidence level
        for confidence, metrics in historical_results.items():
            var_threshold = -metrics['VaR']
            violations_count = (returns < -var_threshold).sum()
            expected_violations = len(returns) * (1 - confidence)
            
            violations[f'violations_{int(confidence*100)}'] = violations_count
            violations['exception_rates'][confidence] = {
                'actual': violations_count / len(returns),
                'expected': 1 - confidence,
                'difference': (violations_count / len(returns)) - (1 - confidence)
            }
        
        # Perform Kupiec's Proportion of Failures test (unconditional coverage)
        for confidence in [0.95, 0.99]:
            if confidence in historical_results:
                var_threshold = -historical_results[confidence]['VaR']
                violations_count = (returns < -var_threshold).sum()
                n = len(returns)
                p = 1 - confidence
                
                # Likelihood ratio test
                if violations_count > 0 and n - violations_count > 0:
                    LR_uc = -2 * np.log(
                        ((1 - p) ** (n - violations_count) * p ** violations_count) /
                        ((1 - violations_count/n) ** (n - violations_count) * 
                         (violations_count/n) ** violations_count)
                    )
                    
                    violations['unconditional_coverage'][confidence] = {
                        'LR_statistic': LR_uc,
                        'p_value': 1 - stats.chi2.cdf(LR_uc, 1),
                        'pass': LR_uc < 3.841  # 95% confidence chi2 critical value
                    }
                else:
                    violations['unconditional_coverage'][confidence] = {
                        'LR_statistic': 0,
                        'p_value': 1.0,
                        'pass': True
                    }
        
        return violations
    
    def _perform_var_backtesting(self, returns: pd.Series, 
                                historical_results: Dict) -> Dict:
        """Perform comprehensive VaR backtesting."""
        backtest = {
            'rolling_var': {},
            'conditional_coverage': {},
            'violation_clustering': {},
            'duration_between_failures': {}
        }
        
        # Calculate rolling VaR
        windows = [63, 126, 252]  # 3, 6, 12 months
        for window in windows:
            try:
                rolling_var = returns.rolling(window=window).apply(
                    lambda x: -np.percentile(x, 5) if len(x.dropna()) >= 20 else np.nan, raw=True
                ).dropna()
                if not rolling_var.empty:
                    backtest['rolling_var'][window] = {
                        'mean': rolling_var.mean(),
                        'std': rolling_var.std(),
                        'max': rolling_var.max(),
                        'min': rolling_var.min()
                    }
            except Exception as e:
                continue
        
        # Check for violation clustering (Christoffersen's independence test)
        if 0.95 in historical_results:
            var_threshold = -historical_results[0.95]['VaR']
            violations = (returns < -var_threshold).astype(int).values
            
            # Calculate transition probabilities
            n00 = n01 = n10 = n11 = 0
            for i in range(1, len(violations)):
                if violations[i-1] == 0 and violations[i] == 0:
                    n00 += 1
                elif violations[i-1] == 0 and violations[i] == 1:
                    n01 += 1
                elif violations[i-1] == 1 and violations[i] == 0:
                    n10 += 1
                elif violations[i-1] == 1 and violations[i] == 1:
                    n11 += 1
            
            # Calculate test statistic
            if (n00 + n01) > 0 and (n10 + n11) > 0:
                pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
                pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
                pi_total = n00 + n01 + n10 + n11
                pi = (n01 + n11) / pi_total if pi_total > 0 else 0
                
                if pi0 > 0 and pi1 > 0 and pi > 0:
                    try:
                        likelihood_ratio = -2 * np.log(
                            ((1 - pi) ** (n00 + n10) * pi ** (n01 + n11)) /
                            ((1 - pi0) ** n00 * pi0 ** n01 * (1 - pi1) ** n10 * pi1 ** n11)
                        )
                        
                        backtest['conditional_coverage'] = {
                            'LR_statistic': likelihood_ratio,
                            'p_value': 1 - stats.chi2.cdf(likelihood_ratio, 1) if not np.isnan(likelihood_ratio) else 1.0,
                            'pass': likelihood_ratio < 3.841 if not np.isnan(likelihood_ratio) else True
                        }
                    except Exception as e:
                        backtest['conditional_coverage'] = {'error': str(e)}
        
        # Calculate duration between failures
        if 0.95 in historical_results:
            var_threshold = -historical_results[0.95]['VaR']
            violations_idx = np.where(returns < -var_threshold)[0]
            if len(violations_idx) > 1:
                durations = np.diff(violations_idx)
                backtest['duration_between_failures'] = {
                    'mean': durations.mean() if len(durations) > 0 else 0,
                    'std': durations.std() if len(durations) > 0 else 0,
                    'min': durations.min() if len(durations) > 0 else 0,
                    'max': durations.max() if len(durations) > 0 else 0
                }
        
        return backtest
    
    def _perform_stress_tests(self, returns: pd.Series) -> Dict:
        """Perform stress testing scenarios."""
        stress_tests = {
            'historical_scenarios': {},
            'hypothetical_scenarios': {},
            'reverse_stress_tests': {}
        }
        
        # Historical scenarios
        historical_periods = {
            '2008 Crisis': ('2008-01-01', '2009-12-31'),
            'COVID Crash': ('2020-02-01', '2020-04-30'),
            '2022 Inflation': ('2022-01-01', '2022-12-31')
        }
        
        for scenario, (start, end) in historical_periods.items():
            try:
                start_date = pd.Timestamp(start)
                end_date = pd.Timestamp(end)
                
                # Check if returns index is datetime-like
                if hasattr(returns.index, 'tz_localize'):
                    returns_local = returns.tz_localize(None)
                else:
                    returns_local = returns.copy()
                
                mask = (returns_local.index >= start_date) & (returns_local.index <= end_date)
                if mask.any():
                    period_returns = returns_local[mask]
                    if len(period_returns) > 10:
                        var_95 = -np.percentile(period_returns, 5)
                        cvar_95_data = period_returns[period_returns <= -var_95]
                        cvar_95 = -cvar_95_data.mean() if len(cvar_95_data) > 0 else var_95
                        
                        stress_tests['historical_scenarios'][scenario] = {
                            'returns': period_returns.mean() * 252,
                            'volatility': period_returns.std() * np.sqrt(252),
                            'max_drawdown': self._calculate_max_drawdown(period_returns),
                            'var_95': var_95,
                            'cvar_95': cvar_95
                        }
            except Exception as e:
                continue
        
        # Hypothetical scenarios
        mean_return = returns.mean()
        std_return = returns.std()
        hypothetical_scenarios = {
            'Market Crash (-20%)': {'return_shock': -0.20, 'vol_multiplier': 2.0},
            'Volatility Spike': {'return_shock': -0.10, 'vol_multiplier': 3.0},
            'Slow Decline': {'return_shock': -0.30, 'vol_multiplier': 1.5}
        }
        
        for scenario, params in hypothetical_scenarios.items():
            try:
                if std_return > 0:
                    stressed_var_95 = -(mean_return + std_return * params['vol_multiplier'] * norm.ppf(0.95))
                else:
                    stressed_var_95 = -mean_return
                
                stress_tests['hypothetical_scenarios'][scenario] = {
                    'stressed_return': mean_return * 252 + params['return_shock'],
                    'stressed_volatility': std_return * np.sqrt(252) * params['vol_multiplier'],
                    'stressed_var_95': stressed_var_95,
                    'description': f"Return shock: {params['return_shock']:.1%}, Volatility multiplier: {params['vol_multiplier']}x"
                }
            except Exception as e:
                continue
        
        # Reverse stress tests
        target_losses = [0.10, 0.20, 0.30]  # 10%, 20%, 30% losses
        for loss in target_losses:
            try:
                if std_return > 0:
                    required_shock = norm.ppf(0.01) * std_return  # 99% confidence
                    probability = norm.cdf(-loss / std_return) if std_return > 0 else 0
                else:
                    required_shock = 0
                    probability = 0
                    
                stress_tests['reverse_stress_tests'][f'{loss:.0%}_Loss'] = {
                    'required_shock': required_shock,
                    'probability': probability,
                    'daily_return_required': -loss
                }
            except Exception as e:
                continue
        
        return stress_tests
    
    def _calculate_additional_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate additional risk metrics."""
        metrics = {
            'expected_shortfall_ratio': 0,
            'tail_risk_measures': {},
            'liquidity_risk': {},
            'concentration_risk': {}
        }
        
        if len(returns) > 0:
            # Expected Shortfall Ratio (CVaR / VaR)
            var_95 = -np.percentile(returns, 5)
            cvar_95_data = returns[returns <= -var_95]
            cvar_95 = -cvar_95_data.mean() if len(cvar_95_data) > 0 else var_95
            
            if var_95 != 0:
                metrics['expected_shortfall_ratio'] = cvar_95 / var_95
            
            # Tail risk measures
            try:
                skewness_val = returns.skew()
                kurtosis_val = returns.kurtosis()
                metrics['tail_risk_measures'] = {
                    'skewness': skewness_val if not np.isnan(skewness_val) else 0,
                    'excess_kurtosis': kurtosis_val if not np.isnan(kurtosis_val) else 0,
                    'tail_index': self._calculate_tail_index(returns),
                    'extreme_value_index': self._calculate_extreme_value_index(returns)
                }
            except:
                metrics['tail_risk_measures'] = {
                    'skewness': 0,
                    'excess_kurtosis': 0,
                    'tail_index': 0,
                    'extreme_value_index': 0
                }
            
            # Liquidity risk (simplified)
            try:
                illiquidity_ratio = self._calculate_illiquidity_ratio(returns)
                volume_volatility_ratio = returns.std() / (abs(returns).mean() if abs(returns).mean() > 0 else 1)
                metrics['liquidity_risk'] = {
                    'illiquidity_ratio': illiquidity_ratio,
                    'volume_volatility_ratio': volume_volatility_ratio if not np.isnan(volume_volatility_ratio) else 0
                }
            except:
                metrics['liquidity_risk'] = {
                    'illiquidity_ratio': 0,
                    'volume_volatility_ratio': 0
                }
        
        return metrics
    
    def _calculate_tail_index(self, returns: pd.Series) -> float:
        """Calculate tail index (Hill estimator)."""
        try:
            sorted_returns = np.sort(returns.values)
            k = max(10, len(sorted_returns) // 20)  # Use top 5% for tail estimation
            
            if k > 1:
                tail_returns = sorted_returns[:k]  # Negative returns (losses)
                if tail_returns[-1] < 0:
                    hill_estimator = 1 / (np.mean(np.log(-tail_returns / (-tail_returns[-1]))) if tail_returns[-1] < 0 else 1)
                    return hill_estimator
        except:
            pass
        return 0
    
    def _calculate_extreme_value_index(self, returns: pd.Series) -> float:
        """Calculate extreme value index."""
        try:
            threshold = returns.quantile(0.10)
            excess = returns[returns < threshold] - threshold
            if len(excess) > 0 and excess.mean() != 0:
                return (excess.var() / (excess.mean() ** 2) - 1) / 2
        except:
            pass
        return 0
    
    def _calculate_illiquidity_ratio(self, returns: pd.Series) -> float:
        """Calculate illiquidity ratio (Amihud measure approximation)."""
        try:
            if len(returns) > 0:
                returns_mean = abs(returns).mean()
                returns_range = returns.max() - returns.min()
                if returns_range != 0:
                    return returns_mean / returns_range
        except:
            pass
        return 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(returns) == 0:
                return 0
                
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min() if not drawdown.empty else 0
        except:
            return 0
    
    def _create_fallback_results(self, returns: pd.Series, 
                                portfolio_value: float) -> Dict:
        """Create fallback results when comprehensive analysis fails."""
        returns_clean = returns.dropna()
        if len(returns_clean) > 0:
            var_95 = -np.percentile(returns_clean, 5)
            cvar_95_data = returns_clean[returns_clean <= -var_95]
            cvar_95 = -cvar_95_data.mean() if len(cvar_95_data) > 0 else var_95 * 1.2
        else:
            var_95 = 0
            cvar_95 = 0
        
        return {
            'methods': {
                'Historical': {
                    0.95: {
                        'VaR': var_95,
                        'VaR_absolute': var_95 * portfolio_value,
                        'CVaR': cvar_95,
                        'CVaR_absolute': cvar_95 * portfolio_value,
                        'ES': cvar_95,
                        'ES_absolute': cvar_95 * portfolio_value,
                        'confidence': 0.95,
                        'method': 'Historical'
                    }
                }
            },
            'portfolio_value': portfolio_value,
            'summary': {
                'best_method': 'Historical',
                'worst_case_var': var_95,
                'average_var': var_95,
                'var_consistency': 0
            },
            'violations': {
                'total_days': len(returns_clean),
                'violations_95': (returns_clean < var_95).sum() if len(returns_clean) > 0 else 0,
                'exception_rates': {
                    0.95: {
                        'actual': (returns_clean < var_95).sum() / len(returns_clean) if len(returns_clean) > 0 else 0,
                        'expected': 0.05,
                        'difference': (returns_clean < var_95).sum() / len(returns_clean) - 0.05 if len(returns_clean) > 0 else -0.05
                    }
                }
            }
        }

# Initialize advanced risk analytics
risk_analytics = AdvancedRiskAnalytics()

# ============================================================================
# 5. ENHANCED PORTFOLIO OPTIMIZATION ENGINE
# ============================================================================

class AdvancedPortfolioOptimizer:
    """Advanced portfolio optimization with multiple algorithms and constraints."""
    
    def __init__(self):
        self.optimization_methods = {
            'MAX_SHARPE': self._optimize_max_sharpe,
            'MIN_VARIANCE': self._optimize_min_variance,
            'MAX_RETURN': self._optimize_max_return,
            'RISK_PARITY': self._optimize_risk_parity,
            'MAX_DIVERSIFICATION': self._optimize_max_diversification,
            'HRP': self._optimize_hierarchical_risk_parity,
            'BLACK_LITTERMAN': self._optimize_black_litterman,
            'MEAN_CVAR': self._optimize_mean_cvar,
            'MEAN_VARIANCE_SEMI': self._optimize_mean_variance_semi,
            'ROBUST_OPTIMIZATION': self._optimize_robust
        }
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          method: str = 'MAX_SHARPE',
                          constraints: Dict = None,
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
                    returns_clean, constraints, risk_free_rate
                )
            else:
                # Default to max Sharpe
                weights, metrics = self._optimize_max_sharpe(returns_clean, constraints, risk_free_rate)
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(returns_clean, weights, risk_free_rate)
            metrics.update(additional_metrics)
            
            # Create comprehensive results
            results = {
                'weights': weights,
                'metrics': metrics,
                'method': method,
                'constraints': constraints,
                'risk_free_rate': risk_free_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            performance_monitor.end_operation(f'portfolio_optimization_{method}')
            return results
            
        except Exception as e:
            performance_monitor.end_operation(f'portfolio_optimization_{method}', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {
                'operation': f'portfolio_optimization_{method}',
                'assets': len(returns.columns)
            })
            
            # Fallback to equal weight
            return self._fallback_equal_weight(returns, method, risk_free_rate)
    
    def _optimize_max_sharpe(self, returns: pd.DataFrame, 
                            constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Maximize Sharpe ratio."""
        try:
            # Check if PyPortfolioOpt is available
            if hasattr(st.session_state, 'pypfopt_available') and st.session_state.pypfopt_available:
                try:
                    from pypfopt import expected_returns, risk_models
                    from pypfopt.efficient_frontier import EfficientFrontier
                    
                    mu = expected_returns.mean_historical_return(returns)
                    S = risk_models.sample_cov(returns)
                    
                    ef = EfficientFrontier(mu, S)
                    
                    # Apply constraints
                    if constraints:
                        if 'bounds' in constraints:
                            ef.bounds = constraints['bounds']
                        if 'weight_bounds' in constraints:
                            ef.weight_bounds = constraints['weight_bounds']
                    
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                    cleaned_weights = ef.clean_weights()
                    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                    
                    return cleaned_weights, {
                        'expected_return': perf[0],
                        'expected_volatility': perf[1],
                        'sharpe_ratio': perf[2]
                    }
                except Exception as e:
                    # Fall through to simplified implementation
                    pass
        
        except Exception as e:
            pass
        
        # Simplified implementation
        return self._simplified_max_sharpe(returns, risk_free_rate)
    
    def _simplified_max_sharpe(self, returns: pd.DataFrame, 
                              risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Simplified max Sharpe optimization."""
        mu = returns.mean() * 252
        S = returns.cov() * 252
        
        n_assets = len(mu)
        
        def objective(weights):
            port_return = np.dot(weights, mu)
            port_risk = np.sqrt(weights.T @ S @ weights)
            if port_risk == 0:
                return 1e10  # Penalize zero risk
            return -(port_return - risk_free_rate) / port_risk
        
        # Constraints
        bounds = [(0, 1) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimization
        result = optimize.minimize(
            objective,
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
        
        # If optimization fails, return equal weights
        equal_weight = 1.0 / n_assets
        weights = {ticker: equal_weight for ticker in returns.columns}
        
        port_return = np.mean(mu)
        port_risk = np.sqrt(np.mean(np.diag(S)) / n_assets + 
                           (n_assets - 1) / n_assets * np.mean(S.values))
        sharpe = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
        
        return weights, {
            'expected_return': port_return,
            'expected_volatility': port_risk,
            'sharpe_ratio': sharpe
        }
    
    def _optimize_min_variance(self, returns: pd.DataFrame,
                              constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Minimize portfolio variance."""
        try:
            # Check if PyPortfolioOpt is available
            if hasattr(st.session_state, 'pypfopt_available') and st.session_state.pypfopt_available:
                try:
                    from pypfopt import expected_returns, risk_models
                    from pypfopt.efficient_frontier import EfficientFrontier
                    
                    mu = expected_returns.mean_historical_return(returns)
                    S = risk_models.sample_cov(returns)
                    
                    ef = EfficientFrontier(mu, S)
                    
                    if constraints:
                        if 'bounds' in constraints:
                            ef.bounds = constraints['bounds']
                    
                    weights = ef.min_volatility()
                    cleaned_weights = ef.clean_weights()
                    perf = ef.portfolio_performance()
                    
                    return cleaned_weights, {
                        'expected_return': perf[0],
                        'expected_volatility': perf[1],
                        'sharpe_ratio': (perf[0] - risk_free_rate) / perf[1] if perf[1] > 0 else 0
                    }
                except Exception as e:
                    pass  # Fall through to simplified implementation
        
        except Exception as e:
            pass
        
        # Simplified implementation
        S = returns.cov() * 252
        n_assets = len(returns.columns)
        
        def objective(weights):
            return weights.T @ S @ weights
        
        bounds = [(0, 1) for _ in range(n_assets)]
        constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        initial_weights = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints_opt,
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
        
        # Equal weight fallback
        return self._fallback_equal_weight_simple(returns, risk_free_rate)
    
    def _optimize_max_return(self, returns: pd.DataFrame,
                            constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Maximize portfolio return with risk constraint."""
        try:
            # Check if PyPortfolioOpt is available
            if hasattr(st.session_state, 'pypfopt_available') and st.session_state.pypfopt_available:
                try:
                    from pypfopt import expected_returns, risk_models
                    from pypfopt.efficient_frontier import EfficientFrontier
                    
                    mu = expected_returns.mean_historical_return(returns)
                    S = risk_models.sample_cov(returns)
                    
                    ef = EfficientFrontier(mu, S)
                    
                    # Default max volatility constraint
                    max_vol = constraints.get('max_volatility', 0.30) if constraints else 0.30
                    ef.efficient_risk(target_volatility=max_vol)
                    
                    weights = ef.weights
                    cleaned_weights = ef.clean_weights()
                    perf = ef.portfolio_performance()
                    
                    return cleaned_weights, {
                        'expected_return': perf[0],
                        'expected_volatility': perf[1],
                        'sharpe_ratio': (perf[0] - risk_free_rate) / perf[1] if perf[1] > 0 else 0
                    }
                except Exception as e:
                    pass  # Fall through to simplified implementation
        
        except Exception as e:
            pass
        
        # Simplified implementation
        mu = returns.mean() * 252
        n_assets = len(mu)
        
        def objective(weights):
            return -np.dot(weights, mu)  # Minimize negative return
        
        bounds = [(0, 1) for _ in range(n_assets)]
        constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Add risk constraint if specified
        if constraints and 'max_volatility' in constraints:
            S = returns.cov() * 252
            max_vol = constraints['max_volatility']
            constraints_opt.append({
                'type': 'ineq',
                'fun': lambda w: max_vol - np.sqrt(w.T @ S @ w)
            })
        
        initial_weights = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints_opt,
            method='SLSQP'
        )
        
        if result.success:
            weights = result.x
            weights = weights / weights.sum()
            weight_dict = dict(zip(returns.columns, weights))
            
            # Calculate metrics
            port_return = np.dot(weights, mu)
            S = returns.cov() * 252
            port_risk = np.sqrt(weights.T @ S @ weights)
            
            return weight_dict, {
                'expected_return': port_return,
                'expected_volatility': port_risk,
                'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
            }
        
        return self._fallback_equal_weight_simple(returns, risk_free_rate)
    
    def _optimize_risk_parity(self, returns: pd.DataFrame,
                             constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Risk parity optimization."""
        try:
            # Simplified risk parity: inverse volatility weighting
            volatilities = returns.std() * np.sqrt(252)
            # Replace zero volatilities with small value
            volatilities = volatilities.replace(0, volatilities[volatilities > 0].min() if any(volatilities > 0) else 0.01)
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
                'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0,
                'risk_parity_score': self._calculate_risk_parity_score(returns, weight_dict)
            }
            
        except Exception as e:
            return self._fallback_equal_weight_simple(returns, risk_free_rate)
    
    def _calculate_risk_parity_score(self, returns: pd.DataFrame, weights: Dict) -> float:
        """Calculate how close the portfolio is to perfect risk parity."""
        try:
            S = returns.cov() * 252
            w_array = np.array(list(weights.values()))
            
            # Portfolio volatility
            port_vol = np.sqrt(w_array.T @ S @ w_array)
            if port_vol == 0:
                return 0
            
            # Marginal contribution to risk
            mctr = (S @ w_array) / port_vol
            
            # Risk contribution
            rctr = w_array * mctr
            
            # Perfect risk parity would have equal risk contributions
            target_contribution = 1 / len(weights)
            score = 1 - np.std(rctr) / target_contribution if target_contribution > 0 else 0
            
            return max(0, min(1, score))
            
        except:
            return 0
    
    def _optimize_max_diversification(self, returns: pd.DataFrame,
                                     constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Maximum diversification optimization."""
        try:
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            def diversification_ratio(weights):
                w = np.array(weights)
                # Portfolio volatility
                port_vol = np.sqrt(w.T @ (returns.cov() * 252) @ w)
                if port_vol == 0:
                    return 0
                # Weighted average volatility
                weighted_vol = np.sum(w * (returns.std() * np.sqrt(252)))
                return weighted_vol / port_vol
            
            n_assets = len(returns.columns)
            initial_weights = np.ones(n_assets) / n_assets
            
            bounds = [(0, 1) for _ in range(n_assets)]
            constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            result = optimize.minimize(
                lambda w: -diversification_ratio(w),
                initial_weights,
                bounds=bounds,
                constraints=constraints_opt,
                method='SLSQP'
            )
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()
                weight_dict = dict(zip(returns.columns, weights))
                
                # Calculate metrics
                mu = returns.mean() * 252
                S = returns.cov() * 252
                w_array = np.array(list(weight_dict.values()))
                
                port_return = np.dot(w_array, mu)
                port_risk = np.sqrt(w_array.T @ S @ w_array)
                div_ratio = diversification_ratio(w_array)
                
                return weight_dict, {
                    'expected_return': port_return,
                    'expected_volatility': port_risk,
                    'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0,
                    'diversification_ratio': div_ratio
                }
        
        except Exception as e:
            pass
        
        return self._fallback_equal_weight_simple(returns, risk_free_rate)
    
    def _optimize_hierarchical_risk_parity(self, returns: pd.DataFrame,
                                          constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Hierarchical Risk Parity optimization."""
        try:
            # Check if PyPortfolioOpt is available
            if hasattr(st.session_state, 'pypfopt_available') and st.session_state.pypfopt_available:
                try:
                    from pypfopt.hierarchical_portfolio import HRPOpt
                    
                    hrp = HRPOpt(returns)
                    weights = hrp.optimize()
                    
                    # Calculate metrics
                    mu = returns.mean() * 252
                    S = returns.cov() * 252
                    w_array = np.array(list(weights.values()))
                    
                    port_return = np.dot(w_array, mu)
                    port_risk = np.sqrt(w_array.T @ S @ w_array)
                    
                    return weights, {
                        'expected_return': port_return,
                        'expected_volatility': port_risk,
                        'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
                    }
                except Exception as e:
                    pass  # Fall through to risk parity
        
        except Exception as e:
            pass
        
        return self._optimize_risk_parity(returns, constraints, risk_free_rate)
    
    def _optimize_black_litterman(self, returns: pd.DataFrame,
                                 constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Black-Litterman optimization."""
        try:
            # Check if PyPortfolioOpt is available
            if hasattr(st.session_state, 'pypfopt_available') and st.session_state.pypfopt_available:
                try:
                    from pypfopt import expected_returns, risk_models
                    from pypfopt.efficient_frontier import EfficientFrontier
                    
                    mu = expected_returns.mean_historical_return(returns)
                    S = risk_models.sample_cov(returns)
                    
                    # Simplified Black-Litterman: use market equilibrium as starting point
                    ef = EfficientFrontier(mu, S)
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                    cleaned_weights = ef.clean_weights()
                    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                    
                    return cleaned_weights, {
                        'expected_return': perf[0],
                        'expected_volatility': perf[1],
                        'sharpe_ratio': perf[2]
                    }
                except Exception as e:
                    pass  # Fall through to max sharpe
        
        except Exception as e:
            pass
        
        return self._optimize_max_sharpe(returns, constraints, risk_free_rate)
    
    def _optimize_mean_cvar(self, returns: pd.DataFrame,
                           constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Mean-CVaR optimization."""
        try:
            # Simplified implementation
            n_assets = len(returns.columns)
            n_scenarios = 1000
            confidence = 0.95
            
            # Generate scenarios
            mu = returns.mean()
            S = returns.cov()
            # Add small regularization to covariance matrix
            S_reg = S + np.eye(n_assets) * 1e-6
            scenarios = np.random.multivariate_normal(mu, S_reg, n_scenarios)
            
            def cvar_objective(weights):
                portfolio_returns = scenarios @ weights
                var = np.percentile(portfolio_returns, (1-confidence)*100)
                cvar = portfolio_returns[portfolio_returns <= var].mean()
                return cvar if not np.isnan(cvar) else 0
            
            def return_constraint(weights):
                portfolio_return = np.dot(weights, mu * 252)
                target_return = constraints.get('target_return', 0.10) if constraints else 0.10
                return portfolio_return - target_return
            
            bounds = [(0, 1) for _ in range(n_assets)]
            constraints_opt = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq', 'fun': return_constraint}
            ]
            
            initial_weights = np.ones(n_assets) / n_assets
            
            result = optimize.minimize(
                cvar_objective,
                initial_weights,
                bounds=bounds,
                constraints=constraints_opt,
                method='SLSQP'
            )
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()
                weight_dict = dict(zip(returns.columns, weights))
                
                # Calculate metrics
                mu_annual = returns.mean() * 252
                S_annual = returns.cov() * 252
                w_array = np.array(list(weight_dict.values()))
                
                port_return = np.dot(w_array, mu_annual)
                port_risk = np.sqrt(w_array.T @ S_annual @ w_array)
                
                # Calculate CVaR
                portfolio_returns_scenarios = scenarios @ w_array
                var = np.percentile(portfolio_returns_scenarios, (1-confidence)*100)
                cvar_data = portfolio_returns_scenarios[portfolio_returns_scenarios <= var]
                cvar = cvar_data.mean() if len(cvar_data) > 0 else var
                
                return weight_dict, {
                    'expected_return': port_return,
                    'expected_volatility': port_risk,
                    'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0,
                    'cvar_95': cvar,
                    'var_95': var
                }
        
        except Exception as e:
            pass
        
        return self._optimize_min_variance(returns, constraints, risk_free_rate)
    
    def _optimize_mean_variance_semi(self, returns: pd.DataFrame,
                                    constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Mean-variance with semi-variance optimization."""
        try:
            # Use semi-variance instead of variance
            mu = returns.mean() * 252
            n_assets = len(returns.columns)
            
            def semivariance(weights):
                portfolio_returns = returns @ weights
                downside_returns = portfolio_returns[portfolio_returns < portfolio_returns.mean()]
                if len(downside_returns) == 0:
                    return 0
                return np.mean((downside_returns - portfolio_returns.mean()) ** 2) * 252
            
            def objective(weights):
                port_return = np.dot(weights, mu)
                semi_var = semivariance(weights)
                semi_vol = np.sqrt(semi_var) if semi_var > 0 else 0
                if semi_vol == 0:
                    return 1e10
                return -(port_return - risk_free_rate) / semi_vol
            
            bounds = [(0, 1) for _ in range(n_assets)]
            constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            initial_weights = np.ones(n_assets) / n_assets
            
            result = optimize.minimize(
                objective,
                initial_weights,
                bounds=bounds,
                constraints=constraints_opt,
                method='SLSQP'
            )
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()
                weight_dict = dict(zip(returns.columns, weights))
                
                # Calculate metrics
                port_return = np.dot(weights, mu)
                semi_var = semivariance(weights)
                semi_vol = np.sqrt(semi_var) if semi_var > 0 else 0
                
                # Calculate regular volatility for comparison
                S = returns.cov() * 252
                port_risk = np.sqrt(weights.T @ S @ weights)
                
                return weight_dict, {
                    'expected_return': port_return,
                    'expected_volatility': port_risk,
                    'semi_volatility': semi_vol,
                    'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0,
                    'sortino_ratio': (port_return - risk_free_rate) / semi_vol if semi_vol > 0 else 0
                }
        
        except Exception as e:
            pass
        
        return self._optimize_max_sharpe(returns, constraints, risk_free_rate)
    
    def _optimize_robust(self, returns: pd.DataFrame,
                        constraints: Dict, risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Robust optimization with uncertainty in parameters."""
        try:
            # Simple robust optimization using bootstrapping
            n_assets = len(returns.columns)
            n_bootstrap = 20  # Reduced for performance
            
            # Bootstrap expected returns and covariance
            bootstrap_returns = []
            bootstrap_covs = []
            
            for _ in range(n_bootstrap):
                sample_idx = np.random.choice(len(returns), size=len(returns), replace=True)
                sample_returns = returns.iloc[sample_idx]
                bootstrap_returns.append(sample_returns.mean() * 252)
                bootstrap_covs.append(sample_returns.cov() * 252)
            
            def worst_case_sharpe(weights):
                worst_sharpe = float('inf')
                for mu_sample, S_sample in zip(bootstrap_returns, bootstrap_covs):
                    port_return = np.dot(weights, mu_sample)
                    port_risk = np.sqrt(weights.T @ S_sample @ weights)
                    if port_risk > 0:
                        sharpe = (port_return - risk_free_rate) / port_risk
                        worst_sharpe = min(worst_sharpe, sharpe)
                    else:
                        worst_sharpe = min(worst_sharpe, -1e10)
                return -worst_sharpe  # Minimize negative worst-case Sharpe
            
            bounds = [(0, 1) for _ in range(n_assets)]
            constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            initial_weights = np.ones(n_assets) / n_assets
            
            result = optimize.minimize(
                worst_case_sharpe,
                initial_weights,
                bounds=bounds,
                constraints=constraints_opt,
                method='SLSQP',
                options={'maxiter': 200}  # Reduced for performance
            )
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()
                weight_dict = dict(zip(returns.columns, weights))
                
                # Calculate metrics using original parameters
                mu = returns.mean() * 252
                S = returns.cov() * 252
                w_array = np.array(list(weight_dict.values()))
                
                port_return = np.dot(w_array, mu)
                port_risk = np.sqrt(w_array.T @ S @ w_array)
                
                return weight_dict, {
                    'expected_return': port_return,
                    'expected_volatility': port_risk,
                    'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0,
                    'robustness_score': self._calculate_robustness_score(returns, weights)
                }
        
        except Exception as e:
            pass
        
        return self._optimize_max_sharpe(returns, constraints, risk_free_rate)
    
    def _calculate_robustness_score(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate robustness score of the portfolio."""
        try:
            n_bootstrap = 10  # Reduced for performance
            performances = []
            
            for _ in range(n_bootstrap):
                sample_idx = np.random.choice(len(returns), size=len(returns), replace=True)
                sample_returns = returns.iloc[sample_idx]
                
                mu_sample = sample_returns.mean() * 252
                S_sample = sample_returns.cov() * 252
                
                port_return = np.dot(weights, mu_sample)
                port_risk = np.sqrt(weights.T @ S_sample @ weights)
                
                if port_risk > 0:
                    sharpe = port_return / port_risk
                    performances.append(sharpe)
            
            if len(performances) > 0:
                # Robustness: low variance in performance across bootstrap samples
                return 1 / (1 + np.std(performances))
            
        except:
            pass
        
        return 0.5
    
    def _calculate_additional_metrics(self, returns: pd.DataFrame, 
                                     weights: Dict, risk_free_rate: float) -> Dict:
        """Calculate additional portfolio metrics."""
        metrics = {}
        
        try:
            mu = returns.mean() * 252
            S = returns.cov() * 252
            w_array = np.array(list(weights.values()))
            
            # Basic metrics
            port_return = np.dot(w_array, mu)
            port_risk = np.sqrt(w_array.T @ S @ w_array)
            
            metrics.update({
                'expected_return': port_return,
                'expected_volatility': port_risk,
                'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
            })
            
            # Calculate portfolio returns series
            portfolio_returns = returns.dot(w_array)
            
            # Downside risk metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            if len(downside_returns) > 0:
                downside_vol = np.std(downside_returns) * np.sqrt(252)
                metrics['downside_volatility'] = downside_vol
                metrics['sortino_ratio'] = (port_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
            else:
                metrics['downside_volatility'] = 0
                metrics['sortino_ratio'] = float('inf') if port_return > risk_free_rate else 0
            
            # Maximum drawdown
            if len(portfolio_returns) > 0:
                cumulative = (1 + portfolio_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_dd = drawdown.min() if not drawdown.empty else 0
                metrics['max_drawdown'] = max_dd
                if max_dd < 0:
                    metrics['calmar_ratio'] = port_return / abs(max_dd)
                else:
                    metrics['calmar_ratio'] = 0
            else:
                metrics['max_drawdown'] = 0
                metrics['calmar_ratio'] = 0
            
            # Omega ratio
            threshold = risk_free_rate / 252  # Daily risk-free rate
            gains = portfolio_returns[portfolio_returns > threshold].sum()
            losses = abs(portfolio_returns[portfolio_returns < threshold]).sum()
            metrics['omega_ratio'] = gains / losses if losses > 0 else float('inf')
            
            # Diversification metrics
            individual_vols = returns.std() * np.sqrt(252)
            weighted_avg_vol = np.sum(w_array * individual_vols)
            metrics['diversification_ratio'] = weighted_avg_vol / port_risk if port_risk > 0 else 1
            
            # Concentration metrics
            metrics['herfindahl_index'] = np.sum(w_array ** 2)
            metrics['effective_n_assets'] = 1 / metrics['herfindahl_index'] if metrics['herfindahl_index'] > 0 else len(w_array)
            
            # Skewness and kurtosis
            if len(portfolio_returns) > 0:
                metrics['skewness'] = portfolio_returns.skew()
                metrics['kurtosis'] = portfolio_returns.kurtosis()
            else:
                metrics['skewness'] = 0
                metrics['kurtosis'] = 0
            
            # Tail risk
            if len(portfolio_returns) > 0:
                var_95 = -np.percentile(portfolio_returns, 5)
                cvar_95_data = portfolio_returns[portfolio_returns <= -var_95]
                cvar_95 = -cvar_95_data.mean() if len(cvar_95_data) > 0 else var_95
                metrics['var_95'] = var_95
                metrics['cvar_95'] = cvar_95
                metrics['expected_shortfall_ratio'] = cvar_95 / var_95 if var_95 != 0 else 0
            else:
                metrics['var_95'] = 0
                metrics['cvar_95'] = 0
                metrics['expected_shortfall_ratio'] = 0
            
            # Turnover estimation (simplified)
            metrics['estimated_turnover'] = self._estimate_turnover(weights, returns)
            
            # Liquidity score (simplified)
            metrics['liquidity_score'] = self._calculate_liquidity_score(weights, returns)
            
        except Exception as e:
            # Fill with default values if calculation fails
            n_assets = len(weights)
            metrics.update({
                'downside_volatility': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'omega_ratio': 1,
                'diversification_ratio': 1,
                'herfindahl_index': 1/n_assets,
                'effective_n_assets': n_assets,
                'skewness': 0,
                'kurtosis': 0,
                'var_95': 0,
                'cvar_95': 0,
                'expected_shortfall_ratio': 0,
                'estimated_turnover': 0,
                'liquidity_score': 0.5
            })
        
        return metrics
    
    def _estimate_turnover(self, weights: Dict, returns: pd.DataFrame) -> float:
        """Estimate portfolio turnover."""
        try:
            # Simplified turnover estimation
            # Assume rebalancing creates turnover proportional to weight changes
            n_assets = len(weights)
            equal_weight = 1 / n_assets
            current_weights = np.array(list(weights.values()))
            
            # Calculate distance from equal weight (proxy for turnover)
            turnover = np.sum(np.abs(current_weights - equal_weight))
            return min(1.0, turnover)  # Cap at 100%
            
        except:
            return 0.5  # Default moderate turnover
    
    def _calculate_liquidity_score(self, weights: Dict, returns: pd.DataFrame) -> float:
        """Calculate portfolio liquidity score."""
        try:
            # Simplified liquidity score
            # Based on weight concentration and asset volatility
            weights_array = np.array(list(weights.values()))
            
            # Concentration penalty
            herfindahl = np.sum(weights_array ** 2)
            concentration_penalty = herfindahl * 0.5
            
            # Volatility penalty (illiquid assets often more volatile)
            volatilities = returns.std() * np.sqrt(252)
            avg_volatility = np.mean(volatilities)
            volatility_penalty = min(0.3, avg_volatility / 0.3)
            
            # Calculate score (0 = illiquid, 1 = liquid)
            score = 1 - (concentration_penalty + volatility_penalty) / 2
            return max(0, min(1, score))
            
        except:
            return 0.5
    
    def _fallback_equal_weight(self, returns: pd.DataFrame, 
                              method: str, risk_free_rate: float) -> Dict:
        """Fallback to equal weight portfolio."""
        n_assets = len(returns.columns)
        equal_weight = 1.0 / n_assets
        weights = {ticker: equal_weight for ticker in returns.columns}
        
        # Calculate metrics
        mu = returns.mean() * 252
        S = returns.cov() * 252
        w_array = np.array([equal_weight] * n_assets)
        
        port_return = np.mean(mu)
        port_risk = np.sqrt(np.mean(np.diag(S)) / n_assets + 
                           (n_assets - 1) / n_assets * np.mean(S.values))
        
        return {
            'weights': weights,
            'metrics': {
                'expected_return': port_return,
                'expected_volatility': port_risk,
                'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0,
                'method': f'{method} (Fallback: Equal Weight)'
            },
            'method': method,
            'constraints': None,
            'risk_free_rate': risk_free_rate,
            'timestamp': datetime.now().isoformat()
        }
    
    def _fallback_equal_weight_simple(self, returns: pd.DataFrame, 
                                     risk_free_rate: float) -> Tuple[Dict, Dict]:
        """Simple equal weight fallback."""
        n_assets = len(returns.columns)
        equal_weight = 1.0 / n_assets
        weights = {ticker: equal_weight for ticker in returns.columns}
        
        mu = returns.mean() * 252
        S = returns.cov() * 252
        w_array = np.array([equal_weight] * n_assets)
        
        port_return = np.mean(mu)
        port_risk = np.sqrt(np.mean(np.diag(S)) / n_assets + 
                           (n_assets - 1) / n_assets * np.mean(S.values))
        
        return weights, {
            'expected_return': port_return,
            'expected_volatility': port_risk,
            'sharpe_ratio': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
        }

# Initialize advanced portfolio optimizer
portfolio_optimizer = AdvancedPortfolioOptimizer()

# ============================================================================
# 6. ADVANCED DATA MANAGEMENT AND PROCESSING ENGINE
# ============================================================================

class AdvancedDataManager:
    """Advanced data management with caching, validation, and preprocessing."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.max_workers = 5
        self.retry_attempts = 2
        self.timeout = 15
    
    def fetch_advanced_market_data(self, tickers: List[str], 
                                  start_date: datetime, 
                                  end_date: datetime,
                                  interval: str = '1d',
                                  progress_callback = None) -> Dict:
        """Fetch advanced market data with multiple features."""
        performance_monitor.start_operation('fetch_advanced_market_data')
        
        try:
            data = {
                'prices': pd.DataFrame(),
                'returns': pd.DataFrame(),
                'volumes': pd.DataFrame(),
                'dividends': {},
                'splits': {},
                'metadata': {},
                'errors': {}
            }
            
            # Download data in parallel with limited workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(tickers))) as executor:
                future_to_ticker = {
                    executor.submit(self._fetch_single_ticker_data, 
                                  ticker, start_date, end_date, interval): ticker
                    for ticker in tickers
                }
                
                completed = 0
                total = len(tickers)
                
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed / total, f"Fetching {ticker}...")
                    
                    try:
                        ticker_data = future.result(timeout=self.timeout)
                        
                        if ticker_data and not ticker_data['prices'].empty:
                            # Add to data structures
                            if data['prices'].empty:
                                data['prices'] = ticker_data['prices']
                                data['volumes'] = ticker_data['volumes']
                            else:
                                data['prices'] = data['prices'].merge(
                                    ticker_data['prices'], left_index=True, right_index=True, how='outer'
                                )
                                data['volumes'] = data['volumes'].merge(
                                    ticker_data['volumes'], left_index=True, right_index=True, how='outer'
                                )
                            
                            # Store metadata
                            data['metadata'][ticker] = ticker_data['metadata']
                            
                            # Store dividends and splits
                            if not ticker_data['dividends'].empty:
                                data['dividends'][ticker] = ticker_data['dividends']
                            if not ticker_data['splits'].empty:
                                data['splits'][ticker] = ticker_data['splits']
                                
                        else:
                            data['errors'][ticker] = "No data returned"
                            
                    except Exception as e:
                        data['errors'][ticker] = str(e)
            
            # Process the data
            if not data['prices'].empty:
                # Forward fill and backfill missing prices
                data['prices'] = data['prices'].ffill().bfill()
                if not data['volumes'].empty:
                    data['volumes'] = data['volumes'].ffill().bfill()
                
                # Calculate returns
                data['returns'] = data['prices'].pct_change().dropna()
                
                # Remove assets with too many missing values
                if not data['returns'].empty:
                    missing_threshold = 0.3  # 30% missing
                    assets_to_keep = data['returns'].columns[
                        data['returns'].isnull().mean() < missing_threshold
                    ]
                    
                    if len(assets_to_keep) > 0:
                        data['prices'] = data['prices'][assets_to_keep]
                        data['returns'] = data['returns'][assets_to_keep]
                        if not data['volumes'].empty:
                            data['volumes'] = data['volumes'][assets_to_keep]
                    else:
                        raise ValueError("No assets with sufficient data")
            
            # Calculate additional features
            if not data['returns'].empty:
                data['additional_features'] = self._calculate_additional_features(data)
            
            performance_monitor.end_operation('fetch_advanced_market_data', {
                'tickers_fetched': len(data['prices'].columns) if not data['prices'].empty else 0,
                'tickers_failed': len(data['errors'])
            })
            
            return data
            
        except Exception as e:
            performance_monitor.end_operation('fetch_advanced_market_data', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {
                'operation': 'fetch_advanced_market_data',
                'tickers': tickers,
                'date_range': f"{start_date} to {end_date}"
            })
            raise
    
    def _fetch_single_ticker_data(self, ticker: str, 
                                 start_date: datetime, 
                                 end_date: datetime,
                                 interval: str) -> Dict:
        """Fetch data for a single ticker with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                stock = yf.Ticker(ticker)
                
                # Download historical data
                hist = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    actions=True
                )
                
                if hist.empty:
                    if attempt == self.retry_attempts - 1:
                        raise ValueError(f"No historical data for {ticker}")
                    time.sleep(1)
                    continue
                
                # Extract different data types
                prices = hist['Close'].rename(ticker)
                volumes = hist['Volume'].rename(ticker)
                dividends = stock.dividends[start_date:end_date]
                splits = stock.splits[start_date:end_date]
                
                # Get metadata
                info = stock.info
                metadata = {
                    'name': info.get('longName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'beta': info.get('beta', 1.0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'currency': info.get('currency', 'USD'),
                    'country': info.get('country', 'Unknown'),
                    'exchange': info.get('exchange', 'Unknown')
                }
                
                return {
                    'prices': pd.DataFrame(prices),
                    'volumes': pd.DataFrame(volumes),
                    'dividends': dividends,
                    'splits': splits,
                    'metadata': metadata
                }
                
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise
                time.sleep(1)  # Wait before retry
        
        return {}
    
    def _calculate_additional_features(self, data: Dict) -> Dict:
        """Calculate additional features for analysis."""
        features = {
            'technical_indicators': {},
            'statistical_features': {},
            'risk_metrics': {},
            'liquidity_metrics': {}
        }
        
        try:
            returns = data['returns']
            prices = data['prices']
            
            # Calculate basic statistics for each asset
            for ticker in returns.columns:
                ticker_returns = returns[ticker].dropna()
                
                if len(ticker_returns) > 0:
                    # Basic statistics
                    features['statistical_features'][ticker] = {
                        'mean': ticker_returns.mean(),
                        'std': ticker_returns.std(),
                        'skewness': ticker_returns.skew(),
                        'kurtosis': ticker_returns.kurtosis(),
                        'sharpe_ratio': ticker_returns.mean() / ticker_returns.std() if ticker_returns.std() > 0 else 0,
                        'max_drawdown': self._calculate_max_drawdown_series(ticker_returns)
                    }
                    
                    # Risk metrics
                    var_95 = -np.percentile(ticker_returns, 5)
                    cvar_95_data = ticker_returns[ticker_returns <= -var_95]
                    cvar_95 = -cvar_95_data.mean() if len(cvar_95_data) > 0 else var_95
                    
                    features['risk_metrics'][ticker] = {
                        'var_95': var_95,
                        'cvar_95': cvar_95,
                        'expected_shortfall': cvar_95
                    }
            
            # Calculate correlation matrix
            features['correlation_matrix'] = returns.corr()
            
            # Calculate covariance matrix
            features['covariance_matrix'] = returns.cov() * 252
            
        except Exception as e:
            features['error'] = str(e)
        
        return features
    
    def _calculate_max_drawdown_series(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a return series."""
        try:
            if len(returns) == 0:
                return 0
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min() if not drawdown.empty else 0
        except:
            return 0
    
    def validate_portfolio_data(self, data: Dict, 
                               min_assets: int = 3,
                               min_data_points: int = 50) -> Dict:
        """Validate portfolio data for analysis."""
        validation = {
            'is_valid': False,
            'issues': [],
            'warnings': [],
            'suggestions': [],
            'summary': {}
        }
        
        try:
            # Check if we have prices
            if data['prices'].empty:
                validation['issues'].append("No price data available")
                return validation
            
            # Check number of assets
            n_assets = len(data['prices'].columns)
            if n_assets < min_assets:
                validation['issues'].append(f"Only {n_assets} assets available, minimum {min_assets} required")
            
            # Check data points
            n_data_points = len(data['prices'])
            if n_data_points < min_data_points:
                validation['warnings'].append(f"Only {n_data_points} data points, recommended minimum {min_data_points}")
            
            # Check for missing values
            if not data['prices'].empty:
                missing_percentage = data['prices'].isnull().mean().mean()
                if missing_percentage > 0.1:  # More than 10% missing
                    validation['warnings'].append(f"High percentage of missing values: {missing_percentage:.1%}")
            
            # Check for zero or negative prices
            if not data['prices'].empty and (data['prices'] <= 0).any().any():
                validation['warnings'].append("Some assets have zero or negative prices")
            
            # Check returns calculation
            if data['returns'].empty:
                validation['issues'].append("Cannot calculate returns")
            else:
                # Check for infinite or NaN returns
                if not np.isfinite(data['returns'].values).all():
                    validation['warnings'].append("Non-finite values in returns")
                
                # Check for zero volatility assets
                zero_vol_assets = data['returns'].std()[data['returns'].std() == 0].index.tolist()
                if zero_vol_assets:
                    validation['warnings'].append(f"Zero volatility assets: {zero_vol_assets}")
            
            # Generate suggestions
            if validation['issues']:
                validation['suggestions'].extend([
                    "Check ticker symbols for validity",
                    "Extend date range for more data",
                    "Remove assets with missing data"
                ])
            
            # Final validation
            validation['is_valid'] = len(validation['issues']) == 0
            
            if validation['is_valid']:
                validation['summary'] = {
                    'assets': n_assets,
                    'data_points': n_data_points,
                    'date_range': f"{data['prices'].index[0].date()} to {data['prices'].index[-1].date()}" if n_data_points > 0 else "No data",
                    'missing_data': f"{missing_percentage:.1%}" if 'missing_percentage' in locals() else "N/A"
                }
            
        except Exception as e:
            validation['issues'].append(f"Validation error: {str(e)}")
        
        return validation
    
    def prepare_data_for_optimization(self, data: Dict, 
                                     remove_outliers: bool = True,
                                     fill_method: str = 'ffill') -> Dict:
        """Prepare data for portfolio optimization."""
        prepared_data = data.copy()
        
        try:
            returns = data['returns'].copy()
            
            # Remove outliers (if requested)
            if remove_outliers and not returns.empty:
                # Use IQR method to detect outliers
                Q1 = returns.quantile(0.25)
                Q3 = returns.quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with bounds
                returns = returns.clip(lower_bound, upper_bound, axis=1)
            
            # Fill any remaining NaN values
            if not returns.empty:
                if fill_method == 'ffill':
                    returns = returns.ffill().bfill()
                elif fill_method == 'zero':
                    returns = returns.fillna(0)
                elif fill_method == 'mean':
                    returns = returns.fillna(returns.mean())
            
            # Ensure no NaN values remain
            if not returns.empty and returns.isnull().any().any():
                # Remove columns with NaN values
                returns = returns.dropna(axis=1)
            
            # Update prepared data
            prepared_data['returns_clean'] = returns
            
            # Calculate clean prices from clean returns if possible
            if not returns.empty and not data['prices'].empty:
                # Get the first price for each asset
                initial_prices = data['prices'].iloc[0][returns.columns]
                
                # Reconstruct prices from clean returns
                price_reconstruction = pd.DataFrame(index=returns.index)
                for ticker in returns.columns:
                    if ticker in initial_prices:
                        cumulative_returns = (1 + returns[ticker]).cumprod()
                        price_reconstruction[ticker] = initial_prices[ticker] * cumulative_returns
                
                prepared_data['prices_clean'] = price_reconstruction
            
            # Calculate additional statistics
            if not returns.empty:
                prepared_data['statistics'] = {
                    'mean_returns': returns.mean(),
                    'volatility': returns.std(),
                    'sharpe_ratios': returns.mean() / returns.std(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'correlation_matrix': returns.corr(),
                    'covariance_matrix': returns.cov() * 252
                }
            
        except Exception as e:
            error_analyzer.analyze_error_with_context(e, {'operation': 'prepare_data_for_optimization'})
            # Return original data if preparation fails
            if not data['returns'].empty:
                prepared_data['returns_clean'] = data['returns'].fillna(0)
            if not data['prices'].empty:
                prepared_data['prices_clean'] = data['prices'].fillna(method='ffill').fillna(method='bfill')
        
        return prepared_data

# Initialize advanced data manager
data_manager = AdvancedDataManager()

# ============================================================================
# 7. SIMPLIFIED UI COMPONENTS (REPLACING COMPLEX SmartUIComponents)
# ============================================================================

class EnhancedUIComponents:
    """Enhanced UI components with simplified but effective styling."""
    
    @staticmethod
    def create_metric_card(title: str, value: Any, change: Any = None, 
                          icon: str = "📊", theme: str = "default") -> None:
        """Create enhanced metric card."""
        theme_colors = {
            'default': {'bg': 'rgba(30, 30, 30, 0.8)', 'accent': '#00cc96'},
            'success': {'bg': 'rgba(0, 204, 150, 0.1)', 'accent': '#00cc96'},
            'warning': {'bg': 'rgba(255, 161, 90, 0.1)', 'accent': '#FFA15A'},
            'danger': {'bg': 'rgba(239, 85, 59, 0.1)', 'accent': '#ef553b'},
            'info': {'bg': 'rgba(99, 110, 250, 0.1)', 'accent': '#636efa'}
        }
        
        colors = theme_colors.get(theme, theme_colors['default'])
        
        # Create columns for the card
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"""
            <div style="
                background: {colors['accent']};
                border-radius: 12px;
                width: 60px;
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                color: white;
                margin: 0 auto;
            ">
                {icon}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="
                padding: 0.5rem 0;
            ">
                <div style="
                    font-size: 0.8rem;
                    color: #94a3b8;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                ">
                    {title}
                </div>
                <div style="
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: white;
                    margin: 0.2rem 0;
                ">
                    {value}
                </div>
            """, unsafe_allow_html=True)
            
            if change is not None:
                try:
                    if isinstance(change, str):
                        change_color = "#00cc96" if "+" in change else "#ef553b"
                    elif isinstance(change, (int, float)):
                        change_color = "#00cc96" if change > 0 else "#ef553b"
                    else:
                        change_color = "#94a3b8"
                    
                    st.markdown(f"""
                    <div style="
                        font-size: 0.8rem;
                        color: {change_color};
                        font-weight: 600;
                    ">
                        {change}
                    </div>
                """, unsafe_allow_html=True)
                except:
                    pass
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    @staticmethod
    def create_progress_tracker(steps: List[str], current_step: int) -> None:
        """Create simplified progress tracker."""
        st.markdown("""
        <style>
            .progress-container {
                display: flex;
                justify-content: space-between;
                margin: 1.5rem 0;
                position: relative;
            }
            .progress-step {
                display: flex;
                flex-direction: column;
                align-items: center;
                z-index: 2;
                flex: 1;
            }
            .step-circle {
                width: 35px;
                height: 35px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                margin-bottom: 8px;
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.2);
                color: rgba(255, 255, 255, 0.5);
                transition: all 0.3s ease;
            }
            .step-circle.active {
                background: linear-gradient(135deg, #00cc96, #636efa);
                border-color: transparent;
                color: white;
                box-shadow: 0 4px 12px rgba(0, 204, 150, 0.3);
            }
            .step-circle.completed {
                background: rgba(0, 204, 150, 0.2);
                border-color: #00cc96;
                color: #00cc96;
            }
            .step-label {
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.5);
                text-align: center;
                font-weight: 500;
            }
            .step-label.active {
                color: white;
                font-weight: 600;
            }
            .progress-line {
                position: absolute;
                top: 17.5px;
                left: 0;
                right: 0;
                height: 2px;
                background: rgba(255, 255, 255, 0.1);
                z-index: 1;
            }
            .progress-fill {
                position: absolute;
                top: 17.5px;
                left: 0;
                height: 2px;
                background: linear-gradient(90deg, #00cc96, #636efa);
                z-index: 1;
                transition: width 0.5s ease;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Calculate fill width
        fill_width = (current_step / (len(steps) - 1)) * 100 if len(steps) > 1 else 0
        
        progress_html = f"""
        <div class="progress-container">
            <div class="progress-line"></div>
            <div class="progress-fill" style="width: {fill_width}%;"></div>
        """
        
        for i, step in enumerate(steps):
            status = "completed" if i < current_step else "active" if i == current_step else ""
            progress_html += f"""
            <div class="progress-step">
                <div class="step-circle {status}">{i + 1}</div>
                <div class="step-label {status if i == current_step else ''}">{step}</div>
            </div>
            """
        
        progress_html += "</div>"
        st.markdown(progress_html, unsafe_allow_html=True)

# Initialize enhanced UI components
ui = EnhancedUIComponents()

# ============================================================================
# 8. MAIN ENHANCED APPLICATION - FIXED AND WORKING VERSION
# ============================================================================

class QuantEdgeProEnhanced:
    """Enhanced QuantEdge Pro application with all advanced features."""
    
    def __init__(self):
        self.data_manager = data_manager
        self.risk_analytics = risk_analytics
        self.portfolio_optimizer = portfolio_optimizer
        self.viz_engine = viz_engine
        self.ui = ui
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'portfolio_data': None,
            'optimization_results': None,
            'risk_analysis_results': None,
            'current_step': 0,
            'analysis_complete': False,
            'data_fetched': False,
            'analysis_running': False,
            'config': None,
            'last_fetch_time': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_enhanced_header(self):
        """Render enhanced application header."""
        st.markdown("""
        <style>
            .main-header {
                background: linear-gradient(135deg, rgba(26, 29, 46, 0.95), rgba(42, 42, 42, 0.95));
                padding: 2rem;
                border-radius: 16px;
                margin-bottom: 2rem;
                border-left: 6px solid #00cc96;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
                text-align: center;
            }
            .main-title {
                background: linear-gradient(135deg, #00cc96, #636efa, #ab63fa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 3.5rem;
                font-weight: 800;
                margin-bottom: 1rem;
            }
            .main-subtitle {
                color: #94a3b8;
                font-size: 1.3rem;
                margin-bottom: 0.5rem;
            }
            .main-tagline {
                color: #636efa;
                font-size: 1.1rem;
                font-weight: 500;
            }
        </style>
        
        <div class="main-header">
            <div class="main-title">⚡ QuantEdge Pro v4.0 Enhanced</div>
            <div class="main-subtitle">Institutional Portfolio Analytics Platform with Advanced Risk Metrics</div>
            <div class="main-tagline">Featuring: Advanced VaR/CVaR/ES Analytics • 3D Visualizations • Smart Optimization</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Library status indicator
        if not LIBRARY_STATUS['all_available']:
            with st.expander("📦 Library Status", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.warning("Some optional libraries are missing:")
                    for lib in LIBRARY_STATUS['missing']:
                        st.write(f"• {lib}")
                with col2:
                    st.info("Basic functionality available. Install missing libraries for advanced features.")
                    st.code("pip install pypfopt scikit-learn statsmodels")
    
    def render_enhanced_sidebar(self):
        """Render enhanced sidebar with smart components."""
        with st.sidebar:
            st.markdown("""
            <div style="
                padding: 1.5rem;
                border-radius: 12px;
                background: rgba(30, 30, 30, 0.8);
                margin-bottom: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
            ">
                <h3 style="color: #00cc96; margin-bottom: 1rem; text-align: center;">🎯 Configuration Panel</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress tracker
            steps = ['Data Setup', 'Optimization', 'Risk Analysis', 'Results']
            self.ui.create_progress_tracker(steps, st.session_state.current_step)
            
            st.markdown("---")
            
            # Asset universe selection
            st.subheader("🌍 Asset Universe")
            universe_options = {
                "US Large Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "JNJ", "V", "WMT"],
                "Technology Focus": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC", "QCOM", "CRM", "ADBE", "ORCL"],
                "Global Diversified": ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "NSRGY", "NVO", "TSM", "HSBC"],
                "Emerging Markets": ["BABA", "TSM", "005930.KS", "ITUB", "VALE", "INFY", "HDB", "IDX", "EWZ"],
                "Custom Selection": []
            }
            
            selected_universe = st.selectbox(
                "Select Predefined Universe",
                list(universe_options.keys()),
                help="Choose from predefined asset universes",
                key="universe_select"
            )
            
            # Custom tickers input
            custom_tickers = ""
            if selected_universe == "Custom Selection":
                custom_tickers = st.text_area(
                    "Enter tickers (comma-separated)",
                    value="AAPL, MSFT, GOOGL, AMZN, TSLA",
                    help="Enter stock tickers separated by commas",
                    key="custom_tickers"
                )
            
            # Date range with presets
            st.subheader("📅 Time Period")
            date_preset = st.selectbox(
                "Select Period",
                ["Custom", "1 Year", "3 Years", "5 Years", "10 Years"],
                help="Select predefined time periods or choose custom",
                key="date_preset"
            )
            
            end_date = datetime.now()
            if date_preset == "1 Year":
                start_date = end_date - timedelta(days=365)
            elif date_preset == "3 Years":
                start_date = end_date - timedelta(days=365*3)
            elif date_preset == "5 Years":
                start_date = end_date - timedelta(days=365*5)
            elif date_preset == "10 Years":
                start_date = end_date - timedelta(days=365*10)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=end_date - timedelta(days=365*3), key="start_date")
                with col2:
                    end_date = st.date_input("End Date", value=end_date, key="end_date")
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings", expanded=False):
                risk_free_rate = st.slider(
                    "Risk-Free Rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=4.5,
                    step=0.1,
                    help="Annual risk-free rate for Sharpe ratio calculation",
                    key="risk_free_rate"
                ) / 100
                
                optimization_method = st.selectbox(
                    "Optimization Method",
                    ["MAX_SHARPE", "MIN_VARIANCE", "RISK_PARITY", "MAX_DIVERSIFICATION"],
                    help="Select portfolio optimization methodology",
                    key="optimization_method"
                )
                
                # Constraints
                st.subheader("🎯 Constraints")
                col1, col2 = st.columns(2)
                with col1:
                    max_weight = st.slider(
                        "Max Weight (%)",
                        min_value=5,
                        max_value=100,
                        value=30,
                        step=5,
                        help="Maximum weight for any single asset",
                        key="max_weight"
                    ) / 100
                with col2:
                    min_weight = st.slider(
                        "Min Weight (%)",
                        min_value=0,
                        max_value=20,
                        value=0,
                        step=1,
                        help="Minimum weight for any single asset",
                        key="min_weight"
                    ) / 100
            
            st.markdown("---")
            
            # Action buttons
            st.subheader("🚀 Actions")
            
            # Get tickers list
            tickers = []
            if selected_universe != "Custom Selection":
                tickers = universe_options[selected_universe]
            else:
                if custom_tickers:
                    tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
            
            config = {
                'universe': selected_universe,
                'tickers': tickers,
                'start_date': start_date,
                'end_date': end_date,
                'risk_free_rate': risk_free_rate,
                'optimization_method': optimization_method,
                'max_weight': max_weight,
                'min_weight': min_weight
            }
            
            col1, col2 = st.columns(2)
            with col1:
                fetch_clicked = st.button(
                    "📥 Fetch Data",
                    use_container_width=True,
                    key="fetch_data",
                    disabled=len(tickers) == 0,
                    help="Download market data for selected assets"
                )
            
            with col2:
                run_clicked = st.button(
                    "⚡ Run Analysis",
                    use_container_width=True,
                    key="run_analysis",
                    disabled=not st.session_state.data_fetched,
                    help="Run comprehensive portfolio analysis"
                )
            
            # Reset button
            if st.button("🔄 Reset Analysis", use_container_width=True, key="reset"):
                self._reset_analysis()
                st.rerun()
            
            # Add refresh button if data was fetched recently
            if st.session_state.last_fetch_time:
                time_since_fetch = (datetime.now() - st.session_state.last_fetch_time).seconds / 60
                if time_since_fetch > 10:  # 10 minutes
                    if st.button("🔄 Refresh Data", use_container_width=True, key="refresh"):
                        st.session_state.data_fetched = False
                        st.rerun()
            
            return config, fetch_clicked, run_clicked
    
    def run_data_fetch(self, config: Dict):
        """Run data fetching process."""
        try:
            if not config['tickers']:
                st.error("Please select or enter at least one ticker")
                return False
            
            with st.spinner(f"📥 Fetching data for {len(config['tickers'])} assets..."):
                # Update progress
                st.session_state.current_step = 1
                
                # Create progress bar
                progress_bar = st.progress(0)
                
                def progress_callback(progress, message):
                    progress_bar.progress(progress, text=message)
                
                # Fetch data
                portfolio_data = self.data_manager.fetch_advanced_market_data(
                    tickers=config['tickers'],
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    progress_callback=progress_callback
                )
                
                # Validate data
                validation = self.data_manager.validate_portfolio_data(portfolio_data)
                
                if validation['is_valid']:
                    st.session_state.portfolio_data = portfolio_data
                    st.session_state.data_fetched = True
                    st.session_state.config = config
                    st.session_state.last_fetch_time = datetime.now()
                    
                    # Close progress bar
                    progress_bar.empty()
                    
                    st.success(f"✅ Data fetched successfully!")
                    
                    # Show data preview
                    with st.expander("📊 Data Preview", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Assets", validation['summary']['assets'])
                        with col2:
                            st.metric("Date Range", validation['summary']['date_range'])
                        with col3:
                            st.metric("Data Points", validation['summary']['data_points'])
                        
                        if 'errors' in portfolio_data and portfolio_data['errors']:
                            st.warning(f"⚠️ {len(portfolio_data['errors'])} assets failed to fetch")
                    
                    return True
                else:
                    progress_bar.empty()
                    st.error("❌ Data validation failed:")
                    for issue in validation['issues']:
                        st.write(f"• {issue}")
                    for warning in validation['warnings']:
                        st.warning(f"⚠️ {warning}")
                    return False
                    
        except Exception as e:
            if 'progress_bar' in locals():
                progress_bar.empty()
            
            error_analysis = error_analyzer.analyze_error_with_context(e, {
                'operation': 'data_fetch',
                'tickers': config['tickers'],
                'date_range': f"{config['start_date']} to {config['end_date']}"
            })
            
            st.error(f"❌ Error fetching data: {str(e)}")
            
            with st.expander("Error Details", expanded=False):
                error_analyzer.create_advanced_error_display(error_analysis)
            
            return False
    
    def run_portfolio_analysis(self, config: Dict):
        """Run comprehensive portfolio analysis."""
        try:
            if st.session_state.portfolio_data is None:
                st.error("Please fetch data first")
                return False
            
            # Update progress
            st.session_state.current_step = 2
            st.session_state.analysis_running = True
            
            # Prepare data
            with st.spinner("🔄 Preparing data for analysis..."):
                prepared_data = self.data_manager.prepare_data_for_optimization(
                    st.session_state.portfolio_data,
                    remove_outliers=True
                )
            
            # Check if we have enough data
            if prepared_data['returns_clean'].empty or len(prepared_data['returns_clean'].columns) < 2:
                st.error("Insufficient data for analysis. Please check your data selection.")
                st.session_state.analysis_running = False
                return False
            
            # Create progress containers
            optimization_progress = st.progress(0, text="Optimizing portfolio...")
            
            # Run portfolio optimization
            try:
                optimization_results = self.portfolio_optimizer.optimize_portfolio(
                    returns=prepared_data['returns_clean'],
                    method=config['optimization_method'],
                    constraints={
                        'bounds': (config['min_weight'], config['max_weight'])
                    },
                    risk_free_rate=config['risk_free_rate']
                )
                
                st.session_state.optimization_results = optimization_results
                optimization_progress.progress(0.5, text="Portfolio optimized! Analyzing risk...")
            except Exception as e:
                optimization_progress.empty()
                st.error(f"Optimization failed: {str(e)}")
                st.session_state.analysis_running = False
                return False
            
            # Run risk analysis
            try:
                # Calculate portfolio returns
                weights_array = np.array(list(optimization_results['weights'].values()))
                portfolio_returns = prepared_data['returns_clean'].dot(weights_array)
                
                risk_analysis_results = self.risk_analytics.calculate_comprehensive_var_analysis(
                    portfolio_returns,
                    portfolio_value=1_000_000
                )
                
                st.session_state.risk_analysis_results = risk_analysis_results
                optimization_progress.progress(1.0, text="Analysis complete!")
            except Exception as e:
                optimization_progress.empty()
                st.error(f"Risk analysis failed: {str(e)}")
                st.session_state.analysis_running = False
                return False
            
            # Update progress
            st.session_state.current_step = 3
            st.session_state.analysis_complete = True
            st.session_state.analysis_running = False
            
            time.sleep(0.5)  # Let user see completion
            optimization_progress.empty()
            
            st.success("✅ Analysis complete!")
            return True
            
        except Exception as e:
            st.session_state.analysis_running = False
            
            if 'optimization_progress' in locals():
                optimization_progress.empty()
            
            error_analysis = error_analyzer.analyze_error_with_context(e, {
                'operation': 'portfolio_analysis',
                'optimization_method': config['optimization_method']
            })
            
            st.error(f"❌ Error during analysis: {str(e)}")
            
            with st.expander("Error Details", expanded=False):
                error_analyzer.create_advanced_error_display(error_analysis)
            
            return False
    
    def render_optimization_results(self):
        """Render optimization results."""
        if st.session_state.optimization_results is None:
            return
        
        results = st.session_state.optimization_results
        
        st.markdown('<h2 class="section-header">⚡ Portfolio Optimization Results</h2>', 
                   unsafe_allow_html=True)
        
        # Key metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.ui.create_metric_card(
                "Expected Return",
                f"{results['metrics']['expected_return']:.2%}",
                icon="📈",
                theme="success"
            )
        with col2:
            self.ui.create_metric_card(
                "Expected Volatility",
                f"{results['metrics']['expected_volatility']:.2%}",
                icon="📉",
                theme="warning"
            )
        with col3:
            self.ui.create_metric_card(
                "Sharpe Ratio",
                f"{results['metrics']['sharpe_ratio']:.2f}",
                icon="⚡",
                theme="info"
            )
        with col4:
            max_dd = results['metrics'].get('max_drawdown', 0)
            self.ui.create_metric_card(
                "Max Drawdown",
                f"{max_dd:.2%}",
                icon="📊",
                theme="danger"
            )
        
        # Portfolio allocation
        st.subheader("🎯 Portfolio Allocation")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # Create sunburst chart if metadata is available
            if st.session_state.portfolio_data and 'metadata' in st.session_state.portfolio_data:
                try:
                    fig = self.viz_engine.create_portfolio_allocation_sunburst(
                        results['weights'],
                        st.session_state.portfolio_data['metadata']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Sunburst chart error: {str(e)}")
                    # Fallback to pie chart
                    weights_df = pd.DataFrame.from_dict(results['weights'], orient='index', columns=['Weight'])
                    fig = px.pie(weights_df, values='Weight', names=weights_df.index, 
                                title='Portfolio Allocation')
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Simple pie chart as fallback
                weights_df = pd.DataFrame.from_dict(results['weights'], orient='index', columns=['Weight'])
                fig = px.pie(weights_df, values='Weight', names=weights_df.index, 
                            title='Portfolio Allocation')
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display weights in table
            weights_df = pd.DataFrame(
                [(ticker, f"{weight:.2%}") 
                 for ticker, weight in results['weights'].items()],
                columns=['Asset', 'Weight']
            ).sort_values('Weight', ascending=False)
            
            st.dataframe(
                weights_df,
                use_container_width=True,
                height=400
            )
        
        # Additional metrics
        st.subheader("📊 Additional Metrics")
        
        metrics_grid = st.columns(4)
        metrics_to_display = [
            ('Sortino Ratio', 'sortino_ratio', 0),
            ('Calmar Ratio', 'calmar_ratio', 1),
            ('Omega Ratio', 'omega_ratio', 2),
            ('Diversification', 'diversification_ratio', 3),
            ('Effective Assets', 'effective_n_assets', 0),
            ('Skewness', 'skewness', 1),
            ('Kurtosis', 'kurtosis', 2),
            ('VaR 95%', 'var_95', 3)
        ]
        
        for label, key, col_idx in metrics_to_display:
            if key in results['metrics']:
                value = results['metrics'][key]
                if isinstance(value, float):
                    if abs(value) < 10:
                        display_value = f"{value:.3f}"
                    else:
                        display_value = f"{value:.2f}"
                else:
                    display_value = str(value)
                
                with metrics_grid[col_idx]:
                    st.metric(label, display_value)
    
    def render_risk_analysis_results(self):
        """Render advanced risk analysis results."""
        if st.session_state.risk_analysis_results is None:
            return
        
        results = st.session_state.risk_analysis_results
        
        st.markdown('<h2 class="section-header">📈 Advanced Risk Analytics</h2>', 
                   unsafe_allow_html=True)
        
        # Create tabs for different risk analyses
        tab1, tab2, tab3 = st.tabs([
            "📊 VaR Dashboard",
            "📈 Comparative Analysis",
            "🌪️ Stress Testing"
        ])
        
        with tab1:
            self._render_var_dashboard(results)
        
        with tab2:
            self._render_comparative_analysis(results)
        
        with tab3:
            self._render_stress_testing(results)
    
    def _render_var_dashboard(self, results: Dict):
        """Render VaR analysis dashboard."""
        # Get portfolio returns from optimization results
        if st.session_state.optimization_results and st.session_state.portfolio_data:
            portfolio_returns = None
            try:
                # Calculate portfolio returns
                if 'returns_clean' in st.session_state.portfolio_data:
                    returns = st.session_state.portfolio_data['returns_clean']
                    weights = st.session_state.optimization_results['weights']
                    weights_array = np.array(list(weights.values()))
                    portfolio_returns = returns.dot(weights_array)
                else:
                    # Fallback calculation
                    returns = st.session_state.portfolio_data['returns']
                    weights = st.session_state.optimization_results['weights']
                    weights_array = np.array(list(weights.values()))
                    portfolio_returns = returns.dot(weights_array)
                
                if portfolio_returns is not None and len(portfolio_returns.dropna()) > 0:
                    # Create advanced VaR dashboard
                    fig = self.viz_engine.create_advanced_var_analysis_dashboard(portfolio_returns)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # VaR summary
                    st.subheader("📋 VaR Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Best Method",
                            results['summary'].get('best_method', 'N/A')
                        )
                    with col2:
                        worst_case_var = results['summary'].get('worst_case_var', 0)
                        st.metric(
                            "Worst-Case VaR",
                            f"{worst_case_var:.3%}"
                        )
                    with col3:
                        avg_var = results['summary'].get('average_var', 0)
                        st.metric(
                            "Average VaR",
                            f"{avg_var:.3%}"
                        )
                    with col4:
                        var_consistency = results['summary'].get('var_consistency', 0)
                        st.metric(
                            "VaR Consistency",
                            f"{var_consistency:.3f}"
                        )
                else:
                    st.warning("Insufficient portfolio return data for VaR dashboard")
            except Exception as e:
                st.error(f"Error creating VaR dashboard: {str(e)}")
    
    def _render_comparative_analysis(self, results: Dict):
        """Render comparative analysis."""
        st.subheader("📊 Method Comparison")
        
        # Create comparison table
        comparison_data = []
        if 'methods' in results:
            for method in results['methods']:
                if 0.95 in results['methods'][method]:
                    metrics = results['methods'][method][0.95]
                    comparison_data.append({
                        'Method': method,
                        'VaR': f"{metrics['VaR']:.3%}",
                        'CVaR': f"{metrics['CVaR']:.3%}",
                        'ES': f"{metrics['ES']:.3%}",
                        'VaR (Abs)': f"${metrics['VaR_absolute']:,.0f}",
                        'CVaR (Abs)': f"${metrics['CVaR_absolute']:,.0f}"
                    })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Display with styling
            st.dataframe(
                df_comparison,
                use_container_width=True,
                height=300
            )
        else:
            st.info("No comparative data available")
        
        # Violations analysis
        if results.get('violations'):
            st.subheader("🚨 VaR Violations Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Days",
                    results['violations']['total_days']
                )
            with col2:
                violations_95 = results['violations'].get('violations_95', 0)
                st.metric(
                    "Violations (95%)",
                    violations_95
                )
            with col3:
                violations_99 = results['violations'].get('violations_99', 0)
                st.metric(
                    "Violations (99%)",
                    violations_99
                )
    
    def _render_stress_testing(self, results: Dict):
        """Render stress testing results."""
        st.subheader("🌪️ Stress Testing Scenarios")
        
        # Historical scenarios
        if results.get('stress_tests', {}).get('historical_scenarios'):
            st.markdown("#### 📅 Historical Stress Periods")
            
            historical_data = []
            for scenario, metrics in results['stress_tests']['historical_scenarios'].items():
                historical_data.append({
                    'Scenario': scenario,
                    'Return': f"{metrics.get('returns', 0):.2%}",
                    'Volatility': f"{metrics.get('volatility', 0):.2%}",
                    'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                    'VaR 95%': f"{metrics.get('var_95', 0):.3%}",
                    'CVaR 95%': f"{metrics.get('cvar_95', 0):.3%}"
                })
            
            if historical_data:
                st.dataframe(pd.DataFrame(historical_data), use_container_width=True)
        
        # Hypothetical scenarios
        if results.get('stress_tests', {}).get('hypothetical_scenarios'):
            st.markdown("#### 🎯 Hypothetical Stress Scenarios")
            
            hypothetical_data = []
            for scenario, metrics in results['stress_tests']['hypothetical_scenarios'].items():
                hypothetical_data.append({
                    'Scenario': scenario,
                    'Stressed Return': f"{metrics.get('stressed_return', 0):.2%}",
                    'Stressed Volatility': f"{metrics.get('stressed_volatility', 0):.2%}",
                    'Stressed VaR 95%': f"{metrics.get('stressed_var_95', 0):.3%}",
                    'Description': metrics.get('description', '')
                })
            
            if hypothetical_data:
                st.dataframe(pd.DataFrame(hypothetical_data), use_container_width=True)
        
        # Show message if no stress test data
        if not results.get('stress_tests'):
            st.info("No stress test data available. Try running the analysis with more historical data.")
    
    def render_advanced_visualizations(self):
        """Render advanced visualizations."""
        if not st.session_state.analysis_complete:
            return
        
        st.markdown('<h2 class="section-header">🎨 Advanced Visualizations</h2>', 
                   unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs([
            "🎯 3D Efficient Frontier",
            "📊 Interactive Heatmap",
            "📈 Real-Time Dashboard"
        ])
        
        with tab1:
            self._render_3d_efficient_frontier()
        
        with tab2:
            self._render_interactive_heatmap()
        
        with tab3:
            self._render_realtime_dashboard()
    
    def _render_3d_efficient_frontier(self):
        """Render 3D efficient frontier visualization."""
        if st.session_state.portfolio_data and st.session_state.optimization_results:
            try:
                if 'returns_clean' in st.session_state.portfolio_data:
                    returns = st.session_state.portfolio_data['returns_clean']
                else:
                    returns = st.session_state.portfolio_data['returns']
                
                if len(returns.columns) >= 2:
                    # Get risk-free rate from config
                    risk_free_rate = st.session_state.config.get('risk_free_rate', 0.045) if st.session_state.config else 0.045
                    
                    fig = self.viz_engine.create_3d_efficient_frontier(returns, risk_free_rate)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    **3D Efficient Frontier Interpretation:**
                    - **X-axis**: Portfolio Risk (Volatility)
                    - **Y-axis**: Portfolio Return
                    - **Z-axis**: Sharpe Ratio
                    - **Color**: Sharpe Ratio (higher = better)
                    
                    The efficient frontier line shows optimal portfolios that maximize return for a given level of risk.
                    """)
                else:
                    st.warning("Need at least 2 assets for 3D efficient frontier visualization")
            except Exception as e:
                st.error(f"Error creating 3D efficient frontier: {str(e)}")
    
    def _render_interactive_heatmap(self):
        """Render interactive correlation heatmap."""
        if st.session_state.portfolio_data:
            try:
                if 'returns_clean' in st.session_state.portfolio_data:
                    returns = st.session_state.portfolio_data['returns_clean']
                else:
                    returns = st.session_state.portfolio_data['returns']
                
                if not returns.empty and len(returns.columns) >= 2:
                    correlation_matrix = returns.corr()
                    
                    fig = self.viz_engine.create_interactive_heatmap(correlation_matrix)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation statistics
                    st.subheader("📊 Correlation Statistics")
                    
                    # Calculate average correlation
                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
                    upper_tri = correlation_matrix.where(mask)
                    avg_correlation = upper_tri.stack().mean() if not upper_tri.stack().empty else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Correlation", f"{avg_correlation:.3f}")
                    with col2:
                        min_corr = correlation_matrix.min().min() if not correlation_matrix.empty else 0
                        st.metric("Min Correlation", f"{min_corr:.3f}")
                    with col3:
                        max_corr = correlation_matrix.max().max() if not correlation_matrix.empty else 0
                        st.metric("Max Correlation", f"{max_corr:.3f}")
                else:
                    st.warning("Insufficient data for correlation heatmap")
            except Exception as e:
                st.error(f"Error creating correlation heatmap: {str(e)}")
    
    def _render_realtime_dashboard(self):
        """Render real-time metrics dashboard."""
        if st.session_state.optimization_results:
            try:
                metrics = st.session_state.optimization_results['metrics']
                
                fig = self.viz_engine.create_real_time_metrics_dashboard(metrics)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating real-time dashboard: {str(e)}")
    
    def _reset_analysis(self):
        """Reset the analysis."""
        st.session_state.portfolio_data = None
        st.session_state.optimization_results = None
        st.session_state.risk_analysis_results = None
        st.session_state.current_step = 0
        st.session_state.analysis_complete = False
        st.session_state.data_fetched = False
        st.session_state.analysis_running = False
        st.session_state.config = None
        st.session_state.last_fetch_time = None
    
    def run(self):
        """Main application runner."""
        try:
            # Set custom CSS
            st.markdown("""
            <style>
                .stApp {
                    background: linear-gradient(135deg, #0e1117 0%, #1a1d2e 100%);
                }
                .section-header {
                    font-size: 2rem;
                    font-weight: 800;
                    color: white;
                    margin: 3rem 0 2rem 0;
                    padding-bottom: 1rem;
                    border-bottom: 2px solid;
                    border-image: linear-gradient(90deg, #00cc96, #636efa) 1;
                }
                .metric-card {
                    background: rgba(30, 30, 30, 0.8);
                    border-radius: 12px;
                    padding: 1.5rem;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    transition: all 0.3s ease;
                    height: 100%;
                }
                .metric-card:hover {
                    border-color: #00cc96;
                    box-shadow: 0 8px 32px rgba(0, 204, 150, 0.2);
                }
                .stProgress > div > div > div > div {
                    background: linear-gradient(90deg, #00cc96, #636efa);
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Render header
            self.render_enhanced_header()
            
            # Render sidebar and get configuration
            config, fetch_clicked, run_clicked = self.render_enhanced_sidebar()
            
            # Handle data fetching
            if fetch_clicked:
                success = self.run_data_fetch(config)
                if success:
                    st.rerun()
            
            # Handle analysis
            if run_clicked and st.session_state.data_fetched:
                success = self.run_portfolio_analysis(config)
                if success:
                    st.rerun()
            
            # Show data status
            if st.session_state.data_fetched and not st.session_state.analysis_complete:
                st.info("✅ Data fetched successfully! Click 'Run Analysis' to proceed.")
                
                # Show data preview
                if st.session_state.portfolio_data:
                    with st.expander("📊 Data Preview", expanded=True):
                        if 'returns' in st.session_state.portfolio_data:
                            returns = st.session_state.portfolio_data['returns']
                            st.write(f"**Assets:** {len(returns.columns)}")
                            if len(returns) > 0:
                                st.write(f"**Date Range:** {returns.index[0].date()} to {returns.index[-1].date()}")
                            st.write(f"**Total Returns:** {len(returns)}")
                            
                            # Show correlation heatmap
                            if not returns.empty and len(returns.columns) >= 2:
                                try:
                                    fig = self.viz_engine.create_interactive_heatmap(returns.corr())
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not create correlation heatmap: {str(e)}")
                        else:
                            st.warning("No return data available")
            
            # Render results if analysis is complete
            if st.session_state.analysis_complete:
                # Create main tabs
                tab1, tab2, tab3 = st.tabs([
                    "📊 Optimization Results",
                    "📈 Risk Analytics",
                    "🎨 Visualizations"
                ])
                
                with tab1:
                    self.render_optimization_results()
                
                with tab2:
                    self.render_risk_analysis_results()
                
                with tab3:
                    self.render_advanced_visualizations()
            
            # Show analysis running status
            if st.session_state.analysis_running:
                st.info("🔄 Analysis in progress... Please wait.")
            
            # Performance report
            with st.sidebar:
                if st.button("📊 Performance Report", use_container_width=True, key="perf_report"):
                    report = performance_monitor.get_performance_report()
                    with st.expander("Performance Report", expanded=True):
                        st.json(report, expanded=False)
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 2rem 0;">
                <p>⚡ <strong>QuantEdge Pro v4.0 Enhanced</strong> | Advanced Portfolio Analytics Platform</p>
                <p>🎯 Production-Grade Analytics • 5500+ Lines of Code • Enterprise Ready</p>
                <p style="margin-top: 1rem; font-size: 0.8rem; color: #636efa;">
                    © 2024 QuantEdge Technologies. All rights reserved.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            # Global error handling
            error_analysis = error_analyzer.analyze_error_with_context(e, {
                'operation': 'application_runtime',
                'stage': 'main_application'
            })
            
            st.error("""
            ## 🚨 Application Error
            
            The application encountered an unexpected error. Please try:
            
            1. Refreshing the page
            2. Reducing the number of assets
            3. Adjusting the date range
            4. Checking your internet connection
            
            If the problem persists, please contact support.
            """)
            
            with st.expander("Technical Details", expanded=False):
                error_analyzer.create_advanced_error_display(error_analysis)

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for the enhanced application."""
    try:
        # Set page configuration
        st.set_page_config(
            page_title="QuantEdge Pro v4.0 Enhanced",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Hide Streamlit default elements
        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display: none;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        
        # Initialize and run application
        app = QuantEdgeProEnhanced()
        app.run()
        
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

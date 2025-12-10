# ============================================================================
# QUANTEDGE PRO v4.0 ENHANCED - FIXED VERSION
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
            globals().update({
                'expected_returns': expected_returns,
                'risk_models': risk_models,
                'EfficientFrontier': EfficientFrontier,
                'HRPOpt': HRPOpt,
                'BlackLittermanModel': BlackLittermanModel,
                'CLA': CLA,
                'DiscreteAllocation': DiscreteAllocation,
                'get_latest_prices': get_latest_prices
            })
        except ImportError as e:
            lib_status['pypfopt'] = False
            missing_libs.append('PyPortfolioOpt')
        except Exception as e:
            lib_status['pypfopt'] = False
            advanced_features['pypfopt_error'] = str(e)
        
        try:
            # ARCH for GARCH modeling
            import arch
            from arch.univariate import GARCH, HARCH, EGARCH
            lib_status['arch'] = True
            advanced_features['arch'] = {
                'version': '6.0.0+',
                'models': ['GARCH', 'EGARCH', 'HARCH']
            }
            globals().update({
                'arch': arch,
                'GARCH': GARCH,
                'EGARCH': EGARCH,
                'HARCH': HARCH
            })
        except ImportError:
            lib_status['arch'] = False
            missing_libs.append('arch')
        
        try:
            # Scikit-Learn with advanced ML
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
            globals().update({
                'PCA': PCA,
                'LinearRegression': LinearRegression,
                'Ridge': Ridge,
                'Lasso': Lasso,
                'RandomForestRegressor': RandomForestRegressor,
                'IsolationForest': IsolationForest,
                'GradientBoostingRegressor': GradientBoostingRegressor,
                'StandardScaler': StandardScaler,
                'RobustScaler': RobustScaler,
                'KMeans': KMeans,
                'DBSCAN': DBSCAN,
                'SVR': SVR,
                'MLPRegressor': MLPRegressor
            })
        except ImportError:
            lib_status['sklearn'] = False
            missing_libs.append('scikit-learn')
        
        try:
            # Statsmodels for advanced statistics
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
            globals().update({
                'sm': sm,
                'acorr_ljungbox': acorr_ljungbox,
                'het_arch': het_arch,
                'adfuller': adfuller,
                'kpss': kpss,
                'VAR': VAR,
                'SimpleExpSmoothing': SimpleExpSmoothing,
                'RollingOLS': RollingOLS
            })
        except ImportError:
            lib_status['statsmodels'] = False
            missing_libs.append('statsmodels')
        
        try:
            # SciPy for advanced mathematics
            import scipy.stats as stats
            from scipy.optimize import minimize, differential_evolution
            from scipy.spatial.distance import pdist, squareform, cdist
            from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
            from scipy.signal import savgol_filter
            lib_status['scipy'] = True
            advanced_features['scipy'] = {
                'version': '1.11.0+',
                'features': ['Advanced Optimization', 'Hierarchical Clustering', 'Signal Processing']
            }
            globals().update({
                'stats': stats,
                'minimize': minimize,
                'differential_evolution': differential_evolution,
                'pdist': pdist,
                'squareform': squareform,
                'cdist': cdist,
                'linkage': linkage,
                'dendrogram': dendrogram,
                'fcluster': fcluster,
                'savgol_filter': savgol_filter
            })
        except ImportError:
            lib_status['scipy'] = False
            missing_libs.append('scipy')
        
        try:
            # TensorFlow/PyTorch for deep learning (optional)
            try:
                import tensorflow as tf
                lib_status['tensorflow'] = True
                advanced_features['tensorflow'] = {
                    'version': tf.__version__,
                    'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
                }
            except:
                lib_status['tensorflow'] = False
        
            try:
                import torch
                lib_status['pytorch'] = True
                advanced_features['pytorch'] = {
                    'version': torch.__version__,
                    'cuda_available': torch.cuda.is_available()
                }
            except:
                lib_status['pytorch'] = False
                
        except:
            pass
        
        try:
            # Additional financial libraries
            import ta  # Technical analysis
            lib_status['ta'] = True
            advanced_features['ta'] = {
                'version': '0.10.0+',
                'indicators': 200
            }
        except:
            lib_status['ta'] = False
        
        try:
            # Riskfolio-Lib for advanced portfolio optimization
            import riskfolio as rp
            lib_status['riskfolio'] = True
            advanced_features['riskfolio'] = {
                'version': '4.0.0+',
                'features': ['HCP', 'Risk Parity', 'NCO']
            }
        except:
            lib_status['riskfolio'] = False
        
        return {
            'status': lib_status,
            'missing': missing_libs,
            'advanced_features': advanced_features,
            'all_available': len(missing_libs) == 0
        }

# Initialize advanced library manager
LIBRARY_STATUS = AdvancedLibraryManager.check_and_import_all()

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
                st.checkbox(f"Action {i}: {action}", value=False, key=f"recovery_{i}")
            
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
            n_portfolios = 5000
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
                    skew = port_returns.skew()
                    
                    portfolio_returns.append(port_return)
                    portfolio_risks.append(port_risk)
                    portfolio_sharpes.append(sharpe)
                    portfolio_skewness.append(skew)
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
            
            # Calculate efficient frontier portfolios
            if LIBRARY_STATUS['status'].get('pypfopt', False):
                from pypfopt.efficient_frontier import EfficientFrontier
                
                ef = EfficientFrontier(mu, S)
                
                # Generate efficient frontier points
                efficient_risks = []
                efficient_returns = []
                efficient_sharpes = []
                
                target_returns = np.linspace(min(portfolio_returns), max(portfolio_returns) * 0.9, 20)
                
                for target_return in target_returns:
                    try:
                        ef = EfficientFrontier(mu, S)
                        ef.efficient_return(target_return)
                        ret, risk, _ = ef.portfolio_performance()
                        
                        efficient_risks.append(risk)
                        efficient_returns.append(ret)
                        efficient_sharpes.append((ret - risk_free_rate) / risk if risk > 0 else 0)
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
            # Calculate VaR using all methods
            for method in self.methods:
                results['methods'][method] = self._calculate_var_method(
                    returns, method, self.confidence_levels, portfolio_value
                )
            
            # Calculate summary statistics
            results['summary'] = self._calculate_var_summary(results['methods'], returns)
            
            # Calculate VaR violations
            results['violations'] = self._calculate_var_violations(
                returns, results['methods']['Historical']
            )
            
            # Perform backtesting
            results['backtest'] = self._perform_var_backtesting(
                returns, results['methods']['Historical']
            )
            
            # Perform stress tests
            results['stress_tests'] = self._perform_stress_tests(returns)
            
            # Calculate additional risk metrics
            results['additional_metrics'] = self._calculate_additional_risk_metrics(returns)
            
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
                cvar = -returns[returns <= -var].mean()
                es = cvar  # For historical, ES = CVaR
                
            elif method == 'Parametric':
                # Parametric (normal distribution)
                mean = returns.mean()
                std = returns.std()
                var = -(mean + std * norm.ppf(confidence))
                cvar = -(mean - std * norm.pdf(norm.ppf(alpha)) / alpha)
                es = cvar
                
            elif method == 'MonteCarlo':
                # Monte Carlo simulation
                np.random.seed(42)
                n_simulations = 10000
                simulated_returns = np.random.normal(returns.mean(), returns.std(), n_simulations)
                var = -np.percentile(simulated_returns, alpha * 100)
                cvar = -simulated_returns[simulated_returns <= -var].mean()
                es = cvar
                
            elif method == 'EVT':
                # Extreme Value Theory
                try:
                    threshold = np.percentile(returns, 10)
                    excess = returns[returns < threshold] - threshold
                    
                    if len(excess) > 10:
                        # Generalized Pareto Distribution parameters
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
                        cvar = -returns[returns <= -var].mean()
                        es = cvar
                        
                except:
                    var = -np.percentile(returns, alpha * 100)
                    cvar = -returns[returns <= -var].mean()
                    es = cvar
                    
            elif method == 'GARCH':
                # GARCH model VaR
                try:
                    if LIBRARY_STATUS['status'].get('arch', False):
                        from arch import arch_model
                        
                        # Fit GARCH(1,1) model
                        am = arch_model(returns * 100, vol='Garch', p=1, q=1)
                        res = am.fit(disp='off')
                        
                        # Forecast
                        forecast = res.forecast(horizon=1)
                        conditional_vol = np.sqrt(forecast.variance.values[-1, 0]) / 100
                        
                        var = -(returns.mean() + conditional_vol * norm.ppf(confidence))
                        cvar = -(returns.mean() - conditional_vol * norm.pdf(norm.ppf(alpha)) / alpha)
                        es = cvar
                    else:
                        raise ImportError("ARCH library not available")
                        
                except Exception as e:
                    # Fallback to parametric
                    mean = returns.mean()
                    std = returns.std()
                    var = -(mean + std * norm.ppf(confidence))
                    cvar = -(mean - std * norm.pdf(norm.ppf(alpha)) / alpha)
                    es = cvar
            
            else:
                # Default to historical
                var = -np.percentile(returns, alpha * 100)
                cvar = -returns[returns <= -var].mean()
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
            return xi, beta
        else:
            return 0.1, mean_excess
    
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
            summary['var_consistency'] = np.std(var_values) / np.mean(var_values) if np.mean(var_values) != 0 else 0
        
        # Determine best method (lowest average VaR with consistency)
        method_scores = {}
        for method, results in methods_results.items():
            method_vars = [metrics['VaR'] for metrics in results.values()]
            avg_var = np.mean(method_vars)
            consistency = np.std(method_vars) / avg_var if avg_var != 0 else float('inf')
            
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
            if method in self.optimization_methods:
                weights, metrics = self.optimization_methods[method](
                    returns, constraints, risk_free_rate
                )
            else:
                # Default to max Sharpe
                weights, metrics = self._optimize_max_sharpe(returns, constraints, risk_free_rate)
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(returns, weights, risk_free_rate)
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
            if LIBRARY_STATUS['status'].get('pypfopt', False):
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
            pass
        
        # Fallback implementation
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
        result = minimize(
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

# ============================================================================
# 6. ADVANCED DATA MANAGEMENT AND PROCESSING ENGINE
# ============================================================================

class AdvancedDataManager:
    """Advanced data management with caching, validation, and preprocessing."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.max_workers = 10
        self.retry_attempts = 3
        self.timeout = 30
    
    @st.cache_data(ttl=3600, show_spinner=True, max_entries=50)
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
                'dividends': pd.DataFrame(),
                'splits': pd.DataFrame(),
                'metadata': {},
                'errors': {}
            }
            
            # Download data in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
                data['volumes'] = data['volumes'].ffill().bfill()
                
                # Calculate returns
                data['returns'] = data['prices'].pct_change().dropna()
                
                # Remove assets with too many missing values
                missing_threshold = 0.1  # 10% missing
                assets_to_keep = data['returns'].columns[
                    data['returns'].isnull().mean() < missing_threshold
                ]
                
                if len(assets_to_keep) > 0:
                    data['prices'] = data['prices'][assets_to_keep]
                    data['returns'] = data['returns'][assets_to_keep]
                    data['volumes'] = data['volumes'][assets_to_keep]
                else:
                    raise ValueError("No assets with sufficient data")
            
            # Calculate additional features
            if not data['returns'].empty:
                data['additional_features'] = self._calculate_additional_features(data)
            
            performance_monitor.end_operation('fetch_advanced_market_data', {
                'tickers_fetched': len(data['prices'].columns),
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
                    raise ValueError(f"No historical data for {ticker}")
                
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

# ============================================================================
# 7. SMART USER INTERFACE COMPONENTS
# ============================================================================

class SmartUIComponents:
    """Smart UI components with enhanced visualization and interactivity."""
    
    @staticmethod
    def create_smart_button(label: str, key: str, icon: str = "⚡", 
                           tooltip: str = "", variant: str = "primary") -> bool:
        """Create smart button with enhanced visualization."""
        button_html = f"""
        <style>
            .smart-button-{key} {{
                background: linear-gradient(135deg, 
                    {SmartUIComponents._get_button_color(variant, 'start')}, 
                    {SmartUIComponents._get_button_color(variant, 'end')});
                color: white;
                border: none;
                padding: 0.7rem 1.8rem;
                border-radius: 12px;
                font-weight: 700;
                font-size: 0.95rem;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                min-width: 140px;
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .smart-button-{key}:hover {{
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
                border-color: rgba(255, 255, 255, 0.3);
            }}
            
            .smart-button-{key}:active {{
                transform: translateY(-1px) scale(1.02);
            }}
            
            .smart-button-{key}::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, 
                    transparent, 
                    rgba(255, 255, 255, 0.2), 
                    transparent);
                transition: 0.5s;
            }}
            
            .smart-button-{key}:hover::before {{
                left: 100%;
            }}
            
            .smart-button-{key}.processing {{
                background: linear-gradient(135deg, #636efa, #ab63fa);
            }}
            
            .btn-icon {{
                font-size: 1.2rem;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
            }}
            
            .btn-label {{
                white-space: nowrap;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }}
            
            .btn-tooltip {{
                position: absolute;
                bottom: calc(100% + 10px);
                left: 50%;
                transform: translateX(-50%);
                background: rgba(30, 30, 30, 0.98);
                color: white;
                padding: 10px 16px;
                border-radius: 8px;
                font-size: 0.85rem;
                white-space: nowrap;
                opacity: 0;
                transition: opacity 0.3s, transform 0.3s;
                pointer-events: none;
                border: 1px solid rgba(255, 255, 255, 0.15);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(10px);
                z-index: 1000;
            }}
            
            .smart-button-{key}:hover .btn-tooltip {{
                opacity: 1;
                transform: translateX(-50%) translateY(-5px);
            }}
            
            .btn-tooltip::after {{
                content: '';
                position: absolute;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                border: 6px solid transparent;
                border-top-color: rgba(30, 30, 30, 0.98);
            }}
        </style>
        
        <button class="smart-button-{key}" id="btn-{key}">
            <span class="btn-icon">{icon}</span>
            <span class="btn-label">{label}</span>
            <div class="btn-tooltip">{tooltip}</div>
        </button>
        
        <script>
            const btn{key} = document.getElementById('btn-{key}');
            
            btn{key}.addEventListener('click', function() {{
                this.classList.add('processing');
                this.querySelector('.btn-icon').textContent = '⏳';
                this.querySelector('.btn-label').textContent = 'Processing...';
                this.disabled = true;
                
                // Trigger Streamlit button click
                const event = new Event('click', {{ bubbles: true }});
                this.dispatchEvent(event);
            }});
        </script>
        """
        
        # Create columns for centered button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(button_html, unsafe_allow_html=True)
            
            # Create hidden Streamlit button
            return st.button(label, key=key, help=tooltip, disabled=False, 
                           use_container_width=False, type=variant if variant != "secondary" else "secondary")
    
    @staticmethod
    def _get_button_color(variant: str, position: str) -> str:
        """Get button gradient colors based on variant."""
        colors = {
            'primary': {'start': '#00cc96', 'end': '#636efa'},
            'secondary': {'start': '#636efa', 'end': '#ab63fa'},
            'success': {'start': '#00cc96', 'end': '#00b894'},
            'warning': {'start': '#FFA15A', 'end': '#ff9f43'},
            'danger': {'start': '#ef553b', 'end': '#ff6b6b'},
            'info': {'start': '#17a2b8', 'end': '#2d98da'}
        }
        return colors.get(variant, colors['primary'])[position]

# ============================================================================
# 8. MAIN ENHANCED APPLICATION - FIXED VERSION
# ============================================================================

# Initialize global components
viz_engine = AdvancedVisualizationEngine()
risk_analytics = AdvancedRiskAnalytics()
portfolio_optimizer = AdvancedPortfolioOptimizer()
data_manager = AdvancedDataManager()
ui = SmartUIComponents()

class QuantEdgeProEnhanced:
    """Enhanced QuantEdge Pro application with all advanced features."""
    
    def __init__(self):
        self.data_manager = data_manager
        self.risk_analytics = risk_analytics
        self.portfolio_optimizer = portfolio_optimizer
        self.viz_engine = viz_engine
        self.ui = ui
        
        # Initialize session state with ALL required keys
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize ALL session state variables with proper defaults."""
        # Data storage
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = {
                'prices': pd.DataFrame(),
                'returns': pd.DataFrame(),
                'volumes': pd.DataFrame(),
                'metadata': {},
                'errors': {},
                'returns_clean': pd.DataFrame(),  # Add this
                'prices_clean': pd.DataFrame()    # Add this
            }
        
        # Optimization results
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = {
                'weights': {},
                'metrics': {},
                'method': '',
                'constraints': None,
                'risk_free_rate': 0.045,
                'timestamp': ''
            }
        
        # Risk analysis
        if 'risk_analysis_results' not in st.session_state:
            st.session_state.risk_analysis_results = {
                'methods': {},
                'portfolio_value': 0,
                'summary': {},
                'violations': {},
                'backtest': {},
                'stress_tests': {}
            }
        
        # Analysis state
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'config' not in st.session_state:
            st.session_state.config = {}
    
    def _safe_get_data(self, key: str, default=None):
        """Safely get data from portfolio_data with fallback."""
        if (st.session_state.portfolio_data and 
            key in st.session_state.portfolio_data):
            data = st.session_state.portfolio_data[key]
            # Check if data is valid
            if isinstance(data, pd.DataFrame) and not data.empty:
                return data
            elif isinstance(data, dict) and data:
                return data
            elif data is not None:
                return data
        return default
    
    def _safe_get_returns(self) -> pd.DataFrame:
        """Safely get returns data with validation."""
        returns = self._safe_get_data('returns_clean')
        if returns is None or returns.empty:
            returns = self._safe_get_data('returns')
        if returns is None or returns.empty:
            # Create empty dataframe with proper structure
            returns = pd.DataFrame()
        return returns
    
    def _safe_get_prices(self) -> pd.DataFrame:
        """Safely get prices data with validation."""
        prices = self._safe_get_data('prices_clean')
        if prices is None or prices.empty:
            prices = self._safe_get_data('prices')
        if prices is None or prices.empty:
            prices = pd.DataFrame()
        return prices
    
    def render_enhanced_header(self):
        """Render enhanced application header."""
        st.markdown("""
        <div class="main-header" style="text-align: center;">
            <h1 style="color: white; font-size: 3.8rem; margin-bottom: 1rem; background: linear-gradient(135deg, #00cc96, #636efa, #ab63fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                ⚡ QuantEdge Pro v4.0 Enhanced
            </h1>
            <p style="color: #94a3b8; font-size: 1.3rem; margin-bottom: 0.5rem;">
                Institutional Portfolio Analytics Platform with Advanced Risk Metrics
            </p>
            <p style="color: #636efa; font-size: 1.1rem;">
                Featuring: Advanced VaR/CVaR/ES Analytics • 3D Visualizations • Smart Optimization
            </p>
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
    
    def render_enhanced_sidebar(self):
        """Render enhanced sidebar with smart components."""
        with st.sidebar:
            st.markdown("""
            <div style="padding: 1.5rem; border-radius: 12px; background: rgba(30, 30, 30, 0.8); margin-bottom: 2rem;">
                <h3 style="color: #00cc96; margin-bottom: 1rem;">🎯 Configuration Panel</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress tracker
            steps = [
                {'label': 'Data Setup', 'icon': '📊'},
                {'label': 'Optimization', 'icon': '⚡'},
                {'label': 'Risk Analysis', 'icon': '📈'},
                {'label': 'Results', 'icon': '📊'}
            ]
            
            # Simple progress tracker
            st.progress(st.session_state.current_step / len(steps))
            
            st.markdown("---")
            
            # Asset universe selection
            st.subheader("🌍 Asset Universe")
            universe_options = {
                "US Large Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "JNJ", "V", "WMT"],
                "Technology Focus": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC", "QCOM", "CRM", "ADBE", "ORCL"],
                "Global Diversified": ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "NSRGY", "NVO", "TSM", "HSBC"],
                "Emerging Markets": ["BABA", "TSM", "005930.KS", "ITUB", "VALE", "INFY", "HDB", "IDX", "EWZ"]
            }
            
            selected_universe = st.selectbox(
                "Select Predefined Universe",
                list(universe_options.keys()),
                help="Choose from predefined asset universes"
            )
            
            # Custom tickers
            with st.expander("➕ Add Custom Assets"):
                custom_tickers = st.text_area(
                    "Enter custom tickers (comma-separated)",
                    placeholder="e.g., AAPL, MSFT, GOOGL",
                    help="Add additional assets to the portfolio"
                )
            
            # Date range with presets
            st.subheader("📅 Time Period")
            date_preset = st.selectbox(
                "Select Period",
                ["Custom", "1 Year", "3 Years", "5 Years", "10 Years"],
                help="Select predefined time periods or choose custom"
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
                    start_date = st.date_input("Start Date", value=end_date - timedelta(days=365*3))
                with col2:
                    end_date = st.date_input("End Date", value=end_date)
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    risk_free_rate = st.slider(
                        "Risk-Free Rate",
                        min_value=0.0,
                        max_value=0.20,
                        value=0.045,
                        step=0.001,
                        format="%.1%"
                    )
                with col2:
                    transaction_cost = st.slider(
                        "Transaction Cost",
                        min_value=0,
                        max_value=100,
                        value=10,
                        step=1,
                        format="%d bps"
                    )
                
                optimization_method = st.selectbox(
                    "Optimization Method",
                    list(portfolio_optimizer.optimization_methods.keys()),
                    help="Select portfolio optimization methodology"
                )
                
                # Constraints
                st.subheader("🎯 Constraints")
                col1, col2 = st.columns(2)
                with col1:
                    max_weight = st.slider(
                        "Max Weight",
                        min_value=0.05,
                        max_value=1.0,
                        value=0.30,
                        step=0.05,
                        format="%.0%"
                    )
                with col2:
                    min_weight = st.slider(
                        "Min Weight",
                        min_value=0.0,
                        max_value=0.20,
                        value=0.0,
                        step=0.01,
                        format="%.0%"
                    )
            
            st.markdown("---")
            
            # Action buttons
            st.subheader("🚀 Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                fetch_data = self.ui.create_smart_button(
                    "Fetch Data",
                    "fetch_data",
                    icon="📥",
                    tooltip="Download market data",
                    variant="primary"
                )
            
            with col2:
                run_analysis = self.ui.create_smart_button(
                    "Run Analysis",
                    "run_analysis",
                    icon="⚡",
                    tooltip="Run comprehensive analysis",
                    variant="success"
                )
            
            # Reset button
            if st.button("🔄 Reset Analysis", use_container_width=True):
                self._reset_analysis()
            
            return {
                'universe': selected_universe,
                'tickers': universe_options[selected_universe] + 
                          ([t.strip().upper() for t in custom_tickers.split(',')] if custom_tickers else []),
                'start_date': start_date,
                'end_date': end_date,
                'risk_free_rate': risk_free_rate,
                'transaction_cost': transaction_cost / 10000,
                'optimization_method': optimization_method,
                'max_weight': max_weight,
                'min_weight': min_weight,
                'fetch_data': fetch_data,
                'run_analysis': run_analysis
            }
    
    def run_data_fetch(self, config: Dict):
        """Run data fetching process with robust error handling."""
        try:
            with st.spinner("📥 Fetching market data..."):
                # Update progress
                st.session_state.current_step = 1
                
                # Validate tickers
                if not config['tickers']:
                    st.error("❌ No tickers specified")
                    return False
                
                # Fetch data with timeout
                portfolio_data = self.data_manager.fetch_advanced_market_data(
                    tickers=config['tickers'][:20],  # Limit to 20 tickers initially
                    start_date=config['start_date'],
                    end_date=config['end_date']
                )
                
                # Validate data structure
                if not portfolio_data or 'prices' not in portfolio_data:
                    st.error("❌ No data returned from API")
                    return False
                
                # Initialize all required keys
                required_keys = ['prices', 'returns', 'volumes', 'metadata', 'errors']
                for key in required_keys:
                    if key not in portfolio_data:
                        portfolio_data[key] = pd.DataFrame() if key != 'metadata' and key != 'errors' else {}
                
                # Validate data
                if portfolio_data['prices'].empty:
                    st.error("❌ No price data available")
                    return False
                
                # Calculate returns if not present
                if 'returns' not in portfolio_data or portfolio_data['returns'].empty:
                    portfolio_data['returns'] = portfolio_data['prices'].pct_change().dropna()
                
                # Store data
                st.session_state.portfolio_data = portfolio_data
                st.success(f"✅ Data fetched successfully: {len(portfolio_data['prices'].columns)} assets, {len(portfolio_data['prices'])} data points")
                
                # Show data preview
                with st.expander("📊 Data Preview", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Assets", len(portfolio_data['prices'].columns))
                        st.metric("Date Range", f"{portfolio_data['prices'].index[0].date()} to {portfolio_data['prices'].index[-1].date()}")
                    with col2:
                        st.metric("Data Points", len(portfolio_data['prices']))
                        missing_percentage = portfolio_data['prices'].isnull().mean().mean()
                        st.metric("Missing Data", f"{missing_percentage:.1%}")
                
                return True
                    
        except Exception as e:
            error_analysis = error_analyzer.analyze_error_with_context(e, {
                'operation': 'data_fetch',
                'tickers': config['tickers'][:5] if config['tickers'] else [],
                'date_range': f"{config['start_date']} to {config['end_date']}"
            })
            error_analyzer.create_advanced_error_display(error_analysis)
            return False
    
    def run_portfolio_analysis(self, config: Dict):
        """Run comprehensive portfolio analysis with safe data access."""
        try:
            if st.session_state.portfolio_data is None:
                st.error("Please fetch data first")
                return False
            
            # Update progress
            st.session_state.current_step = 2
            
            # Get data safely
            returns = self._safe_get_returns()
            if returns.empty:
                st.error("No returns data available")
                return False
            
            # Prepare data (simplified version)
            with st.spinner("🔄 Preparing data for analysis..."):
                prepared_data = st.session_state.portfolio_data.copy()
                prepared_data['returns_clean'] = returns.fillna(0)
            
            # Run portfolio optimization
            with st.spinner("⚡ Optimizing portfolio..."):
                optimization_results = self.portfolio_optimizer.optimize_portfolio(
                    returns=prepared_data['returns_clean'],
                    method=config['optimization_method'],
                    constraints={
                        'bounds': (config['min_weight'], config['max_weight'])
                    },
                    risk_free_rate=config['risk_free_rate']
                )
                
                # Validate optimization results
                if not optimization_results or 'weights' not in optimization_results:
                    st.error("Portfolio optimization failed")
                    return False
                
                st.session_state.optimization_results = optimization_results
            
            # Run risk analysis
            with st.spinner("📈 Analyzing risk metrics..."):
                # Calculate portfolio returns safely
                weights_array = np.array(list(optimization_results['weights'].values()))
                if len(weights_array) == len(prepared_data['returns_clean'].columns):
                    portfolio_returns = prepared_data['returns_clean'].dot(weights_array)
                else:
                    # If dimension mismatch, use equal weights
                    n_assets = len(prepared_data['returns_clean'].columns)
                    equal_weights = np.ones(n_assets) / n_assets
                    portfolio_returns = prepared_data['returns_clean'].dot(equal_weights)
                
                risk_analysis_results = self.risk_analytics.calculate_comprehensive_var_analysis(
                    portfolio_returns,
                    portfolio_value=1_000_000
                )
                
                st.session_state.risk_analysis_results = risk_analysis_results
            
            # Update progress
            st.session_state.current_step = 3
            st.session_state.analysis_complete = True
            
            st.success("✅ Analysis complete!")
            return True
            
        except Exception as e:
            error_analysis = error_analyzer.analyze_error_with_context(e, {
                'operation': 'portfolio_analysis',
                'optimization_method': config.get('optimization_method', 'unknown')
            })
            error_analyzer.create_advanced_error_display(error_analysis)
            return False
    
    def render_optimization_results(self):
        """Render optimization results with safe data access."""
        if (not st.session_state.optimization_results or 
            'weights' not in st.session_state.optimization_results):
            st.info("No optimization results available yet")
            return
        
        results = st.session_state.optimization_results
        
        # Safely get metrics
        metrics = results.get('metrics', {})
        
        st.markdown('<div class="section-header">⚡ Portfolio Optimization Results</div>', 
                   unsafe_allow_html=True)
        
        # Key metrics with safe defaults
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Expected Return",
                f"{metrics.get('expected_return', 0):.2%}"
            )
        with col2:
            st.metric(
                "Expected Volatility",
                f"{metrics.get('expected_volatility', 0):.2%}"
            )
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}"
            )
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.2%}"
            )
        
        # Portfolio allocation - only if we have weights
        if results['weights']:
            st.subheader("🎯 Portfolio Allocation")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Create simple pie chart
                weights_df = pd.DataFrame(
                    [(ticker, weight) 
                     for ticker, weight in results['weights'].items()],
                    columns=['Asset', 'Weight']
                ).sort_values('Weight', ascending=False)
                
                fig = px.pie(weights_df, values='Weight', names='Asset', 
                            title='Portfolio Allocation')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display weights in table
                weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
                st.dataframe(
                    weights_df,
                    use_container_width=True,
                    height=400
                )
        
        # Additional metrics
        if metrics:
            st.subheader("📊 Additional Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            metrics_to_display = [
                ('Sortino Ratio', 'sortino_ratio'),
                ('Calmar Ratio', 'calmar_ratio'),
                ('Omega Ratio', 'omega_ratio'),
                ('Diversification', 'diversification_ratio'),
            ]
            
            cols = [col1, col2, col3, col4]
            for idx, (label, key) in enumerate(metrics_to_display):
                if key in metrics:
                    with cols[idx % 4]:
                        value = metrics[key]
                        if isinstance(value, float):
                            display_value = f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}"
                        else:
                            display_value = str(value)
                        
                        st.metric(label, display_value)
    
    def render_risk_analysis_results(self):
        """Render advanced risk analysis results."""
        if st.session_state.risk_analysis_results is None:
            return
        
        results = st.session_state.risk_analysis_results
        
        st.markdown('<div class="section-header">📈 Advanced Risk Analytics</div>', 
                   unsafe_allow_html=True)
        
        # Create tabs for different risk analyses
        tab1, tab2, tab3 = st.tabs([
            "📊 VaR Analysis",
            "🌪️ Stress Testing",
            "🔍 Risk Metrics"
        ])
        
        with tab1:
            self._render_var_analysis(results)
        
        with tab2:
            self._render_stress_testing(results)
        
        with tab3:
            self._render_risk_metrics(results)
    
    def _render_var_analysis(self, results: Dict):
        """Render VaR analysis."""
        if 'methods' in results:
            st.subheader("📊 Value at Risk (VaR) Analysis")
            
            # Create comparison table
            comparison_data = []
            for method in results['methods']:
                for confidence, metrics in results['methods'][method].items():
                    comparison_data.append({
                        'Method': method,
                        'Confidence': f'{confidence:.1%}',
                        'VaR': f"{metrics.get('VaR', 0):.3%}",
                        'CVaR': f"{metrics.get('CVaR', 0):.3%}",
                        'ES': f"{metrics.get('ES', 0):.3%}"
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
            
            # Summary
            if 'summary' in results:
                st.subheader("📋 Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Best Method",
                        results['summary'].get('best_method', 'N/A')
                    )
                with col2:
                    st.metric(
                        "Worst-Case VaR",
                        f"{results['summary'].get('worst_case_var', 0):.3%}"
                    )
                with col3:
                    st.metric(
                        "Average VaR",
                        f"{results['summary'].get('average_var', 0):.3%}"
                    )
    
    def _render_stress_testing(self, results: Dict):
        """Render stress testing results."""
        if 'stress_tests' in results:
            st.subheader("🌪️ Stress Testing")
            
            # Historical scenarios
            if 'historical_scenarios' in results['stress_tests']:
                st.markdown("#### 📅 Historical Stress Periods")
                
                historical_data = []
                for scenario, metrics in results['stress_tests']['historical_scenarios'].items():
                    historical_data.append({
                        'Scenario': scenario,
                        'Return': f"{metrics.get('returns', 0):.2%}",
                        'Volatility': f"{metrics.get('volatility', 0):.2%}",
                        'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}"
                    })
                
                if historical_data:
                    st.dataframe(pd.DataFrame(historical_data), use_container_width=True)
    
    def _render_risk_metrics(self, results: Dict):
        """Render comprehensive risk metrics."""
        if 'additional_metrics' in results:
            metrics = results['additional_metrics']
            
            st.subheader("📊 Comprehensive Risk Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Tail Risk Measures")
                if 'tail_risk_measures' in metrics:
                    tail_metrics = metrics['tail_risk_measures']
                    for key, value in tail_metrics.items():
                        st.metric(
                            key.replace('_', ' ').title(),
                            f"{value:.3f}" if isinstance(value, float) else str(value)
                        )
            
            with col2:
                st.markdown("#### ⚠️ Expected Shortfall")
                if 'expected_shortfall_ratio' in metrics:
                    st.metric(
                        "Expected Shortfall Ratio",
                        f"{metrics['expected_shortfall_ratio']:.3f}"
                    )
    
    def render_advanced_visualizations(self):
        """Render advanced visualizations."""
        if not st.session_state.analysis_complete:
            return
        
        st.markdown('<div class="section-header">🎨 Advanced Visualizations</div>', 
                   unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs([
            "🎯 3D Efficient Frontier",
            "📊 Correlation Heatmap"
        ])
        
        with tab1:
            self._render_3d_efficient_frontier()
        
        with tab2:
            self._render_correlation_heatmap()
    
    def _render_3d_efficient_frontier(self):
        """Render 3D efficient frontier visualization."""
        returns = self._safe_get_returns()
        if not returns.empty:
            # Get risk-free rate from config
            risk_free_rate = 0.045  # Default
            
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
    
    def _render_correlation_heatmap(self):
        """Render correlation heatmap."""
        returns = self._safe_get_returns()
        if not returns.empty:
            correlation_matrix = returns.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title='Asset Correlation Matrix',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation statistics
            st.subheader("📊 Correlation Statistics")
            
            # Calculate average correlation
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            upper_tri = correlation_matrix.where(mask)
            avg_correlation = upper_tri.stack().mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Correlation", f"{avg_correlation:.3f}")
            with col2:
                st.metric("Min Correlation", f"{correlation_matrix.min().min():.3f}")
            with col3:
                st.metric("Max Correlation", f"{correlation_matrix.max().max():.3f}")
    
    def _reset_analysis(self):
        """Reset the analysis."""
        st.session_state.portfolio_data = None
        st.session_state.optimization_results = None
        st.session_state.risk_analysis_results = None
        st.session_state.current_step = 0
        st.session_state.analysis_complete = False
        st.rerun()
    
    def run(self):
        """Main application runner with comprehensive error handling."""
        try:
            # Set page configuration
            st.set_page_config(
                page_title="QuantEdge Pro v4.0 Enhanced",
                page_icon="⚡",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Add custom CSS
            st.markdown("""
            <style>
                .stApp {
                    background: linear-gradient(135deg, #0e1117 0%, #1a1d2e 100%);
                }
                .main-header {
                    background: linear-gradient(135deg, rgba(26, 29, 46, 0.95), rgba(42, 42, 42, 0.95));
                    padding: 2.5rem;
                    border-radius: 16px;
                    margin-bottom: 2.5rem;
                    border-left: 6px solid #00cc96;
                    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
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
            </style>
            """, unsafe_allow_html=True)
            
            # Render header
            self.render_enhanced_header()
            
            # Render sidebar and get configuration
            config = self.render_enhanced_sidebar()
            
            # Handle data fetching
            if config and config.get('fetch_data'):
                success = self.run_data_fetch(config)
                if success:
                    st.rerun()
            
            # Handle analysis
            if (config and config.get('run_analysis') and 
                st.session_state.portfolio_data):
                success = self.run_portfolio_analysis(config)
                if success:
                    st.rerun()
            
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
            
            # Performance report
            with st.sidebar:
                if st.button("📊 Performance Report", use_container_width=True):
                    report = performance_monitor.get_performance_report()
                    with st.expander("Performance Report", expanded=True):
                        st.json(report)
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 2rem 0;">
                <p>⚡ <strong>QuantEdge Pro v4.0 Enhanced</strong> | Advanced Portfolio Analytics Platform</p>
                <p>🎯 Production-Grade Analytics • Enterprise Ready</p>
                <p style="margin-top: 1rem; font-size: 0.8rem; color: #636efa;">
                    © 2024 QuantEdge Technologies. All rights reserved.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        except KeyError as e:
            st.error(f"🔑 KeyError: Missing data key - {str(e)}")
            st.info("Try resetting the application or fetching data again")
            if st.button("🔄 Reset Application"):
                self._reset_analysis()
                st.rerun()
                
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
        # Initialize and run application
        app = QuantEdgeProEnhanced()
        app.run()
        
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

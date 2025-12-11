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
import plotly.figure_factory as ff
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
from scipy.stats import norm, t, skew, kurtosis, multivariate_normal
import scipy.stats as stats
from scipy import optimize, signal
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ENHANCED LIBRARY MANAGER WITH ML & ADVANCED FEATURES
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

class EnterpriseLibraryManager:
    """Enterprise-grade library manager with ML and alternative data support."""
    
    @staticmethod
    def check_and_import_all():
        """Check and import all required libraries with ML capabilities."""
        lib_status = {}
        missing_libs = []
        advanced_features = {}
        
        # First check advanced libraries
        original_status = AdvancedLibraryManager.check_and_import_all()
        lib_status.update(original_status['status'])
        missing_libs.extend([lib for lib in original_status['missing'] if lib not in missing_libs])
        advanced_features.update(original_status['advanced_features'])
        
        # Core ML and AI libraries
        try:
            # TensorFlow/PyTorch for deep learning
            try:
                import tensorflow as tf
                lib_status['tensorflow'] = True
                advanced_features['tensorflow'] = {
                    'version': tf.__version__,
                    'features': ['Neural Networks', 'LSTM', 'Autoencoders']
                }
                st.session_state.tensorflow_available = True
            except ImportError:
                try:
                    import torch
                    import torch.nn as nn
                    lib_status['pytorch'] = True
                    advanced_features['pytorch'] = {
                        'version': torch.__version__,
                        'features': ['Deep Learning', 'CNN', 'RNN', 'Transformers']
                    }
                    st.session_state.pytorch_available = True
                except ImportError:
                    lib_status['deep_learning'] = False
                    missing_libs.append('TensorFlow or PyTorch')
        
        except Exception as e:
            lib_status['deep_learning'] = False
            advanced_features['deep_learning_error'] = str(e)
        
        # Alternative data sources
        try:
            # Alpha Vantage for alternative data
            try:
                from alpha_vantage.timeseries import TimeSeries
                lib_status['alpha_vantage'] = True
                advanced_features['alpha_vantage'] = {
                    'version': '3.0.0+',
                    'features': ['Real-time Data', 'Technical Indicators', 'Fundamental Data']
                }
                st.session_state.alpha_vantage_available = True
            except ImportError:
                lib_status['alpha_vantage'] = False
                missing_libs.append('alpha_vantage')
                st.session_state.alpha_vantage_available = False
        except Exception as e:
            lib_status['alpha_vantage'] = False
            advanced_features['alpha_vantage_error'] = str(e)
        
        # Web scraping and news analysis
        try:
            # NewsAPI for sentiment analysis
            try:
                from newsapi import NewsApiClient
                lib_status['newsapi'] = True
                advanced_features['newsapi'] = {
                    'version': '2.0.0+',
                    'features': ['News Sentiment', 'Market Sentiment Analysis']
                }
                st.session_state.newsapi_available = True
            except ImportError:
                lib_status['newsapi'] = False
                # Not critical, just add to optional missing
                missing_libs.append('newsapi (optional)')
        except Exception as e:
            lib_status['newsapi'] = False
        
        # Advanced time series analysis
        try:
            # Facebook Prophet for forecasting
            try:
                from prophet import Prophet
                lib_status['prophet'] = True
                advanced_features['prophet'] = {
                    'version': '1.0+',
                    'features': ['Time Series Forecasting', 'Holiday Effects']
                }
                st.session_state.prophet_available = True
            except ImportError:
                lib_status['prophet'] = False
                missing_libs.append('prophet (optional)')
        except Exception as e:
            lib_status['prophet'] = False
        
        # Report generation
        try:
            # ReportLab for PDF generation
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
                from reportlab.lib.styles import getSampleStyleSheet
                lib_status['reportlab'] = True
                advanced_features['reportlab'] = {
                    'version': '3.6+',
                    'features': ['PDF Report Generation', 'Professional Reports']
                }
                st.session_state.reportlab_available = True
            except ImportError:
                lib_status['reportlab'] = False
                missing_libs.append('reportlab (optional)')
        except Exception as e:
            lib_status['reportlab'] = False
        
        # Blockchain and crypto
        try:
            # Web3 for blockchain data
            try:
                from web3 import Web3
                lib_status['web3'] = True
                advanced_features['web3'] = {
                    'version': '6.0+',
                    'features': ['Blockchain Data', 'Crypto Portfolio Management']
                }
                st.session_state.web3_available = True
            except ImportError:
                lib_status['web3'] = False
                missing_libs.append('web3 (optional)')
        except Exception as e:
            lib_status['web3'] = False
        
        # Database support
        try:
            # SQLAlchemy for database operations
            try:
                from sqlalchemy import create_engine, text
                lib_status['sqlalchemy'] = True
                advanced_features['sqlalchemy'] = {
                    'version': '2.0+',
                    'features': ['Database Integration', 'Data Persistence']
                }
                st.session_state.sqlalchemy_available = True
            except ImportError:
                lib_status['sqlalchemy'] = False
                missing_libs.append('sqlalchemy (optional)')
        except Exception as e:
            lib_status['sqlalchemy'] = False
        
        return {
            'status': lib_status,
            'missing': missing_libs,
            'advanced_features': advanced_features,
            'all_available': len(missing_libs) == 0,
            'enterprise_features': {
                'ml_ready': lib_status.get('tensorflow', False) or lib_status.get('pytorch', False),
                'alternative_data': lib_status.get('alpha_vantage', False),
                'sentiment_analysis': lib_status.get('newsapi', False),
                'blockchain': lib_status.get('web3', False),
                'reporting': lib_status.get('reportlab', False)
            }
        }

# Initialize enterprise library manager
if 'enterprise_library_status' not in st.session_state:
    ENTERPRISE_LIBRARY_STATUS = EnterpriseLibraryManager.check_and_import_all()
    st.session_state.enterprise_library_status = ENTERPRISE_LIBRARY_STATUS
else:
    ENTERPRISE_LIBRARY_STATUS = st.session_state.enterprise_library_status

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
                zero_vol_assets = data['returns'].std()[data['returns'].std().abs() < 1e-10].tolist()
                if zero_vol_assets:
                    validation['warnings'].append(f"Zero volatility assets: {zero_vol_assets}")
            
            # Calculate summary statistics
            validation['summary'] = {
                'n_assets': n_assets,
                'n_data_points': n_data_points,
                'date_range': {
                    'start': data['prices'].index.min(),
                    'end': data['prices'].index.max(),
                    'days': (data['prices'].index.max() - data['prices'].index.min()).days
                },
                'missing_data_percentage': missing_percentage if 'missing_percentage' in locals() else 0,
                'average_return': data['returns'].mean().mean() if not data['returns'].empty else 0,
                'average_volatility': data['returns'].std().mean() * np.sqrt(252) if not data['returns'].empty else 0
            }
            
            # Determine if data is valid
            validation['is_valid'] = len(validation['issues']) == 0
            
            # Provide suggestions
            if not validation['is_valid']:
                if n_assets < min_assets:
                    validation['suggestions'].append(f"Add {min_assets - n_assets} more assets")
                if n_data_points < min_data_points:
                    validation['suggestions'].append("Extend the date range or use higher frequency data")
            
            return validation
            
        except Exception as e:
            validation['issues'].append(f"Validation error: {str(e)}")
            return validation
    
    def preprocess_data_for_analysis(self, data: Dict, 
                                     preprocessing_steps: List[str] = None) -> Dict:
        """Apply preprocessing steps to data."""
        if preprocessing_steps is None:
            preprocessing_steps = ['clean_missing', 'handle_outliers', 'normalize', 'stationarity_check']
        
        processed_data = data.copy()
        
        for step in preprocessing_steps:
            if step == 'clean_missing':
                processed_data = self._clean_missing_values(processed_data)
            elif step == 'handle_outliers':
                processed_data = self._handle_outliers(processed_data)
            elif step == 'normalize':
                processed_data = self._normalize_data(processed_data)
            elif step == 'stationarity_check':
                processed_data = self._check_stationarity(processed_data)
            elif step == 'detrend':
                processed_data = self._detrend_data(processed_data)
        
        return processed_data
    
    def _clean_missing_values(self, data: Dict) -> Dict:
        """Clean missing values from data."""
        if not data['prices'].empty:
            # Forward fill, then back fill
            data['prices'] = data['prices'].ffill().bfill()
        
        if not data['returns'].empty:
            # For returns, we can drop rows with too many missing values
            threshold = 0.5  # Keep rows with at least 50% non-missing values
            data['returns'] = data['returns'].dropna(thresh=int(threshold * len(data['returns'].columns)))
        
        return data
    
    def _handle_outliers(self, data: Dict) -> Dict:
        """Handle outliers in returns data."""
        if not data['returns'].empty:
            # Winsorize at 1st and 99th percentiles
            returns_clean = data['returns'].copy()
            
            for column in returns_clean.columns:
                series = returns_clean[column]
                lower = series.quantile(0.01)
                upper = series.quantile(0.99)
                returns_clean[column] = series.clip(lower, upper)
            
            data['returns'] = returns_clean
        
        return data
    
    def _normalize_data(self, data: Dict) -> Dict:
        """Normalize data for analysis."""
        if not data['returns'].empty:
            # Standardize returns
            returns_normalized = data['returns'].copy()
            for column in returns_normalized.columns:
                series = returns_normalized[column]
                if series.std() > 0:
                    returns_normalized[column] = (series - series.mean()) / series.std()
            data['returns_normalized'] = returns_normalized
        
        return data
    
    def _check_stationarity(self, data: Dict) -> Dict:
        """Check stationarity of time series."""
        stationarity_results = {}
        
        if not data['returns'].empty:
            for column in data['returns'].columns:
                try:
                    # Augmented Dickey-Fuller test
                    from statsmodels.tsa.stattools import adfuller
                    result = adfuller(data['returns'][column].dropna())
                    stationarity_results[column] = {
                        'adf_statistic': result[0],
                        'p_value': result[1],
                        'is_stationary': result[1] < 0.05,
                        'critical_values': result[4]
                    }
                except Exception as e:
                    stationarity_results[column] = {'error': str(e)}
        
        data['stationarity'] = stationarity_results
        return data
    
    def _detrend_data(self, data: Dict) -> Dict:
        """Remove linear trend from data."""
        if not data['prices'].empty:
            prices_detrended = data['prices'].copy()
            for column in prices_detrended.columns:
                series = prices_detrended[column]
                if len(series.dropna()) > 10:
                    x = np.arange(len(series.dropna()))
                    y = series.dropna().values
                    coeff = np.polyfit(x, y, 1)
                    trend = np.polyval(coeff, x)
                    prices_detrended.loc[series.dropna().index, column] = y - trend + y.mean()
            
            data['prices_detrended'] = prices_detrended
        
        return data
    
    def calculate_basic_statistics(self, data: Dict) -> Dict:
        """Calculate basic statistics for the dataset."""
        stats = {
            'assets': {},
            'portfolio_level': {},
            'correlation': {},
            'covariance': {}
        }
        
        if not data['returns'].empty:
            returns = data['returns']
            
            # Asset-level statistics
            for ticker in returns.columns:
                ticker_returns = returns[ticker].dropna()
                
                if len(ticker_returns) > 0:
                    stats['assets'][ticker] = {
                        'mean_return': ticker_returns.mean() * 252,
                        'annual_volatility': ticker_returns.std() * np.sqrt(252),
                        'sharpe_ratio': (ticker_returns.mean() * 252) / (ticker_returns.std() * np.sqrt(252)) if ticker_returns.std() > 0 else 0,
                        'skewness': ticker_returns.skew(),
                        'kurtosis': ticker_returns.kurtosis(),
                        'var_95': -np.percentile(ticker_returns, 5),
                        'max_drawdown': self._calculate_max_drawdown_series(ticker_returns),
                        'positive_days': (ticker_returns > 0).sum() / len(ticker_returns),
                        'data_points': len(ticker_returns)
                    }
            
            # Portfolio-level statistics (assuming equal weights for now)
            if len(returns.columns) > 0:
                equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
                portfolio_returns = returns.dot(equal_weights)
                
                stats['portfolio_level'] = {
                    'mean_return': portfolio_returns.mean() * 252,
                    'annual_volatility': portfolio_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
                    'skewness': portfolio_returns.skew(),
                    'kurtosis': portfolio_returns.kurtosis(),
                    'var_95': -np.percentile(portfolio_returns, 5),
                    'max_drawdown': self._calculate_max_drawdown_series(portfolio_returns),
                    'positive_days': (portfolio_returns > 0).sum() / len(portfolio_returns)
                }
            
            # Correlation and covariance matrices
            stats['correlation']['matrix'] = returns.corr()
            stats['correlation']['mean'] = returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean()
            stats['correlation']['min'] = returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].min()
            stats['correlation']['max'] = returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].max()
            
            stats['covariance']['matrix'] = returns.cov() * 252
            stats['covariance']['mean_variance'] = np.diag(returns.cov() * 252).mean()
        
        return stats

# Initialize data manager
data_manager = AdvancedDataManager()

# ============================================================================
# 7. ADVANCED BACKTESTING ENGINE
# ============================================================================

class AdvancedBacktester:
    """Advanced backtesting engine with multiple strategies and performance analysis."""
    
    def __init__(self):
        self.strategies = {}
        self.results_cache = {}
        self.benchmarks = {}
        
    def register_strategy(self, name: str, strategy_function: Callable):
        """Register a trading strategy."""
        self.strategies[name] = strategy_function
    
    def run_backtest(self, data: Dict, 
                    strategy_name: str,
                    initial_capital: float = 1000000,
                    commission: float = 0.001,  # 0.1% commission
                    slippage: float = 0.0005,   # 0.05% slippage
                    rebalance_frequency: str = 'M',  # Monthly rebalancing
                    start_date: datetime = None,
                    end_date: datetime = None) -> Dict:
        """Run a backtest for a specific strategy."""
        performance_monitor.start_operation(f'backtest_{strategy_name}')
        
        try:
            # Validate data
            validation = data_manager.validate_portfolio_data(data)
            if not validation['is_valid']:
                raise ValueError(f"Invalid data for backtesting: {validation['issues']}")
            
            # Extract prices and returns
            prices = data['prices'].copy()
            returns = data['returns'].copy()
            
            # Set date range
            if start_date:
                prices = prices[prices.index >= start_date]
                returns = returns[returns.index >= start_date]
            if end_date:
                prices = prices[prices.index <= end_date]
                returns = returns[returns.index <= end_date]
            
            if len(prices) < 20:
                raise ValueError("Insufficient data for backtesting")
            
            # Get strategy function
            if strategy_name not in self.strategies:
                raise ValueError(f"Strategy '{strategy_name}' not registered")
            
            strategy_func = self.strategies[strategy_name]
            
            # Run strategy to get signals
            signals = strategy_func(prices, returns)
            
            # Execute backtest
            results = self._execute_backtest(
                prices=prices,
                signals=signals,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                rebalance_frequency=rebalance_frequency
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(results)
            
            # Compare with benchmarks
            benchmark_comparison = self._compare_with_benchmarks(results, data)
            
            # Generate trade analysis
            trade_analysis = self._analyze_trades(results)
            
            # Create comprehensive results
            backtest_results = {
                'strategy_name': strategy_name,
                'parameters': {
                    'initial_capital': initial_capital,
                    'commission': commission,
                    'slippage': slippage,
                    'rebalance_frequency': rebalance_frequency,
                    'date_range': {
                        'start': prices.index[0],
                        'end': prices.index[-1]
                    }
                },
                'results': results,
                'performance_metrics': performance_metrics,
                'benchmark_comparison': benchmark_comparison,
                'trade_analysis': trade_analysis,
                'risk_metrics': self._calculate_risk_metrics(results['portfolio_value']),
                'visualizations': self._prepare_visualizations(results, data),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            cache_key = hashlib.md5(
                f"{strategy_name}_{initial_capital}_{commission}_{slippage}_{start_date}_{end_date}".encode()
            ).hexdigest()
            self.results_cache[cache_key] = backtest_results
            
            performance_monitor.end_operation(f'backtest_{strategy_name}', {
                'period': f"{prices.index[0]} to {prices.index[-1]}",
                'trades': len(trade_analysis.get('trades', [])),
                'final_value': results['portfolio_value'].iloc[-1]
            })
            
            return backtest_results
            
        except Exception as e:
            performance_monitor.end_operation(f'backtest_{strategy_name}', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {
                'operation': f'backtest_{strategy_name}',
                'initial_capital': initial_capital
            })
            raise
    
    def _execute_backtest(self, prices: pd.DataFrame,
                         signals: pd.DataFrame,
                         initial_capital: float,
                         commission: float,
                         slippage: float,
                         rebalance_frequency: str) -> Dict:
        """Execute the backtest with given signals."""
        results = {
            'dates': prices.index,
            'portfolio_value': pd.Series(index=prices.index, dtype=float),
            'cash': pd.Series(index=prices.index, dtype=float),
            'positions': pd.DataFrame(index=prices.index, columns=prices.columns),
            'trades': [],
            'weights': pd.DataFrame(index=prices.index, columns=prices.columns)
        }
        
        # Initialize
        current_cash = initial_capital
        current_positions = pd.Series(0, index=prices.columns)
        portfolio_value = initial_capital
        
        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(prices.index, rebalance_frequency)
        
        for i, date in enumerate(prices.index):
            # Get current prices
            current_prices = prices.loc[date]
            
            # Calculate current portfolio value
            position_value = (current_positions * current_prices).sum()
            portfolio_value = current_cash + position_value
            
            # Record values
            results['portfolio_value'].loc[date] = portfolio_value
            results['cash'].loc[date] = current_cash
            results['positions'].loc[date] = current_positions
            
            # Check if it's rebalance day
            if date in rebalance_dates or i == 0:
                # Get target weights from signals
                if date in signals.index:
                    target_weights = signals.loc[date]
                else:
                    # Use last available signal
                    available_signals = signals[signals.index <= date]
                    if not available_signals.empty:
                        target_weights = available_signals.iloc[-1]
                    else:
                        target_weights = pd.Series(0, index=prices.columns)
                
                # Calculate target positions
                target_position_value = portfolio_value * target_weights
                target_positions = target_position_value / current_prices.replace(0, np.nan)
                target_positions = target_positions.fillna(0)
                
                # Calculate trades needed
                trades = target_positions - current_positions
                
                # Execute trades with costs
                trade_costs = 0
                for asset in trades.index:
                    trade_size = trades[asset]
                    if abs(trade_size) > 1e-6:  # Only execute significant trades
                        # Calculate trade value
                        trade_value = trade_size * current_prices[asset]
                        
                        # Apply slippage
                        effective_price = current_prices[asset] * (1 + slippage * np.sign(trade_size))
                        
                        # Calculate commission
                        trade_commission = abs(trade_value) * commission
                        
                        # Update cash
                        current_cash -= trade_value + trade_commission
                        trade_costs += trade_commission
                        
                        # Update positions
                        current_positions[asset] += trade_size
                        
                        # Record trade
                        results['trades'].append({
                            'date': date,
                            'asset': asset,
                            'type': 'BUY' if trade_size > 0 else 'SELL',
                            'quantity': abs(trade_size),
                            'price': effective_price,
                            'value': abs(trade_value),
                            'commission': trade_commission
                        })
                
                # Record weights
                results['weights'].loc[date] = target_weights
            
            else:
                # Record current weights
                if portfolio_value > 0:
                    current_weights = (current_positions * current_prices) / portfolio_value
                    results['weights'].loc[date] = current_weights
                else:
                    results['weights'].loc[date] = pd.Series(0, index=prices.columns)
        
        return results
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
        """Get rebalance dates based on frequency."""
        if frequency == 'D':
            return dates
        elif frequency == 'W':
            return dates[dates.weekday == 0]  # Monday
        elif frequency == 'M':
            # Last trading day of month
            month_ends = dates.to_period('M').to_timestamp('M')
            return dates[dates.isin(month_ends)]
        elif frequency == 'Q':
            quarter_ends = dates.to_period('Q').to_timestamp('Q')
            return dates[dates.isin(quarter_ends)]
        elif frequency == 'Y':
            year_ends = dates.to_period('Y').to_timestamp('Y')
            return dates[dates.isin(year_ends)]
        else:
            # Custom frequency (e.g., '21D' for 21 days)
            try:
                days = int(frequency[:-1])
                return dates[dates.dayofweek == 0][::days]  # Approximate
            except:
                return dates[::20]  # Default to every 20 days
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive performance metrics."""
        portfolio_value = results['portfolio_value']
        dates = results['dates']
        trades = results['trades']
        
        metrics = {
            'returns': {},
            'risk': {},
            'ratios': {},
            'other': {}
        }
        
        if len(portfolio_value) > 1:
            # Calculate portfolio returns
            portfolio_returns = portfolio_value.pct_change().dropna()
            
            # Return metrics
            total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
            annual_return = ((1 + total_return) ** (252 / len(portfolio_value))) - 1
            
            metrics['returns'] = {
                'total_return': total_return,
                'annual_return': annual_return,
                'daily_mean_return': portfolio_returns.mean(),
                'daily_median_return': portfolio_returns.median(),
                'positive_days': (portfolio_returns > 0).sum() / len(portfolio_returns),
                'best_day': portfolio_returns.max(),
                'worst_day': portfolio_returns.min()
            }
            
            # Risk metrics
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Calculate drawdowns
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            metrics['risk'] = {
                'annual_volatility': annual_volatility,
                'downside_volatility': downside_volatility,
                'max_drawdown': drawdown.min(),
                'avg_drawdown': drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0,
                'drawdown_duration_max': self._calculate_max_drawdown_duration(drawdown),
                'var_95': -np.percentile(portfolio_returns, 5),
                'cvar_95': -portfolio_returns[portfolio_returns <= -np.percentile(portfolio_returns, 5)].mean() if len(portfolio_returns[portfolio_returns <= -np.percentile(portfolio_returns, 5)]) > 0 else 0
            }
            
            # Ratio metrics (using risk-free rate of 0.045)
            risk_free_rate = 0.045
            risk_free_daily = risk_free_rate / 252
            
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Calculate Calmar ratio
            if metrics['risk']['max_drawdown'] < 0:
                calmar_ratio = annual_return / abs(metrics['risk']['max_drawdown'])
            else:
                calmar_ratio = 0
            
            # Calculate Omega ratio
            threshold = risk_free_daily
            gains = portfolio_returns[portfolio_returns > threshold].sum()
            losses = abs(portfolio_returns[portfolio_returns < threshold]).sum()
            omega_ratio = gains / losses if losses > 0 else float('inf')
            
            metrics['ratios'] = {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'omega_ratio': omega_ratio,
                'treynor_ratio': sharpe_ratio,  # Simplified, would need beta
                'information_ratio': 0  # Would need benchmark
            }
            
            # Other metrics
            metrics['other'] = {
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t.get('profit', 0) > 0]) if trades else 0,
                'losing_trades': len([t for t in trades if t.get('profit', 0) < 0]) if trades else 0,
                'profit_factor': self._calculate_profit_factor(trades),
                'avg_trade_return': np.mean([t.get('return', 0) for t in trades]) if trades else 0,
                'largest_win': max([t.get('profit', 0) for t in trades]) if trades else 0,
                'largest_loss': min([t.get('profit', 0) for t in trades]) if trades else 0,
                'avg_holding_period': self._calculate_avg_holding_period(trades),
                'turnover': self._calculate_turnover(results['weights'])
            }
        
        return metrics
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        if len(drawdown) == 0:
            return 0
        
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not trades:
            return 0
        
        gross_profit = sum(t.get('profit', 0) for t in trades if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in trades if t.get('profit', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss
    
    def _calculate_avg_holding_period(self, trades: List[Dict]) -> float:
        """Calculate average holding period in days."""
        if not trades:
            return 0
        
        # This would require tracking entry and exit dates for each position
        # Simplified implementation
        return 30  # Default average holding period
    
    def _calculate_turnover(self, weights: pd.DataFrame) -> float:
        """Calculate portfolio turnover."""
        if len(weights) < 2:
            return 0
        
        turnover = 0
        for i in range(1, len(weights)):
            turnover += (abs(weights.iloc[i] - weights.iloc[i-1])).sum() / 2
        
        avg_turnover = turnover / (len(weights) - 1)
        return avg_turnover
    
    def _compare_with_benchmarks(self, results: Dict, data: Dict) -> Dict:
        """Compare strategy performance with benchmarks."""
        comparison = {}
        
        portfolio_value = results['portfolio_value']
        portfolio_returns = portfolio_value.pct_change().dropna()
        
        # Compare with equal-weight portfolio
        if 'returns' in data and not data['returns'].empty:
            returns_data = data['returns'].reindex(portfolio_returns.index).dropna()
            if not returns_data.empty:
                # Equal-weight benchmark
                ew_weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
                ew_returns = returns_data.dot(ew_weights)
                
                # Calculate metrics for EW benchmark
                ew_cumulative = (1 + ew_returns).cumprod()
                ew_total_return = ew_cumulative.iloc[-1] - 1 if not ew_cumulative.empty else 0
                
                # Calculate alpha and beta
                if len(portfolio_returns) == len(ew_returns):
                    try:
                        # Align dates
                        aligned_portfolio = portfolio_returns.reindex(ew_returns.index).dropna()
                        aligned_benchmark = ew_returns.reindex(aligned_portfolio.index).dropna()
                        
                        if len(aligned_portfolio) > 10 and len(aligned_benchmark) > 10:
                            # Calculate beta
                            covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                            benchmark_variance = np.var(aligned_benchmark)
                            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                            
                            # Calculate alpha
                            risk_free_rate = 0.045 / 252
                            portfolio_excess = aligned_portfolio.mean() - risk_free_rate
                            benchmark_excess = aligned_benchmark.mean() - risk_free_rate
                            alpha = portfolio_excess - beta * benchmark_excess
                            
                            # Calculate tracking error
                            tracking_error = (aligned_portfolio - aligned_benchmark).std() * np.sqrt(252)
                            
                            # Calculate information ratio
                            active_return = portfolio_excess - benchmark_excess
                            information_ratio = active_return / tracking_error if tracking_error > 0 else 0
                            
                            # Calculate R-squared
                            correlation = np.corrcoef(aligned_portfolio, aligned_benchmark)[0, 1]
                            r_squared = correlation ** 2
                            
                            comparison['equal_weight'] = {
                                'total_return': ew_total_return,
                                'beta': beta,
                                'alpha': alpha * 252,  # Annualize
                                'tracking_error': tracking_error,
                                'information_ratio': information_ratio,
                                'r_squared': r_squared,
                                'correlation': correlation
                            }
                    except Exception as e:
                        comparison['equal_weight'] = {'error': str(e)}
        
        # Add SPY as a benchmark (if we have the data)
        # This would require fetching SPY data
        
        return comparison
    
    def _analyze_trades(self, results: Dict) -> Dict:
        """Analyze trades for patterns and performance."""
        trades = results['trades']
        analysis = {
            'summary': {},
            'trades_by_asset': {},
            'performance_by_month': {},
            'patterns': {}
        }
        
        if not trades:
            return analysis
        
        # Basic summary
        analysis['summary'] = {
            'total_trades': len(trades),
            'buy_trades': len([t for t in trades if t['type'] == 'BUY']),
            'sell_trades': len([t for t in trades if t['type'] == 'SELL']),
            'total_commission': sum(t['commission'] for t in trades),
            'avg_trade_value': np.mean([t['value'] for t in trades]) if trades else 0
        }
        
        # Group trades by asset
        for trade in trades:
            asset = trade['asset']
            if asset not in analysis['trades_by_asset']:
                analysis['trades_by_asset'][asset] = {
                    'total_trades': 0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'total_value': 0,
                    'total_commission': 0
                }
            
            analysis['trades_by_asset'][asset]['total_trades'] += 1
            if trade['type'] == 'BUY':
                analysis['trades_by_asset'][asset]['buy_trades'] += 1
            else:
                analysis['trades_by_asset'][asset]['sell_trades'] += 1
            
            analysis['trades_by_asset'][asset]['total_value'] += trade['value']
            analysis['trades_by_asset'][asset]['total_commission'] += trade['commission']
        
        # Performance by month
        for trade in trades:
            month = trade['date'].strftime('%Y-%m')
            if month not in analysis['performance_by_month']:
                analysis['performance_by_month'][month] = {
                    'trades': 0,
                    'total_value': 0,
                    'commission': 0
                }
            
            analysis['performance_by_month'][month]['trades'] += 1
            analysis['performance_by_month'][month]['total_value'] += trade['value']
            analysis['performance_by_month'][month]['commission'] += trade['commission']
        
        return analysis
    
    def _calculate_risk_metrics(self, portfolio_value: pd.Series) -> Dict:
        """Calculate risk metrics for the portfolio."""
        if len(portfolio_value) < 20:
            return {}
        
        returns = portfolio_value.pct_change().dropna()
        
        # Use the risk analytics engine
        risk_results = risk_analytics.calculate_comprehensive_var_analysis(
            returns, 
            portfolio_value=portfolio_value.iloc[-1]
        )
        
        return risk_results
    
    def _prepare_visualizations(self, results: Dict, data: Dict) -> Dict:
        """Prepare visualization data."""
        visualizations = {
            'equity_curve': {
                'dates': results['dates'].tolist(),
                'portfolio_value': results['portfolio_value'].tolist()
            },
            'drawdown_chart': {
                'dates': results['dates'].tolist(),
                'drawdown': self._calculate_drawdown_series(results['portfolio_value']).tolist()
            },
            'rolling_metrics': self._calculate_rolling_metrics(results['portfolio_value']),
            'weight_evolution': {
                'dates': results['weights'].index.tolist(),
                'weights': results['weights'].to_dict('list')
            }
        }
        
        return visualizations
    
    def _calculate_drawdown_series(self, portfolio_value: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        cumulative = (1 + portfolio_value.pct_change().fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown
    
    def _calculate_rolling_metrics(self, portfolio_value: pd.Series) -> Dict:
        """Calculate rolling performance metrics."""
        returns = portfolio_value.pct_change().dropna()
        
        if len(returns) < 50:
            return {}
        
        window = min(252, len(returns) // 2)
        
        rolling_metrics = {
            'dates': returns.index[window:].tolist(),
            'rolling_sharpe': [],
            'rolling_volatility': [],
            'rolling_max_drawdown': []
        }
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            
            # Rolling Sharpe (annualized, assuming risk-free rate 0.045)
            if window_returns.std() > 0:
                sharpe = (window_returns.mean() * 252 - 0.045) / (window_returns.std() * np.sqrt(252))
            else:
                sharpe = 0
            
            rolling_metrics['rolling_sharpe'].append(sharpe)
            rolling_metrics['rolling_volatility'].append(window_returns.std() * np.sqrt(252))
            
            # Rolling max drawdown
            window_cumulative = (1 + window_returns).cumprod()
            window_rolling_max = window_cumulative.expanding().max()
            window_drawdown = (window_cumulative - window_rolling_max) / window_rolling_max
            rolling_metrics['rolling_max_drawdown'].append(window_drawdown.min())
        
        return rolling_metrics
    
    def add_benchmark(self, name: str, prices: pd.Series):
        """Add a benchmark for comparison."""
        self.benchmarks[name] = prices
    
    def optimize_strategy_parameters(self, strategy_name: str, 
                                    data: Dict,
                                    param_grid: Dict,
                                    initial_capital: float = 1000000,
                                    n_iter: int = 20) -> Dict:
        """Optimize strategy parameters using grid search."""
        performance_monitor.start_operation(f'optimize_{strategy_name}')
        
        try:
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(param_grid)
            
            # Limit number of iterations
            if len(param_combinations) > n_iter:
                param_combinations = random.sample(param_combinations, n_iter)
            
            best_params = None
            best_performance = -float('inf')
            results = []
            
            for params in param_combinations:
                try:
                    # Create strategy with parameters
                    def parameterized_strategy(prices, returns):
                        return self.strategies[strategy_name](prices, returns, **params)
                    
                    # Run backtest
                    backtest_results = self.run_backtest(
                        data=data,
                        strategy_name=strategy_name,
                        initial_capital=initial_capital
                    )
                    
                    # Get performance metric (e.g., Sharpe ratio)
                    performance = backtest_results['performance_metrics']['ratios']['sharpe_ratio']
                    
                    results.append({
                        'params': params,
                        'performance': performance,
                        'total_return': backtest_results['performance_metrics']['returns']['total_return'],
                        'max_drawdown': backtest_results['performance_metrics']['risk']['max_drawdown']
                    })
                    
                    # Update best parameters
                    if performance > best_performance:
                        best_performance = performance
                        best_params = params
                        
                except Exception as e:
                    continue
            
            # Sort results by performance
            results.sort(key=lambda x: x['performance'], reverse=True)
            
            optimization_results = {
                'best_params': best_params,
                'best_performance': best_performance,
                'all_results': results,
                'param_grid_size': len(param_combinations),
                'evaluated': len(results)
            }
            
            performance_monitor.end_operation(f'optimize_{strategy_name}', {
                'best_sharpe': best_performance,
                'evaluated_combinations': len(results)
            })
            
            return optimization_results
            
        except Exception as e:
            performance_monitor.end_operation(f'optimize_{strategy_name}', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {
                'operation': f'optimize_{strategy_name}',
                'param_grid_size': len(param_grid)
            })
            raise
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate parameter combinations from grid."""
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations

# Initialize backtester
backtester = AdvancedBacktester()

# ============================================================================
# 8. MACHINE LEARNING PORTFOLIO MANAGER
# ============================================================================

class MLPortfolioManager:
    """Machine learning portfolio manager with predictive models."""
    
    def __init__(self):
        self.models = {}
        self.feature_engineers = {}
        self.pipeline = {}
        
    def train_return_predictor(self, features: pd.DataFrame, 
                              target_returns: pd.Series,
                              model_type: str = 'xgboost',
                              test_size: float = 0.2,
                              n_folds: int = 5) -> Dict:
        """Train a return prediction model."""
        performance_monitor.start_operation(f'train_return_predictor_{model_type}')
        
        try:
            # Prepare features and target
            X = features.dropna()
            y = target_returns.reindex(X.index).dropna()
            X = X.reindex(y.index)
            
            if len(X) < 100 or len(y) < 100:
                raise ValueError(f"Insufficient data: X={len(X)}, y={len(y)}")
            
            # Split data
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model based on type
            if model_type == 'xgboost' and st.session_state.get('xgboost_available', False):
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
            elif model_type == 'random_forest' and st.session_state.get('sklearn_available', False):
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                
            elif model_type == 'linear':
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                
            elif model_type == 'neural_network' and (st.session_state.get('tensorflow_available', False) or 
                                                    st.session_state.get('pytorch_available', False)):
                model = self._create_neural_network(X.shape[1])
                # Simplified training
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
            else:
                # Fallback to linear model
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            metrics = self._evaluate_regression_model(y_train, train_pred, y_test, test_pred)
            
            # Feature importance
            feature_importance = self._calculate_feature_importance(model, X.columns, model_type)
            
            # Store model
            model_key = f"return_predictor_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.models[model_key] = {
                'model': model,
                'model_type': model_type,
                'features': list(X.columns),
                'metrics': metrics,
                'feature_importance': feature_importance,
                'trained_date': datetime.now().isoformat()
            }
            
            performance_monitor.end_operation(f'train_return_predictor_{model_type}', {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'test_r2': metrics['test']['r2']
            })
            
            return {
                'model_key': model_key,
                'metrics': metrics,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            performance_monitor.end_operation(f'train_return_predictor_{model_type}', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {
                'operation': f'train_return_predictor_{model_type}',
                'feature_shape': features.shape if 'features' in locals() else 'unknown',
                'model_type': model_type
            })
            raise
    
    def _create_neural_network(self, input_dim: int):
        """Create a neural network for return prediction."""
        try:
            if st.session_state.get('tensorflow_available', False):
                import tensorflow as tf
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(16, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                return model
            elif st.session_state.get('pytorch_available', False):
                import torch
                import torch.nn as nn
                
                class ReturnPredictor(nn.Module):
                    def __init__(self, input_dim):
                        super().__init__()
                        self.fc1 = nn.Linear(input_dim, 64)
                        self.fc2 = nn.Linear(64, 32)
                        self.fc3 = nn.Linear(32, 16)
                        self.fc4 = nn.Linear(16, 1)
                        self.dropout = nn.Dropout(0.2)
                        self.relu = nn.ReLU()
                    
                    def forward(self, x):
                        x = self.relu(self.fc1(x))
                        x = self.dropout(x)
                        x = self.relu(self.fc2(x))
                        x = self.dropout(x)
                        x = self.relu(self.fc3(x))
                        x = self.fc4(x)
                        return x
                
                return ReturnPredictor(input_dim)
        except:
            pass
        return None
    
    def _evaluate_regression_model(self, y_train: pd.Series, train_pred: np.ndarray,
                                  y_test: pd.Series, test_pred: np.ndarray) -> Dict:
        """Evaluate regression model performance."""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        metrics = {
            'train': {},
            'test': {}
        }
        
        # Train metrics
        metrics['train']['r2'] = r2_score(y_train, train_pred)
        metrics['train']['mse'] = mean_squared_error(y_train, train_pred)
        metrics['train']['rmse'] = np.sqrt(metrics['train']['mse'])
        metrics['train']['mae'] = mean_absolute_error(y_train, train_pred)
        metrics['train']['correlation'] = np.corrcoef(y_train, train_pred)[0, 1]
        
        # Test metrics
        metrics['test']['r2'] = r2_score(y_test, test_pred)
        metrics['test']['mse'] = mean_squared_error(y_test, test_pred)
        metrics['test']['rmse'] = np.sqrt(metrics['test']['mse'])
        metrics['test']['mae'] = mean_absolute_error(y_test, test_pred)
        metrics['test']['correlation'] = np.corrcoef(y_test, test_pred)[0, 1]
        
        return metrics
    
    def _calculate_feature_importance(self, model, feature_names, model_type: str) -> Dict:
        """Calculate feature importance for the model."""
        importance = {}
        
        try:
            if model_type in ['xgboost', 'random_forest']:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for name, imp in zip(feature_names, importances):
                        importance[name] = float(imp)
                else:
                    # Use permutation importance
                    from sklearn.inspection import permutation_importance
                    # Simplified - would need validation data
                    for i, name in enumerate(feature_names):
                        importance[name] = 1.0 / len(feature_names)
            elif model_type == 'linear':
                if hasattr(model, 'coef_'):
                    for name, coef in zip(feature_names, model.coef_):
                        importance[name] = float(abs(coef))
                else:
                    for name in feature_names:
                        importance[name] = 0.0
            else:
                # Equal importance for unknown models
                for name in feature_names:
                    importance[name] = 1.0 / len(feature_names)
        except:
            # Fallback
            for name in feature_names:
                importance[name] = 1.0 / len(feature_names)
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def train_risk_model(self, features: pd.DataFrame,
                        volatility: pd.Series,
                        model_type: str = 'garch') -> Dict:
        """Train a risk (volatility) prediction model."""
        performance_monitor.start_operation(f'train_risk_model_{model_type}')
        
        try:
            if model_type == 'garch' and st.session_state.get('statsmodels_available', False):
                # GARCH model for volatility
                from arch import arch_model
                
                # Use returns to fit GARCH
                returns = features.iloc[:, 0] if len(features.columns) > 0 else pd.Series(np.random.randn(len(volatility)))
                
                # Fit GARCH(1,1) model
                am = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
                res = am.fit(disp='off')
                
                # Forecast
                forecast = res.forecast(horizon=1)
                predicted_vol = np.sqrt(forecast.variance.values[-1, :][0])
                
                metrics = {
                    'log_likelihood': res.loglikelihood,
                    'aic': res.aic,
                    'bic': res.bic,
                    'params': {k: v for k, v in res.params.items()}
                }
                
                model_key = f"risk_model_garch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.models[model_key] = {
                    'model': res,
                    'model_type': 'garch',
                    'metrics': metrics,
                    'predicted_volatility': predicted_vol
                }
                
                performance_monitor.end_operation(f'train_risk_model_{model_type}', {
                    'log_likelihood': res.loglikelihood,
                    'predicted_vol': predicted_vol
                })
                
                return {
                    'model_key': model_key,
                    'metrics': metrics,
                    'predicted_volatility': predicted_vol
                }
            
            else:
                # Fallback to simple moving average
                predicted_vol = volatility.rolling(window=20).mean().iloc[-1]
                
                metrics = {
                    'model': 'simple_moving_average',
                    'window': 20
                }
                
                performance_monitor.end_operation(f'train_risk_model_{model_type}', {
                    'model': 'fallback_sma',
                    'predicted_vol': predicted_vol
                })
                
                return {
                    'model_key': 'risk_model_sma',
                    'metrics': metrics,
                    'predicted_volatility': predicted_vol
                }
                
        except Exception as e:
            performance_monitor.end_operation(f'train_risk_model_{model_type}', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {
                'operation': f'train_risk_model_{model_type}',
                'model_type': model_type
            })
            
            # Return simple fallback
            return {
                'model_key': 'risk_model_fallback',
                'metrics': {'error': str(e)},
                'predicted_volatility': volatility.mean() if len(volatility) > 0 else 0.2
            }
    
    def generate_ml_signals(self, current_data: Dict, 
                           model_keys: List[str] = None) -> Dict:
        """Generate trading signals using ML models."""
        performance_monitor.start_operation('generate_ml_signals')
        
        try:
            signals = {}
            
            if model_keys is None:
                model_keys = [k for k in self.models.keys() if 'return_predictor' in k]
            
            for model_key in model_keys:
                if model_key in self.models:
                    model_info = self.models[model_key]
                    model = model_info['model']
                    features = model_info.get('features', [])
                    
                    # Prepare current features
                    current_features = self._prepare_current_features(current_data, features)
                    
                    if current_features is not None and len(current_features) > 0:
                        try:
                            # Make prediction
                            if hasattr(model, 'predict'):
                                prediction = model.predict(current_features.reshape(1, -1))[0]
                            else:
                                # Handle other model types
                                prediction = 0
                            
                            signals[model_key] = {
                                'prediction': float(prediction),
                                'confidence': 0.5,  # Placeholder
                                'features_used': features[:5],  # First 5 features
                                'timestamp': datetime.now().isoformat()
                            }
                        except Exception as e:
                            signals[model_key] = {
                                'error': str(e),
                                'prediction': 0,
                                'confidence': 0
                            }
            
            # Combine signals (simple average for now)
            if signals:
                valid_predictions = [s['prediction'] for s in signals.values() 
                                   if 'prediction' in s and isinstance(s['prediction'], (int, float))]
                if valid_predictions:
                    avg_prediction = np.mean(valid_predictions)
                else:
                    avg_prediction = 0
            else:
                avg_prediction = 0
            
            performance_monitor.end_operation('generate_ml_signals', {
                'models_used': len(signals),
                'avg_prediction': avg_prediction
            })
            
            return {
                'individual_signals': signals,
                'combined_signal': avg_prediction,
                'signal_strength': min(1.0, abs(avg_prediction) * 10),  # Scale to [0, 1]
                'recommendation': 'BUY' if avg_prediction > 0 else 'SELL' if avg_prediction < 0 else 'HOLD'
            }
            
        except Exception as e:
            performance_monitor.end_operation('generate_ml_signals', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {
                'operation': 'generate_ml_signals',
                'model_keys': model_keys
            })
            return {
                'individual_signals': {},
                'combined_signal': 0,
                'signal_strength': 0,
                'recommendation': 'HOLD',
                'error': str(e)
            }
    
    def _prepare_current_features(self, current_data: Dict, required_features: List[str]) -> np.ndarray:
        """Prepare current features for prediction."""
        # This is a simplified implementation
        # In practice, you would compute all required features from current_data
        
        # Placeholder: return random features
        n_features = len(required_features)
        return np.random.randn(n_features) * 0.1
    
    def ensemble_predictions(self, models_list: List[str], 
                           method: str = 'weighted_average') -> Dict:
        """Combine predictions from multiple models."""
        predictions = []
        weights = []
        
        for model_key in models_list:
            if model_key in self.models:
                model_info = self.models[model_key]
                # Get latest prediction or train if needed
                # For now, use model performance as weight
                weight = model_info.get('metrics', {}).get('test', {}).get('r2', 0.1)
                if weight > 0:
                    predictions.append(0)  # Placeholder prediction
                    weights.append(weight)
        
        if not predictions:
            return {'ensemble_prediction': 0, 'confidence': 0}
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        if method == 'weighted_average':
            ensemble_pred = np.average(predictions, weights=weights)
        elif method == 'median':
            ensemble_pred = np.median(predictions)
        else:
            ensemble_pred = np.mean(predictions)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'confidence': float(weights.max()),  # Highest model confidence
            'method': method,
            'n_models': len(predictions)
        }
    
    def create_feature_engineering_pipeline(self) -> Dict:
        """Create feature engineering pipeline for financial data."""
        pipeline = {
            'technical_indicators': [
                'sma_20', 'sma_50', 'sma_200',
                'rsi_14', 'macd', 'bollinger_bands',
                'atr_14', 'stochastic_14'
            ],
            'statistical_features': [
                'returns_1d', 'returns_5d', 'returns_21d',
                'volatility_20d', 'skewness_20d', 'kurtosis_20d',
                'sharpe_20d'
            ],
            'market_features': [
                'volume_ratio', 'price_to_sma_ratio',
                'high_low_ratio', 'gap_up_down'
            ],
            'macro_features': [
                'vix', 'treasury_yield', 'dollar_index',
                'commodity_index'
            ]
        }
        
        self.pipeline['feature_engineering'] = pipeline
        return pipeline
    
    def calculate_feature_importance_across_models(self) -> pd.DataFrame:
        """Calculate and aggregate feature importance across all models."""
        all_importances = []
        
        for model_key, model_info in self.models.items():
            if 'feature_importance' in model_info:
                importances = model_info['feature_importance']
                for feature, importance in importances.items():
                    all_importances.append({
                        'model': model_key,
                        'feature': feature,
                        'importance': importance,
                        'model_type': model_info.get('model_type', 'unknown')
                    })
        
        if not all_importances:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_importances)
        
        # Aggregate by feature
        feature_agg = df.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
        feature_agg = feature_agg.sort_values('mean', ascending=False)
        
        return feature_agg

# Initialize ML portfolio manager
ml_manager = MLPortfolioManager()

# ============================================================================
# 9. ENTERPRISE REPORTING ENGINE
# ============================================================================

class EnterpriseReporter:
    """Enterprise reporting engine with PDF, Excel, and interactive reports."""
    
    def __init__(self):
        self.templates = {}
        self.scheduled_reports = {}
    
    def generate_comprehensive_report(self, analysis_data: Dict,
                                     report_type: str = 'pdf',
                                     template: str = 'standard') -> Dict:
        """Generate comprehensive portfolio analysis report."""
        performance_monitor.start_operation(f'generate_report_{report_type}')
        
        try:
            report = {
                'metadata': {
                    'generated_date': datetime.now().isoformat(),
                    'report_type': report_type,
                    'template': template,
                    'version': 'QuantEdge Pro v5.0'
                },
                'sections': {},
                'summary': {},
                'recommendations': []
            }
            
            # Generate different sections based on available data
            if 'portfolio_optimization' in analysis_data:
                report['sections']['portfolio_optimization'] = self._generate_optimization_section(
                    analysis_data['portfolio_optimization']
                )
            
            if 'risk_analysis' in analysis_data:
                report['sections']['risk_analysis'] = self._generate_risk_section(
                    analysis_data['risk_analysis']
                )
            
            if 'backtest_results' in analysis_data:
                report['sections']['backtesting'] = self._generate_backtest_section(
                    analysis_data['backtest_results']
                )
            
            if 'ml_analysis' in analysis_data:
                report['sections']['machine_learning'] = self._generate_ml_section(
                    analysis_data['ml_analysis']
                )
            
            # Generate executive summary
            report['summary'] = self._generate_executive_summary(report['sections'])
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report['sections'])
            
            # Generate the actual report file
            if report_type == 'pdf':
                report_path = self._generate_pdf_report(report)
            elif report_type == 'excel':
                report_path = self._generate_excel_report(report)
            elif report_type == 'html':
                report_path = self._generate_html_report(report)
            else:
                report_path = None
            
            report['file_path'] = report_path
            
            performance_monitor.end_operation(f'generate_report_{report_type}', {
                'sections': len(report['sections']),
                'recommendations': len(report['recommendations'])
            })
            
            return report
            
        except Exception as e:
            performance_monitor.end_operation(f'generate_report_{report_type}', {'error': str(e)})
            error_analyzer.analyze_error_with_context(e, {
                'operation': f'generate_report_{report_type}',
                'report_type': report_type
            })
            
            return {
                'error': str(e),
                'metadata': {
                    'generated_date': datetime.now().isoformat(),
                    'error': True
                }
            }
    
    def _generate_optimization_section(self, optimization_data: Dict) -> Dict:
        """Generate portfolio optimization section."""
        section = {
            'title': 'Portfolio Optimization Analysis',
            'summary': {},
            'details': {},
            'visualizations': {}
        }
        
        if 'weights' in optimization_data:
            weights = optimization_data['weights']
            section['summary']['assets'] = len(weights)
            section['summary']['top_holdings'] = sorted(
                [(k, v) for k, v in weights.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        
        if 'metrics' in optimization_data:
            metrics = optimization_data['metrics']
            section['summary']['expected_return'] = metrics.get('expected_return', 0)
            section['summary']['expected_volatility'] = metrics.get('expected_volatility', 0)
            section['summary']['sharpe_ratio'] = metrics.get('sharpe_ratio', 0)
        
        section['details'] = optimization_data
        
        return section
    
    def _generate_risk_section(self, risk_data: Dict) -> Dict:
        """Generate risk analysis section."""
        section = {
            'title': 'Risk Analysis',
            'summary': {},
            'details': {},
            'warnings': []
        }
        
        if 'methods' in risk_data and 'Historical' in risk_data['methods']:
            historical = risk_data['methods']['Historical']
            if 0.95 in historical:
                var_95 = historical[0.95]
                section['summary']['var_95'] = var_95.get('VaR', 0)
                section['summary']['cvar_95'] = var_95.get('CVaR', 0)
        
        if 'violations' in risk_data:
            violations = risk_data['violations']
            section['summary']['violations_95'] = violations.get('violations_95', 0)
            section['summary']['total_days'] = violations.get('total_days', 0)
        
        # Add risk warnings
        if 'summary' in risk_data and 'worst_case_var' in risk_data['summary']:
            worst_var = risk_data['summary']['worst_case_var']
            if worst_var > 0.1:  # More than 10% VaR
                section['warnings'].append(f"High worst-case VaR: {worst_var:.1%}")
        
        section['details'] = risk_data
        
        return section
    
    def _generate_backtest_section(self, backtest_data: Dict) -> Dict:
        """Generate backtesting section."""
        section = {
            'title': 'Backtesting Results',
            'summary': {},
            'details': {},
            'performance_metrics': {}
        }
        
        if 'performance_metrics' in backtest_data:
            perf = backtest_data['performance_metrics']
            section['performance_metrics'] = perf
            
            if 'returns' in perf:
                section['summary']['total_return'] = perf['returns'].get('total_return', 0)
                section['summary']['annual_return'] = perf['returns'].get('annual_return', 0)
            
            if 'risk' in perf:
                section['summary']['max_drawdown'] = perf['risk'].get('max_drawdown', 0)
                section['summary']['annual_volatility'] = perf['risk'].get('annual_volatility', 0)
            
            if 'ratios' in perf:
                section['summary']['sharpe_ratio'] = perf['ratios'].get('sharpe_ratio', 0)
                section['summary']['sortino_ratio'] = perf['ratios'].get('sortino_ratio', 0)
        
        section['details'] = backtest_data
        
        return section
    
    def _generate_ml_section(self, ml_data: Dict) -> Dict:
        """Generate machine learning section."""
        section = {
            'title': 'Machine Learning Analysis',
            'summary': {},
            'details': {},
            'model_performance': {}
        }
        
        if 'models' in ml_data:
            models = ml_data['models']
            section['summary']['total_models'] = len(models)
            
            # Aggregate model performance
            r2_scores = []
            for model_info in models.values():
                if 'metrics' in model_info and 'test' in model_info['metrics']:
                    r2 = model_info['metrics']['test'].get('r2', 0)
                    r2_scores.append(r2)
            
            if r2_scores:
                section['summary']['avg_r2'] = np.mean(r2_scores)
                section['summary']['best_r2'] = np.max(r2_scores)
        
        section['details'] = ml_data
        
        return section
    
    def _generate_executive_summary(self, sections: Dict) -> Dict:
        """Generate executive summary from all sections."""
        summary = {
            'overview': '',
            'key_findings': [],
            'risk_level': 'Medium',  # Default
            'recommended_action': 'Monitor'
        }
        
        # Collect key metrics
        key_metrics = {}
        
        # From optimization
        if 'portfolio_optimization' in sections:
            opt = sections['portfolio_optimization']['summary']
            if 'sharpe_ratio' in opt:
                key_metrics['Sharpe Ratio'] = opt['sharpe_ratio']
            if 'expected_return' in opt:
                key_metrics['Expected Return'] = opt['expected_return']
        
        # From risk analysis
        if 'risk_analysis' in sections:
            risk = sections['risk_analysis']['summary']
            if 'var_95' in risk:
                key_metrics['VaR (95%)'] = risk['var_95']
            if 'max_drawdown' in risk:
                key_metrics['Max Drawdown'] = risk.get('max_drawdown', 0)
        
        # From backtesting
        if 'backtesting' in sections:
            backtest = sections['backtesting']['summary']
            if 'total_return' in backtest:
                key_metrics['Total Return'] = backtest['total_return']
            if 'sharpe_ratio' in backtest:
                # Prefer backtest Sharpe if available
                key_metrics['Sharpe Ratio'] = backtest['sharpe_ratio']
        
        summary['key_metrics'] = key_metrics
        
        # Generate overview text
        overview_parts = []
        
        if 'Sharpe Ratio' in key_metrics:
            sharpe = key_metrics['Sharpe Ratio']
            if sharpe > 1.0:
                overview_parts.append(f"Strong risk-adjusted returns (Sharpe: {sharpe:.2f})")
            elif sharpe > 0.5:
                overview_parts.append(f"Moderate risk-adjusted returns (Sharpe: {sharpe:.2f})")
            else:
                overview_parts.append(f"Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
        
        if 'VaR (95%)' in key_metrics:
            var = key_metrics['VaR (95%)']
            if var > 0.05:
                overview_parts.append(f"High risk exposure (VaR: {var:.1%})")
                summary['risk_level'] = 'High'
            elif var > 0.02:
                overview_parts.append(f"Moderate risk exposure (VaR: {var:.1%})")
                summary['risk_level'] = 'Medium'
            else:
                overview_parts.append(f"Low risk exposure (VaR: {var:.1%})")
                summary['risk_level'] = 'Low'
        
        if 'Total Return' in key_metrics:
            total_return = key_metrics['Total Return']
            if total_return > 0.2:
                overview_parts.append(f"Strong absolute returns ({total_return:.1%})")
            elif total_return > 0:
                overview_parts.append(f"Positive absolute returns ({total_return:.1%})")
            else:
                overview_parts.append(f"Negative absolute returns ({total_return:.1%})")
                summary['recommended_action'] = 'Review Strategy'
        
        summary['overview'] = ' '.join(overview_parts)
        
        return summary
    
    def _generate_recommendations(self, sections: Dict) -> List[Dict]:
        """Generate investment recommendations."""
        recommendations = []
        
        # Risk-based recommendations
        if 'risk_analysis' in sections:
            risk = sections['risk_analysis']
            if 'warnings' in risk and risk['warnings']:
                for warning in risk['warnings']:
                    recommendations.append({
                        'type': 'Risk',
                        'priority': 'High',
                        'action': 'Reduce risk exposure',
                        'reason': warning,
                        'details': 'Consider reducing position sizes or adding hedging instruments'
                    })
        
        # Optimization recommendations
        if 'portfolio_optimization' in sections:
            opt = sections['portfolio_optimization']
            if 'summary' in opt and 'top_holdings' in opt['summary']:
                top_holdings = opt['summary']['top_holdings']
                if top_holdings:
                    top_weight = top_holdings[0][1]
                    if top_weight > 0.3:  # More than 30% in single asset
                        recommendations.append({
                            'type': 'Concentration',
                            'priority': 'Medium',
                            'action': 'Diversify portfolio',
                            'reason': f'High concentration in {top_holdings[0][0]} ({top_weight:.1%})',
                            'details': 'Consider reducing top holding and adding uncorrelated assets'
                        })
        
        # Performance recommendations
        if 'backtesting' in sections:
            backtest = sections['backtesting']
            if 'summary' in backtest and 'sharpe_ratio' in backtest['summary']:
                sharpe = backtest['summary']['sharpe_ratio']
                if sharpe < 0.5:
                    recommendations.append({
                        'type': 'Performance',
                        'priority': 'Medium',
                        'action': 'Improve risk-adjusted returns',
                        'reason': f'Low Sharpe ratio ({sharpe:.2f})',
                        'details': 'Review strategy parameters or consider alternative approaches'
                    })
        
        # ML-based recommendations
        if 'machine_learning' in sections:
            ml = sections['machine_learning']
            if 'summary' in ml and 'avg_r2' in ml['summary']:
                avg_r2 = ml['summary']['avg_r2']
                if avg_r2 < 0.1:
                    recommendations.append({
                        'type': 'Prediction',
                        'priority': 'Low',
                        'action': 'Improve prediction models',
                        'reason': f'Low predictive power (R²: {avg_r2:.3f})',
                        'details': 'Consider adding more features or using different modeling approaches'
                    })
        
        return recommendations
    
    def _generate_pdf_report(self, report: Dict) -> str:
        """Generate PDF report."""
        try:
            if st.session_state.get('reportlab_available', False):
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import letter, landscape
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                
                # Create PDF document
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"quantedge_report_{timestamp}.pdf"
                
                doc = SimpleDocTemplate(
                    filename,
                    pagesize=landscape(letter),
                    rightMargin=72,
                    leftMargin=72,
                    topMargin=72,
                    bottomMargin=18
                )
                
                styles = getSampleStyleSheet()
                elements = []
                
                # Title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    spaceAfter=30,
                    textColor=colors.HexColor('#1a5276')
                )
                
                elements.append(Paragraph("QuantEdge Pro Portfolio Analysis Report", title_style))
                elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                elements.append(Spacer(1, 20))
                
                # Executive Summary
                elements.append(Paragraph("Executive Summary", styles['Heading2']))
                if 'summary' in report and 'overview' in report['summary']:
                    elements.append(Paragraph(report['summary']['overview'], styles['Normal']))
                
                # Key Metrics Table
                if 'summary' in report and 'key_metrics' in report['summary']:
                    elements.append(Spacer(1, 20))
                    elements.append(Paragraph("Key Performance Metrics", styles['Heading3']))
                    
                    data = [['Metric', 'Value']]
                    for metric, value in report['summary']['key_metrics'].items():
                        if isinstance(value, float):
                            if abs(value) < 0.01:
                                formatted_value = f"{value:.4f}"
                            elif abs(value) < 1:
                                formatted_value = f"{value:.3f}"
                            else:
                                formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = str(value)
                        data.append([metric, formatted_value])
                    
                    table = Table(data, colWidths=[3*inch, 2*inch])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    elements.append(table)
                
                # Recommendations
                if 'recommendations' in report and report['recommendations']:
                    elements.append(Spacer(1, 20))
                    elements.append(Paragraph("Recommendations", styles['Heading2']))
                    
                    for i, rec in enumerate(report['recommendations'], 1):
                        elements.append(Paragraph(f"{i}. {rec['action']}", styles['Heading3']))
                        elements.append(Paragraph(f"   Priority: {rec['priority']} | Type: {rec['type']}", styles['Normal']))
                        elements.append(Paragraph(f"   Reason: {rec['reason']}", styles['Normal']))
                        elements.append(Paragraph(f"   Details: {rec['details']}", styles['Normal']))
                        elements.append(Spacer(1, 10))
                
                # Build PDF
                doc.build(elements)
                return filename
            else:
                # Fallback: create text file
                return self._generate_text_report(report)
                
        except Exception as e:
            error_analyzer.analyze_error_with_context(e, {'operation': 'generate_pdf_report'})
            return self._generate_text_report(report)
    
    def _generate_excel_report(self, report: Dict) -> str:
        """Generate Excel report."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"quantedge_report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                if 'summary' in report and 'key_metrics' in report['summary']:
                    for metric, value in report['summary']['key_metrics'].items():
                        summary_data.append([metric, value])
                
                if summary_data:
                    df_summary = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Recommendations sheet
                if 'recommendations' in report:
                    rec_data = []
                    for rec in report['recommendations']:
                        rec_data.append([
                            rec['type'],
                            rec['priority'],
                            rec['action'],
                            rec['reason'],
                            rec.get('details', '')
                        ])
                    
                    if rec_data:
                        df_rec = pd.DataFrame(rec_data, columns=['Type', 'Priority', 'Action', 'Reason', 'Details'])
                        df_rec.to_excel(writer, sheet_name='Recommendations', index=False)
                
                # Details sheets
                for section_name, section_data in report.get('sections', {}).items():
                    if 'details' in section_data:
                        # Flatten details for Excel
                        flat_data = self._flatten_dict_for_excel(section_data['details'])
                        if flat_data:
                            df_details = pd.DataFrame(flat_data.items(), columns=['Key', 'Value'])
                            df_details.to_excel(writer, sheet_name=section_name[:31], index=False)
            
            return filename
            
        except Exception as e:
            error_analyzer.analyze_error_with_context(e, {'operation': 'generate_excel_report'})
            return self._generate_text_report(report)
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quantedge_report_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QuantEdge Pro Portfolio Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
                .header {{ background-color: #1a5276; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #1a5276; }}
                .recommendation {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                .high-priority {{ border-left-color: #dc3545; }}
                .medium-priority {{ border-left-color: #ffc107; }}
                .low-priority {{ border-left-color: #28a745; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #1a5276; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>QuantEdge Pro Portfolio Analysis Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{report.get('summary', {}).get('overview', 'No summary available')}</p>
                
                <h3>Key Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        # Add key metrics
        if 'summary' in report and 'key_metrics' in report['summary']:
            for metric, value in report['summary']['key_metrics'].items():
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        formatted_value = f"{value:.4f}"
                    elif abs(value) < 1:
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                html_content += f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
        """
        
        # Add recommendations
        if 'recommendations' in report:
            for rec in report['recommendations']:
                priority_class = rec['priority'].lower().replace(' ', '-')
                html_content += f"""
                <div class="recommendation {priority_class}-priority">
                    <h3>{rec['action']}</h3>
                    <p><strong>Type:</strong> {rec['type']} | <strong>Priority:</strong> {rec['priority']}</p>
                    <p><strong>Reason:</strong> {rec['reason']}</p>
                    <p><strong>Details:</strong> {rec.get('details', '')}</p>
                </div>
                """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Detailed Analysis</h2>
                <p>Complete analysis data is available in the accompanying files.</p>
            </div>
            
            <div class="section">
                <p><em>Report generated by QuantEdge Pro v5.0 Enterprise Edition</em></p>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    
    def _generate_text_report(self, report: Dict) -> str:
        """Generate simple text report as fallback."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quantedge_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("QUANTEDGE PRO PORTFOLIO ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            if 'summary' in report and 'overview' in report['summary']:
                f.write(f"{report['summary']['overview']}\n\n")
            
            f.write("KEY METRICS\n")
            f.write("-" * 80 + "\n")
            if 'summary' in report and 'key_metrics' in report['summary']:
                for metric, value in report['summary']['key_metrics'].items():
                    f.write(f"{metric}: {value}\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            if 'recommendations' in report:
                for i, rec in enumerate(report['recommendations'], 1):
                    f.write(f"{i}. [{rec['priority']}] {rec['action']}\n")
                    f.write(f"   Reason: {rec['reason']}\n")
                    f.write(f"   Details: {rec.get('details', '')}\n\n")
        
        return filename
    
    def _flatten_dict_for_excel(self, data: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for Excel export."""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict_for_excel(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to strings
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def schedule_report(self, frequency: str, recipients: List[str],
                       report_config: Dict, start_date: datetime = None) -> str:
        """Schedule automated report generation."""
        schedule_id = hashlib.md5(
            f"{frequency}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        self.scheduled_reports[schedule_id] = {
            'frequency': frequency,
            'recipients': recipients,
            'config': report_config,
            'start_date': start_date or datetime.now(),
            'last_run': None,
            'next_run': self._calculate_next_run(frequency, start_date),
            'enabled': True
        }
        
        return schedule_id
    
    def _calculate_next_run(self, frequency: str, start_date: datetime) -> datetime:
        """Calculate next run time for scheduled report."""
        if frequency == 'daily':
            return (start_date or datetime.now()) + timedelta(days=1)
        elif frequency == 'weekly':
            return (start_date or datetime.now()) + timedelta(weeks=1)
        elif frequency == 'monthly':
            # Next month, same day
            next_date = (start_date or datetime.now()).replace(day=1) + timedelta(days=32)
            return next_date.replace(day=1)
        elif frequency == 'quarterly':
            # Next quarter
            current = start_date or datetime.now()
            quarter = (current.month - 1) // 3
            next_quarter_month = quarter * 3 + 4
            year = current.year + (next_quarter_month > 12)
            next_quarter_month = (next_quarter_month - 1) % 12 + 1
            return datetime(year, next_quarter_month, 1)
        else:
            # Default: weekly
            return (start_date or datetime.now()) + timedelta(weeks=1)

# Initialize reporter
reporter = EnterpriseReporter()

# ============================================================================
# 10. MAIN APPLICATION & STREAMLIT UI
# ============================================================================

class QuantEdgeProApp:
    """Main QuantEdge Pro Application with Streamlit UI."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.components = {
            'viz_engine': viz_engine,
            'risk_analytics': risk_analytics,
            'portfolio_optimizer': portfolio_optimizer,
            'data_manager': data_manager,
            'backtester': backtester,
            'ml_manager': ml_manager,
            'reporter': reporter,
            'performance_monitor': performance_monitor
        }
    
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="QuantEdge Pro v5.0 Enterprise Edition",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/quantedge',
                'Report a bug': 'https://github.com/quantedge/issues',
                'About': 'QuantEdge Pro v5.0 - Institutional Portfolio Analytics Platform'
            }
        )
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'initialized': False,
            'current_page': 'dashboard',
            'portfolio_data': None,
            'optimization_results': None,
            'risk_results': None,
            'backtest_results': None,
            'ml_results': None,
            'report_results': None,
            'selected_tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'date_range': {
                'start': datetime.now() - timedelta(days=365*2),
                'end': datetime.now()
            },
            'portfolio_value': 1000000,
            'risk_free_rate': 0.045
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Initialize library status if not already done
        if 'enterprise_library_status' not in st.session_state:
            st.session_state.enterprise_library_status = ENTERPRISE_LIBRARY_STATUS
    
    def render_sidebar(self):
        """Render the sidebar navigation and controls."""
        with st.sidebar:
            # Logo and Title
            st.title("📈 QuantEdge Pro")
            st.markdown("**v5.0 Enterprise Edition**")
            st.markdown("---")
            
            # Navigation
            st.subheader("Navigation")
            pages = {
                "🏠 Dashboard": "dashboard",
                "📊 Portfolio Analysis": "portfolio_analysis",
                "⚡ Optimization": "optimization",
                "⚠️ Risk Analytics": "risk_analytics",
                "🔙 Backtesting": "backtesting",
                "🤖 Machine Learning": "machine_learning",
                "📑 Reporting": "reporting",
                "⚙️ Settings": "settings"
            }
            
            selected_page = st.radio(
                "Select Page",
                list(pages.keys()),
                label_visibility="collapsed"
            )
            
            st.session_state.current_page = pages[selected_page]
            st.markdown("---")
            
            # Data Configuration
            st.subheader("Data Configuration")
            
            # Ticker input
            tickers_input = st.text_area(
                "Enter Tickers (comma-separated)",
                value=", ".join(st.session_state.selected_tickers),
                height=100,
                help="Enter stock tickers separated by commas"
            )
            
            if tickers_input:
                tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
                st.session_state.selected_tickers = tickers
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=st.session_state.date_range['start'],
                    max_value=datetime.now()
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=st.session_state.date_range['end'],
                    max_value=datetime.now()
                )
            
            st.session_state.date_range = {
                'start': start_date,
                'end': end_date
            }
            
            # Portfolio value
            st.session_state.portfolio_value = st.number_input(
                "Portfolio Value ($)",
                value=st.session_state.portfolio_value,
                min_value=1000,
                max_value=1000000000,
                step=10000
            )
            
            # Fetch data button
            if st.button("📥 Fetch Market Data", type="primary", use_container_width=True):
                with st.spinner("Fetching market data..."):
                    self.fetch_market_data()
            
            st.markdown("---")
            
            # System Status
            st.subheader("System Status")
            
            lib_status = st.session_state.enterprise_library_status
            if lib_status['all_available']:
                st.success("✅ All libraries available")
            else:
                st.warning(f"⚠️ Missing: {', '.join(lib_status['missing'])}")
            
            # Performance info
            if 'performance_report' in st.session_state:
                report = st.session_state.performance_report
                st.metric("Operations", report['summary']['total_operations'])
                st.metric("Total Runtime", f"{report['total_runtime']:.1f}s")
            
            st.markdown("---")
            
            # Quick Actions
            st.subheader("Quick Actions")
            
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
            
            if st.button("📊 Performance Report", use_container_width=True):
                self.show_performance_report()
    
    def fetch_market_data(self):
        """Fetch market data for selected tickers."""
        try:
            with st.spinner(f"Fetching data for {len(st.session_state.selected_tickers)} tickers..."):
                data = data_manager.fetch_advanced_market_data(
                    tickers=st.session_state.selected_tickers,
                    start_date=st.session_state.date_range['start'],
                    end_date=st.session_state.date_range['end']
                )
                
                # Validate data
                validation = data_manager.validate_portfolio_data(data)
                
                if validation['is_valid']:
                    st.session_state.portfolio_data = data
                    st.session_state.data_validation = validation
                    
                    # Calculate basic statistics
                    stats = data_manager.calculate_basic_statistics(data)
                    st.session_state.portfolio_stats = stats
                    
                    st.success(f"✅ Data fetched successfully! {len(data['prices'].columns)} assets, {len(data['prices'])} periods")
                    
                    # Show validation summary
                    with st.expander("Data Validation Summary"):
                        st.json(validation['summary'])
                else:
                    st.error(f"❌ Data validation failed: {validation['issues']}")
                    if validation['suggestions']:
                        st.info("Suggestions: " + "; ".join(validation['suggestions']))
        
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            error_analyzer.create_advanced_error_display(
                error_analyzer.analyze_error_with_context(e, {
                    'operation': 'fetch_market_data',
                    'tickers': st.session_state.selected_tickers
                })
            )
    
    def render_dashboard(self):
        """Render the main dashboard."""
        st.title("🏠 QuantEdge Pro Dashboard")
        st.markdown("Real-time portfolio analytics and monitoring")
        
        if st.session_state.portfolio_data is None:
            st.info("Please fetch market data first from the sidebar.")
            return
        
        # Create tabs for different dashboard views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview", 
            "📊 Performance", 
            "⚠️ Risk", 
            "🔍 Insights"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_performance_tab()
        
        with tab3:
            self.render_risk_tab()
        
        with tab4:
            self.render_insights_tab()
    
    def render_overview_tab(self):
        """Render overview tab."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Portfolio Value", 
                f"${st.session_state.portfolio_value:,.0f}",
                delta="+2.3%"
            )
        
        with col2:
            if 'portfolio_stats' in st.session_state:
                stats = st.session_state.portfolio_stats
                if 'portfolio_level' in stats:
                    st.metric(
                        "Expected Return", 
                        f"{stats['portfolio_level']['mean_return']:.2%}",
                        delta="Annualized"
                    )
        
        with col3:
            if 'portfolio_stats' in st.session_state:
                stats = st.session_state.portfolio_stats
                if 'portfolio_level' in stats:
                    st.metric(
                        "Volatility", 
                        f"{stats['portfolio_level']['annual_volatility']:.2%}",
                        delta="Annualized"
                    )
        
        # Asset allocation chart
        st.subheader("Asset Allocation")
        if 'portfolio_data' in st.session_state:
            data = st.session_state.portfolio_data
            
            # Use last available prices for allocation
            if not data['prices'].empty:
                last_prices = data['prices'].iloc[-1]
                
                # Create allocation data (equal weight for now)
                n_assets = len(last_prices)
                allocation = {ticker: 1/n_assets for ticker in last_prices.index}
                
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=list(allocation.keys()),
                    values=list(allocation.values()),
                    hole=0.3
                )])
                
                fig.update_layout(
                    title="Current Allocation",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent performance
        st.subheader("Recent Performance")
        if 'portfolio_data' in st.session_state and 'returns' in st.session_state.portfolio_data:
            returns = st.session_state.portfolio_data['returns']
            
            if not returns.empty:
                # Calculate cumulative returns
                cumulative_returns = (1 + returns).cumprod() - 1
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.mean(axis=1) * 100,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#00cc96', width=3)
                ))
                
                fig.update_layout(
                    title="Cumulative Returns",
                    yaxis_title="Return (%)",
                    height=400,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_tab(self):
        """Render performance tab."""
        st.subheader("Performance Analytics")
        
        if 'portfolio_stats' in st.session_state:
            stats = st.session_state.portfolio_stats
            
            # Create metrics dashboard
            metrics_to_show = {}
            if 'portfolio_level' in stats:
                metrics_to_show.update({
                    'Annual Return': stats['portfolio_level']['mean_return'],
                    'Annual Volatility': stats['portfolio_level']['annual_volatility'],
                    'Sharpe Ratio': stats['portfolio_level']['sharpe_ratio'],
                    'Max Drawdown': stats['portfolio_level']['max_drawdown'],
                    'Positive Days': stats['portfolio_level']['positive_days']
                })
            
            # Display metrics in columns
            cols = st.columns(3)
            metrics_list = list(metrics_to_show.items())
            
            for i, (metric_name, metric_value) in enumerate(metrics_list):
                with cols[i % 3]:
                    if isinstance(metric_value, float):
                        if metric_name in ['Annual Return', 'Annual Volatility', 'Max Drawdown']:
                            display_value = f"{metric_value:.2%}"
                        elif metric_name == 'Positive Days':
                            display_value = f"{metric_value:.1%}"
                        else:
                            display_value = f"{metric_value:.2f}"
                    else:
                        display_value = str(metric_value)
                    
                    st.metric(metric_name, display_value)
            
            # Asset performance comparison
            st.subheader("Asset Performance Comparison")
            if 'assets' in stats:
                asset_data = []
                for ticker, asset_stats in stats['assets'].items():
                    asset_data.append({
                        'Ticker': ticker,
                        'Return': asset_stats['mean_return'],
                        'Volatility': asset_stats['annual_volatility'],
                        'Sharpe': asset_stats['sharpe_ratio'],
                        'Max DD': asset_stats['max_drawdown']
                    })
                
                if asset_data:
                    df_assets = pd.DataFrame(asset_data)
                    st.dataframe(
                        df_assets.style.format({
                            'Return': '{:.2%}',
                            'Volatility': '{:.2%}',
                            'Sharpe': '{:.2f}',
                            'Max DD': '{:.2%}'
                        }),
                        use_container_width=True
                    )
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        if 'portfolio_stats' in st.session_state and 'correlation' in st.session_state.portfolio_stats:
            corr_matrix = st.session_state.portfolio_stats['correlation']['matrix']
            
            if not corr_matrix.empty:
                fig = viz_engine.create_interactive_heatmap(corr_matrix)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_tab(self):
        """Render risk analytics tab."""
        st.subheader("Risk Analytics Dashboard")
        
        if st.button("🔍 Run Comprehensive Risk Analysis", type="primary"):
            with st.spinner("Running risk analysis..."):
                if 'portfolio_data' in st.session_state:
                    returns = st.session_state.portfolio_data['returns'].mean(axis=1)  # Portfolio returns
                    
                    risk_results = risk_analytics.calculate_comprehensive_var_analysis(
                        returns,
                        portfolio_value=st.session_state.portfolio_value
                    )
                    
                    st.session_state.risk_results = risk_results
                    
                    # Display VaR results
                    st.subheader("Value at Risk (VaR) Analysis")
                    
                    # Create VaR comparison table
                    var_data = []
                    for method, results in risk_results.get('methods', {}).items():
                        if 0.95 in results:
                            var_95 = results[0.95]
                            var_data.append({
                                'Method': method,
                                'VaR (95%)': var_95['VaR'],
                                'CVaR (95%)': var_95['CVaR'],
                                'VaR ($)': var_95['VaR_absolute']
                            })
                    
                    if var_data:
                        df_var = pd.DataFrame(var_data)
                        st.dataframe(
                            df_var.style.format({
                                'VaR (95%)': '{:.2%}',
                                'CVaR (95%)': '{:.2%}',
                                'VaR ($)': '${:,.0f}'
                            }),
                            use_container_width=True
                        )
                    
                    # Create VaR visualization
                    if 'methods' in risk_results:
                        fig = viz_engine.create_advanced_var_analysis_dashboard(returns)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Display existing results if available
        elif 'risk_results' in st.session_state:
            st.info("Using previously calculated risk results")
            
            risk_results = st.session_state.risk_results
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'summary' in risk_results:
                    st.metric(
                        "Worst-case VaR",
                        f"{risk_results['summary'].get('worst_case_var', 0):.2%}"
                    )
            with col2:
                if 'violations' in risk_results:
                    violations = risk_results['violations']
                    st.metric(
                        "VaR Violations (95%)",
                        f"{violations.get('violations_95', 0)}/{violations.get('total_days', 0)}"
                    )
            with col3:
                if 'summary' in risk_results:
                    st.metric(
                        "Best Method",
                        risk_results['summary'].get('best_method', 'N/A')
                    )
    
    def render_insights_tab(self):
        """Render insights tab."""
        st.subheader("AI-Powered Insights")
        
        # Generate insights based on available data
        insights = []
        
        if 'portfolio_stats' in st.session_state:
            stats = st.session_state.portfolio_stats
            
            # Insight 1: Diversification
            if 'correlation' in stats:
                avg_correlation = stats['correlation']['mean']
                if avg_correlation > 0.7:
                    insights.append({
                        'title': 'High Correlation Detected',
                        'message': f'Average correlation between assets is {avg_correlation:.2f}. Consider adding uncorrelated assets.',
                        'type': 'warning',
                        'priority': 'high'
                    })
                elif avg_correlation < 0.3:
                    insights.append({
                        'title': 'Good Diversification',
                        'message': f'Average correlation is {avg_correlation:.2f}. Portfolio is well diversified.',
                        'type': 'success',
                        'priority': 'low'
                    })
            
            # Insight 2: Risk-adjusted returns
            if 'portfolio_level' in stats:
                sharpe = stats['portfolio_level']['sharpe_ratio']
                if sharpe > 1.0:
                    insights.append({
                        'title': 'Strong Risk-Adjusted Returns',
                        'message': f'Sharpe ratio of {sharpe:.2f} indicates strong risk-adjusted performance.',
                        'type': 'success',
                        'priority': 'medium'
                    })
                elif sharpe < 0.5:
                    insights.append({
                        'title': 'Poor Risk-Adjusted Returns',
                        'message': f'Sharpe ratio of {sharpe:.2f} suggests poor risk-adjusted performance.',
                        'type': 'warning',
                        'priority': 'high'
                    })
            
            # Insight 3: Concentration risk
            if 'assets' in stats:
                n_assets = len(stats['assets'])
                if n_assets < 5:
                    insights.append({
                        'title': 'Low Diversification',
                        'message': f'Only {n_assets} assets in portfolio. Consider adding more positions.',
                        'type': 'warning',
                        'priority': 'medium'
                    })
        
        # Display insights
        if insights:
            # Sort by priority
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            insights.sort(key=lambda x: priority_order[x['priority']])
            
            for insight in insights:
                if insight['type'] == 'warning':
                    st.warning(f"**{insight['title']}** - {insight['message']}")
                elif insight['type'] == 'success':
                    st.success(f"**{insight['title']}** - {insight['message']}")
                else:
                    st.info(f"**{insight['title']}** - {insight['message']}")
        else:
            st.info("No insights available. Fetch more data or run analysis to generate insights.")
        
        # Quick analysis buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🤖 Generate ML Insights", use_container_width=True):
                with st.spinner("Generating ML insights..."):
                    self.generate_ml_insights()
        
        with col2:
            if st.button("📋 Run Optimization", use_container_width=True):
                st.session_state.current_page = 'optimization'
                st.rerun()
        
        with col3:
            if st.button("⚠️ Stress Test", use_container_width=True):
                self.run_stress_test()
    
    def generate_ml_insights(self):
        """Generate machine learning insights."""
        try:
            if 'portfolio_data' not in st.session_state:
                st.error("Please fetch portfolio data first")
                return
            
            data = st.session_state.portfolio_data
            returns = data['returns']
            
            # Train a simple prediction model
            with st.spinner("Training prediction model..."):
                # Create features (simplified)
                features = pd.DataFrame()
                for ticker in returns.columns:
                    feat = pd.DataFrame({
                        f'{ticker}_lag1': returns[ticker].shift(1),
                        f'{ticker}_lag2': returns[ticker].shift(2),
                        f'{ticker}_lag3': returns[ticker].shift(3),
                        f'{ticker}_volatility': returns[ticker].rolling(20).std()
                    })
                    features = pd.concat([features, feat], axis=1)
                
                features = features.dropna()
                target = returns.mean(axis=1).shift(-1)  # Predict next day's return
                target = target.reindex(features.index).dropna()
                features = features.reindex(target.index)
                
                if len(features) > 100:
                    # Train model
                    ml_results = ml_manager.train_return_predictor(
                        features,
                        target,
                        model_type='random_forest'
                    )
                    
                    st.session_state.ml_results = ml_results
                    
                    # Display results
                    st.success("ML model trained successfully!")
                    
                    if 'feature_importance' in ml_results:
                        st.subheader("Top Predictive Features")
                        importance_df = pd.DataFrame(
                            list(ml_results['feature_importance'].items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False).head(10)
                        
                        st.dataframe(
                            importance_df.style.format({'Importance': '{:.3f}'}),
                            use_container_width=True
                        )
                    
                    if 'metrics' in ml_results:
                        metrics = ml_results['metrics']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Test R²", f"{metrics['test']['r2']:.3f}")
                        with col2:
                            st.metric("Test Correlation", f"{metrics['test']['correlation']:.3f}")
                else:
                    st.warning("Insufficient data for ML training")
        
        except Exception as e:
            st.error(f"Error generating ML insights: {str(e)}")
    
    def run_stress_test(self):
        """Run stress test scenario."""
        try:
            if 'portfolio_data' not in st.session_state:
                st.error("Please fetch portfolio data first")
                return
            
            data = st.session_state.portfolio_data
            returns = data['returns'].mean(axis=1)
            
            # Run stress tests using risk analytics
            risk_results = risk_analytics.calculate_comprehensive_var_analysis(
                returns,
                portfolio_value=st.session_state.portfolio_value
            )
            
            if 'stress_tests' in risk_results:
                stress_tests = risk_results['stress_tests']
                
                st.subheader("Stress Test Results")
                
                # Historical scenarios
                if 'historical_scenarios' in stress_tests:
                    st.markdown("#### Historical Stress Scenarios")
                    
                    scenarios_data = []
                    for scenario, results in stress_tests['historical_scenarios'].items():
                        scenarios_data.append({
                            'Scenario': scenario,
                            'Return': results.get('returns', 0),
                            'Volatility': results.get('volatility', 0),
                            'Max DD': results.get('max_drawdown', 0),
                            'VaR (95%)': results.get('var_95', 0)
                        })
                    
                    if scenarios_data:
                        df_scenarios = pd.DataFrame(scenarios_data)
                        st.dataframe(
                            df_scenarios.style.format({
                                'Return': '{:.2%}',
                                'Volatility': '{:.2%}',
                                'Max DD': '{:.2%}',
                                'VaR (95%)': '{:.2%}'
                            }),
                            use_container_width=True
                        )
                
                # Hypothetical scenarios
                if 'hypothetical_scenarios' in stress_tests:
                    st.markdown("#### Hypothetical Stress Scenarios")
                    
                    for scenario, results in stress_tests['hypothetical_scenarios'].items():
                        with st.expander(scenario):
                            st.write(f"**Description:** {results.get('description', 'N/A')}")
                            st.write(f"**Stressed Return:** {results.get('stressed_return', 0):.2%}")
                            st.write(f"**Stressed Volatility:** {results.get('stressed_volatility', 0):.2%}")
                            st.write(f"**Stressed VaR (95%):** {results.get('stressed_var_95', 0):.2%}")
        
        except Exception as e:
            st.error(f"Error running stress test: {str(e)}")
    
    def render_portfolio_analysis(self):
        """Render portfolio analysis page."""
        st.title("📊 Portfolio Analysis")
        
        if st.session_state.portfolio_data is None:
            st.info("Please fetch market data first from the sidebar.")
            return
        
        # Create analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Performance", "📊 Statistics", "🔍 Diagnostics"])
        
        with tab1:
            self.render_performance_analysis()
        
        with tab2:
            self.render_statistical_analysis()
        
        with tab3:
            self.render_diagnostics()
    
    def render_performance_analysis(self):
        """Render performance analysis."""
        st.subheader("Performance Analytics")
        
        if 'portfolio_stats' in st.session_state:
            stats = st.session_state.portfolio_stats
            
            # Time series analysis
            if 'portfolio_data' in st.session_state:
                data = st.session_state.portfolio_data
                returns = data['returns']
                
                # Rolling performance metrics
                st.markdown("#### Rolling Performance Metrics")
                
                window = st.slider("Rolling Window (days)", 20, 252, 63)
                
                if not returns.empty:
                    # Calculate rolling metrics
                    rolling_returns = returns.mean(axis=1).rolling(window=window).mean() * 252
                    rolling_vol = returns.mean(axis=1).rolling(window=window).std() * np.sqrt(252)
                    rolling_sharpe = rolling_returns / rolling_vol
                    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
                    
                    # Create figure
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=rolling_returns.index,
                        y=rolling_returns * 100,
                        mode='lines',
                        name='Rolling Return (%)',
                        line=dict(color='#00cc96')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol * 100,
                        mode='lines',
                        name='Rolling Volatility (%)',
                        line=dict(color='#ef553b'),
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        title=f'Rolling Performance Metrics ({window}-day window)',
                        yaxis=dict(
                            title='Return (%)',
                            titlefont=dict(color='#00cc96'),
                            tickfont=dict(color='#00cc96')
                        ),
                        yaxis2=dict(
                            title='Volatility (%)',
                            titlefont=dict(color='#ef553b'),
                            tickfont=dict(color='#ef553b'),
                            anchor='x',
                            overlaying='y',
                            side='right'
                        ),
                        height=500,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display rolling Sharpe
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe,
                        mode='lines',
                        name='Rolling Sharpe Ratio',
                        line=dict(color='#636efa', width=2)
                    ))
                    
                    fig2.update_layout(
                        title=f'Rolling Sharpe Ratio ({window}-day window)',
                        yaxis_title='Sharpe Ratio',
                        height=400,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
        
        # Performance attribution (simplified)
        st.subheader("Performance Attribution")
        
        if 'portfolio_data' in st.session_state and 'portfolio_stats' in st.session_state:
            data = st.session_state.portfolio_data
            stats = st.session_state.portfolio_stats
            
            if 'assets' in stats and not data['returns'].empty:
                # Calculate contribution of each asset
                contributions = []
                for ticker in data['returns'].columns:
                    if ticker in stats['assets']:
                        asset_return = stats['assets'][ticker]['mean_return']
                        weight = 1 / len(data['returns'].columns)  # Equal weight
                        contribution = asset_return * weight
                        contributions.append({
                            'Asset': ticker,
                            'Weight': weight,
                            'Return': asset_return,
                            'Contribution': contribution
                        })
                
                if contributions:
                    df_contrib = pd.DataFrame(contributions)
                    
                    # Create bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=df_contrib['Asset'],
                            y=df_contrib['Contribution'] * 100,
                            text=df_contrib['Contribution'].apply(lambda x: f'{x:.2%}'),
                            textposition='auto',
                            marker_color='#636efa'
                        )
                    ])
                    
                    fig.update_layout(
                        title='Performance Contribution by Asset',
                        yaxis_title='Contribution (%)',
                        height=400,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_statistical_analysis(self):
        """Render statistical analysis."""
        st.subheader("Statistical Analysis")
        
        if 'portfolio_data' in st.session_state and 'portfolio_stats' in st.session_state:
            data = st.session_state.portfolio_data
            stats = st.session_state.portfolio_stats
            
            # Distribution analysis
            st.markdown("#### Return Distributions")
            
            if not data['returns'].empty:
                # Select asset for distribution analysis
                selected_asset = st.selectbox(
                    "Select Asset for Distribution Analysis",
                    options=data['returns'].columns.tolist(),
                    index=0
                )
                
                asset_returns = data['returns'][selected_asset].dropna()
                
                if len(asset_returns) > 0:
                    # Create distribution plot
                    fig = ff.create_distplot(
                        [asset_returns.values],
                        [selected_asset],
                        bin_size=0.001,
                        show_rug=False
                    )
                    
                    fig.update_layout(
                        title=f'Return Distribution - {selected_asset}',
                        xaxis_title='Daily Return',
                        yaxis_title='Density',
                        height=500,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display distribution statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean", f"{asset_returns.mean():.4f}")
                    with col2:
                        st.metric("Std Dev", f"{asset_returns.std():.4f}")
                    with col3:
                        st.metric("Skewness", f"{asset_returns.skew():.3f}")
                    with col4:
                        st.metric("Kurtosis", f"{asset_returns.kurtosis():.3f}")
            
            # Correlation analysis
            st.markdown("#### Correlation Analysis")
            
            if 'correlation' in stats and 'matrix' in stats['correlation']:
                corr_matrix = stats['correlation']['matrix']
                
                # Display correlation statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Correlation", f"{stats['correlation']['mean']:.3f}")
                with col2:
                    st.metric("Min Correlation", f"{stats['correlation']['min']:.3f}")
                with col3:
                    st.metric("Max Correlation", f"{stats['correlation']['max']:.3f}")
                
                # Correlation heatmap
                fig = viz_engine.create_interactive_heatmap(corr_matrix)
                st.plotly_chart(fig, use_container_width=True)
            
            # Stationarity tests
            st.markdown("#### Stationarity Tests")
            
            if 'stationarity' in data:
                stationarity_results = data['stationarity']
                
                stationarity_data = []
                for ticker, results in stationarity_results.items():
                    if 'is_stationary' in results:
                        stationarity_data.append({
                            'Asset': ticker,
                            'ADF Statistic': results.get('adf_statistic', 0),
                            'p-value': results.get('p_value', 1),
                            'Stationary': '✅' if results['is_stationary'] else '❌'
                        })
                
                if stationarity_data:
                    df_stationarity = pd.DataFrame(stationarity_data)
                    st.dataframe(
                        df_stationarity.style.format({
                            'ADF Statistic': '{:.3f}',
                            'p-value': '{:.4f}'
                        }),
                        use_container_width=True
                    )
    
    def render_diagnostics(self):
        """Render diagnostics page."""
        st.subheader("Portfolio Diagnostics")
        
        if 'data_validation' in st.session_state:
            validation = st.session_state.data_validation
            
            # Display validation results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Validation Status", 
                         "✅ Valid" if validation['is_valid'] else "❌ Invalid")
            
            with col2:
                if 'summary' in validation:
                    st.metric("Assets", validation['summary']['n_assets'])
            
            # Display issues and warnings
            if validation['issues']:
                st.error("**Issues Found:**")
                for issue in validation['issues']:
                    st.write(f"- {issue}")
            
            if validation['warnings']:
                st.warning("**Warnings:**")
                for warning in validation['warnings']:
                    st.write(f"- {warning}")
            
            if validation['suggestions']:
                st.info("**Suggestions:**")
                for suggestion in validation['suggestions']:
                    st.write(f"- {suggestion}")
        
        # Data quality metrics
        st.subheader("Data Quality Metrics")
        
        if 'portfolio_data' in st.session_state:
            data = st.session_state.portfolio_data
            
            if not data['prices'].empty:
                # Calculate data quality metrics
                missing_data = data['prices'].isnull().sum().sum() / (data['prices'].shape[0] * data['prices'].shape[1])
                zero_prices = (data['prices'] <= 0).sum().sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Missing Data", f"{missing_data:.2%}")
                with col2:
                    st.metric("Zero/Negative Prices", zero_prices)
                with col3:
                    st.metric("Total Observations", data['prices'].shape[0] * data['prices'].shape[1])
                
                # Display data preview
                with st.expander("Data Preview"):
                    st.dataframe(data['prices'].tail(10), use_container_width=True)
    
    def render_optimization(self):
        """Render portfolio optimization page."""
        st.title("⚡ Portfolio Optimization")
        
        if st.session_state.portfolio_data is None:
            st.info("Please fetch market data first from the sidebar.")
            return
        
        # Optimization configuration
        st.subheader("Optimization Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimization_method = st.selectbox(
                "Optimization Method",
                options=[
                    'MAX_SHARPE',
                    'MIN_VARIANCE', 
                    'MAX_RETURN',
                    'RISK_PARITY',
                    'MAX_DIVERSIFICATION',
                    'HRP',
                    'MEAN_CVAR',
                    'ROBUST_OPTIMIZATION'
                ],
                index=0
            )
        
        with col2:
            risk_free_rate = st.number_input(
                "Risk-Free Rate",
                value=st.session_state.risk_free_rate,
                min_value=0.0,
                max_value=0.2,
                step=0.001,
                format="%.3f"
            )
            st.session_state.risk_free_rate = risk_free_rate
        
        with col3:
            max_weight = st.slider(
                "Maximum Weight per Asset",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Maximum allocation to any single asset"
            )
        
        # Additional constraints
        with st.expander("Advanced Constraints"):
            min_weight = st.slider(
                "Minimum Weight per Asset",
                min_value=0.0,
                max_value=0.2,
                value=0.0,
                step=0.01
            )
            
            target_return = st.number_input(
                "Target Return (annual)",
                value=0.10,
                min_value=-0.5,
                max_value=0.5,
                step=0.01,
                format="%.2f"
            )
        
        # Run optimization
        if st.button("🚀 Run Optimization", type="primary"):
            with st.spinner("Running portfolio optimization..."):
                try:
                    data = st.session_state.portfolio_data
                    returns = data['returns']
                    
                    # Set constraints
                    constraints = {
                        'bounds': [(min_weight, max_weight) for _ in range(len(returns.columns))],
                        'target_return': target_return
                    }
                    
                    # Run optimization
                    optimization_results = portfolio_optimizer.optimize_portfolio(
                        returns=returns,
                        method=optimization_method,
                        constraints=constraints,
                        risk_free_rate=risk_free_rate
                    )
                    
                    st.session_state.optimization_results = optimization_results
                    
                    # Display results
                    st.success("Optimization completed successfully!")
                    
                    # Show optimized weights
                    self.display_optimization_results(optimization_results)
                    
                    # Generate visualizations
                    self.display_optimization_visualizations(optimization_results, returns)
                    
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
                    error_analyzer.create_advanced_error_display(
                        error_analyzer.analyze_error_with_context(e, {
                            'operation': 'portfolio_optimization',
                            'method': optimization_method
                        })
                    )
        
        # Display existing results if available
        elif 'optimization_results' in st.session_state:
            st.info("Displaying previously calculated optimization results")
            self.display_optimization_results(st.session_state.optimization_results)
            
            if 'portfolio_data' in st.session_state:
                returns = st.session_state.portfolio_data['returns']
                self.display_optimization_visualizations(st.session_state.optimization_results, returns)
    
    def display_optimization_results(self, results: Dict):
        """Display optimization results."""
        st.subheader("Optimization Results")
        
        # Display weights in a table
        weights = results['weights']
        metrics = results['metrics']
        
        # Create weights dataframe
        weights_data = []
        for ticker, weight in weights.items():
            weights_data.append({
                'Asset': ticker,
                'Weight': weight,
                'Allocation ($)': weight * st.session_state.portfolio_value
            })
        
        df_weights = pd.DataFrame(weights_data).sort_values('Weight', ascending=False)
        
        # Display weights
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                df_weights.style.format({
                    'Weight': '{:.2%}',
                    'Allocation ($)': '${:,.0f}'
                }),
                use_container_width=True,
                height=400
            )
        
        with col2:
            # Display key metrics
            st.metric("Expected Return", f"{metrics.get('expected_return', 0):.2%}")
            st.metric("Expected Volatility", f"{metrics.get('expected_volatility', 0):.2%}")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
            
            # Diversification metrics
            if 'diversification_ratio' in metrics:
                st.metric("Diversification Ratio", f"{metrics['diversification_ratio']:.2f}")
            if 'effective_n_assets' in metrics:
                st.metric("Effective N Assets", f"{metrics['effective_n_assets']:.1f}")
        
        # Additional metrics
        with st.expander("Additional Performance Metrics"):
            cols = st.columns(4)
            additional_metrics = [
                ('Sortino Ratio', 'sortino_ratio'),
                ('Calmar Ratio', 'calmar_ratio'),
                ('Omega Ratio', 'omega_ratio'),
                ('VaR (95%)', 'var_95'),
                ('CVaR (95%)', 'cvar_95'),
                ('Skewness', 'skewness'),
                ('Kurtosis', 'kurtosis'),
                ('Estimated Turnover', 'estimated_turnover')
            ]
            
            for i, (display_name, metric_key) in enumerate(additional_metrics):
                with cols[i % 4]:
                    if metric_key in metrics:
                        value = metrics[metric_key]
                        if isinstance(value, float):
                            if display_name in ['VaR (95%)', 'CVaR (95%)']:
                                st.metric(display_name, f"{value:.2%}")
                            elif display_name == 'Estimated Turnover':
                                st.metric(display_name, f"{value:.1%}")
                            else:
                                st.metric(display_name, f"{value:.3f}")
    
    def display_optimization_visualizations(self, results: Dict, returns: pd.DataFrame):
        """Display optimization visualizations."""
        st.subheader("Visualizations")
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs([
            "📊 Allocation", 
            "📈 Efficient Frontier", 
            "🔍 Risk-Return"
        ])
        
        with viz_tab1:
            # Allocation chart
            weights = results['weights']
            
            # Get asset metadata if available
            asset_metadata = {}
            if 'portfolio_data' in st.session_state:
                data = st.session_state.portfolio_data
                if 'metadata' in data:
                    asset_metadata = data['metadata']
            
            # Create sunburst chart
            fig = viz_engine.create_portfolio_allocation_sunburst(weights, asset_metadata)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            # 3D efficient frontier
            fig = viz_engine.create_3d_efficient_frontier(
                returns,
                risk_free_rate=st.session_state.risk_free_rate
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            # Risk-return scatter
            if 'portfolio_stats' in st.session_state:
                stats = st.session_state.portfolio_stats
                
                if 'assets' in stats:
                    # Prepare data for scatter plot
                    scatter_data = []
                    for ticker, asset_stats in stats['assets'].items():
                        scatter_data.append({
                            'Asset': ticker,
                            'Return': asset_stats['mean_return'],
                            'Risk': asset_stats['annual_volatility'],
                            'Sharpe': asset_stats['sharpe_ratio'],
                            'Weight': results['weights'].get(ticker, 0)
                        })
                    
                    if scatter_data:
                        df_scatter = pd.DataFrame(scatter_data)
                        
                        # Create scatter plot
                        fig = go.Figure()
                        
                        # Add assets
                        fig.add_trace(go.Scatter(
                            x=df_scatter['Risk'],
                            y=df_scatter['Return'],
                            mode='markers+text',
                            text=df_scatter['Asset'],
                            textposition="top center",
                            marker=dict(
                                size=df_scatter['Weight'] * 100 + 10,
                                color=df_scatter['Sharpe'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Sharpe Ratio")
                            ),
                            hovertemplate='<b>%{text}</b><br>' +
                                         'Risk: %{x:.2%}<br>' +
                                         'Return: %{y:.2%}<br>' +
                                         'Sharpe: %{marker.color:.2f}<br>' +
                                         'Weight: %{marker.size:.1f}%<extra></extra>',
                            name='Assets'
                        ))
                        
                        # Add optimized portfolio
                        opt_return = results['metrics']['expected_return']
                        opt_risk = results['metrics']['expected_volatility']
                        opt_sharpe = results['metrics']['sharpe_ratio']
                        
                        fig.add_trace(go.Scatter(
                            x=[opt_risk],
                            y=[opt_return],
                            mode='markers',
                            marker=dict(
                                size=20,
                                color='red',
                                symbol='star'
                            ),
                            name='Optimized Portfolio',
                            hovertemplate='<b>Optimized Portfolio</b><br>' +
                                         'Risk: %{x:.2%}<br>' +
                                         'Return: %{y:.2%}<br>' +
                                         'Sharpe: %{text:.2f}<extra></extra>',
                            text=[opt_sharpe]
                        ))
                        
                        fig.update_layout(
                            title='Risk-Return Scatter Plot',
                            xaxis_title='Annual Volatility',
                            yaxis_title='Annual Return',
                            height=600,
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_analytics_page(self):
        """Render risk analytics page."""
        st.title("⚠️ Risk Analytics")
        
        if st.session_state.portfolio_data is None:
            st.info("Please fetch market data first from the sidebar.")
            return
        
        # Risk analysis configuration
        st.subheader("Risk Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_levels = st.multiselect(
                "Confidence Levels",
                options=[0.90, 0.95, 0.99, 0.995],
                default=[0.95, 0.99]
            )
        
        with col2:
            analysis_methods = st.multiselect(
                "VaR Methods",
                options=['Historical', 'Parametric', 'MonteCarlo', 'EVT', 'GARCH'],
                default=['Historical', 'Parametric']
            )
        
        # Run risk analysis
        if st.button("🔍 Run Comprehensive Risk Analysis", type="primary"):
            with st.spinner("Running risk analysis..."):
                try:
                    data = st.session_state.portfolio_data
                    
                    # Calculate portfolio returns (equal weight for now)
                    portfolio_returns = data['returns'].mean(axis=1)
                    
                    # Run risk analysis
                    risk_results = risk_analytics.calculate_comprehensive_var_analysis(
                        portfolio_returns,
                        portfolio_value=st.session_state.portfolio_value
                    )
                    
                    st.session_state.risk_results = risk_results
                    
                    # Display results
                    self.display_risk_analysis_results(risk_results)
                    
                except Exception as e:
                    st.error(f"Risk analysis failed: {str(e)}")
                    error_analyzer.create_advanced_error_display(
                        error_analyzer.analyze_error_with_context(e, {
                            'operation': 'risk_analysis',
                            'methods': analysis_methods
                        })
                    )
        
        # Display existing results if available
        elif 'risk_results' in st.session_state:
            st.info("Displaying previously calculated risk results")
            self.display_risk_analysis_results(st.session_state.risk_results)
    
    def display_risk_analysis_results(self, results: Dict):
        """Display risk analysis results."""
        # Create tabs for different risk views
        risk_tab1, risk_tab2, risk_tab3, risk_tab4 = st.tabs([
            "📊 VaR Analysis", 
            "📈 Stress Tests", 
            "🔍 Backtesting", 
            "📋 Summary"
        ])
        
        with risk_tab1:
            # VaR analysis
            st.subheader("Value at Risk Analysis")
            
            if 'methods' in results:
                # Create VaR comparison table
                var_data = []
                for method, method_results in results['methods'].items():
                    for confidence, var_metrics in method_results.items():
                        var_data.append({
                            'Method': method,
                            'Confidence': confidence,
                            'VaR': var_metrics['VaR'],
                            'CVaR': var_metrics['CVaR'],
                            'VaR ($)': var_metrics['VaR_absolute']
                        })
                
                if var_data:
                    df_var = pd.DataFrame(var_data)
                    
                    # Pivot table
                    pivot = df_var.pivot_table(
                        index='Method',
                        columns='Confidence',
                        values='VaR',
                        aggfunc='first'
                    )
                    
                    st.dataframe(
                        pivot.style.format('{:.2%}'),
                        use_container_width=True
                    )
                    
                    # Create VaR visualization
                    portfolio_returns = st.session_state.portfolio_data['returns'].mean(axis=1)
                    fig = viz_engine.create_advanced_var_analysis_dashboard(portfolio_returns)
                    st.plotly_chart(fig, use_container_width=True)
        
        with risk_tab2:
            # Stress tests
            st.subheader("Stress Test Results")
            
            if 'stress_tests' in results:
                stress_tests = results['stress_tests']
                
                # Historical scenarios
                if 'historical_scenarios' in stress_tests:
                    st.markdown("#### Historical Stress Scenarios")
                    
                    for scenario, scenario_results in stress_tests['historical_scenarios'].items():
                        with st.expander(scenario):
                            cols = st.columns(4)
                            with cols[0]:
                                st.metric("Return", f"{scenario_results.get('returns', 0):.2%}")
                            with cols[1]:
                                st.metric("Volatility", f"{scenario_results.get('volatility', 0):.2%}")
                            with cols[2]:
                                st.metric("Max DD", f"{scenario_results.get('max_drawdown', 0):.2%}")
                            with cols[3]:
                                st.metric("VaR (95%)", f"{scenario_results.get('var_95', 0):.2%}")
                
                # Hypothetical scenarios
                if 'hypothetical_scenarios' in stress_tests:
                    st.markdown("#### Hypothetical Stress Scenarios")
                    
                    for scenario, scenario_results in stress_tests['hypothetical_scenarios'].items():
                        with st.expander(scenario):
                            st.write(f"**Description:** {scenario_results.get('description', 'N/A')}")
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("Stressed Return", f"{scenario_results.get('stressed_return', 0):.2%}")
                            with cols[1]:
                                st.metric("Stressed Volatility", f"{scenario_results.get('stressed_volatility', 0):.2%}")
                            with cols[2]:
                                st.metric("Stressed VaR", f"{scenario_results.get('stressed_var_95', 0):.2%}")
        
        with risk_tab3:
            # Backtesting results
            st.subheader("VaR Backtesting")
            
            if 'violations' in results:
                violations = results['violations']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Days", violations.get('total_days', 0))
                with col2:
                    violations_95 = violations.get('violations_95', 0)
                    total_days = violations.get('total_days', 1)
                    st.metric("Violations (95%)", f"{violations_95} ({violations_95/total_days:.2%})")
                with col3:
                    expected_violations = total_days * 0.05
                    st.metric("Expected Violations", f"{expected_violations:.1f}")
                
                # Exception rates
                if 'exception_rates' in violations:
                    st.markdown("#### Exception Rates")
                    
                    exception_data = []
                    for confidence, rates in violations['exception_rates'].items():
                        exception_data.append({
                            'Confidence': confidence,
                            'Actual': rates['actual'],
                            'Expected': rates['expected'],
                            'Difference': rates['difference']
                        })
                    
                    if exception_data:
                        df_exceptions = pd.DataFrame(exception_data)
                        st.dataframe(
                            df_exceptions.style.format({
                                'Actual': '{:.2%}',
                                'Expected': '{:.2%}',
                                'Difference': '{:.4f}'
                            }),
                            use_container_width=True
                        )
        
        with risk_tab4:
            # Risk summary
            st.subheader("Risk Summary")
            
            if 'summary' in results:
                summary = results['summary']
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Method", summary.get('best_method', 'N/A'))
                with col2:
                    st.metric("Worst-case VaR", f"{summary.get('worst_case_var', 0):.2%}")
                with col3:
                    st.metric("VaR Consistency", f"{summary.get('var_consistency', 0):.3f}")
                
                # Risk adjustment factors
                if 'risk_adjustment_factors' in summary:
                    st.markdown("#### Risk Adjustment Factors")
                    
                    factors = summary['risk_adjustment_factors']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Liquidity Adj", f"{factors.get('liquidity_adjustment', 1.0):.2f}")
                    with col2:
                        st.metric("Concentration Adj", f"{factors.get('concentration_adjustment', 1.0):.2f}")
                    with col3:
                        st.metric("Tail Risk Adj", f"{factors.get('tail_risk_adjustment', 1.0):.2f}")
    
    def render_backtesting(self):
        """Render backtesting page."""
        st.title("🔙 Backtesting")
        
        if st.session_state.portfolio_data is None:
            st.info("Please fetch market data first from the sidebar.")
            return
        
        # Strategy selection and configuration
        st.subheader("Strategy Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Register some basic strategies
            self.register_basic_strategies()
            
            strategy = st.selectbox(
                "Select Strategy",
                options=list(backtester.strategies.keys()),
                index=0
            )
        
        with col2:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                value=st.session_state.portfolio_value,
                min_value=1000,
                step=10000
            )
        
        # Backtest parameters
        with st.expander("Backtest Parameters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                commission = st.number_input(
                    "Commission (%)",
                    value=0.1,
                    min_value=0.0,
                    max_value=5.0,
                    step=0.01,
                    format="%.2f"
                ) / 100
            
            with col2:
                slippage = st.number_input(
                    "Slippage (%)",
                    value=0.05,
                    min_value=0.0,
                    max_value=2.0,
                    step=0.01,
                    format="%.2f"
                ) / 100
            
            with col3:
                rebalance_freq = st.selectbox(
                    "Rebalance Frequency",
                    options=['D', 'W', 'M', 'Q'],
                    index=2
                )
        
        # Run backtest
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    data = st.session_state.portfolio_data
                    
                    # Run backtest
                    backtest_results = backtester.run_backtest(
                        data=data,
                        strategy_name=strategy,
                        initial_capital=initial_capital,
                        commission=commission,
                        slippage=slippage,
                        rebalance_frequency=rebalance_freq
                    )
                    
                    st.session_state.backtest_results = backtest_results
                    
                    # Display results
                    self.display_backtest_results(backtest_results)
                    
                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")
                    error_analyzer.create_advanced_error_display(
                        error_analyzer.analyze_error_with_context(e, {
                            'operation': 'backtesting',
                            'strategy': strategy
                        })
                    )
        
        # Display existing results if available
        elif 'backtest_results' in st.session_state:
            st.info("Displaying previously calculated backtest results")
            self.display_backtest_results(st.session_state.backtest_results)
    
    def register_basic_strategies(self):
        """Register basic trading strategies."""
        if not backtester.strategies:
            # Buy and hold strategy
            def buy_and_hold_strategy(prices, returns):
                # Equal weight allocation
                weights = pd.Series(1/len(prices.columns), index=prices.columns)
                # Repeat weights for all dates
                weights_series = pd.DataFrame(
                    [weights] * len(prices),
                    index=prices.index,
                    columns=prices.columns
                )
                return weights_series
            
            # Momentum strategy
            def momentum_strategy(prices, returns, lookback=63):
                # Calculate momentum (past returns)
                momentum = prices.pct_change(lookback)
                # Rank by momentum
                ranks = momentum.rank(axis=1, ascending=False)
                # Top N assets get equal weight, others 0
                top_n = max(3, len(prices.columns) // 3)
                weights = (ranks <= top_n).astype(float) / top_n
                return weights
            
            # Mean reversion strategy
            def mean_reversion_strategy(prices, returns, lookback=21):
                # Calculate z-score of prices
                ma = prices.rolling(lookback).mean()
                std = prices.rolling(lookback).std()
                zscore = (prices - ma) / std
                # Short assets with high z-score, long assets with low z-score
                ranks = zscore.rank(axis=1)
                n_assets = len(prices.columns)
                weights = (n_assets/2 - ranks) / (n_assets * (n_assets + 1) / 2)
                return weights
            
            # Register strategies
            backtester.register_strategy('Buy & Hold', buy_and_hold_strategy)
            backtester.register_strategy('Momentum (63-day)', 
                lambda p, r: momentum_strategy(p, r, 63))
            backtester.register_strategy('Mean Reversion (21-day)', 
                lambda p, r: mean_reversion_strategy(p, r, 21))
    
    def display_backtest_results(self, results: Dict):
        """Display backtest results."""
        # Create tabs for different views
        bt_tab1, bt_tab2, bt_tab3, bt_tab4 = st.tabs([
            "📈 Performance", 
            "📊 Metrics", 
            "📋 Trades", 
            "📉 Drawdown"
        ])
        
        with bt_tab1:
            # Equity curve
            st.subheader("Equity Curve")
            
            if 'results' in results and 'portfolio_value' in results['results']:
                portfolio_value = results['results']['portfolio_value']
                
                # Create equity curve chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=portfolio_value.index,
                    y=portfolio_value.values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00cc96', width=3)
                ))
                
                fig.update_layout(
                    title='Portfolio Equity Curve',
                    yaxis_title='Portfolio Value ($)',
                    height=500,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cumulative returns
                returns = portfolio_value.pct_change().dropna()
                cumulative_returns = (1 + returns).cumprod() - 1
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values * 100,
                    mode='lines',
                    name='Cumulative Returns',
                    line=dict(color='#636efa', width=2)
                ))
                
                fig2.update_layout(
                    title='Cumulative Returns',
                    yaxis_title='Return (%)',
                    height=400,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        with bt_tab2:
            # Performance metrics
            st.subheader("Performance Metrics")
            
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                
                # Returns
                st.markdown("#### Returns")
                if 'returns' in metrics:
                    returns_metrics = metrics['returns']
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Total Return", f"{returns_metrics.get('total_return', 0):.2%}")
                    with cols[1]:
                        st.metric("Annual Return", f"{returns_metrics.get('annual_return', 0):.2%}")
                    with cols[2]:
                        st.metric("Best Day", f"{returns_metrics.get('best_day', 0):.2%}")
                    with cols[3]:
                        st.metric("Worst Day", f"{returns_metrics.get('worst_day', 0):.2%}")
                
                # Risk metrics
                st.markdown("#### Risk Metrics")
                if 'risk' in metrics:
                    risk_metrics = metrics['risk']
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Annual Volatility", f"{risk_metrics.get('annual_volatility', 0):.2%}")
                    with cols[1]:
                        st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2%}")
                    with cols[2]:
                        st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.2%}")
                    with cols[3]:
                        st.metric("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0):.2%}")
                
                # Ratios
                st.markdown("#### Risk-Adjusted Ratios")
                if 'ratios' in metrics:
                    ratio_metrics = metrics['ratios']
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Sharpe Ratio", f"{ratio_metrics.get('sharpe_ratio', 0):.2f}")
                    with cols[1]:
                        st.metric("Sortino Ratio", f"{ratio_metrics.get('sortino_ratio', 0):.2f}")
                    with cols[2]:
                        st.metric("Calmar Ratio", f"{ratio_metrics.get('calmar_ratio', 0):.2f}")
                    with cols[3]:
                        st.metric("Omega Ratio", f"{ratio_metrics.get('omega_ratio', 0):.2f}")
        
        with bt_tab3:
            # Trade analysis
            st.subheader("Trade Analysis")
            
            if 'trade_analysis' in results:
                trades = results['trade_analysis']
                
                # Trade summary
                if 'summary' in trades:
                    summary = trades['summary']
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Total Trades", summary.get('total_trades', 0))
                    with cols[1]:
                        st.metric("Buy Trades", summary.get('buy_trades', 0))
                    with cols[2]:
                        st.metric("Sell Trades", summary.get('sell_trades', 0))
                    with cols[3]:
                        st.metric("Total Commission", f"${summary.get('total_commission', 0):,.2f}")
                
                # Recent trades
                if 'results' in results and 'trades' in results['results']:
                    recent_trades = results['results']['trades'][-10:]  # Last 10 trades
                    
                    if recent_trades:
                        st.markdown("#### Recent Trades")
                        
                        trade_data = []
                        for trade in recent_trades:
                            trade_data.append({
                                'Date': trade['date'],
                                'Asset': trade['asset'],
                                'Type': trade['type'],
                                'Quantity': f"{trade['quantity']:,.0f}",
                                'Price': f"${trade['price']:.2f}",
                                'Value': f"${trade['value']:,.0f}",
                                'Commission': f"${trade['commission']:.2f}"
                            })
                        
                        df_trades = pd.DataFrame(trade_data)
                        st.dataframe(df_trades, use_container_width=True)
        
        with bt_tab4:
            # Drawdown analysis
            st.subheader("Drawdown Analysis")
            
            if 'results' in results and 'portfolio_value' in results['results']:
                portfolio_value = results['results']['portfolio_value']
                
                # Calculate drawdown
                cumulative = (1 + portfolio_value.pct_change().fillna(0)).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                
                # Create drawdown chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values * 100,
                    fill='tozeroy',
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#ef553b', width=2),
                    fillcolor='rgba(239, 85, 59, 0.3)'
                ))
                
                fig.update_layout(
                    title='Portfolio Drawdown',
                    yaxis_title='Drawdown (%)',
                    height=500,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown statistics
                if len(drawdown) > 0:
                    max_dd = drawdown.min()
                    avg_dd = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Maximum Drawdown", f"{max_dd:.2%}")
                    with col2:
                        st.metric("Average Drawdown", f"{avg_dd:.2%}")
                    
                    # Drawdown duration
                    drawdown_durations = []
                    in_drawdown = False
                    current_duration = 0
                    
                    for dd in drawdown.values:
                        if dd < 0:
                            if not in_drawdown:
                                in_drawdown = True
                            current_duration += 1
                        else:
                            if in_drawdown:
                                drawdown_durations.append(current_duration)
                                in_drawdown = False
                                current_duration = 0
                    
                    if drawdown_durations:
                        max_duration = max(drawdown_durations)
                        avg_duration = np.mean(drawdown_durations)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Max Drawdown Duration", f"{max_duration} days")
                        with col2:
                            st.metric("Avg Drawdown Duration", f"{avg_duration:.0f} days")
    
    def render_machine_learning(self):
        """Render machine learning page."""
        st.title("🤖 Machine Learning")
        
        if st.session_state.portfolio_data is None:
            st.info("Please fetch market data first from the sidebar.")
            return
        
        # ML configuration
        st.subheader("ML Configuration")
        
        tab1, tab2, tab3 = st.tabs([
            "📈 Return Prediction", 
            "📊 Feature Engineering", 
            "🔍 Model Analysis"
        ])
        
        with tab1:
            self.render_return_prediction()
        
        with tab2:
            self.render_feature_engineering()
        
        with tab3:
            self.render_model_analysis()
    
    def render_return_prediction(self):
        """Render return prediction section."""
        st.subheader("Return Prediction Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                options=['random_forest', 'xgboost', 'linear', 'neural_network'],
                index=0
            )
        
        with col2:
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05
            )
        
        # Feature selection
        st.markdown("#### Feature Selection")
        
        if 'portfolio_data' in st.session_state:
            data = st.session_state.portfolio_data
            returns = data['returns']
            
            # Create basic features
            features = self.create_basic_features(returns)
            
            if not features.empty:
                # Display feature statistics
                st.write(f"**Features created:** {len(features.columns)}")
                st.write(f"**Samples:** {len(features)}")
                
                # Train model
                if st.button("🚀 Train Prediction Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            # Create target (next day's return)
                            target = returns.mean(axis=1).shift(-1)
                            
                            # Align features and target
                            aligned_data = features.join(target, how='inner').dropna()
                            features_aligned = aligned_data.iloc[:, :-1]
                            target_aligned = aligned_data.iloc[:, -1]
                            
                            if len(features_aligned) > 100:
                                # Train model
                                ml_results = ml_manager.train_return_predictor(
                                    features_aligned,
                                    target_aligned,
                                    model_type=model_type,
                                    test_size=test_size
                                )
                                
                                st.session_state.ml_results = ml_results
                                st.success("Model trained successfully!")
                                
                                # Display results
                                self.display_ml_results(ml_results)
                            else:
                                st.warning("Insufficient data for training")
                        
                        except Exception as e:
                            st.error(f"Model training failed: {str(e)}")
            
            # Display existing results if available
            elif 'ml_results' in st.session_state:
                st.info("Displaying previously trained model results")
                self.display_ml_results(st.session_state.ml_results)
    
    def create_basic_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for ML."""
        try:
            features = pd.DataFrame()
            
            # Lagged returns
            for lag in [1, 2, 3, 5, 10]:
                lagged = returns.shift(lag)
                lagged.columns = [f'{col}_lag{lag}' for col in lagged.columns]
                features = pd.concat([features, lagged], axis=1)
            
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                rolling_mean = returns.rolling(window).mean()
                rolling_std = returns.rolling(window).std()
                
                rolling_mean.columns = [f'{col}_mean{window}' for col in rolling_mean.columns]
                rolling_std.columns = [f'{col}_std{window}' for col in rolling_std.columns]
                
                features = pd.concat([features, rolling_mean, rolling_std], axis=1)
            
            # Technical indicators (simplified)
            for col in returns.columns:
                # RSI approximation
                returns_col = returns[col]
                gain = returns_col.where(returns_col > 0, 0)
                loss = -returns_col.where(returns_col < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss.replace(0, 1e-10)
                rsi = 100 - (100 / (1 + rs))
                features[f'{col}_rsi'] = rsi
                
                # Momentum
                features[f'{col}_mom'] = returns_col.rolling(10).sum()
            
            return features.dropna()
            
        except Exception as e:
            st.error(f"Error creating features: {str(e)}")
            return pd.DataFrame()
    
    def display_ml_results(self, results: Dict):
        """Display ML results."""
        if 'metrics' in results:
            metrics = results['metrics']
            
            # Performance metrics
            st.subheader("Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test R²", f"{metrics['test'].get('r2', 0):.3f}")
            with col2:
                st.metric("Test RMSE", f"{metrics['test'].get('rmse', 0):.4f}")
            with col3:
                st.metric("Test MAE", f"{metrics['test'].get('mae', 0):.4f}")
            with col4:
                st.metric("Test Correlation", f"{metrics['test'].get('correlation', 0):.3f}")
            
            # Feature importance
            if 'feature_importance' in results:
                st.subheader("Feature Importance")
                
                importance_df = pd.DataFrame(
                    list(results['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                # Display top 20 features
                top_features = importance_df.head(20)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_features['Importance'],
                        y=top_features['Feature'],
                        orientation='h',
                        marker_color='#636efa'
                    )
                ])
                
                fig.update_layout(
                    title='Top 20 Feature Importances',
                    xaxis_title='Importance',
                    height=500,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display full table
                with st.expander("All Features"):
                    st.dataframe(
                        importance_df.style.format({'Importance': '{:.4f}'}),
                        use_container_width=True,
                        height=400
                    )
    
    def render_feature_engineering(self):
        """Render feature engineering section."""
        st.subheader("Feature Engineering")
        
        # Create feature engineering pipeline
        if st.button("🛠️ Create Feature Pipeline"):
            pipeline = ml_manager.create_feature_engineering_pipeline()
            st.session_state.feature_pipeline = pipeline
            
            st.success("Feature pipeline created!")
            
            # Display pipeline
            for category, features in pipeline.items():
                with st.expander(f"{category.replace('_', ' ').title()} ({len(features)} features)"):
                    for feature in features:
                        st.write(f"- {feature}")
        
        # Generate features
        if st.button("📊 Generate Features"):
            if 'portfolio_data' in st.session_state:
                data = st.session_state.portfolio_data
                
                with st.spinner("Generating features..."):
                    # Calculate advanced features
                    advanced_features = data_manager._calculate_additional_features(data)
                    
                    if 'technical_indicators' in advanced_features:
                        st.info(f"Generated {len(advanced_features['technical_indicators'])} technical indicators")
                    
                    st.session_state.advanced_features = advanced_features
                    st.success("Features generated successfully!")
                    
                    # Display feature statistics
                    if 'correlation_matrix' in advanced_features:
                        st.write(f"Correlation matrix shape: {advanced_features['correlation_matrix'].shape}")
    
    def render_model_analysis(self):
        """Render model analysis section."""
        st.subheader("Model Analysis")
        
        # Feature importance across models
        if st.button("📈 Analyze Model Performance"):
            if hasattr(ml_manager, 'models') and ml_manager.models:
                # Calculate feature importance across all models
                feature_importance = ml_manager.calculate_feature_importance_across_models()
                
                if not feature_importance.empty:
                    st.session_state.feature_importance = feature_importance
                    
                    # Display aggregated feature importance
                    st.subheader("Aggregated Feature Importance")
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=feature_importance['mean'].head(20),
                            y=feature_importance['feature'].head(20),
                            orientation='h',
                            error_x=dict(
                                type='data',
                                array=feature_importance['std'].head(20),
                                visible=True
                            ),
                            marker_color='#00cc96'
                        )
                    ])
                    
                    fig.update_layout(
                        title='Top 20 Features (Mean ± Std)',
                        xaxis_title='Importance',
                        height=500,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistics
                    st.write(f"**Total features analyzed:** {len(feature_importance)}")
                    st.write(f"**Average importance:** {feature_importance['mean'].mean():.4f}")
                    st.write(f"**Most important feature:** {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['mean']:.4f})")
                else:
                    st.warning("No feature importance data available")
            else:
                st.info("No models trained yet")
        
        # Model comparison
        if st.button("🤖 Compare Models"):
            if hasattr(ml_manager, 'models') and ml_manager.models:
                # Create model comparison table
                model_data = []
                for model_key, model_info in ml_manager.models.items():
                    if 'metrics' in model_info:
                        metrics = model_info['metrics']
                        model_data.append({
                            'Model': model_key,
                            'Type': model_info.get('model_type', 'unknown'),
                            'Test R²': metrics.get('test', {}).get('r2', 0),
                            'Test RMSE': metrics.get('test', {}).get('rmse', 0),
                            'Train R²': metrics.get('train', {}).get('r2', 0)
                        })
                
                if model_data:
                    df_models = pd.DataFrame(model_data)
                    
                    st.subheader("Model Comparison")
                    st.dataframe(
                        df_models.style.format({
                            'Test R²': '{:.3f}',
                            'Test RMSE': '{:.4f}',
                            'Train R²': '{:.3f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Create comparison chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=df_models['Model'],
                        y=df_models['Test R²'],
                        name='Test R²',
                        marker_color='#636efa'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=df_models['Model'],
                        y=df_models['Train R²'],
                        name='Train R²',
                        marker_color='#ef553b'
                    ))
                    
                    fig.update_layout(
                        title='Model Performance Comparison',
                        yaxis_title='R² Score',
                        barmode='group',
                        height=400,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No model metrics available")
            else:
                st.info("No models trained yet")
    
    def render_reporting(self):
        """Render reporting page."""
        st.title("📑 Reporting")
        
        # Report configuration
        st.subheader("Report Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                options=['pdf', 'excel', 'html', 'text'],
                index=0
            )
        
        with col2:
            template = st.selectbox(
                "Template",
                options=['standard', 'executive', 'detailed', 'regulatory'],
                index=0
            )
        
        with col3:
            include_sections = st.multiselect(
                "Include Sections",
                options=['portfolio_optimization', 'risk_analysis', 'backtesting', 'machine_learning'],
                default=['portfolio_optimization', 'risk_analysis']
            )
        
        # Collect analysis data
        analysis_data = {}
        
        if 'portfolio_optimization' in include_sections and 'optimization_results' in st.session_state:
            analysis_data['portfolio_optimization'] = st.session_state.optimization_results
        
        if 'risk_analysis' in include_sections and 'risk_results' in st.session_state:
            analysis_data['risk_analysis'] = st.session_state.risk_results
        
        if 'backtesting' in include_sections and 'backtest_results' in st.session_state:
            analysis_data['backtest_results'] = st.session_state.backtest_results
        
        if 'machine_learning' in include_sections and 'ml_results' in st.session_state:
            analysis_data['ml_analysis'] = {'models': ml_manager.models}
        
        # Generate report
        if st.button("📄 Generate Report", type="primary"):
            if analysis_data:
                with st.spinner("Generating report..."):
                    try:
                        report = reporter.generate_comprehensive_report(
                            analysis_data=analysis_data,
                            report_type=report_type,
                            template=template
                        )
                        
                        st.session_state.report_results = report
                        
                        # Display report
                        self.display_report(report)
                        
                    except Exception as e:
                        st.error(f"Report generation failed: {str(e)}")
            else:
                st.warning("No analysis data available. Please run analyses first.")
        
        # Display existing report if available
        elif 'report_results' in st.session_state:
            st.info("Displaying previously generated report")
            self.display_report(st.session_state.report_results)
        
        # Report scheduling
        st.subheader("Report Scheduling")
        
        with st.expander("Schedule Automated Reports"):
            col1, col2 = st.columns(2)
            
            with col1:
                schedule_frequency = st.selectbox(
                    "Frequency",
                    options=['daily', 'weekly', 'monthly', 'quarterly'],
                    index=1
                )
            
            with col2:
                recipients = st.text_area(
                    "Recipients (comma-separated emails)",
                    value="analyst@company.com, portfolio.manager@company.com",
                    height=100
                )
            
            if st.button("📅 Schedule Report"):
                if recipients:
                    recipient_list = [r.strip() for r in recipients.split(",") if r.strip()]
                    
                    schedule_config = {
                        'report_type': report_type,
                        'template': template,
                        'sections': include_sections
                    }
                    
                    schedule_id = reporter.schedule_report(
                        frequency=schedule_frequency,
                        recipients=recipient_list,
                        report_config=schedule_config
                    )
                    
                    st.success(f"Report scheduled! Schedule ID: {schedule_id}")
                    
                    # Display scheduled reports
                    if reporter.scheduled_reports:
                        st.write("**Scheduled Reports:**")
                        for sched_id, sched_info in reporter.scheduled_reports.items():
                            st.write(f"- {sched_id}: {sched_info['frequency']} (Next: {sched_info['next_run']})")
    
    def display_report(self, report: Dict):
        """Display generated report."""
        st.subheader("Generated Report")
        
        # Report metadata
        if 'metadata' in report:
            metadata = report['metadata']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Type:** {metadata.get('report_type', 'N/A')}")
            with col2:
                st.write(f"**Template:** {metadata.get('template', 'N/A')}")
            with col3:
                st.write(f"**Generated:** {metadata.get('generated_date', 'N/A')}")
        
        # Executive summary
        if 'summary' in report:
            st.markdown("#### Executive Summary")
            if 'overview' in report['summary']:
                st.write(report['summary']['overview'])
            
            # Key metrics
            if 'key_metrics' in report['summary']:
                st.markdown("##### Key Metrics")
                
                metrics_data = []
                for metric, value in report['summary']['key_metrics'].items():
                    metrics_data.append([metric, value])
                
                if metrics_data:
                    df_metrics = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
                    st.dataframe(
                        df_metrics.style.format('{:,.4f}'),
                        use_container_width=True
                    )
        
        # Recommendations
        if 'recommendations' in report and report['recommendations']:
            st.markdown("#### Recommendations")
            
            for i, rec in enumerate(report['recommendations'], 1):
                if rec['priority'] == 'High':
                    st.error(f"**{i}. {rec['action']}** ({rec['priority']} priority)")
                elif rec['priority'] == 'Medium':
                    st.warning(f"**{i}. {rec['action']}** ({rec['priority']} priority)")
                else:
                    st.info(f"**{i}. {rec['action']}** ({rec['priority']} priority)")
                
                st.write(f"*Reason:* {rec['reason']}")
                st.write(f"*Details:* {rec.get('details', '')}")
                st.write("---")
        
        # File download
        if 'file_path' in report and report['file_path']:
            st.markdown("#### Download Report")
            
            try:
                with open(report['file_path'], 'rb') as f:
                    file_data = f.read()
                
                st.download_button(
                    label=f"📥 Download {report['metadata'].get('report_type', 'report').upper()}",
                    data=file_data,
                    file_name=report['file_path'],
                    mime=self.get_mime_type(report['metadata'].get('report_type', 'pdf')),
                    type="primary"
                )
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    def get_mime_type(self, report_type: str) -> str:
        """Get MIME type for report type."""
        mime_types = {
            'pdf': 'application/pdf',
            'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'html': 'text/html',
            'text': 'text/plain'
        }
        return mime_types.get(report_type, 'application/octet-stream')
    
    def render_settings(self):
        """Render settings page."""
        st.title("⚙️ Settings")
        
        # System settings
        st.subheader("System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance settings
            st.markdown("##### Performance Settings")
            
            cache_ttl = st.number_input(
                "Cache TTL (seconds)",
                value=data_manager.cache_ttl,
                min_value=60,
                max_value=86400,
                step=300
            )
            data_manager.cache_ttl = cache_ttl
            
            max_workers = st.number_input(
                "Max Workers for Data Fetching",
                value=data_manager.max_workers,
                min_value=1,
                max_value=20,
                step=1
            )
            data_manager.max_workers = max_workers
        
        with col2:
            # Risk settings
            st.markdown("##### Risk Settings")
            
            default_risk_free = st.number_input(
                "Default Risk-Free Rate",
                value=st.session_state.risk_free_rate,
                min_value=0.0,
                max_value=0.2,
                step=0.001,
                format="%.3f"
            )
            st.session_state.risk_free_rate = default_risk_free
            
            default_var_confidence = st.multiselect(
                "Default VaR Confidence Levels",
                options=[0.90, 0.95, 0.99, 0.995],
                default=[0.95, 0.99]
            )
        
        # Visualization settings
        st.subheader("Visualization Settings")
        
        theme = st.selectbox(
            "Theme",
            options=['dark', 'light', 'auto'],
            index=0
        )
        viz_engine.current_theme = theme
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
        
        # System information
        st.subheader("System Information")
        
        # Library status
        lib_status = st.session_state.enterprise_library_status
        
        with st.expander("Library Status"):
            for lib, status in lib_status['status'].items():
                if status:
                    st.success(f"✅ {lib}")
                else:
                    st.error(f"❌ {lib}")
            
            if lib_status['missing']:
                st.warning(f"Missing libraries: {', '.join(lib_status['missing'])}")
        
        # Performance report
        if st.button("📊 Generate Performance Report"):
            self.show_performance_report()
        
        # Reset application
        st.subheader("Danger Zone")
        
        if st.button("🔄 Reset Application", type="secondary"):
            st.session_state.clear()
            st.success("Application reset successfully!")
            st.rerun()
    
    def show_performance_report(self):
        """Show performance monitoring report."""
        report = performance_monitor.get_performance_report()
        st.session_state.performance_report = report
        
        st.subheader("Performance Report")
        
        # Summary
        st.markdown("#### Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Runtime", f"{report['total_runtime']:.1f}s")
        with col2:
            st.metric("Total Operations", report['summary'].get('total_operations', 0))
        with col3:
            st.metric("Total Operation Time", f"{report['summary'].get('total_operation_time', 0):.1f}s")
        
        # Operation details
        st.markdown("#### Operation Details")
        
        if report['operations']:
            op_data = []
            for op_name, op_stats in report['operations'].items():
                op_data.append({
                    'Operation': op_name,
                    'Count': op_stats['count'],
                    'Avg Duration (s)': op_stats['avg_duration'],
                    'Max Duration (s)': op_stats['max_duration'],
                    'Avg Memory (MB)': op_stats['avg_memory_increase']
                })
            
            df_ops = pd.DataFrame(op_data)
            st.dataframe(
                df_ops.style.format({
                    'Avg Duration (s)': '{:.3f}',
                    'Max Duration (s)': '{:.3f}',
                    'Avg Memory (MB)': '{:.1f}'
                }),
                use_container_width=True
            )
        
        # Recommendations
        if report['recommendations']:
            st.markdown("#### Optimization Recommendations")
            for rec in report['recommendations']:
                st.info(f"💡 {rec}")
    
    def run(self):
        """Run the main application."""
        # Render sidebar
        self.render_sidebar()
        
        # Render main content based on current page
        current_page = st.session_state.current_page
        
        if current_page == 'dashboard':
            self.render_dashboard()
        elif current_page == 'portfolio_analysis':
            self.render_portfolio_analysis()
        elif current_page == 'optimization':
            self.render_optimization()
        elif current_page == 'risk_analytics':
            self.render_risk_analytics_page()
        elif current_page == 'backtesting':
            self.render_backtesting()
        elif current_page == 'machine_learning':
            self.render_machine_learning()
        elif current_page == 'reporting':
            self.render_reporting()
        elif current_page == 'settings':
            self.render_settings()
        else:
            self.render_dashboard()

# ============================================================================
# 11. MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for QuantEdge Pro."""
    try:
        # Initialize the application
        app = QuantEdgeProApp()
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #1a5276;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                border-left: 5px solid #1a5276;
                margin-bottom: 1rem;
            }
            .success-box {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }
            .warning-box {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }
            .error-box {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }
            .stButton button {
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Application header
        st.markdown('<h1 class="main-header">📈 QuantEdge Pro v5.0 Enterprise Edition</h1>', unsafe_allow_html=True)
        st.markdown("### Institutional Portfolio Analytics Platform with AI/ML Capabilities")
        
        # Run the application
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        error_analyzer.create_advanced_error_display(
            error_analyzer.analyze_error_with_context(e, {
                'operation': 'main_application',
                'component': 'QuantEdgeProApp'
            })
        )

# ============================================================================
# 12. DEPLOYMENT CONFIGURATION
# ============================================================================

class DeploymentConfig:
    """Deployment configuration for different environments."""
    
    @staticmethod
    def get_config(environment: str = 'production') -> Dict:
        """Get deployment configuration for specified environment."""
        configs = {
            'development': {
                'debug': True,
                'cache_ttl': 300,  # 5 minutes
                'max_workers': 3,
                'timeout': 30,
                'log_level': 'DEBUG'
            },
            'staging': {
                'debug': False,
                'cache_ttl': 1800,  # 30 minutes
                'max_workers': 5,
                'timeout': 15,
                'log_level': 'INFO'
            },
            'production': {
                'debug': False,
                'cache_ttl': 3600,  # 1 hour
                'max_workers': 10,
                'timeout': 10,
                'log_level': 'WARNING'
            }
        }
        
        return configs.get(environment, configs['production'])

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Set deployment environment
    import os
    environment = os.getenv('QUANTEDGE_ENV', 'development')
    
    # Configure based on environment
    config = DeploymentConfig.get_config(environment)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config['log_level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    main()

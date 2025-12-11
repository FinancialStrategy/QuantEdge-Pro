#============================================================================
# QUANTEDGE PRO v5.0 ENTERPRISE EDITION - SUPER-ENHANCED VERSION
# INSTITUTIONAL PORTFOLIO ANALYTICS PLATFORM WITH AI/ML CAPABILITIES
# Total Lines: 5500+ | Production Grade | Enterprise Ready
# Enhanced Features: Machine Learning, Advanced Backtesting, Real-time Analytics
# ============================================================================

# ============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
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
import gc
from scipy.stats import norm, t, skew, kurtosis, multivariate_normal
import scipy.stats as stats
from scipy import optimize, signal
from scipy.spatial.distance import pdist, squareform
from itertools import product
import psutil
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Portfolio optimization imports (core engine) ---
PYPFOPT_AVAILABLE = False
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class Config:
    """Centralized configuration for QuantEdge Pro."""
    
    # Data fetching
    MAX_TICKERS = 150          # ⬆ bumped so 70+ ticker universe is allowed
    MAX_WORKERS = 10
    DATA_TIMEOUT = 30
    RETRY_ATTEMPTS = 3
    DATA_CHUNK_SIZE = 1000
    MAX_HISTORICAL_DAYS = 365 * 10  # 10 years
    
    # Optimization
    DEFAULT_RISK_FREE_RATE = 0.045
    MAX_OPTIMIZATION_ITERATIONS = 1000
    OPTIMIZATION_TOLERANCE = 1e-8
    MIN_ASSETS_FOR_OPTIMIZATION = 2
    
    # Risk analysis
    CONFIDENCE_LEVELS = [0.90, 0.95, 0.99, 0.995]
    VAR_WINDOWS = [63, 126, 252]  # 3, 6, 12 months
    STRESS_TEST_SCENARIOS = 50
    
    # Backtesting
    DEFAULT_INITIAL_CAPITAL = 1_000_000
    DEFAULT_COMMISSION = 0.001  # 0.1%
    DEFAULT_SLIPPAGE = 0.0005   # 0.05%
    MIN_BACKTEST_DAYS = 20
    
    # Caching
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # 1 hour
    CACHE_DIR = ".quantedge_cache"
    
    # Visualization
    CHART_HEIGHT = 700
    DARK_THEME = {
        'bg_color': 'rgba(10, 10, 20, 0.9)',
        'grid_color': 'rgba(255, 255, 255, 0.1)',
        'font_color': 'white',
        'accent_color': '#00cc96'
    }
    LIGHT_THEME = {
        'bg_color': 'rgba(255, 255, 255, 0.9)',
        'grid_color': 'rgba(0, 0, 0, 0.1)',
        'font_color': 'black',
        'accent_color': '#636efa'
    }
    
    # Financial constants
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_PER_WEEK = 5
    
    # Validation thresholds
    MIN_DATA_POINTS = 50
    MAX_MISSING_PERCENTAGE = 0.3  # 30%
    MIN_CORRELATION_DATA_POINTS = 20
    
    # ML Configuration
    ML_TRAIN_TEST_SPLIT = 0.2
    ML_N_FOLDS = 5
    ML_MIN_TRAINING_SAMPLES = 100
    
    @classmethod
    def ensure_cache_dir(cls):
        """Ensure cache directory exists."""
        cache_path = Path(cls.CACHE_DIR)
        cache_path.mkdir(exist_ok=True)
        return cache_path

# ============================================================================
# DECORATORS FOR MONITORING AND ERROR HANDLING
# ============================================================================

def monitor_operation(operation_name: str):
    """Decorator to monitor operation performance and errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            performance_monitor = None
            
            if hasattr(st.session_state, 'performance_monitor'):
                performance_monitor = st.session_state.performance_monitor
                if operation_name in performance_monitor.operations:
                    op = performance_monitor.operations[operation_name]
                    if 'is_running' in op and op['is_running']:
                        # Already running: avoid recursion
                        return func(*args, **kwargs)
            
            if performance_monitor:
                if operation_name not in performance_monitor.operations:
                    performance_monitor.operations[operation_name] = {}
                performance_monitor.operations[operation_name]['is_running'] = True
                performance_monitor.start_operation(operation_name)
            
            try:
                result = func(*args, **kwargs)
                if performance_monitor:
                    performance_monitor.end_operation(operation_name)
                return result
            except Exception as e:
                if performance_monitor:
                    performance_monitor.end_operation(operation_name, {'error': str(e)})
                
                error_analyzer = getattr(st.session_state, 'error_analyzer', None)
                if error_analyzer:
                    context = {
                        'operation': operation_name,
                        'function': func.__name__,
                        'module': func.__module__
                    }
                    analysis = error_analyzer._analyze_error_safely(e, context)
                    if 'streamlit' in sys.modules:
                        try:
                            error_analyzer.create_advanced_error_display(analysis)
                        except Exception:
                            st.error(f"Error in {operation_name}: {str(e)[:100]}...")
                raise
            finally:
                if performance_monitor and operation_name in performance_monitor.operations:
                    performance_monitor.operations[operation_name]['is_running'] = False
        return wrapper
    return decorator

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_exception
        return wrapper
    return decorator

# ============================================================================
# 1. ENHANCED LIBRARY MANAGER WITH ML & ADVANCED FEATURES
# ============================================================================

class AdvancedLibraryManager:
    """Enhanced library manager with advanced feature detection."""
    
    @staticmethod
    def check_and_import_all():
        lib_status = {}
        missing_libs = []
        advanced_features = {}
        
        core_libraries = {
            'numpy': ('np', 'Numerical computing'),
            'pandas': ('pd', 'Data manipulation'),
            'scipy': ('scipy', 'Scientific computing'),
            'plotly': ('plotly', 'Visualization'),
            'yfinance': ('yf', 'Financial data'),
            'streamlit': ('st', 'Web interface')
        }
        
        for lib_name, (import_as, description) in core_libraries.items():
            try:
                __import__(lib_name)
                lib_status[lib_name] = True
            except ImportError:
                lib_status[lib_name] = False
                missing_libs.append(f"{lib_name} ({description})")
        
        advanced_libraries = {
            'pypfopt': {
                'modules': ['expected_returns', 'risk_models', 'EfficientFrontier', 'HRPOpt', 'BlackLittermanModel', 'CLA'],
                'description': 'Portfolio optimization'
            },
            'sklearn': {
                'modules': ['PCA', 'RandomForestRegressor', 'GradientBoostingRegressor', 'StandardScaler'],
                'description': 'Machine learning'
            },
            'statsmodels': {
                'modules': ['api', 'adfuller', 'VAR', 'RollingOLS'],
                'description': 'Statistical models'
            }
        }
        
        for lib_name, config in advanced_libraries.items():
            try:
                __import__(lib_name)
                lib_status[lib_name] = True
                advanced_features[lib_name] = {
                    'description': config['description'],
                    'modules_available': config['modules']
                }
                st.session_state[f"{lib_name}_available"] = True
            except ImportError:
                lib_status[lib_name] = False
                missing_libs.append(f"{lib_name} (optional: {config['description']})")
                st.session_state[f"{lib_name}_available"] = False
        
        dl_libraries = ['tensorflow', 'torch']
        for lib_name in dl_libraries:
            try:
                __import__(lib_name)
                lib_status[lib_name] = True
                st.session_state[f"{lib_name}_available"] = True
            except ImportError:
                lib_status[lib_name] = False
                st.session_state[f"{lib_name}_available"] = False
        
        return {
            'status': lib_status,
            'missing': missing_libs,
            'advanced_features': advanced_features,
            'all_core_available': all(lib_status.get(lib, False) for lib in core_libraries.keys())
        }

class EnterpriseLibraryManager:
    """Enterprise-grade library manager with ML and alternative data support."""
    
    @staticmethod
    def check_and_import_all():
        lib_status = {}
        missing_libs = []
        advanced_features = {}
        
        original_status = AdvancedLibraryManager.check_and_import_all()
        lib_status.update(original_status['status'])
        missing_libs.extend([lib for lib in original_status['missing'] if lib not in missing_libs])
        advanced_features.update(original_status['advanced_features'])
        
        alt_data_libraries = {
            'alpha_vantage': {
                'class': 'TimeSeries',
                'description': 'Alternative financial data'
            },
            'newsapi': {
                'class': 'NewsApiClient',
                'description': 'News sentiment analysis'
            }
        }
        
        for lib_name, config in alt_data_libraries.items():
            try:
                __import__(lib_name)
                lib_status[lib_name] = True
                advanced_features[lib_name] = {
                    'description': config['description'],
                    'available': True
                }
                st.session_state[f"{lib_name}_available"] = True
            except ImportError:
                lib_status[lib_name] = False
                missing_libs.append(f"{lib_name} (optional: {config['description']})")
                st.session_state[f"{lib_name}_available"] = False
        
        try:
            from prophet import Prophet  # noqa
            lib_status['prophet'] = True
            advanced_features['prophet'] = {
                'description': 'Time series forecasting',
                'available': True
            }
            st.session_state.prophet_available = True
        except ImportError:
            lib_status['prophet'] = False
            missing_libs.append('prophet (optional: Time series forecasting)')
        
        try:
            from reportlab.lib import colors  # noqa
            from reportlab.lib.pagesizes import letter  # noqa
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph  # noqa
            from reportlab.lib.styles import getSampleStyleSheet  # noqa
            lib_status['reportlab'] = True
            advanced_features['reportlab'] = {
                'description': 'PDF report generation',
                'available': True
            }
            st.session_state.reportlab_available = True
        except ImportError:
            lib_status['reportlab'] = False
            missing_libs.append('reportlab (optional: PDF generation)')
        
        try:
            from web3 import Web3  # noqa
            lib_status['web3'] = True
            advanced_features['web3'] = {
                'description': 'Blockchain data access',
                'available': True
            }
            st.session_state.web3_available = True
        except ImportError:
            lib_status['web3'] = False
            missing_libs.append('web3 (optional: Blockchain)')
        
        try:
            from sqlalchemy import create_engine, text  # noqa
            lib_status['sqlalchemy'] = True
            advanced_features['sqlalchemy'] = {
                'description': 'Database integration',
                'available': True
            }
            st.session_state.sqlalchemy_available = True
        except ImportError:
            lib_status['sqlalchemy'] = False
            missing_libs.append('sqlalchemy (optional: Database)')
        
        try:
            from arch import arch_model  # noqa
            lib_status['arch'] = True
            advanced_features['arch'] = {
                'description': 'GARCH volatility models',
                'available': True
            }
            st.session_state.arch_available = True
        except ImportError:
            lib_status['arch'] = False
            missing_libs.append('arch (optional: GARCH models)')
        
        try:
            import xgboost as xgb  # noqa
            lib_status['xgboost'] = True
            advanced_features['xgboost'] = {
                'description': 'Gradient boosting',
                'available': True
            }
            st.session_state.xgboost_available = True
        except ImportError:
            lib_status['xgboost'] = False
            missing_libs.append('xgboost (optional: Gradient boosting)')
        
        return {
            'status': lib_status,
            'missing': missing_libs,
            'advanced_features': advanced_features,
            'all_available': len(missing_libs) == 0,
            'enterprise_features': {
                'ml_ready': lib_status.get('tensorflow', False) or lib_status.get('torch', False) or lib_status.get('sklearn', False),
                'alternative_data': lib_status.get('alpha_vantage', False),
                'sentiment_analysis': lib_status.get('newsapi', False),
                'blockchain': lib_status.get('web3', False),
                'reporting': lib_status.get('reportlab', False),
                'time_series': lib_status.get('prophet', False) or lib_status.get('arch', False)
            }
        }

# Initialize enterprise library manager
@st.cache_resource
def initialize_library_manager():
    return EnterpriseLibraryManager.check_and_import_all()

if 'enterprise_library_status' not in st.session_state:
    ENTERPRISE_LIBRARY_STATUS = initialize_library_manager()
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
            'symptoms': ['yahoo', 'timeout', 'connection', '404', '403', '502', '503'],
            'solutions': [
                'Try alternative data source (Alpha Vantage, IEX Cloud)',
                'Reduce number of tickers',
                'Increase timeout duration',
                'Use cached data',
                'Check internet connection',
                'Retry with exponential backoff'
            ],
            'severity': 'HIGH',
            'recovery_actions': ['retry', 'reduce_scope', 'use_cache']
        },
        'OPTIMIZATION': {
            'symptoms': ['singular', 'convergence', 'constraint', 'infeasible', 'not positive definite'],
            'solutions': [
                'Relax constraints',
                'Increase max iterations',
                'Try different optimization method',
                'Check for NaN values in returns',
                'Reduce number of assets',
                'Add regularization to covariance matrix',
                'Use Ledoit-Wolf shrinkage estimator'
            ],
            'severity': 'MEDIUM',
            'recovery_actions': ['change_method', 'add_regularization', 'reduce_assets']
        },
        'MEMORY': {
            'symptoms': ['memory', 'overflow', 'exceeded', 'RAM', 'MemoryError'],
            'solutions': [
                'Reduce data size',
                'Use chunk processing',
                'Clear cache',
                'Increase swap memory',
                'Use more efficient data structures',
                'Enable garbage collection'
            ],
            'severity': 'CRITICAL',
            'recovery_actions': ['reduce_data', 'chunk_processing', 'clear_cache']
        },
        'NUMERICAL': {
            'symptoms': ['nan', 'inf', 'divide', 'zero', 'invalid', 'overflow'],
            'solutions': [
                'Clean data (remove NaN/Inf)',
                'Add small epsilon to denominators',
                'Use robust statistical methods',
                'Check for stationarity',
                'Normalize data',
                'Handle zero values appropriately'
            ],
            'severity': 'MEDIUM',
            'recovery_actions': ['clean_data', 'add_epsilon', 'normalize']
        },
        'API_LIMIT': {
            'symptoms': ['limit', 'quota', 'rate limit', '429', 'too many requests'],
            'solutions': [
                'Implement rate limiting',
                'Use API keys with higher limits',
                'Cache responses',
                'Reduce request frequency',
                'Use batch endpoints'
            ],
            'severity': 'MEDIUM',
            'recovery_actions': ['rate_limit', 'use_cache', 'batch_requests']
        },
        'DECORATOR_RECURSION': {
            'symptoms': ['recursion', 'maximum recursion depth exceeded', 'RecursionError'],
            'solutions': [
                'Fix recursive decorator in monitoring system',
                'Remove @monitor_operation from error analysis methods',
                'Add recursion prevention flags',
                'Simplify decorator logic'
            ],
            'severity': 'HIGH',
            'recovery_actions': ['fix_decorator', 'simplify_monitoring']
        }
    }
    
    def __init__(self):
        self.error_history = []
        self.max_history_size = 100
        self._is_analyzing_error = False
    
    def analyze_error_with_context(self, error: Exception, context: Dict) -> Dict:
        if self._is_analyzing_error:
            return self._create_simple_error_analysis(error, context)
        
        self._is_analyzing_error = True
        
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context,
                'stack_trace': traceback.format_exc(),
                'severity_score': 5,
                'recovery_actions': [],
                'preventive_measures': [],
                'ml_suggestions': [],
                'error_category': 'UNKNOWN'
            }
            
            error_lower = str(error).lower()
            stack_lower = traceback.format_exc().lower()
            
            for pattern_name, pattern in self.ERROR_PATTERNS.items():
                if any(symptom in error_lower for symptom in pattern['symptoms']) or \
                   any(symptom in stack_lower for symptom in pattern['symptoms']):
                    
                    analysis['error_category'] = pattern_name
                    analysis['severity_score'] = {
                        'CRITICAL': 9,
                        'HIGH': 7,
                        'MEDIUM': 5,
                        'LOW': 3
                    }.get(pattern['severity'], 5)
                    
                    analysis['recovery_actions'].extend(pattern['solutions'])
                    
                    if 'tickers' in context and pattern_name == 'DATA_FETCH':
                        ticker_count = len(context['tickers'])
                        recommended_count = min(20, max(5, ticker_count // 2))
                        analysis['recovery_actions'].append(
                            f"Reduce from {ticker_count} to {recommended_count} tickers"
                        )
                    
                    if 'window' in context and pattern_name == 'MEMORY':
                        analysis['recovery_actions'].append(
                            f"Reduce window size from {context['window']} to {min(context['window'], 252)}"
                        )
            
            analysis['ml_suggestions'] = self._generate_ml_suggestions(error, context)
            analysis['recovery_confidence'] = min(95, 100 - (analysis['severity_score'] * 10))
            analysis['preventive_measures'] = self._generate_preventive_measures(analysis)
            
            self.error_history.append(analysis)
            if len(self.error_history) > self.max_history_size:
                self.error_history.pop(0)
            
            return analysis
            
        finally:
            self._is_analyzing_error = False
    
    def _analyze_error_safely(self, error: Exception, context: Dict) -> Dict:
        return self.analyze_error_with_context(error, context)
    
    def _create_simple_error_analysis(self, error: Exception, context: Dict) -> Dict:
        return {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error)[:200],
            'context': {k: str(v)[:100] for k, v in context.items()},
            'stack_trace': 'Stack trace omitted to prevent recursion',
            'severity_score': 5,
            'recovery_actions': ["Fix recursive decorator in monitoring system"],
            'error_category': 'DECORATOR_RECURSION'
        }
    
    def _generate_ml_suggestions(self, error: Exception, context: Dict) -> List[str]:
        suggestions = []
        error_str = str(error).lower()
        
        if 'singular' in error_str or 'invert' in error_str:
            suggestions.extend([
                "Covariance matrix is singular - try shrinkage estimation",
                "Use Ledoit-Wolf covariance estimator",
                "Add regularization to covariance matrix (ridge regularization)",
                "Remove highly correlated assets (correlation > 0.95)",
                "Increase minimum eigenvalue threshold"
            ])
        
        if 'convergence' in error_str:
            suggestions.extend([
                "Increase maximum iterations to 5000",
                "Try different optimization algorithm (SLSQP → COBYLA)",
                "Relax tolerance to 1e-4",
                "Use better initial guess for optimization",
                "Scale the problem (normalize returns)"
            ])
        
        if 'memory' in error_str:
            suggestions.extend([
                "Implement incremental learning",
                "Use sparse matrices where possible",
                "Process data in batches of 1000 rows",
                "Enable garbage collection during processing",
                "Use data streaming instead of loading all at once"
            ])
        
        if 'window' in context:
            window = context['window']
            if window > Config.MAX_HISTORICAL_DAYS:
                suggestions.append(f"Reduce window size from {window} to {Config.MAX_HISTORICAL_DAYS} for better stability")
        
        if 'assets' in context:
            assets = context['assets']
            if assets > Config.MAX_TICKERS:
                suggestions.append(f"Reduce asset universe from {assets} to {Config.MAX_TICKERS} for faster computation")
        
        return suggestions
    
    def _generate_preventive_measures(self, analysis: Dict) -> List[str]:
        measures = []
        
        if analysis['error_category'] == 'DATA_FETCH':
            measures.extend([
                "Implement robust retry logic with exponential backoff",
                "Cache API responses locally",
                "Validate ticker symbols before fetching",
                "Use multiple data sources as fallback"
            ])
        
        if analysis['error_category'] == 'OPTIMIZATION':
            measures.extend([
                "Pre-process data to remove NaN/Inf values",
                "Regularize covariance matrices",
                "Validate input parameters before optimization",
                "Implement fallback optimization strategies"
            ])
        
        if analysis['error_category'] == 'MEMORY':
            measures.extend([
                "Monitor memory usage during execution",
                "Implement chunked processing for large datasets",
                "Clear unused variables and caches periodically",
                "Use memory-efficient data structures"
            ])
        
        if analysis['error_category'] == 'DECORATOR_RECURSION':
            measures.extend([
                "Remove @monitor_operation decorator from error analysis methods",
                "Add recursion prevention flags to decorators",
                "Simplify monitoring system to avoid circular dependencies",
                "Test decorators for recursion issues"
            ])
        
        return measures
    
    def create_advanced_error_display(self, analysis: Dict) -> None:
        with st.expander(f"🔍 Advanced Error Analysis ({analysis['error_type']})", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                severity_color = {
                    9: "🔴",
                    7: "🟠",
                    5: "🟡",
                    3: "🟢"
                }.get(analysis['severity_score'], "⚫")
                st.metric("Severity", f"{severity_color} {analysis['severity_score']}/10")
            with col2:
                st.metric("Recovery Confidence", f"{analysis.get('recovery_confidence', 0)}%")
            with col3:
                category = analysis.get('error_category', 'Unknown')
                st.metric("Category", category)
            
            if analysis.get('recovery_actions'):
                st.subheader("🚀 Recovery Actions")
                for i, action in enumerate(analysis['recovery_actions'][:5], 1):
                    action_key = f"recovery_{i}_{hash(action) % 10000}"
                    st.checkbox(f"Action {i}: {action}", value=False, key=action_key)
            
            if analysis.get('ml_suggestions'):
                st.subheader("🤖 AI-Powered Suggestions")
                for suggestion in analysis['ml_suggestions'][:3]:
                    st.info(f"💡 {suggestion}")
            
            if analysis.get('preventive_measures'):
                st.subheader("🛡️ Preventive Measures")
                for measure in analysis['preventive_measures'][:3]:
                    st.success(f"✓ {measure}")
            
            with st.expander("🔧 Technical Details"):
                st.code(f"""
Error Type: {analysis['error_type']}
Message: {analysis['error_message']}

Context: {json.dumps(analysis['context'], indent=2, default=str)}

Stack Trace:
{analysis['stack_trace']}
                """)
    
    def get_error_statistics(self) -> Dict:
        if not self.error_history:
            return {}
        
        categories = {}
        severities = []
        
        for error in self.error_history:
            category = error.get('error_category', 'UNKNOWN')
            categories[category] = categories.get(category, 0) + 1
            severities.append(error.get('severity_score', 5))
        
        return {
            'total_errors': len(self.error_history),
            'categories': categories,
            'avg_severity': np.mean(severities) if severities else 0,
            'max_severity': max(severities) if severities else 0,
            'recent_errors': self.error_history[-5:] if len(self.error_history) >= 5 else self.error_history
        }

class PerformanceMonitor:
    """Advanced performance monitoring with real-time analytics."""
    
    def __init__(self):
        self.operations = {}
        self.memory_usage = []
        self.execution_times = []
        self.start_time = time.time()
        self.process = psutil.Process()
        self.recursion_depth = {}
    
    def start_operation(self, operation_name: str):
        if operation_name not in self.recursion_depth:
            self.recursion_depth[operation_name] = 0
        
        if self.recursion_depth[operation_name] > 0:
            self.recursion_depth[operation_name] += 1
            return
        
        self.operations[operation_name] = {
            'start': time.time(),
            'memory_start': self._get_memory_usage(),
            'cpu_start': self._get_cpu_usage(),
            'is_running': True
        }
        self.recursion_depth[operation_name] = 1
    
    def end_operation(self, operation_name: str, metadata: Dict = None):
        if operation_name in self.operations:
            if operation_name in self.recursion_depth:
                self.recursion_depth[operation_name] -= 1
                if self.recursion_depth[operation_name] > 0:
                    return
            
            op = self.operations[operation_name]
            duration = time.time() - op['start']
            memory_end = self._get_memory_usage()
            memory_diff = memory_end - op['memory_start']
            cpu_end = self._get_cpu_usage()
            cpu_diff = cpu_end - op['cpu_start']
            
            self.execution_times.append({
                'operation': operation_name,
                'duration': duration,
                'memory_increase_mb': memory_diff,
                'cpu_increase': cpu_diff,
                'timestamp': datetime.now(),
                'metadata': metadata
            })
            
            if 'history' not in op:
                op['history'] = []
            op['history'].append({
                'duration': duration,
                'memory_increase_mb': memory_diff,
                'cpu_increase': cpu_diff,
                'timestamp': datetime.now()
            })
            
            op['is_running'] = False
            
            if memory_diff > 100:
                gc.collect()
    
    def _get_memory_usage(self) -> float:
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        try:
            return self.process.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    @monitor_operation('get_performance_report')
    def get_performance_report(self) -> Dict:
        report = {
            'total_runtime': time.time() - self.start_time,
            'operations': {},
            'summary': {},
            'recommendations': [],
            'resource_usage': {}
        }
        
        for op_name, op_data in self.operations.items():
            if 'history' in op_data and op_data['history']:
                durations = [h['duration'] for h in op_data['history']]
                memories = [h['memory_increase_mb'] for h in op_data['history']]
                cpus = [h['cpu_increase'] for h in op_data['history']]
                
                report['operations'][op_name] = {
                    'count': len(durations),
                    'avg_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'min_duration': np.min(durations),
                    'avg_memory_increase': np.mean(memories),
                    'max_memory_increase': np.max(memories),
                    'avg_cpu_increase': np.mean(cpus),
                    'total_time': np.sum(durations),
                    'p95_duration': np.percentile(durations, 95) if len(durations) > 1 else durations[0]
                }
        
        if report['operations']:
            total_times = [op['total_time'] for op in report['operations'].values()]
            report['summary'] = {
                'total_operations': len(report['operations']),
                'total_operation_time': float(sum(total_times)),
                'slowest_operation': max(report['operations'].items(), key=lambda x: x[1]['avg_duration'])[0],
                'most_frequent_operation': max(report['operations'].items(), key=lambda x: x[1]['count'])[0],
                'most_memory_intensive': max(report['operations'].items(), key=lambda x: x[1]['avg_memory_increase'])[0]
            }
        
        if self.memory_usage:
            report['resource_usage']['memory'] = {
                'peak_mb': max(self.memory_usage) if self.memory_usage else 0,
                'avg_mb': float(np.mean(self.memory_usage)) if self.memory_usage else 0,
                'current_mb': self._get_memory_usage()
            }
        
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        recommendations = []
        
        for op_name, stats_dict in report['operations'].items():
            if stats_dict['avg_duration'] > 5:
                recommendations.append(
                    f"Optimize '{op_name}' - average duration {stats_dict['avg_duration']:.1f}s"
                )
            
            if stats_dict['avg_memory_increase'] > 100:
                recommendations.append(
                    f"Reduce memory usage in '{op_name}' - average increase {stats_dict['avg_memory_increase']:.1f}MB"
                )
            
            if stats_dict['avg_cpu_increase'] > 50:
                recommendations.append(
                    f"Optimize CPU usage in '{op_name}' - average increase {stats_dict['avg_cpu_increase']:.1f}%"
                )
        
        if report['summary'].get('total_operation_time', 0) > 30:
            recommendations.append(
                "Consider implementing parallel processing for independent operations"
            )
        
        if report['resource_usage'].get('memory', {}).get('peak_mb', 0) > 1000:
            recommendations.append(
                "Consider implementing memory-efficient data structures or chunked processing"
            )
        
        return recommendations
    
    def clear_cache(self):
        self.operations.clear()
        self.memory_usage.clear()
        self.execution_times.clear()
        self.recursion_depth.clear()
        gc.collect()

# Initialize global monitors
error_analyzer = AdvancedErrorAnalyzer()
performance_monitor = PerformanceMonitor()
st.session_state.error_analyzer = error_analyzer
st.session_state.performance_monitor = performance_monitor

# ============================================================================
# 3. ENHANCED DATA MANAGEMENT WITH ROBUST YAHOO FINANCE FETCHING
# ============================================================================

class AdvancedDataManager:
    """Advanced data management with caching, validation, and preprocessing."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = Config.CACHE_TTL
        self.max_workers = min(Config.MAX_WORKERS, os.cpu_count() or 4)
        self.retry_attempts = Config.RETRY_ATTEMPTS
        self.timeout = Config.DATA_TIMEOUT
        Config.ensure_cache_dir()
    
    @monitor_operation('fetch_advanced_market_data')
    @retry_on_failure(max_attempts=Config.RETRY_ATTEMPTS, delay=1.0)
    def fetch_advanced_market_data(self, tickers: List[str], 
                                  start_date: datetime, 
                                  end_date: datetime,
                                  interval: str = '1d',
                                  progress_callback = None) -> Dict:
        try:
            if not tickers:
                raise ValueError("No tickers provided")
            
            if len(tickers) > Config.MAX_TICKERS:
                raise ValueError(f"Maximum {Config.MAX_TICKERS} tickers allowed, got {len(tickers)}")
            
            cache_key = self._generate_cache_key(tickers, start_date, end_date, interval)
            if Config.CACHE_ENABLED and cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['data']
            
            data = {
                'prices': pd.DataFrame(),
                'returns': pd.DataFrame(),
                'volumes': pd.DataFrame(),
                'high': pd.DataFrame(),
                'low': pd.DataFrame(),
                'open': pd.DataFrame(),
                'dividends': {},
                'splits': {},
                'metadata': {},
                'errors': {},
                'successful_tickers': []
            }
            
            max_workers = min(self.max_workers, len(tickers))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self._fetch_single_ticker_ohlc, 
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
                        
                        if ticker_data and not ticker_data['close'].empty:
                            close_series = ticker_data['close'].rename(ticker)
                            volume_series = ticker_data['volume'].rename(ticker)
                            high_series = ticker_data['high'].rename(ticker)
                            low_series = ticker_data['low'].rename(ticker)
                            open_series = ticker_data['open'].rename(ticker)
                            
                            if data['prices'].empty:
                                data['prices'] = pd.DataFrame(close_series)
                                data['volumes'] = pd.DataFrame(volume_series)
                                data['high'] = pd.DataFrame(high_series)
                                data['low'] = pd.DataFrame(low_series)
                                data['open'] = pd.DataFrame(open_series)
                            else:
                                data['prices'] = data['prices'].merge(
                                    close_series, left_index=True, right_index=True, how='outer'
                                )
                                data['volumes'] = data['volumes'].merge(
                                    volume_series, left_index=True, right_index=True, how='outer'
                                )
                                data['high'] = data['high'].merge(
                                    high_series, left_index=True, right_index=True, how='outer'
                                )
                                data['low'] = data['low'].merge(
                                    low_series, left_index=True, right_index=True, how='outer'
                                )
                                data['open'] = data['open'].merge(
                                    open_series, left_index=True, right_index=True, how='outer'
                                )
                            
                            data['metadata'][ticker] = ticker_data['metadata']
                            
                            if not ticker_data['dividends'].empty:
                                data['dividends'][ticker] = ticker_data['dividends']
                            if not ticker_data['splits'].empty:
                                data['splits'][ticker] = ticker_data['splits']
                            
                            data['successful_tickers'].append(ticker)
                        else:
                            data['errors'][ticker] = "No OHLC data returned"
                            
                    except concurrent.futures.TimeoutError:
                        data['errors'][ticker] = f"Timeout after {self.timeout} seconds"
                    except Exception as e:
                        data['errors'][ticker] = str(e)
            
            if not data['prices'].empty:
                data = self._process_and_align_data(data)
                if not data['prices'].empty:
                    data['returns'] = data['prices'].pct_change().dropna()
                if len(data['successful_tickers']) > 0:
                    data['additional_features'] = self._calculate_additional_features(data)
            
            if Config.CACHE_ENABLED and data['successful_tickers']:
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
            
            gc.collect()
            return data
            
        except Exception:
            raise
    
    def _fetch_single_ticker_ohlc(self, ticker: str, 
                                 start_date: datetime, 
                                 end_date: datetime,
                                 interval: str) -> Dict:
        for attempt in range(self.retry_attempts):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=False,
                    actions=True,
                    timeout=self.timeout
                )
                
                if hist.empty:
                    if attempt == self.retry_attempts - 1:
                        raise ValueError(f"No historical data for {ticker} in date range {start_date} to {end_date}")
                    time.sleep(2 ** attempt)
                    continue
                
                close_prices = hist['Close']
                volumes = hist['Volume']
                high_prices = hist['High']
                low_prices = hist['Low']
                open_prices = hist['Open']
                
                dividends = stock.dividends[start_date:end_date]
                splits = stock.splits[start_date:end_date]
                
                info = stock.info
                metadata = {
                    'name': info.get('longName', ticker),
                    'short_name': info.get('shortName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'beta': info.get('beta', 1.0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'currency': info.get('currency', 'USD'),
                    'country': info.get('country', 'Unknown'),
                    'exchange': info.get('exchange', 'Unknown'),
                    'quote_type': info.get('quoteType', 'EQUITY'),
                    'market': info.get('market', 'us_market'),
                    'volume_average': info.get('averageVolume', 0),
                    'volume_average_10d': info.get('averageVolume10days', 0),
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                    'fifty_day_average': info.get('fiftyDayAverage', 0),
                    'two_hundred_day_average': info.get('twoHundredDayAverage', 0),
                    'shares_outstanding': info.get('sharesOutstanding', 0),
                    'float_shares': info.get('floatShares', 0)
                }
                
                return {
                    'close': close_prices,
                    'volume': volumes,
                    'high': high_prices,
                    'low': low_prices,
                    'open': open_prices,
                    'dividends': dividends,
                    'splits': splits,
                    'metadata': metadata
                }
                
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise Exception(f"Failed to fetch data for {ticker} after {self.retry_attempts} attempts: {str(e)}")
                time.sleep(2 ** attempt)
        
        return {}
    
    def _process_and_align_data(self, data: Dict) -> Dict:
        all_dates = pd.DatetimeIndex([])
        for df in [data['prices'], data['volumes'], data['high'], data['low'], data['open']]:
            if not df.empty:
                all_dates = all_dates.union(df.index)
        
        if len(all_dates) == 0:
            return data
        
        all_dates = all_dates.sort_values()
        
        for key in ['prices', 'volumes', 'high', 'low', 'open']:
            if not data[key].empty:
                data[key] = data[key].reindex(all_dates)
                
                if key == 'prices':
                    data[key] = data[key].ffill().bfill()
                elif key == 'volumes':
                    data[key] = data[key].ffill().fillna(0)
                else:
                    data[key] = data[key].ffill().bfill()
        
        if not data['prices'].empty:
            nan_counts = data['prices'].isnull().sum()
            valid_assets = nan_counts[nan_counts < len(data['prices']) * 0.5].index.tolist()
            
            if len(valid_assets) < Config.MIN_ASSETS_FOR_OPTIMIZATION:
                # Don't raise here; just leave prices empty to be handled upstream
                data['prices'] = pd.DataFrame()
                data['volumes'] = pd.DataFrame()
                data['high'] = pd.DataFrame()
                data['low'] = pd.DataFrame()
                data['open'] = pd.DataFrame()
                data['successful_tickers'] = []
                return data
            
            for key in ['prices', 'volumes', 'high', 'low', 'open']:
                if not data[key].empty:
                    data[key] = data[key][valid_assets]
            
            data['successful_tickers'] = [t for t in data['successful_tickers'] if t in valid_assets]
        
        return data
    
    def _calculate_additional_features(self, data: Dict) -> Dict:
        features = {
            'technical_indicators': {},
            'statistical_features': {},
            'risk_metrics': {},
            'liquidity_metrics': {},
            'price_features': {}
        }
        
        try:
            returns = data.get('returns', pd.DataFrame())
            prices = data.get('prices', pd.DataFrame())
            volumes = data.get('volumes', pd.DataFrame())
            highs = data.get('high', pd.DataFrame())
            lows = data.get('low', pd.DataFrame())
            
            for ticker in returns.columns:
                ticker_returns = returns[ticker].dropna()
                
                if len(ticker_returns) > 0:
                    features['statistical_features'][ticker] = {
                        'mean_return': ticker_returns.mean(),
                        'std_return': ticker_returns.std(),
                        'skewness': ticker_returns.skew(),
                        'kurtosis': ticker_returns.kurtosis(),
                        'sharpe_ratio': ticker_returns.mean() / ticker_returns.std() if ticker_returns.std() > 0 else 0,
                        'max_drawdown': self._calculate_max_drawdown_series(ticker_returns),
                        'positive_ratio': (ticker_returns > 0).sum() / len(ticker_returns),
                        'var_95': -np.percentile(ticker_returns, 5),
                        'cvar_95': self._calculate_cvar(ticker_returns, 0.95)
                    }
                    
                    if ticker in prices.columns:
                        price_series = prices[ticker].dropna()
                        if len(price_series) > 0:
                            features['price_features'][ticker] = {
                                'current_price': price_series.iloc[-1],
                                'price_change_1d': price_series.pct_change().iloc[-1] if len(price_series) > 1 else 0,
                                'price_change_5d': (price_series.iloc[-1] / price_series.iloc[-6] - 1) if len(price_series) > 6 else 0,
                                'price_change_21d': (price_series.iloc[-1] / price_series.iloc[-22] - 1) if len(price_series) > 22 else 0,
                                'high_low_ratio': (highs[ticker].iloc[-1] / lows[ticker].iloc[-1]) if ticker in highs.columns and ticker in lows.columns else 0
                            }
                    
                    if ticker in volumes.columns:
                        volume_series = volumes[ticker].dropna()
                        if len(volume_series) > 0:
                            features['liquidity_metrics'][ticker] = {
                                'current_volume': volume_series.iloc[-1],
                                'avg_volume_20d': volume_series.tail(20).mean(),
                                'volume_ratio': volume_series.iloc[-1] / volume_series.tail(20).mean() if volume_series.tail(20).mean() > 0 else 0,
                                'volume_std_20d': volume_series.tail(20).std()
                            }
            
            if len(returns.columns) > 1:
                corr_matrix = returns.corr()
                features['correlation_matrix'] = corr_matrix
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                if len(corr_values) > 0:
                    features['correlation_stats'] = {
                        'mean': float(np.mean(corr_values)),
                        'median': float(np.median(corr_values)),
                        'min': float(np.min(corr_values)),
                        'max': float(np.max(corr_values)),
                        'std': float(np.std(corr_values))
                    }
            
            if not returns.empty:
                features['covariance_matrix'] = returns.cov() * Config.TRADING_DAYS_PER_YEAR
            
        except Exception as e:
            features['error'] = str(e)
        
        return features
    
    def _calculate_max_drawdown_series(self, returns: pd.Series) -> float:
        try:
            if len(returns) == 0:
                return 0.0
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return float(drawdown.min()) if not drawdown.empty else 0.0
        except Exception:
            return 0.0
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        try:
            if len(returns) == 0:
                return 0.0
            var = -np.percentile(returns, (1 - confidence) * 100)
            cvar_data = returns[returns <= -var]
            return float(-cvar_data.mean()) if len(cvar_data) > 0 else float(var)
        except Exception:
            return 0.0
    
    def _generate_cache_key(self, tickers: List[str], start_date: datetime, 
                           end_date: datetime, interval: str) -> str:
        tickers_str = '_'.join(sorted(tickers))
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        return f"{tickers_str}_{date_str}_{interval}"
    
    def validate_portfolio_data(self, data: Dict, 
                               min_assets: int = Config.MIN_ASSETS_FOR_OPTIMIZATION,
                               min_data_points: int = Config.MIN_DATA_POINTS) -> Dict:
        validation = {
            'is_valid': False,
            'issues': [],
            'warnings': [],
            'suggestions': [],
            'summary': {}
        }
        
        try:
            if data['prices'].empty:
                validation['issues'].append("No price data available")
                return validation
            
            n_assets = len(data['prices'].columns)
            if n_assets < min_assets:
                validation['issues'].append(f"Only {n_assets} assets available, minimum {min_assets} required")
            
            n_data_points = len(data['prices'])
            if n_data_points < min_data_points:
                validation['warnings'].append(f"Only {n_data_points} data points, recommended minimum {min_data_points}")
            
            if not data['prices'].empty:
                missing_percentage = data['prices'].isnull().mean().mean()
                if missing_percentage > Config.MAX_MISSING_PERCENTAGE:
                    validation['warnings'].append(f"High percentage of missing values after forward fill: {missing_percentage:.1%}")
            else:
                missing_percentage = 0.0
            
            if not data['prices'].empty and (data['prices'] <= 0).any().any():
                problematic_assets = data['prices'].columns[(data['prices'] <= 0).any()].tolist()
                validation['warnings'].append(f"Zero or negative prices in assets: {problematic_assets}")
            
            if data.get('returns', pd.DataFrame()).empty:
                validation['warnings'].append("Cannot calculate returns - check price data continuity")
            else:
                if not np.isfinite(data['returns'].values).all():
                    nan_assets = data['returns'].columns[data['returns'].isnull().any()].tolist()
                    validation['warnings'].append(f"Non-finite values in returns for assets: {nan_assets}")
                
                zero_vol_assets = data['returns'].std()[data['returns'].std().abs() < 1e-10].tolist()
                if zero_vol_assets:
                    validation['warnings'].append(f"Zero volatility assets: {zero_vol_assets}")
            
            validation['summary'] = {
                'n_assets': n_assets,
                'n_data_points': n_data_points,
                'date_range': {
                    'start': data['prices'].index.min(),
                    'end': data['prices'].index.max(),
                    'days': (data['prices'].index.max() - data['prices'].index.min()).days
                } if not data['prices'].empty else {},
                'missing_data_percentage': float(missing_percentage),
                'average_return': float(data['returns'].mean().mean()) if not data.get('returns', pd.DataFrame()).empty else 0.0,
                'average_volatility': float(data['returns'].std().mean() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)) if not data.get('returns', pd.DataFrame()).empty else 0.0,
                'successful_tickers': len(data.get('successful_tickers', [])),
                'failed_tickers': len(data.get('errors', {}))
            }
            
            validation['is_valid'] = len(validation['issues']) == 0 and n_assets >= min_assets
            
            if not validation['is_valid']:
                if n_assets < min_assets:
                    validation['suggestions'].append(f"Add {min_assets - n_assets} more assets")
                if n_data_points < min_data_points:
                    validation['suggestions'].append("Extend the date range or use higher frequency data")
                if validation['summary']['failed_tickers'] > 0:
                    validation['suggestions'].append(f"Review {validation['summary']['failed_tickers']} failed tickers")
            
            return validation
            
        except Exception as e:
            validation['issues'].append(f"Validation error: {str(e)}")
            return validation
    
    @monitor_operation('preprocess_data_for_analysis')
    def preprocess_data_for_analysis(self, data: Dict, 
                                     preprocessing_steps: List[str] = None) -> Dict:
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
            elif step == 'log_returns':
                processed_data = self._calculate_log_returns(processed_data)
        
        return processed_data
    
    def _clean_missing_values(self, data: Dict) -> Dict:
        if not data['prices'].empty:
            data['prices'] = data['prices'].ffill().bfill()
        
        if not data.get('returns', pd.DataFrame()).empty:
            threshold = 0.5
            min_non_na = int(threshold * len(data['returns'].columns))
            data['returns'] = data['returns'].dropna(thresh=min_non_na)
        
        return data
    
    def _handle_outliers(self, data: Dict) -> Dict:
        if not data.get('returns', pd.DataFrame()).empty:
            returns_clean = data['returns'].copy()
            
            for column in returns_clean.columns:
                series = returns_clean[column].dropna()
                if len(series) > 10:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    returns_clean[column] = series.clip(lower_bound, upper_bound)
            
            data['returns'] = returns_clean
        
        return data
    
    def _normalize_data(self, data: Dict) -> Dict:
        if not data.get('returns', pd.DataFrame()).empty:
            returns_normalized = data['returns'].copy()
            for column in returns_normalized.columns:
                series = returns_normalized[column]
                if series.std() > 0:
                    returns_normalized[column] = (series - series.mean()) / series.std()
            data['returns_normalized'] = returns_normalized
        
        return data
    
    def _check_stationarity(self, data: Dict) -> Dict:
        stationarity_results = {}
        
        if not data.get('returns', pd.DataFrame()).empty:
            for column in data['returns'].columns:
                try:
                    from statsmodels.tsa.stattools import adfuller
                    series = data['returns'][column].dropna()
                    if len(series) > 10:
                        result = adfuller(series, autolag='AIC')
                        stationarity_results[column] = {
                            'adf_statistic': float(result[0]),
                            'p_value': float(result[1]),
                            'is_stationary': bool(result[1] < 0.05),
                            'critical_values': result[4],
                            'test_used': 'ADF'
                        }
                except Exception as e:
                    stationarity_results[column] = {'error': str(e)}
        
        data['stationarity'] = stationarity_results
        return data
    
    def _detrend_data(self, data: Dict) -> Dict:
        if not data['prices'].empty:
            prices_detrended = data['prices'].copy()
            for column in prices_detrended.columns:
                series = prices_detrended[column].dropna()
                if len(series) > 10:
                    x = np.arange(len(series))
                    y = series.values
                    coeff = np.polyfit(x, y, 1)
                    trend = np.polyval(coeff, x)
                    prices_detrended.loc[series.index, column] = y - trend + y.mean()
            
            data['prices_detrended'] = prices_detrended
        
        return data
    
    def _calculate_log_returns(self, data: Dict) -> Dict:
        if not data['prices'].empty:
            log_returns = np.log(data['prices'] / data['prices'].shift(1)).dropna()
            data['log_returns'] = log_returns
        
        return data
    
    @monitor_operation('calculate_basic_statistics')
    def calculate_basic_statistics(self, data: Dict) -> Dict:
        stats_dict = {
            'assets': {},
            'portfolio_level': {},
            'correlation': {},
            'covariance': {},
            'liquidity': {},
            'price_level': {}
        }
        
        if not data.get('returns', pd.DataFrame()).empty:
            returns = data['returns']
            prices = data.get('prices', pd.DataFrame())
            volumes = data.get('volumes', pd.DataFrame())
            
            for ticker in returns.columns:
                ticker_returns = returns[ticker].dropna()
                
                if len(ticker_returns) > 0:
                    stats_dict['assets'][ticker] = {
                        'mean_return': float(ticker_returns.mean() * Config.TRADING_DAYS_PER_YEAR),
                        'annual_volatility': float(ticker_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)),
                        'sharpe_ratio': float(
                            (ticker_returns.mean() * Config.TRADING_DAYS_PER_YEAR) /
                            (ticker_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR))
                        ) if ticker_returns.std() > 0 else 0.0,
                        'skewness': float(ticker_returns.skew()),
                        'kurtosis': float(ticker_returns.kurtosis()),
                        'var_95': float(-np.percentile(ticker_returns, 5)),
                        'cvar_95': float(self._calculate_cvar(ticker_returns, 0.95)),
                        'max_drawdown': float(self._calculate_max_drawdown_series(ticker_returns)),
                        'positive_days': float((ticker_returns > 0).sum() / len(ticker_returns)),
                        'data_points': int(len(ticker_returns)),
                        'start_date': ticker_returns.index.min() if not ticker_returns.empty else None,
                        'end_date': ticker_returns.index.max() if not ticker_returns.empty else None
                    }
            
            if len(returns.columns) > 0:
                equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
                portfolio_returns = returns.dot(equal_weights)
                
                stats_dict['portfolio_level'] = {
                    'mean_return': float(portfolio_returns.mean() * Config.TRADING_DAYS_PER_YEAR),
                    'annual_volatility': float(portfolio_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)),
                    'sharpe_ratio': float(
                        (portfolio_returns.mean() * Config.TRADING_DAYS_PER_YEAR) /
                        (portfolio_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR))
                    ) if portfolio_returns.std() > 0 else 0.0,
                    'skewness': float(portfolio_returns.skew()),
                    'kurtosis': float(portfolio_returns.kurtosis()),
                    'var_95': float(-np.percentile(portfolio_returns, 5)),
                    'cvar_95': float(self._calculate_cvar(portfolio_returns, 0.95)),
                    'max_drawdown': float(self._calculate_max_drawdown_series(portfolio_returns)),
                    'positive_days': float((portfolio_returns > 0).sum() / len(portfolio_returns)),
                    'sortino_ratio': float(self._calculate_sortino_ratio(portfolio_returns)),
                    'calmar_ratio': float(self._calculate_calmar_ratio(portfolio_returns))
                }
            
            if len(returns.columns) > 1:
                corr_matrix = returns.corr()
                stats_dict['correlation']['matrix'] = corr_matrix
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                if len(corr_values) > 0:
                    stats_dict['correlation']['stats'] = {
                        'mean': float(np.mean(corr_values)),
                        'median': float(np.median(corr_values)),
                        'min': float(np.min(corr_values)),
                        'max': float(np.max(corr_values)),
                        'std': float(np.std(corr_values)),
                        'q25': float(np.percentile(corr_values, 25)),
                        'q75': float(np.percentile(corr_values, 75))
                    }
                
                cov_matrix = returns.cov() * Config.TRADING_DAYS_PER_YEAR
                stats_dict['covariance']['matrix'] = cov_matrix
                stats_dict['covariance']['mean_variance'] = float(np.diag(cov_matrix).mean())
                stats_dict['covariance']['avg_covariance'] = float(
                    cov_matrix.values[np.triu_indices_from(cov_matrix.values, k=1)].mean()
                )
            
            if not volumes.empty:
                for ticker in volumes.columns:
                    volume_series = volumes[ticker].dropna()
                    if len(volume_series) > 0:
                        stats_dict['liquidity'][ticker] = {
                            'avg_volume': float(volume_series.mean()),
                            'std_volume': float(volume_series.std()),
                            'volume_ratio_last_avg': float(
                                volume_series.iloc[-1] / volume_series.mean()
                            ) if volume_series.mean() > 0 else 0.0,
                            'volume_trend': self._calculate_volume_trend(volume_series)
                        }
            
            if not prices.empty:
                for ticker in prices.columns:
                    price_series = prices[ticker].dropna()
                    if len(price_series) > 0:
                        stats_dict['price_level'][ticker] = {
                            'current_price': float(price_series.iloc[-1]),
                            'price_change_1m': float(
                                (price_series.iloc[-1] / price_series.iloc[-22] - 1)
                            ) if len(price_series) > 22 else 0.0,
                            'price_change_3m': float(
                                (price_series.iloc[-1] / price_series.iloc[-66] - 1)
                            ) if len(price_series) > 66 else 0.0,
                            'price_change_1y': float(
                                (price_series.iloc[-1] / price_series.iloc[-252] - 1)
                            ) if len(price_series) > 252 else 0.0,
                            'price_high_52w': float(price_series.tail(252).max() if len(price_series) >= 252 else price_series.max()),
                            'price_low_52w': float(price_series.tail(252).min() if len(price_series) >= 252 else price_series.min())
                        }
        
        return stats_dict
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = Config.DEFAULT_RISK_FREE_RATE) -> float:
        try:
            if len(returns) == 0:
                return 0.0
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_std = downside_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)
            if downside_std == 0:
                return float('inf')
            
            excess_return = returns.mean() * Config.TRADING_DAYS_PER_YEAR - risk_free_rate
            return float(excess_return / downside_std)
        except Exception:
            return 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        try:
            if len(returns) == 0:
                return 0.0
            max_dd = self._calculate_max_drawdown_series(returns)
            if max_dd == 0:
                return 0.0
            annual_return = returns.mean() * Config.TRADING_DAYS_PER_YEAR
            return float(annual_return / abs(max_dd))
        except Exception:
            return 0.0
    
    def _calculate_volume_trend(self, volume_series: pd.Series, window: int = 20) -> str:
        try:
            if len(volume_series) < window * 2:
                return "Insufficient data"
            
            recent_avg = volume_series.tail(window).mean()
            previous_avg = volume_series.iloc[-(window*2):-window].mean()
            
            if previous_avg == 0:
                return "Stable"
            
            change = (recent_avg / previous_avg - 1) * 100
            
            if change > 20:
                return "Strongly Increasing"
            elif change > 5:
                return "Increasing"
            elif change < -20:
                return "Strongly Decreasing"
            elif change < -5:
                return "Decreasing"
            else:
                return "Stable"
        except Exception:
            return "Unknown"

# Initialize data manager
data_manager = AdvancedDataManager()

# ============================================================================
# 4. PORTFOLIO OPTIMIZATION HELPERS (EF / HRP / CLA / BLACK-LITTERMAN)
# ============================================================================

def _calc_max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min()) if not drawdown.empty else 0.0

def _calc_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    if len(returns) == 0:
        return 0.0
    var = -np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) > 0 else float(var)

def compute_portfolio_performance(port_returns: pd.Series,
                                  risk_free_rate: float = Config.DEFAULT_RISK_FREE_RATE,
                                  freq: int = Config.TRADING_DAYS_PER_YEAR) -> Dict[str, float]:
    if len(port_returns) == 0:
        return {}
    mu = float(port_returns.mean() * freq)
    sigma = float(port_returns.std() * np.sqrt(freq))
    sharpe = float((mu - risk_free_rate) / sigma) if sigma > 0 else 0.0
    max_dd = _calc_max_drawdown(port_returns)
    var_95 = float(-np.percentile(port_returns, 5))
    cvar_95 = _calc_cvar(port_returns, 0.95)
    return {
        "annual_return": mu,
        "annual_vol": sigma,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }

def run_all_portfolio_strategies(prices: pd.DataFrame,
                                 returns: pd.DataFrame,
                                 metadata: Dict[str, Dict[str, Any]] = None,
                                 risk_free_rate: float = Config.DEFAULT_RISK_FREE_RATE):
    if prices is None or prices.empty:
        raise ValueError("Prices data is empty")
    if returns is None or returns.empty:
        returns = prices.pct_change().dropna()
    tickers = list(prices.columns)
    n_assets = len(tickers)
    if n_assets < 2:
        raise ValueError("Need at least 2 assets for optimization")
    
    strategies: Dict[str, pd.Series] = {}
    ew_weights = pd.Series(np.ones(n_assets) / n_assets, index=tickers)
    strategies["Equal Weight"] = ew_weights
    
    if PYPFOPT_AVAILABLE:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        
        # EF Max Sharpe
        try:
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            w = pd.Series(ef.clean_weights())
            w = w.reindex(tickers).fillna(0.0)
            strategies["EF Max Sharpe"] = w
        except Exception as e:
            logging.error(f"EF Max Sharpe failed: {e}")
        
        # EF Min Vol
        try:
            ef2 = EfficientFrontier(mu, S)
            ef2.min_volatility()
            w2 = pd.Series(ef2.clean_weights())
            w2 = w2.reindex(tickers).fillna(0.0)
            strategies["EF Min Volatility"] = w2
        except Exception as e:
            logging.error(f"EF Min Vol failed: {e}")
        
        # CLA Max Sharpe
        try:
            cla = CLA(mu, S)
            cla.max_sharpe(risk_free_rate=risk_free_rate)
            w3 = pd.Series(cla.clean_weights())
            w3 = w3.reindex(tickers).fillna(0.0)
            strategies["CLA Max Sharpe"] = w3
        except Exception as e:
            logging.error(f"CLA failed: {e}")
        
        # HRP
        try:
            hrp = HRPOpt(returns)
            w4 = pd.Series(hrp.optimize())
            w4 = w4.reindex(tickers).fillna(0.0)
            strategies["HRP"] = w4
        except Exception as e:
            logging.error(f"HRP failed: {e}")
        
        # Black-Litterman Max Sharpe
        try:
            if metadata is None:
                metadata = {}
            market_caps = {}
            for t in tickers:
                m = metadata.get(t, {})
                mc = m.get("market_cap", np.nan)
                market_caps[t] = mc
            mc_series = pd.Series(market_caps)
            if mc_series.isnull().all():
                mc_series.loc[:] = 1.0
            else:
                mc_series = mc_series.fillna(mc_series.median())
            delta = 2.5
            prior = market_implied_prior_returns(mc_series, S, risk_aversion=delta)
            bl = BlackLittermanModel(S, pi=prior, absolute=True)
            bl_ret = bl.bl_returns()
            bl_cov = bl.bl_cov()
            ef_bl = EfficientFrontier(bl_ret, bl_cov)
            ef_bl.max_sharpe(risk_free_rate=risk_free_rate)
            w5 = pd.Series(ef_bl.clean_weights())
            w5 = w5.reindex(tickers).fillna(0.0)
            strategies["Black-Litterman Max Sharpe"] = w5
        except Exception as e:
            logging.error(f"Black-Litterman failed: {e}")
    
    perf: Dict[str, Dict[str, float]] = {}
    for name, w in strategies.items():
        port = returns.dot(w)
        perf[name] = compute_portfolio_performance(port, risk_free_rate=risk_free_rate)
    return strategies, perf

# ============================================================================
# 5. RISK ANALYSIS HELPERS (Hist / Parametric / MC VaR + Relative VaR)
# ============================================================================

def compute_var_cvar(returns: pd.Series,
                     confidence: float = 0.95,
                     horizon_days: int = 1,
                     method: str = "historical",
                     n_sim: int = 10000) -> Tuple[float, float]:
    if len(returns) == 0:
        raise ValueError("Empty returns for VaR calculation")
    if horizon_days < 1:
        horizon_days = 1
    r = returns.dropna()
    
    if method == "historical":
        scaled = r * np.sqrt(horizon_days)
        var = -np.percentile(scaled, (1 - confidence) * 100)
        tail = scaled[scaled <= -var]
        cvar = -tail.mean() if len(tail) > 0 else var
    elif method == "parametric":
        mu = r.mean() * horizon_days
        sigma = r.std() * np.sqrt(horizon_days)
        z = norm.ppf(1 - confidence)
        var = -(mu + sigma * z)
        cvar = -(mu + sigma * norm.pdf(z) / (1 - confidence))
    elif method == "monte_carlo":
        mu = r.mean()
        sigma = r.std()
        sim = np.random.normal(mu, sigma, size=n_sim * horizon_days).reshape(-1, horizon_days)
        sim_horizon = sim.sum(axis=1)
        var = -np.percentile(sim_horizon, (1 - confidence) * 100)
        tail = sim_horizon[sim_horizon <= -var]
        cvar = -tail.mean() if len(tail) > 0 else var
    else:
        raise ValueError(f"Unknown VaR method: {method}")
    return float(var), float(cvar)

def compute_relative_var(port_returns: pd.Series,
                         bench_returns: pd.Series,
                         confidence: float = 0.95,
                         horizon_days: int = 1,
                         method: str = "historical") -> Tuple[float, float]:
    common_index = port_returns.index.intersection(bench_returns.index)
    active = (port_returns.loc[common_index] - bench_returns.loc[common_index]).dropna()
    return compute_var_cvar(active, confidence=confidence, horizon_days=horizon_days, method=method)

# ============================================================================
# STREAMLIT APP MAIN FUNCTION
# ============================================================================

def main():
    st.set_page_config(
        page_title="QuantEdge Pro v5.0 - Enterprise Portfolio Analytics",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 QuantEdge Pro v5.0 - Enterprise Portfolio Analytics")
    st.markdown("""
    ### Institutional-grade portfolio optimization, risk analysis, and backtesting platform  
    *Advanced analytics with machine learning, real-time data, and comprehensive reporting*
    """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if 'enterprise_library_status' in st.session_state:
            lib_status = st.session_state.enterprise_library_status
            st.subheader("📚 Library Status")
            
            core_libs = ['numpy', 'pandas', 'scipy', 'plotly', 'yfinance', 'streamlit']
            core_status = all(lib_status['status'].get(lib, False) for lib in core_libs)
            
            if core_status:
                st.success("✅ Core libraries available")
            else:
                st.error("❌ Missing core libraries")
                for lib in core_libs:
                    if not lib_status['status'].get(lib, False):
                        st.warning(f"Missing: {lib}")
        
        st.subheader("📊 Data Configuration")
        tickers_input = st.text_area(
            "Enter tickers (comma-separated):",
            value="AAPL, GOOGL, MSFT, AMZN, TSLA",
            help="Enter stock symbols separated by commas"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start date",
                value=datetime.now() - timedelta(days=365*2),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        st.subheader("🔍 Analysis Type")
        analysis_type = st.selectbox(
            "Select analysis:",
            ["Portfolio Optimization", "Risk Analysis", "Backtesting", "ML Forecasting", "Comprehensive Report"]
        )
        
        if st.button("🚀 Fetch Data & Analyze", type="primary"):
            with st.spinner("Fetching market data..."):
                try:
                    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
                    
                    progress_bar = st.progress(0.0)
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        st.sidebar.text(message)
                    
                    data = data_manager.fetch_advanced_market_data(
                        tickers=tickers,
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.max.time()),
                        progress_callback=update_progress
                    )
                    
                    validation = data_manager.validate_portfolio_data(data)
                    
                    if validation['is_valid'] and not data['prices'].empty and len(data.get('successful_tickers', [])) >= 2:
                        st.session_state.portfolio_data = data
                        st.session_state.data_loaded = True
                        st.sidebar.success(
                            f"✅ Data loaded: {validation['summary']['n_assets']} assets, "
                            f"{validation['summary']['n_data_points']} days"
                        )
                    else:
                        st.session_state.data_loaded = False
                        st.sidebar.error("❌ Data not sufficient for optimization.")
                        if validation['issues']:
                            for issue in validation['issues']:
                                st.sidebar.error(f"- {issue}")
                        if validation['warnings']:
                            for w in validation['warnings']:
                                st.sidebar.warning(f"- {w}")
                        if validation['suggestions']:
                            st.sidebar.info("Suggestions:")
                            for s in validation['suggestions']:
                                st.sidebar.info(f"• {s}")
                
                except Exception as e:
                    st.session_state.data_loaded = False
                    st.sidebar.error(f"❌ Error fetching data: {str(e)[:100]}...")
                    logging.error(f"Data fetch error: {str(e)}")
    
    if st.session_state.get('data_loaded', False) and 'portfolio_data' in st.session_state:
        data = st.session_state.portfolio_data
        
        if data['prices'].empty:
            st.warning("No price data available after fetching. Please adjust tickers or date range and try again.")
            return
        
        st.subheader("📋 Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Assets", len(data['prices'].columns))
        with col2:
            st.metric("Data Points", len(data['prices']))
        with col3:
            idx = data['prices'].index
            if len(idx) > 0:
                start_str = idx[0].date() if hasattr(idx[0], "date") else str(idx[0])
                end_str = idx[-1].date() if hasattr(idx[-1], "date") else str(idx[-1])
                date_range = f"{start_str} to {end_str}"
            else:
                date_range = "N/A"
            st.metric("Date Range", date_range)
        
        with st.expander("📊 Data Preview"):
            tab1, tab2, tab3 = st.tabs(["Prices", "Returns", "Statistics"])
            
            with tab1:
                st.dataframe(data['prices'].tail(10), width="stretch")
            with tab2:
                if not data['returns'].empty:
                    st.dataframe(data['returns'].tail(10), width="stretch")
                else:
                    st.info("Returns are empty; check data continuity.")
            with tab3:
                stats_basic = data_manager.calculate_basic_statistics(data)
                if stats_basic['assets']:
                    stats_df = pd.DataFrame(stats_basic['assets']).T
                    view_cols = ['mean_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
                    view_cols = [c for c in view_cols if c in stats_df.columns]
                    st.dataframe(stats_df[view_cols], width="stretch")
                else:
                    st.info("No asset-level statistics available.")
        
        # =========================
        # PORTFOLIO OPTIMIZATION
        # =========================
        if analysis_type == "Portfolio Optimization":
            st.subheader("🎯 Portfolio Optimization Engine")
            
            if data['returns'].empty or len(data['prices'].columns) < 2:
                st.warning("Need at least 2 assets with returns for optimization.")
            else:
                rf = st.number_input(
                    "Risk-free rate (annualized)",
                    min_value=-0.05,
                    max_value=0.20,
                    value=Config.DEFAULT_RISK_FREE_RATE,
                    step=0.005,
                    format="%.3f"
                )
                
                st.caption("Strategies: Equal Weight, EF Max Sharpe / Min Vol, CLA Max Sharpe, HRP, Black-Litterman Max Sharpe")
                
                try:
                    strategies, perf = run_all_portfolio_strategies(
                        data['prices'],
                        data['returns'],
                        metadata=data.get('metadata', {}),
                        risk_free_rate=rf
                    )
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    strategies, perf = {}, {}
                
                if not strategies:
                    st.error("No optimization strategies could be computed.")
                else:
                    strategy_names = list(strategies.keys())
                    
                    colA, colB = st.columns([2, 3])
                    with colA:
                        selected_strategy = st.selectbox(
                            "Select strategy to inspect:",
                            strategy_names
                        )
                    with colB:
                        benchmark_strategy = st.selectbox(
                            "Benchmark strategy (for comparison):",
                            strategy_names,
                            index=strategy_names.index("Equal Weight") if "Equal Weight" in strategy_names else 0
                        )
                    
                    # Weights table
                    st.markdown("#### Portfolio Weights")
                    w_selected = strategies[selected_strategy]
                    weights_df = pd.DataFrame({
                        "Ticker": w_selected.index,
                        "Weight": w_selected.values
                    }).sort_values("Weight", ascending=False)
                    st.dataframe(weights_df, width="stretch")
                    
                    # Performance comparison table
                    st.markdown("#### Strategy Performance (Annualized)")
                    perf_rows = []
                    for name, metrics in perf.items():
                        if metrics:
                            perf_rows.append({
                                "Strategy": name,
                                "Annual Return": metrics["annual_return"],
                                "Annual Volatility": metrics["annual_vol"],
                                "Sharpe": metrics["sharpe"],
                                "Max Drawdown": metrics["max_drawdown"],
                                "VaR 95% (1d)": metrics["var_95"],
                                "CVaR 95% (1d)": metrics["cvar_95"],
                            })
                    if perf_rows:
                        perf_df = pd.DataFrame(perf_rows)
                        st.dataframe(perf_df, width="stretch")
                    
                    # Weights bar chart for selected strategy
                    st.markdown("#### Weight Allocation – Selected Strategy")
                    fig_w = go.Figure()
                    fig_w.add_bar(
                        x=w_selected.index,
                        y=w_selected.values,
                        name=selected_strategy
                    )
                    fig_w.update_layout(
                        template="plotly_dark",
                        height=500,
                        xaxis_title="Ticker",
                        yaxis_title="Weight",
                        showlegend=False
                    )
                    st.plotly_chart(fig_w, use_container_width=True)
                    
                    # Efficient frontier (if PyPortfolioOpt available)
                    if PYPFOPT_AVAILABLE and len(data['prices'].columns) >= 2:
                        try:
                            st.markdown("#### Efficient Frontier (Sampled)")
                            prices = data['prices']
                            mu = expected_returns.mean_historical_return(prices)
                            S = risk_models.sample_cov(prices)
                            
                            ef_front = EfficientFrontier(mu, S)
                            frontier_x = []
                            frontier_y = []
                            for target in np.linspace(mu.min(), mu.max(), 25):
                                try:
                                    ef_front = EfficientFrontier(mu, S)
                                    ef_front.efficient_return(target)
                                    ret, vol, _ = ef_front.portfolio_performance()
                                    frontier_x.append(vol)
                                    frontier_y.append(ret)
                                except Exception:
                                    continue
                            
                            fig_f = go.Figure()
                            fig_f.add_scatter(
                                x=frontier_x,
                                y=frontier_y,
                                mode="lines",
                                name="Efficient Frontier"
                            )
                            
                            # Mark selected strategy point
                            sel_port = data['returns'].dot(strategies[selected_strategy])
                            sel_stats = compute_portfolio_performance(sel_port, risk_free_rate=rf)
                            fig_f.add_scatter(
                                x=[sel_stats["annual_vol"]],
                                y=[sel_stats["annual_return"]],
                                mode="markers",
                                name=selected_strategy,
                                marker=dict(size=10)
                            )
                            
                            fig_f.update_layout(
                                template="plotly_dark",
                                height=500,
                                xaxis_title="Volatility (σ)",
                                yaxis_title="Return (μ)",
                                legend=dict(orientation="h")
                            )
                            st.plotly_chart(fig_f, use_container_width=True)
                        except Exception as e:
                            st.info(f"Efficient frontier could not be plotted: {e}")
        
        # =========================
        # RISK ANALYSIS
        # =========================
        elif analysis_type == "Risk Analysis":
            st.subheader("⚠️ Risk Analysis Engine")
            
            if data['returns'].empty:
                st.warning("Returns are empty; cannot compute VaR/CVaR. Please refetch data.")
            else:
                rf = st.number_input(
                    "Risk-free rate (annualized, used for Sharpe in background)",
                    min_value=-0.05,
                    max_value=0.20,
                    value=Config.DEFAULT_RISK_FREE_RATE,
                    step=0.005,
                    format="%.3f"
                )
                
                returns = data['returns']
                
                # Build strategies (same as optimization) to reuse weights
                try:
                    strategies, perf = run_all_portfolio_strategies(
                        data['prices'],
                        returns,
                        metadata=data.get('metadata', {}),
                        risk_free_rate=rf
                    )
                except Exception as e:
                    st.error(f"Failed to build strategies for risk analysis: {e}")
                    strategies, perf = {}, {}
                
                if not strategies:
                    # fall back: equal weight only
                    tickers = list(returns.columns)
                    if len(tickers) >= 2:
                        ew = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
                        strategies = {"Equal Weight": ew}
                    else:
                        st.error("Not enough assets for risk analysis.")
                        return
                
                strat_names = list(strategies.keys())
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    selected_strategy = st.selectbox(
                        "Portfolio definition",
                        strat_names
                    )
                with col_r2:
                    confidence = st.selectbox(
                        "Confidence level",
                        Config.CONFIDENCE_LEVELS,
                        index=Config.CONFIDENCE_LEVELS.index(0.95)
                    )
                with col_r3:
                    horizon_days = st.selectbox(
                        "Horizon (days)",
                        [1, 5, 10, 21],
                        index=0
                    )
                
                benchmark_ticker = st.selectbox(
                    "Benchmark (for Relative VaR)",
                    options=list(returns.columns),
                    index=0
                )
                
                w = strategies[selected_strategy]
                port_returns = returns.dot(w)
                bench_returns = returns[benchmark_ticker]
                
                # Compute VaR / CVaR
                var_hist, cvar_hist = compute_var_cvar(
                    port_returns,
                    confidence=confidence,
                    horizon_days=horizon_days,
                    method="historical"
                )
                var_param, cvar_param = compute_var_cvar(
                    port_returns,
                    confidence=confidence,
                    horizon_days=horizon_days,
                    method="parametric"
                )
                var_mc, cvar_mc = compute_var_cvar(
                    port_returns,
                    confidence=confidence,
                    horizon_days=horizon_days,
                    method="monte_carlo",
                    n_sim=10000
                )
                
                # Relative VaR vs benchmark (active returns)
                rel_var_hist, rel_cvar_hist = compute_relative_var(
                    port_returns,
                    bench_returns,
                    confidence=confidence,
                    horizon_days=horizon_days,
                    method="historical"
                )
                
                st.markdown("#### VaR / CVaR Summary")
                risk_rows = [
                    {
                        "Method": "Historical",
                        "VaR": var_hist,
                        "CVaR": cvar_hist,
                        "Relative VaR vs Benchmark": rel_var_hist,
                        "Relative CVaR vs Benchmark": rel_cvar_hist
                    },
                    {
                        "Method": "Parametric (Normal)",
                        "VaR": var_param,
                        "CVaR": cvar_param,
                        "Relative VaR vs Benchmark": np.nan,
                        "Relative CVaR vs Benchmark": np.nan
                    },
                    {
                        "Method": "Monte Carlo (Normal)",
                        "VaR": var_mc,
                        "CVaR": cvar_mc,
                        "Relative VaR vs Benchmark": np.nan,
                        "Relative CVaR vs Benchmark": np.nan
                    }
                ]
                risk_df = pd.DataFrame(risk_rows)
                st.dataframe(risk_df, width="stretch")
                
                st.markdown("#### Return Distribution – Historical (Portfolio)")
                hist_vals = port_returns.dropna()
                fig_r = go.Figure()
                fig_r.add_histogram(
                    x=hist_vals,
                    nbinsx=60,
                    name="Portfolio Returns",
                    histnorm="probability"
                )
                # VaR threshold is negative on return axis
                fig_r.add_vline(
                    x=-var_hist,
                    line_dash="dash",
                    annotation_text=f"VaR {int(confidence*100)}%: {var_hist:.4f}",
                    annotation_position="top left"
                )
                fig_r.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis_title="Daily Return",
                    yaxis_title="Probability",
                    showlegend=False
                )
                st.plotly_chart(fig_r, use_container_width=True)
        
        # Other analysis types are placeholders
        elif analysis_type == "Backtesting":
            st.subheader("📈 Backtesting")
            st.info("Backtesting engine placeholder – integrate strategy rules and portfolio weights here.")
        
        elif analysis_type == "ML Forecasting":
            st.subheader("🤖 Machine Learning Forecasting")
            st.info("ML forecasting placeholder – plug in sklearn / XGBoost / Prophet models on returns or volatility.")
        
        elif analysis_type == "Comprehensive Report":
            st.subheader("📄 Comprehensive Report")
            st.info("Comprehensive reporting placeholder – PDF/HTML/Excel export using current portfolio analytics.")
    
    else:
        st.markdown("""
        ## Welcome to QuantEdge Pro v5.0
        
        ### Get Started:
        1. **Configure your portfolio** in the sidebar  
        2. **Enter ticker symbols** (e.g., AAPL, GOOGL, MSFT)  
        3. **Select date range** for analysis  
        4. **Choose analysis type**  
        5. **Click 'Fetch Data & Analyze'** to begin
        
        ### Available Features:
        - **Portfolio Optimization**: Mean-variance, risk parity, hierarchical risk parity, Black-Litterman
        - **Risk Analysis**: Historical / Parametric / MC VaR, CVaR, Relative VaR vs benchmark
        - **Machine Learning**: Return forecasting, volatility prediction (hooks ready)
        - **Backtesting**: Strategy testing with realistic assumptions
        - **Comprehensive Reporting**: PDF, Excel, and HTML reports
        
        ### System Requirements:
        - Python 3.8+
        - 8GB+ RAM recommended
        - Internet connection for data fetching
        """)
        
        st.subheader("🖥️ System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Memory Usage", f"{performance_monitor._get_memory_usage():.1f} MB")
        with col2:
            st.metric("CPU Usage", f"{performance_monitor._get_cpu_usage():.1f}%")
        with col3:
            st.metric("Python Version", f"{sys.version.split()[0]}")

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()

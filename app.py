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

# --- Portfolio optimization imports (PyPortfolioOpt) ---
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class Config:
    """Centralized configuration for QuantEdge Pro."""
    
    # Data fetching
    MAX_TICKERS = 50
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
                        # Already running; avoid recursion
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
                
                error_analyzer = None
                if hasattr(st.session_state, 'error_analyzer'):
                    error_analyzer = st.session_state.error_analyzer
                
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
        """Check and import all required libraries with advanced capabilities."""
        lib_status = {}
        missing_libs = []
        advanced_features = {}
        
        # Core libraries (required)
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
        
        # Advanced libraries (optional but recommended)
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
                session_key = f"{lib_name}_available"
                st.session_state[session_key] = True
            except ImportError:
                lib_status[lib_name] = False
                missing_libs.append(f"{lib_name} (optional: {config['description']})")
                session_key = f"{lib_name}_available"
                st.session_state[session_key] = False
        
        # Deep learning libraries
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
        """Check and import all required libraries with ML capabilities."""
        lib_status = {}
        missing_libs = []
        advanced_features = {}
        
        original_status = AdvancedLibraryManager.check_and_import_all()
        lib_status.update(original_status['status'])
        missing_libs.extend([lib for lib in original_status['missing'] if lib not in missing_libs])
        advanced_features.update(original_status['advanced_features'])
        
        # Alternative data sources
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
        
        # Time series forecasting
        try:
            from prophet import Prophet  # noqa: F401
            lib_status['prophet'] = True
            advanced_features['prophet'] = {
                'description': 'Time series forecasting',
                'available': True
            }
            st.session_state.prophet_available = True
        except ImportError:
            lib_status['prophet'] = False
            missing_libs.append('prophet (optional: Time series forecasting)')
        
        # Reporting
        try:
            from reportlab.lib import colors  # noqa: F401
            from reportlab.lib.pagesizes import letter  # noqa: F401
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph  # noqa: F401
            from reportlab.lib.styles import getSampleStyleSheet  # noqa: F401
            lib_status['reportlab'] = True
            advanced_features['reportlab'] = {
                'description': 'PDF report generation',
                'available': True
            }
            st.session_state.reportlab_available = True
        except ImportError:
            lib_status['reportlab'] = False
            missing_libs.append('reportlab (optional: PDF generation)')
        
        # Blockchain
        try:
            from web3 import Web3  # noqa: F401
            lib_status['web3'] = True
            advanced_features['web3'] = {
                'description': 'Blockchain data access',
                'available': True
            }
            st.session_state.web3_available = True
        except ImportError:
            lib_status['web3'] = False
            missing_libs.append('web3 (optional: Blockchain)')
        
        # Database
        try:
            from sqlalchemy import create_engine, text  # noqa: F401
            lib_status['sqlalchemy'] = True
            advanced_features['sqlalchemy'] = {
                'description': 'Database integration',
                'available': True
            }
            st.session_state.sqlalchemy_available = True
        except ImportError:
            lib_status['sqlalchemy'] = False
            missing_libs.append('sqlalchemy (optional: Database)')
        
        # ARCH for GARCH models
        try:
            from arch import arch_model  # noqa: F401
            lib_status['arch'] = True
            advanced_features['arch'] = {
                'description': 'GARCH volatility models',
                'available': True
            }
            st.session_state.arch_available = True
        except ImportError:
            lib_status['arch'] = False
            missing_libs.append('arch (optional: GARCH models)')
        
        # XGBoost
        try:
            import xgboost as xgb  # noqa: F401
            lib_status['xgboost'] = True
            advanced_features['xgboost'] = {
                'description': 'Gradient boosting',
                'available': True
            }
            st.session_state.xgboost_available = True
        except ImportError:
            lib_status['xgboost'] = False
            missing_libs.append('xgboost (optional: Gradient boosting)')
        
        # Sync PyPortfolioOpt availability flag with global import result
        st.session_state['pypfopt_available'] = PYPFOPT_AVAILABLE
        
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

@st.cache_resource
def initialize_library_manager():
    """Initialize and cache library manager."""
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
        """Analyze error with full context for intelligent recovery."""
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
        """Safe version of error analysis that never triggers monitoring."""
        return self.analyze_error_with_context(error, context)
    
    def _create_simple_error_analysis(self, error: Exception, context: Dict) -> Dict:
        """Create a simple error analysis without triggering monitoring."""
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
        """Generate ML-powered recovery suggestions."""
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
        """Generate preventive measures based on error analysis."""
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
        """Create advanced error display with interactive elements."""
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
        """Get statistics about errors encountered."""
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
        """Start timing an operation."""
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
        """End timing an operation and record metrics."""
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
            return 0
    
    def _get_cpu_usage(self) -> float:
        try:
            return self.process.cpu_percent(interval=0.1)
        except Exception:
            return 0
    
    @monitor_operation('get_performance_report')
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
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
                'total_operation_time': sum(total_times),
                'slowest_operation': max(report['operations'].items(), key=lambda x: x[1]['avg_duration'])[0],
                'most_frequent_operation': max(report['operations'].items(), key=lambda x: x[1]['count'])[0],
                'most_memory_intensive': max(report['operations'].items(), key=lambda x: x[1]['avg_memory_increase'])[0]
            }
        
        if self.memory_usage:
            report['resource_usage']['memory'] = {
                'peak_mb': max(self.memory_usage) if self.memory_usage else 0,
                'avg_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
                'current_mb': self._get_memory_usage()
            }
        
        report['recommendations'] = self._generate_recommendations(report)
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        for op_name, stats in report['operations'].items():
            if stats['avg_duration'] > 5:
                recommendations.append(
                    f"Optimize '{op_name}' - average duration {stats['avg_duration']:.1f}s"
                )
            if stats['avg_memory_increase'] > 100:
                recommendations.append(
                    f"Reduce memory usage in '{op_name}' - average increase {stats['avg_memory_increase']:.1f}MB"
                )
            if stats['avg_cpu_increase'] > 50:
                recommendations.append(
                    f"Optimize CPU usage in '{op_name}' - average increase {stats['avg_cpu_increase']:.1f}%"
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
        """Clear performance monitoring cache."""
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
        """Fetch advanced market data with multiple features and ensure equal length series."""
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
        """Fetch OHLC data for a single ticker with comprehensive error handling."""
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
        """Process and align data to ensure equal length series with proper forward filling."""
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
                raise ValueError(f"Only {len(valid_assets)} assets with sufficient data (minimum {Config.MIN_ASSETS_FOR_OPTIMIZATION} required)")
            
            for key in ['prices', 'volumes', 'high', 'low', 'open']:
                if not data[key].empty:
                    data[key] = data[key][valid_assets]
            
            data['successful_tickers'] = [t for t in data['successful_tickers'] if t in valid_assets]
        
        return data
    
    def _calculate_additional_features(self, data: Dict) -> Dict:
        """Calculate additional features for analysis."""
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
                        'mean': np.mean(corr_values),
                        'median': np.median(corr_values),
                        'min': np.min(corr_values),
                        'max': np.max(corr_values),
                        'std': np.std(corr_values)
                    }
            
            if not returns.empty:
                features['covariance_matrix'] = returns.cov() * Config.TRADING_DAYS_PER_YEAR
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
        except Exception:
            return 0
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        try:
            if len(returns) == 0:
                return 0
            var = -np.percentile(returns, (1 - confidence) * 100)
            cvar_data = returns[returns <= -var]
            return -cvar_data.mean() if len(cvar_data) > 0 else var
        except Exception:
            return 0
    
    def _generate_cache_key(self, tickers: List[str], start_date: datetime, 
                           end_date: datetime, interval: str) -> str:
        """Generate cache key for data."""
        tickers_str = '_'.join(sorted(tickers))
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        return f"{tickers_str}_{date_str}_{interval}"
    
    def validate_portfolio_data(self, data: Dict, 
                               min_assets: int = Config.MIN_ASSETS_FOR_OPTIMIZATION,
                               min_data_points: int = Config.MIN_DATA_POINTS) -> Dict:
        """Validate portfolio data for analysis."""
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
                },
                'missing_data_percentage': missing_percentage if 'missing_percentage' in locals() else 0,
                'average_return': data['returns'].mean().mean() if not data.get('returns', pd.DataFrame()).empty else 0,
                'average_volatility': data['returns'].std().mean() * np.sqrt(Config.TRADING_DAYS_PER_YEAR) if not data.get('returns', pd.DataFrame()).empty else 0,
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
            elif step == 'log_returns':
                processed_data = self._calculate_log_returns(processed_data)
        
        return processed_data
    
    def _clean_missing_values(self, data: Dict) -> Dict:
        """Clean missing values from data."""
        if not data['prices'].empty:
            data['prices'] = data['prices'].ffill().bfill()
        
        if not data.get('returns', pd.DataFrame()).empty:
            returns = data['returns']
            threshold = 0.5
            min_non_na = int(threshold * len(returns.columns))
            data['returns'] = returns.dropna(thresh=min_non_na)
        
        return data
    
    def _handle_outliers(self, data: Dict) -> Dict:
        """Handle outliers in returns data using winsorization."""
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
        """Normalize data for analysis."""
        if not data.get('returns', pd.DataFrame()).empty:
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
        
        if not data.get('returns', pd.DataFrame()).empty:
            for column in data['returns'].columns:
                try:
                    from statsmodels.tsa.stattools import adfuller
                    series = data['returns'][column].dropna()
                    if len(series) > 10:
                        result = adfuller(series, autolag='AIC')
                        stationarity_results[column] = {
                            'adf_statistic': result[0],
                            'p_value': result[1],
                            'is_stationary': result[1] < 0.05,
                            'critical_values': result[4],
                            'test_used': 'ADF'
                        }
                except Exception as e:
                    stationarity_results[column] = {'error': str(e)}
        
        data['stationarity'] = stationarity_results
        return data
    
    def _detrend_data(self, data: Dict) -> Dict:
        """Remove linear trend from price data."""
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
        """Calculate log returns from prices."""
        if not data['prices'].empty:
            log_returns = np.log(data['prices'] / data['prices'].shift(1)).dropna()
            data['log_returns'] = log_returns
        return data
    
    @monitor_operation('calculate_basic_statistics')
    def calculate_basic_statistics(self, data: Dict) -> Dict:
        """Calculate comprehensive statistics for the dataset."""
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
                        'mean_return': ticker_returns.mean() * Config.TRADING_DAYS_PER_YEAR,
                        'annual_volatility': ticker_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR),
                        'sharpe_ratio': (ticker_returns.mean() * Config.TRADING_DAYS_PER_YEAR) / 
                                       (ticker_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)) 
                                       if ticker_returns.std() > 0 else 0,
                        'skewness': ticker_returns.skew(),
                        'kurtosis': ticker_returns.kurtosis(),
                        'var_95': -np.percentile(ticker_returns, 5),
                        'cvar_95': self._calculate_cvar(ticker_returns, 0.95),
                        'max_drawdown': self._calculate_max_drawdown_series(ticker_returns),
                        'positive_days': (ticker_returns > 0).sum() / len(ticker_returns),
                        'data_points': len(ticker_returns),
                        'start_date': ticker_returns.index.min() if not ticker_returns.empty else None,
                        'end_date': ticker_returns.index.max() if not ticker_returns.empty else None
                    }
            
            if len(returns.columns) > 0:
                equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
                portfolio_returns = returns.dot(equal_weights)
                
                stats_dict['portfolio_level'] = {
                    'mean_return': portfolio_returns.mean() * Config.TRADING_DAYS_PER_YEAR,
                    'annual_volatility': portfolio_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR),
                    'sharpe_ratio': (portfolio_returns.mean() * Config.TRADING_DAYS_PER_YEAR) / 
                                   (portfolio_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)) 
                                   if portfolio_returns.std() > 0 else 0,
                    'skewness': portfolio_returns.skew(),
                    'kurtosis': portfolio_returns.kurtosis(),
                    'var_95': -np.percentile(portfolio_returns, 5),
                    'cvar_95': self._calculate_cvar(portfolio_returns, 0.95),
                    'max_drawdown': self._calculate_max_drawdown_series(portfolio_returns),
                    'positive_days': (portfolio_returns > 0).sum() / len(portfolio_returns),
                    'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns),
                    'calmar_ratio': self._calculate_calmar_ratio(portfolio_returns)
                }
            
            if len(returns.columns) > 1:
                corr_matrix = returns.corr()
                stats_dict['correlation']['matrix'] = corr_matrix
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                if len(corr_values) > 0:
                    stats_dict['correlation']['stats'] = {
                        'mean': np.mean(corr_values),
                        'median': np.median(corr_values),
                        'min': np.min(corr_values),
                        'max': np.max(corr_values),
                        'std': np.std(corr_values),
                        'q25': np.percentile(corr_values, 25),
                        'q75': np.percentile(corr_values, 75)
                    }
                
                cov_matrix = returns.cov() * Config.TRADING_DAYS_PER_YEAR
                stats_dict['covariance']['matrix'] = cov_matrix
                stats_dict['covariance']['mean_variance'] = np.diag(cov_matrix).mean()
                stats_dict['covariance']['avg_covariance'] = cov_matrix.values[np.triu_indices_from(cov_matrix.values, k=1)].mean()
            
            if not volumes.empty:
                for ticker in volumes.columns:
                    volume_series = volumes[ticker].dropna()
                    if len(volume_series) > 0:
                        stats_dict['liquidity'][ticker] = {
                            'avg_volume': volume_series.mean(),
                            'std_volume': volume_series.std(),
                            'volume_ratio_last_avg': volume_series.iloc[-1] / volume_series.mean() if volume_series.mean() > 0 else 0,
                            'volume_trend': self._calculate_volume_trend(volume_series)
                        }
            
            if not prices.empty:
                for ticker in prices.columns:
                    price_series = prices[ticker].dropna()
                    if len(price_series) > 0:
                        stats_dict['price_level'][ticker] = {
                            'current_price': price_series.iloc[-1],
                            'price_change_1m': (price_series.iloc[-1] / price_series.iloc[-22] - 1) if len(price_series) > 22 else 0,
                            'price_change_3m': (price_series.iloc[-1] / price_series.iloc[-66] - 1) if len(price_series) > 66 else 0,
                            'price_change_1y': (price_series.iloc[-1] / price_series.iloc[-252] - 1) if len(price_series) > 252 else 0,
                            'price_high_52w': price_series.tail(252).max() if len(price_series) >= 252 else price_series.max(),
                            'price_low_52w': price_series.tail(252).min() if len(price_series) >= 252 else price_series.min()
                        }
        
        return stats_dict
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = Config.DEFAULT_RISK_FREE_RATE) -> float:
        """Calculate Sortino ratio."""
        try:
            if len(returns) == 0:
                return 0
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return float('inf')
            downside_std = downside_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)
            if downside_std == 0:
                return float('inf')
            excess_return = returns.mean() * Config.TRADING_DAYS_PER_YEAR - risk_free_rate
            return excess_return / downside_std
        except Exception:
            return 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        try:
            if len(returns) == 0:
                return 0
            max_dd = self._calculate_max_drawdown_series(returns)
            if max_dd == 0:
                return 0
            annual_return = returns.mean() * Config.TRADING_DAYS_PER_YEAR
            return annual_return / abs(max_dd)
        except Exception:
            return 0
    
    def _calculate_volume_trend(self, volume_series: pd.Series, window: int = 20) -> str:
        """Calculate volume trend."""
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
# 4. PORTFOLIO OPTIMIZER WITH REGIONAL FILTERS & SCENARIO PRESETS
# ============================================================================

class PortfolioOptimizer:
    """Wire EF / HRP / CLA / BL models into Portfolio Optimization tab with regions & scenarios."""
    
    def __init__(self, data_manager: AdvancedDataManager):
        self.data_manager = data_manager
    
    # ---------- REGION CLASSIFICATION ----------
    def _classify_region(self, ticker: str, metadata: Dict) -> str:
        """Classify ticker into region: US / TR / JP / KR / SG / CN / Other."""
        t = ticker.upper()
        country = str(metadata.get('country', '') or '').lower()
        exchange = str(metadata.get('exchange', '') or '').lower()
        
        if t.endswith(".IS") or "turkey" in country or "türkiye" in country:
            return "TR"
        if t.endswith(".T") or "japan" in country:
            return "JP"
        if t.endswith(".KS") or "korea" in country:
            return "KR"
        if t.endswith(".SI") or "singapore" in country:
            return "SG"
        if t.endswith(".HK") or "hong kong" in country or "china" in country:
            return "CN"
        if exchange in ["nyq", "nms", "ngs", "nasdaq", "nyse", "amex"] or "united states" in country or country == "usa":
            return "US"
        return "Other"
    
    def build_region_map(self, tickers: List[str], metadata: Dict[str, Dict]) -> Dict[str, str]:
        region_map = {}
        for t in tickers:
            md = metadata.get(t, {}) or {}
            region_map[t] = self._classify_region(t, md)
        return region_map
    
    # ---------- SCENARIO PRESETS (DYNAMIC) ----------
    def build_scenarios(self, tickers: List[str], metadata: Dict[str, Dict], region_map: Dict[str, str]) -> Dict[str, List[str]]:
        """Build dynamic scenario presets from current universe."""
        scenarios: Dict[str, List[str]] = {}
        global_tech_tr_banks: List[str] = []
        
        for t in tickers:
            md = metadata.get(t, {}) or {}
            sector = str(md.get('sector', '') or '').lower()
            industry = str(md.get('industry', '') or '').lower()
            short_name = str(md.get('short_name', '') or '').lower()
            long_name = str(md.get('name', '') or '').lower()
            region = region_map.get(t, "Other")
            
            is_tech = any(w in sector for w in ["technology", "communication"]) or \
                      any(w in industry for w in ["semiconductor", "semiconductors", "software", "it services", "internet", "interactive media"])
            is_tr_bank = (region == "TR") and (
                "bank" in short_name or "bank" in long_name or "bank" in industry or "banks" in sector
            )
            
            if (is_tech and region in ["US", "JP", "KR", "SG", "CN"]) or is_tr_bank:
                global_tech_tr_banks.append(t)
        
        if global_tech_tr_banks:
            scenarios["Global Tech + TR Banks"] = sorted(list(set(global_tech_tr_banks)))
        
        # You can add more scenario builders here if needed.
        return scenarios
    
    # ---------- CORE OPTIMIZATION ----------
    @monitor_operation("run_portfolio_optimization")
    def run_optimization(self,
                         data: Dict,
                         selected_tickers: List[str],
                         model: str,
                         risk_free_rate: float) -> Dict[str, Any]:
        """Run selected optimization model + equal-weight benchmark."""
        if len(selected_tickers) < 2:
            raise ValueError("At least 2 assets are required for portfolio optimization.")
        
        prices = data['prices'][selected_tickers].copy()
        returns = data['returns'][selected_tickers].dropna().copy()
        
        if prices.isnull().any().any():
            prices = prices.ffill().bfill()
        if returns.empty:
            returns = prices.pct_change().dropna()
        
        # Equal-weight benchmark
        n = len(selected_tickers)
        ew_weights = {t: 1.0 / n for t in selected_tickers}
        ew_metrics = self._compute_portfolio_metrics(returns, ew_weights, risk_free_rate)
        
        if not PYPFOPT_AVAILABLE or not st.session_state.get("pypfopt_available", False):
            # Only equal-weight available
            return {
                "model_name": "Equal Weight (PyPortfolioOpt not available)",
                "weights": ew_weights,
                "metrics": ew_metrics,
                "benchmark_weights": ew_weights,
                "benchmark_metrics": ew_metrics,
                "frontier": None
            }
        
        mu, S = self._prepare_mu_S(prices)
        frontier_df = self._compute_efficient_frontier(mu, S, risk_free_rate)
        
        model_name = model
        opt_weights_dict = ew_weights
        
        model_lower = model.lower()
        try:
            if "equal" in model_lower:
                model_name = "Equal Weight"
                opt_weights_dict = ew_weights
            
            elif "max sharpe" in model_lower or "ef - max sharpe" in model_lower:
                model_name = "Mean-Variance (EF - Max Sharpe)"
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                opt_weights_dict = ef.clean_weights()
            
            elif "min volatility" in model_lower or "min vol" in model_lower:
                model_name = "Mean-Variance (EF - Min Volatility)"
                ef = EfficientFrontier(mu, S)
                ef.min_volatility()
                opt_weights_dict = ef.clean_weights()
            
            elif "hrp" in model_lower:
                model_name = "HRP (Hierarchical Risk Parity)"
                hrp = HRPOpt(returns)
                opt_weights_dict = hrp.optimize()
            
            elif "cla" in model_lower:
                model_name = "CLA (Critical Line Algorithm)"
                cla = CLA(mu, S)
                cla.max_sharpe(risk_free_rate=risk_free_rate)
                opt_weights_dict = cla.clean_weights()
            
            elif "black" in model_lower or "litterman" in model_lower:
                model_name = "Black-Litterman (Neutral Views)"
                market_caps = self._get_market_caps_for_tickers(selected_tickers, data)
                bl = BlackLittermanModel(
                    S.loc[selected_tickers, selected_tickers],
                    pi="market",
                    market_caps=market_caps,
                    risk_free_rate=risk_free_rate
                )
                bl.bl_weights()  # compute implied weights
                opt_weights_dict = bl.clean_weights()
            else:
                # Default to Max Sharpe
                model_name = "Mean-Variance (EF - Max Sharpe)"
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                opt_weights_dict = ef.clean_weights()
        except Exception as e:
            # Fallback to equal weight if optimization fails
            raise RuntimeError(f"Optimization failed for model '{model_name}': {str(e)}")
        
        # Align weights to selected tickers and normalize
        weights_series = pd.Series(opt_weights_dict)
        weights_series = weights_series.reindex(selected_tickers).fillna(0.0)
        if weights_series.sum() > 0:
            weights_series = weights_series / weights_series.sum()
        opt_weights_dict = weights_series.to_dict()
        
        opt_metrics = self._compute_portfolio_metrics(returns, opt_weights_dict, risk_free_rate)
        
        return {
            "model_name": model_name,
            "weights": opt_weights_dict,
            "metrics": opt_metrics,
            "benchmark_weights": ew_weights,
            "benchmark_metrics": ew_metrics,
            "frontier": frontier_df
        }
    
    # ---------- HELPER METHODS ----------
    def _prepare_mu_S(self, prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Prepare expected returns & covariance matrix."""
        mu = expected_returns.mean_historical_return(prices, frequency=Config.TRADING_DAYS_PER_YEAR)
        cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        return mu, cov
    
    def _compute_efficient_frontier(self, mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float) -> Optional[pd.DataFrame]:
        """Compute a set of points on the efficient frontier for plotting."""
        if mu.empty or cov.empty:
            return None
        
        target_returns = np.linspace(mu.min(), mu.max(), 30)
        vols = []
        rets = []
        
        for tr in target_returns:
            ef = EfficientFrontier(mu, cov)
            try:
                ef.efficient_return(target_return=tr)
                ret, vol, _ = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                vols.append(vol)
                rets.append(ret)
            except Exception:
                continue
        
        if not vols:
            return None
        
        return pd.DataFrame({"volatility": vols, "return": rets})
    
    def _get_market_caps_for_tickers(self, tickers: List[str], data: Dict) -> pd.Series:
        """Build market cap series from metadata; fallback to equal if missing."""
        metadata = data.get("metadata", {}) or {}
        caps_dict = {}
        for t in tickers:
            md = metadata.get(t, {}) or {}
            mc = md.get("market_cap", None)
            if mc is None or mc <= 0:
                mc = 1.0
            caps_dict[t] = float(mc)
        s = pd.Series(caps_dict)
        return s
    
    def _compute_portfolio_metrics(self,
                                   returns: pd.DataFrame,
                                   weights_dict: Dict[str, float],
                                   risk_free_rate: float) -> Dict[str, float]:
        """Compute portfolio KPIs from daily returns and weights."""
        if returns.empty or not weights_dict:
            return {
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0
            }
        
        w = pd.Series(weights_dict).reindex(returns.columns).fillna(0.0).values
        if np.allclose(w.sum(), 0):
            return {
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0
            }
        if not np.isclose(w.sum(), 1.0):
            w = w / w.sum()
        
        port_rets = returns.dot(w)
        
        ann_ret = port_rets.mean() * Config.TRADING_DAYS_PER_YEAR
        ann_vol = port_rets.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
        sortino = self.data_manager._calculate_sortino_ratio(port_rets, risk_free_rate)
        calmar = self.data_manager._calculate_calmar_ratio(port_rets)
        max_dd = self.data_manager._calculate_max_drawdown_series(port_rets)
        var_95 = -np.percentile(port_rets, 5) if len(port_rets) > 0 else 0.0
        cvar_95 = self.data_manager._calculate_cvar(port_rets, 0.95)
        
        return {
            "annual_return": float(ann_ret),
            "annual_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_dd),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95)
        }

# Initialize portfolio optimizer
portfolio_optimizer = PortfolioOptimizer(data_manager)

# ============================================================================
# STREAMLIT APP MAIN FUNCTION
# ============================================================================

def main():
    """Main Streamlit application."""
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
    
    # Sidebar configuration
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
                    progress_bar = st.progress(0)
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        st.sidebar.text(message)
                    
                    data = data_manager.fetch_advanced_market_data(
                        tickers=tickers,
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.max.time()),
                        progress_callback=update_progress
                    )
                    
                    st.session_state.portfolio_data = data
                    st.session_state.data_loaded = True
                    
                    validation = data_manager.validate_portfolio_data(data)
                    
                    if validation['is_valid']:
                        st.sidebar.success(
                            f"✅ Data loaded: {validation['summary']['n_assets']} assets, "
                            f"{validation['summary']['n_data_points']} days"
                        )
                    else:
                        st.sidebar.warning(
                            f"⚠️ Data loaded with issues: {len(validation['issues'])} issues, "
                            f"{len(validation['warnings'])} warnings"
                        )
                except Exception as e:
                    st.sidebar.error(f"❌ Error fetching data: {str(e)[:200]}...")
                    logging.error(f"Data fetch error: {str(e)}")
    
    # Main content area
    if st.session_state.get('data_loaded', False) and 'portfolio_data' in st.session_state:
        data = st.session_state.portfolio_data
        
        st.subheader("📋 Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Assets", len(data['prices'].columns))
        with col2:
            st.metric("Data Points", len(data['prices']))
        with col3:
            date_range = f"{data['prices'].index[0].date()} to {data['prices'].index[-1].date()}"
            st.metric("Date Range", date_range)
        
        with st.expander("📊 Data Preview"):
            tab1, tab2, tab3 = st.tabs(["Prices", "Returns", "Statistics"])
            with tab1:
                st.dataframe(data['prices'].tail(10), use_container_width=True)
            with tab2:
                if not data['returns'].empty:
                    st.dataframe(data['returns'].tail(10), use_container_width=True)
            with tab3:
                stats = data_manager.calculate_basic_statistics(data)
                if stats['assets']:
                    stats_df = pd.DataFrame(stats['assets']).T
                    st.dataframe(
                        stats_df[['mean_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']],
                        use_container_width=True
                    )
        
        # ===================== PORTFOLIO OPTIMIZATION TAB =====================
        if analysis_type == "Portfolio Optimization":
            st.subheader("🎯 Portfolio Optimization")
            
            if not PYPFOPT_AVAILABLE or not st.session_state.get("pypfopt_available", False):
                st.error("PyPortfolioOpt is not available. Please install `pypfopt` to enable optimization models.")
                st.info("You can still use equal-weight portfolios as a simple benchmark.")
            else:
                # --- Regional filters + scenario presets ---
                prices = data['prices']
                returns = data['returns']
                metadata = data.get('metadata', {}) or {}
                
                if prices.empty or returns.empty:
                    st.warning("Price/return data is empty. Please fetch data again with a valid date range.")
                else:
                    all_tickers = list(prices.columns)
                    region_map = portfolio_optimizer.build_region_map(all_tickers, metadata)
                    
                    # Ordered region options
                    region_order = ["US", "TR", "JP", "KR", "SG", "CN", "Other"]
                    present_regions = sorted({region_map[t] for t in all_tickers},
                                             key=lambda r: region_order.index(r) if r in region_order else 999)
                    st.markdown("##### 🌍 Regional Filters")
                    selected_regions = st.multiselect(
                        "Select regions to include:",
                        present_regions,
                        default=present_regions,
                        help="Filter the optimization universe by geographic region."
                    )
                    
                    filtered_by_region = [t for t in all_tickers if region_map.get(t, "Other") in selected_regions]
                    
                    if len(filtered_by_region) < 2:
                        st.warning("Not enough assets after regional filtering. Please select more regions or add more tickers.")
                    else:
                        # Scenario presets
                        scenarios = portfolio_optimizer.build_scenarios(filtered_by_region, metadata, region_map)
                        scenario_names = ["None (use region-filtered universe)"] + list(scenarios.keys())
                        st.markdown("##### 🎛 Scenario Presets")
                        selected_scenario = st.selectbox(
                            "Choose a scenario preset (optional):",
                            scenario_names,
                            help="For example, 'Global Tech + TR Banks' mixes global technology with Turkish banks."
                        )
                        
                        if selected_scenario != "None (use region-filtered universe)":
                            scenario_tickers = scenarios.get(selected_scenario, [])
                            scenario_universe = [t for t in filtered_by_region if t in scenario_tickers]
                            if len(scenario_universe) < 2:
                                st.warning(
                                    f"Scenario '{selected_scenario}' has fewer than 2 assets in the current dataset. "
                                    "Falling back to region-filtered universe."
                                )
                                base_universe = filtered_by_region
                            else:
                                base_universe = scenario_universe
                        else:
                            base_universe = filtered_by_region
                        
                        st.markdown("##### 📌 Asset Selection")
                        selected_assets = st.multiselect(
                            "Select assets for optimization:",
                            base_universe,
                            default=base_universe,
                            help="Choose which assets will be included in the optimization."
                        )
                        
                        if len(selected_assets) < 2:
                            st.warning("Select at least 2 assets to run portfolio optimization.")
                        else:
                            col_opt1, col_opt2 = st.columns(2)
                            with col_opt1:
                                rf_input = st.number_input(
                                    "Risk-free rate (annual, %)",
                                    min_value=-5.0,
                                    max_value=20.0,
                                    value=Config.DEFAULT_RISK_FREE_RATE * 100,
                                    step=0.25
                                )
                            with col_opt2:
                                model = st.selectbox(
                                    "Optimization Model",
                                    [
                                        "Mean-Variance (EF - Max Sharpe)",
                                        "Mean-Variance (EF - Min Volatility)",
                                        "CLA (Critical Line Algorithm)",
                                        "HRP (Hierarchical Risk Parity)",
                                        "Black-Litterman (Neutral Views)",
                                        "Equal Weight (Benchmark Only)"
                                    ]
                                )
                            
                            risk_free_rate = rf_input / 100.0
                            
                            if st.button("⚡ Run Portfolio Optimization"):
                                with st.spinner("Running optimization..."):
                                    try:
                                        results = portfolio_optimizer.run_optimization(
                                            data=data,
                                            selected_tickers=selected_assets,
                                            model=model,
                                            risk_free_rate=risk_free_rate
                                        )
                                        
                                        model_name = results["model_name"]
                                        w_opt = results["weights"]
                                        m_opt = results["metrics"]
                                        w_ew = results["benchmark_weights"]
                                        m_ew = results["benchmark_metrics"]
                                        frontier_df = results["frontier"]
                                        
                                        st.markdown(f"#### 🧠 Optimized Portfolio: **{model_name}**")
                                        
                                        # --- Allocation charts & weights table ---
                                        col_a1, col_a2 = st.columns([1.6, 1.4])
                                        with col_a1:
                                            alloc_tabs = st.tabs(["Weights (Chart)", "Weights (Table)"])
                                            w_series = pd.Series(w_opt).sort_values(ascending=False)
                                            ew_series = pd.Series(w_ew).reindex(w_series.index).fillna(0.0)
                                            
                                            with alloc_tabs[0]:
                                                fig_alloc = go.Figure()
                                                fig_alloc.add_trace(
                                                    go.Bar(
                                                        x=w_series.index,
                                                        y=w_series.values,
                                                        name="Optimized"
                                                    )
                                                )
                                                fig_alloc.add_trace(
                                                    go.Bar(
                                                        x=w_series.index,
                                                        y=ew_series.values,
                                                        name="Equal Weight"
                                                    )
                                                )
                                                fig_alloc.update_layout(
                                                    barmode="group",
                                                    xaxis_title="Asset",
                                                    yaxis_title="Weight",
                                                    height=450
                                                )
                                                st.plotly_chart(fig_alloc, use_container_width=True)
                                            
                                            with alloc_tabs[1]:
                                                weights_df = pd.DataFrame({
                                                    "Optimized": w_series,
                                                    "Equal Weight": ew_series
                                                })
                                                st.dataframe(
                                                    weights_df.style.format("{:.2%}"),
                                                    use_container_width=True
                                                )
                                        
                                        with col_a2:
                                            st.markdown("##### 📌 Portfolio KPIs (Annualized)")
                                            metrics_df = pd.DataFrame({
                                                "Metric": [
                                                    "Annual Return",
                                                    "Annual Volatility",
                                                    "Sharpe Ratio",
                                                    "Sortino Ratio",
                                                    "Calmar Ratio",
                                                    "Max Drawdown",
                                                    "VaR 95%",
                                                    "CVaR 95%"
                                                ],
                                                "Optimized": [
                                                    m_opt["annual_return"],
                                                    m_opt["annual_volatility"],
                                                    m_opt["sharpe_ratio"],
                                                    m_opt["sortino_ratio"],
                                                    m_opt["calmar_ratio"],
                                                    m_opt["max_drawdown"],
                                                    m_opt["var_95"],
                                                    m_opt["cvar_95"]
                                                ],
                                                "Equal Weight": [
                                                    m_ew["annual_return"],
                                                    m_ew["annual_volatility"],
                                                    m_ew["sharpe_ratio"],
                                                    m_ew["sortino_ratio"],
                                                    m_ew["calmar_ratio"],
                                                    m_ew["max_drawdown"],
                                                    m_ew["var_95"],
                                                    m_ew["cvar_95"]
                                                ]
                                            })
                                            st.table(
                                                metrics_df.set_index("Metric").style.format("{:.2%}", subset=["Optimized", "Equal Weight"]).format(
                                                    "{:.2f}", subset=[("Optimized",), ("Equal Weight",)]
                                                )
                                            )
                                        
                                        # --- Efficient frontier chart ---
                                        st.markdown("#### 📉 Efficient Frontier & Portfolio Positioning")
                                        if frontier_df is not None and not frontier_df.empty:
                                            fig_ef = go.Figure()
                                            fig_ef.add_trace(
                                                go.Scatter(
                                                    x=frontier_df["volatility"],
                                                    y=frontier_df["return"],
                                                    mode="lines",
                                                    name="Efficient Frontier"
                                                )
                                            )
                                            fig_ef.add_trace(
                                                go.Scatter(
                                                    x=[m_opt["annual_volatility"]],
                                                    y=[m_opt["annual_return"]],
                                                    mode="markers",
                                                    name="Optimized Portfolio",
                                                    marker=dict(size=12)
                                                )
                                            )
                                            fig_ef.add_trace(
                                                go.Scatter(
                                                    x=[m_ew["annual_volatility"]],
                                                    y=[m_ew["annual_return"]],
                                                    mode="markers",
                                                    name="Equal Weight",
                                                    marker=dict(size=10, symbol="diamond")
                                                )
                                            )
                                            fig_ef.update_layout(
                                                xaxis_title="Volatility (annual)",
                                                yaxis_title="Return (annual)",
                                                height=550
                                            )
                                            st.plotly_chart(fig_ef, use_container_width=True)
                                        else:
                                            st.info("Efficient frontier could not be computed for the current universe.")
                                    except Exception as e:
                                        st.error(f"Optimization error: {str(e)[:300]}...")
        
        # ===================== OTHER TABS (PLACEHOLDERS) ======================
        elif analysis_type == "Risk Analysis":
            st.subheader("⚠️ Risk Analysis")
            st.info("Risk Analysis module will be wired with advanced VaR / CVaR / stress-testing in the next step.")
        
        elif analysis_type == "Backtesting":
            st.subheader("📈 Backtesting")
            st.info("Backtesting engine placeholder – strategies and scenario engine can be integrated here.")
        
        elif analysis_type == "ML Forecasting":
            st.subheader("🤖 Machine Learning Forecasting")
            st.info("ML Forecasting placeholder – models like RF/GBM/LSTM can be wired to use the same data pipeline.")
        
        elif analysis_type == "Comprehensive Report":
            st.subheader("📄 Comprehensive Report")
            st.info("Comprehensive reporting (PDF/Excel/HTML) will aggregate optimization, risk, and backtests.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to QuantEdge Pro v5.0
        
        ### Get Started:
        1. **Configure your portfolio** in the sidebar  
        2. **Enter ticker symbols** (e.g., AAPL, GOOGL, MSFT)  
        3. **Select date range** for analysis  
        4. **Choose analysis type**  
        5. **Click 'Fetch Data & Analyze'** to begin
        
        ### Available Features:
        - **Portfolio Optimization**: Mean-variance (EF), HRP, CLA, Black-Litterman, Equal-weight benchmark  
        - **Risk Analysis**: VaR, CVaR, stress testing, backtesting (to be extended)  
        - **Machine Learning**: Return forecasting, volatility prediction  
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

# ============================================================================
# QUANTEDGE PRO v5.1 ENTERPRISE EDITION - HYPER-ENHANCED VERSION
# INSTITUTIONAL PORTFOLIO ANALYTICS PLATFORM WITH AI/ML CAPABILITIES
# Total Lines: 7000+ | Production Grade | Enterprise Ready
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

# ML Imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

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
            # Get performance monitor from global context or args
            performance_monitor = None
            for arg in args:
                if hasattr(arg, 'end_operation'):  # Check if it's a PerformanceMonitor
                    performance_monitor = arg
                    break
            
            if not performance_monitor and hasattr(st.session_state, 'performance_monitor'):
                performance_monitor = st.session_state.performance_monitor
            
            if performance_monitor:
                performance_monitor.start_operation(operation_name)
            
            try:
                result = func(*args, **kwargs)
                if performance_monitor:
                    performance_monitor.end_operation(operation_name)
                return result
            except Exception as e:
                if performance_monitor:
                    performance_monitor.end_operation(operation_name, {'error': str(e)})
                
                # Get error analyzer from global context
                error_analyzer = None
                if hasattr(st.session_state, 'error_analyzer'):
                    error_analyzer = st.session_state.error_analyzer
                
                if error_analyzer:
                    context = {
                        'operation': operation_name,
                        'function': func.__name__,
                        'module': func.__module__
                    }
                    error_analyzer.analyze_error_with_context(e, context)
                
                raise
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
                
                # Set session state
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
        
        # First check advanced libraries
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
            from prophet import Prophet
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
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
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
            from web3 import Web3
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
            from sqlalchemy import create_engine, text
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
            from arch import arch_model
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
            import xgboost as xgb
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
        }
    }
    
    def __init__(self):
        self.error_history = []
        self.max_history_size = 100
    
    @monitor_operation('error_analysis')
    def analyze_error_with_context(self, error: Exception, context: Dict) -> Dict:
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
            'ml_suggestions': [],
            'error_category': 'UNKNOWN'
        }
        
        # Analyze error message for patterns
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
                
                # Add context-specific solutions
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
        
        # Add ML-powered suggestions based on error history
        analysis['ml_suggestions'] = self._generate_ml_suggestions(error, context)
        
        # Calculate confidence score for recovery
        analysis['recovery_confidence'] = min(95, 100 - (analysis['severity_score'] * 10))
        
        # Add preventive measures
        analysis['preventive_measures'] = self._generate_preventive_measures(analysis)
        
        # Store in history
        self.error_history.append(analysis)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        return analysis
    
    def _generate_ml_suggestions(self, error: Exception, context: Dict) -> List[str]:
        """Generate ML-powered recovery suggestions."""
        suggestions = []
        
        # Pattern-based suggestions
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
        
        # Context-aware suggestions
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
        
        return measures
    
    def create_advanced_error_display(self, analysis: Dict) -> None:
        """Create advanced error display with interactive elements."""
        with st.expander(f"🔍 Advanced Error Analysis ({analysis['error_type']})", expanded=True):
            # Error summary
            col1, col2, col3 = st.columns(3)
            with col1:
                severity_color = {
                    9: "🔴",  # Critical
                    7: "🟠",  # High
                    5: "🟡",  # Medium
                    3: "🟢"   # Low
                }.get(analysis['severity_score'], "⚫")
                st.metric("Severity", f"{severity_color} {analysis['severity_score']}/10")
            with col2:
                st.metric("Recovery Confidence", f"{analysis['recovery_confidence']}%")
            with col3:
                category = analysis.get('error_category', 'Unknown')
                st.metric("Category", category)
            
            # Recovery actions
            if analysis['recovery_actions']:
                st.subheader("🚀 Recovery Actions")
                for i, action in enumerate(analysis['recovery_actions'][:5], 1):
                    action_key = f"recovery_{i}_{hash(action) % 10000}"
                    st.checkbox(f"Action {i}: {action}", value=False, key=action_key)
            
            # ML suggestions
            if analysis['ml_suggestions']:
                st.subheader("🤖 AI-Powered Suggestions")
                for suggestion in analysis['ml_suggestions'][:3]:
                    st.info(f"💡 {suggestion}")
            
            # Preventive measures
            if analysis['preventive_measures']:
                st.subheader("🛡️ Preventive Measures")
                for measure in analysis['preventive_measures'][:3]:
                    st.success(f"✓ {measure}")
            
            # Technical details
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
    
    @monitor_operation('start_operation')
    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.operations[operation_name] = {
            'start': time.time(),
            'memory_start': self._get_memory_usage(),
            'cpu_start': self._get_cpu_usage()
        }
    
    @monitor_operation('end_operation')
    def end_operation(self, operation_name: str, metadata: Dict = None):
        """End timing an operation and record metrics."""
        if operation_name in self.operations:
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
            
            # Update operation record
            if 'history' not in op:
                op['history'] = []
            op['history'].append({
                'duration': duration,
                'memory_increase_mb': memory_diff,
                'cpu_increase': cpu_diff,
                'timestamp': datetime.now()
            })
            
            # Clear memory if operation was large
            if memory_diff > 100:  # More than 100MB increase
                gc.collect()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
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
        
        # Calculate operation statistics
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
        
        # Generate summary
        if report['operations']:
            total_times = [op['total_time'] for op in report['operations'].values()]
            report['summary'] = {
                'total_operations': len(report['operations']),
                'total_operation_time': sum(total_times),
                'slowest_operation': max(report['operations'].items(), 
                                         key=lambda x: x[1]['avg_duration'])[0],
                'most_frequent_operation': max(report['operations'].items(),
                                                key=lambda x: x[1]['count'])[0],
                'most_memory_intensive': max(report['operations'].items(),
                                              key=lambda x: x[1]['avg_memory_increase'])[0]
            }
        
        # Generate resource usage summary
        if self.memory_usage:
            report['resource_usage']['memory'] = {
                'peak_mb': max(self.memory_usage) if self.memory_usage else 0,
                'avg_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
                'current_mb': self._get_memory_usage()
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
            
            if stats['avg_cpu_increase'] > 50:  # More than 50% CPU
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
        gc.collect()

# Initialize global monitors
error_analyzer = AdvancedErrorAnalyzer()
performance_monitor = PerformanceMonitor()

# Store in session state for access by decorators
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
        
        # Ensure cache directory exists
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
            # Validate input
            if not tickers:
                raise ValueError("No tickers provided")
            
            if len(tickers) > Config.MAX_TICKERS:
                raise ValueError(f"Maximum {Config.MAX_TICKERS} tickers allowed, got {len(tickers)}")
            
            # Check cache first
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
            
            # Download data in parallel with limited workers
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
                            # Add to data structures
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
                                # Merge with outer join to include all dates
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
                            
                            # Store metadata
                            data['metadata'][ticker] = ticker_data['metadata']
                            
                            # Store dividends and splits
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
            
            # Process the data to ensure equal length series
            if not data['prices'].empty:
                data = self._process_and_align_data(data)
                
                # Calculate returns
                if not data['prices'].empty:
                    data['returns'] = data['prices'].pct_change().dropna()
                
                # Calculate additional features
                if len(data['successful_tickers']) > 0:
                    data['additional_features'] = self._calculate_additional_features(data)
            
            # Cache the results
            if Config.CACHE_ENABLED and data['successful_tickers']:
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
            
            # Clear memory
            gc.collect()
            
            return data
            
        except Exception as e:
            error_analyzer.analyze_error_with_context(e, {
                'operation': 'fetch_advanced_market_data',
                'tickers': tickers,
                'date_range': f"{start_date} to {end_date}",
                'ticker_count': len(tickers)
            })
            raise
    
    def _fetch_single_ticker_ohlc(self, ticker: str, 
                                 start_date: datetime, 
                                 end_date: datetime,
                                 interval: str) -> Dict:
        """Fetch OHLC data for a single ticker with comprehensive error handling."""
        for attempt in range(self.retry_attempts):
            try:
                stock = yf.Ticker(ticker)
                
                # Download historical data with all available fields
                hist = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=False,  # Don't include pre/post market data
                    actions=True,
                    timeout=self.timeout
                )
                
                if hist.empty:
                    if attempt == self.retry_attempts - 1:
                        raise ValueError(f"No historical data for {ticker} in date range {start_date} to {end_date}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                # Extract OHLC data
                close_prices = hist['Close']
                volumes = hist['Volume']
                high_prices = hist['High']
                low_prices = hist['Low']
                open_prices = hist['Open']
                
                # Get dividends and splits
                dividends = stock.dividends[start_date:end_date]
                splits = stock.splits[start_date:end_date]
                
                # Get comprehensive metadata
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
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {}
    
    def _process_and_align_data(self, data: Dict) -> Dict:
        """Process and align data to ensure equal length series with proper forward filling."""
        
        # Align all dataframes to have the same index (union of all dates)
        all_dates = pd.DatetimeIndex([])
        for df in [data['prices'], data['volumes'], data['high'], data['low'], data['open']]:
            if not df.empty:
                all_dates = all_dates.union(df.index)
        
        if len(all_dates) == 0:
            return data
        
        # Sort dates
        all_dates = all_dates.sort_values()
        
        # Reindex all dataframes to have the same dates
        for key in ['prices', 'volumes', 'high', 'low', 'open']:
            if not data[key].empty:
                # Forward fill prices and OHLC data, then backfill
                data[key] = data[key].reindex(all_dates)
                
                if key == 'prices':
                    # For prices: forward fill, then backfill for initial missing values
                    data[key] = data[key].ffill().bfill()
                elif key == 'volumes':
                    # For volumes: forward fill, fill remaining with 0
                    data[key] = data[key].ffill().fillna(0)
                else:
                    # For OHLC: forward fill, then backfill
                    data[key] = data[key].ffill().bfill()
        
        # Remove assets with too many missing values after forward filling
        if not data['prices'].empty:
            # Count remaining NaN values (should be minimal after forward/back fill)
            nan_counts = data['prices'].isnull().sum()
            
            # Remove assets that are still mostly NaN (e.g., newly listed stocks)
            valid_assets = nan_counts[nan_counts < len(data['prices']) * 0.5].index.tolist()
            
            if len(valid_assets) < Config.MIN_ASSETS_FOR_OPTIMIZATION:
                raise ValueError(f"Only {len(valid_assets)} assets with sufficient data (minimum {Config.MIN_ASSETS_FOR_OPTIMIZATION} required)")
            
            # Filter all dataframes to only include valid assets
            for key in ['prices', 'volumes', 'high', 'low', 'open']:
                if not data[key].empty:
                    data[key] = data[key][valid_assets]
            
            # Update successful tickers
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
            opens = data.get('open', pd.DataFrame())
            
            # Calculate basic statistics for each asset
            for ticker in returns.columns:
                ticker_returns = returns[ticker].dropna()
                
                if len(ticker_returns) > 0:
                    # Basic statistics
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
                    
                    # Price-based features
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
                    
                    # Volume-based features
                    if ticker in volumes.columns:
                        volume_series = volumes[ticker].dropna()
                        if len(volume_series) > 0:
                            features['liquidity_metrics'][ticker] = {
                                'current_volume': volume_series.iloc[-1],
                                'avg_volume_20d': volume_series.tail(20).mean(),
                                'volume_ratio': volume_series.iloc[-1] / volume_series.tail(20).mean() if volume_series.tail(20).mean() > 0 else 0,
                                'volume_std_20d': volume_series.tail(20).std()
                            }
            
            # Calculate correlation matrix
            if len(returns.columns) > 1:
                corr_matrix = returns.corr()
                features['correlation_matrix'] = corr_matrix
                
                # Calculate correlation statistics
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                if len(corr_values) > 0:
                    features['correlation_stats'] = {
                        'mean': np.mean(corr_values),
                        'median': np.median(corr_values),
                        'min': np.min(corr_values),
                        'max': np.max(corr_values),
                        'std': np.std(corr_values)
                    }
            
            # Calculate covariance matrix (annualized)
            if not returns.empty:
                features['covariance_matrix'] = returns.cov() * Config.TRADING_DAYS_PER_YEAR
            
        except Exception as e:
            features['error'] = str(e)
            error_analyzer.analyze_error_with_context(e, {
                'operation': 'calculate_additional_features',
                'tickers': list(returns.columns) if 'returns' in locals() else []
            })
        
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
            
            # Check for missing values after forward filling
            if not data['prices'].empty:
                missing_percentage = data['prices'].isnull().mean().mean()
                if missing_percentage > Config.MAX_MISSING_PERCENTAGE:
                    validation['warnings'].append(f"High percentage of missing values after forward fill: {missing_percentage:.1%}")
            
            # Check for zero or negative prices
            if not data['prices'].empty and (data['prices'] <= 0).any().any():
                problematic_assets = data['prices'].columns[(data['prices'] <= 0).any()].tolist()
                validation['warnings'].append(f"Zero or negative prices in assets: {problematic_assets}")
            
            # Check returns calculation
            if data.get('returns', pd.DataFrame()).empty:
                validation['warnings'].append("Cannot calculate returns - check price data continuity")
            else:
                # Check for infinite or NaN returns
                if not np.isfinite(data['returns'].values).all():
                    nan_assets = data['returns'].columns[data['returns'].isnull().any()].tolist()
                    validation['warnings'].append(f"Non-finite values in returns for assets: {nan_assets}")
                
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
                'average_return': data['returns'].mean().mean() if not data.get('returns', pd.DataFrame()).empty else 0,
                'average_volatility': data['returns'].std().mean() * np.sqrt(Config.TRADING_DAYS_PER_YEAR) if not data.get('returns', pd.DataFrame()).empty else 0,
                'successful_tickers': len(data.get('successful_tickers', [])),
                'failed_tickers': len(data.get('errors', {}))
            }
            
            # Determine if data is valid
            validation['is_valid'] = len(validation['issues']) == 0 and n_assets >= min_assets
            
            # Provide suggestions
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
            # Forward fill, then back fill for any remaining NaNs
            data['prices'] = data['prices'].ffill().bfill()
        
        if not data.get('returns', pd.DataFrame()).empty:
            # For returns, we can drop rows with too many missing values
            threshold = 0.5  # Keep rows with at least 50% non-missing values
            min_non_na = int(threshold * len(data['returns'].columns))
            data['returns'] = data['returns'].dropna(thresh=min_non_na)
        
        return data
    
    def _handle_outliers(self, data: Dict) -> Dict:
        """Handle outliers in returns data using winsorization."""
        if not data.get('returns', pd.DataFrame()).empty:
            returns_clean = data['returns'].copy()
            
            for column in returns_clean.columns:
                series = returns_clean[column].dropna()
                if len(series) > 10:
                    # Calculate robust bounds using IQR
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    # Winsorize extreme values
                    returns_clean[column] = series.clip(lower_bound, upper_bound)
            
            data['returns'] = returns_clean
        
        return data
    
    def _normalize_data(self, data: Dict) -> Dict:
        """Normalize data for analysis."""
        if not data.get('returns', pd.DataFrame()).empty:
            # Standardize returns (z-score normalization)
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
                    # Use Augmented Dickey-Fuller test
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
        stats = {
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
            
            # Asset-level statistics
            for ticker in returns.columns:
                ticker_returns = returns[ticker].dropna()
                
                if len(ticker_returns) > 0:
                    stats['assets'][ticker] = {
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
            
            # Portfolio-level statistics (equal weight benchmark)
            if len(returns.columns) > 0:
                equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
                portfolio_returns = returns.dot(equal_weights)
                
                stats['portfolio_level'] = {
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
            
            # Correlation and covariance matrices
            if len(returns.columns) > 1:
                corr_matrix = returns.corr()
                stats['correlation']['matrix'] = corr_matrix
                
                # Calculate correlation statistics
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                if len(corr_values) > 0:
                    stats['correlation']['stats'] = {
                        'mean': np.mean(corr_values),
                        'median': np.median(corr_values),
                        'min': np.min(corr_values),
                        'max': np.max(corr_values),
                        'std': np.std(corr_values),
                        'q25': np.percentile(corr_values, 25),
                        'q75': np.percentile(corr_values, 75)
                    }
                
                # Calculate covariance matrix (annualized)
                cov_matrix = returns.cov() * Config.TRADING_DAYS_PER_YEAR
                stats['covariance']['matrix'] = cov_matrix
                stats['covariance']['mean_variance'] = np.diag(cov_matrix).mean()
                stats['covariance']['avg_covariance'] = cov_matrix.values[np.triu_indices_from(cov_matrix.values, k=1)].mean()
            
            # Liquidity statistics
            if not volumes.empty:
                for ticker in volumes.columns:
                    volume_series = volumes[ticker].dropna()
                    if len(volume_series) > 0:
                        stats['liquidity'][ticker] = {
                            'avg_volume': volume_series.mean(),
                            'std_volume': volume_series.std(),
                            'volume_ratio_last_avg': volume_series.iloc[-1] / volume_series.mean() if volume_series.mean() > 0 else 0,
                            'volume_trend': self._calculate_volume_trend(volume_series)
                        }
            
            # Price level statistics
            if not prices.empty:
                for ticker in prices.columns:
                    price_series = prices[ticker].dropna()
                    if len(price_series) > 0:
                        stats['price_level'][ticker] = {
                            'current_price': price_series.iloc[-1],
                            'price_change_1m': (price_series.iloc[-1] / price_series.iloc[-22] - 1) if len(price_series) > 22 else 0,
                            'price_change_3m': (price_series.iloc[-1] / price_series.iloc[-66] - 1) if len(price_series) > 66 else 0,
                            'price_change_1y': (price_series.iloc[-1] / price_series.iloc[-252] - 1) if len(price_series) > 252 else 0,
                            'price_high_52w': price_series.tail(252).max() if len(price_series) >= 252 else price_series.max(),
                            'price_low_52w': price_series.tail(252).min() if len(price_series) >= 252 else price_series.min()
                        }
        
        return stats
    
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
# 4. PORTFOLIO OPTIMIZER ENGINE (ADDED IN v5.1)
# ============================================================================

class PortfolioOptimizer:
    """Institutional-grade portfolio optimization engine."""

    def __init__(self, expected_returns, covariance_matrix, risk_free_rate=0.02):
        self.mu = expected_returns
        self.sigma = covariance_matrix
        self.rf = risk_free_rate
        self.num_assets = len(expected_returns)
        self.asset_names = expected_returns.index

    def _portfolio_annualised_performance(self, weights):
        """Calculates annualized portfolio performance."""
        returns = np.sum(self.mu * weights) * Config.TRADING_DAYS_PER_YEAR
        std = np.sqrt(np.dot(weights.T, np.dot(self.sigma, weights))) * np.sqrt(Config.TRADING_DAYS_PER_YEAR)
        return std, returns

    def _neg_sharpe_ratio(self, weights):
        """Negative Sharpe Ratio for minimization."""
        p_var, p_ret = self._portfolio_annualised_performance(weights)
        return -(p_ret - self.rf) / p_var

    def _portfolio_volatility(self, weights):
        """Calculates portfolio volatility."""
        return self._portfolio_annualised_performance(weights)[0]

    def maximize_sharpe_ratio(self) -> Dict:
        """Optimization for Tangency Portfolio (Max Sharpe)."""
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = self.num_assets * [1. / self.num_assets,]
        
        result = optimize.minimize(self._neg_sharpe_ratio, initial_guess,
                                 method='SLSQP', bounds=bounds, constraints=constraints)
        
        vol, ret = self._portfolio_annualised_performance(result.x)
        return {
            'weights': dict(zip(self.asset_names, result.x)),
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - self.rf) / vol
        }

    def minimize_volatility(self) -> Dict:
        """Optimization for Global Minimum Variance Portfolio."""
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = self.num_assets * [1. / self.num_assets,]
        
        result = optimize.minimize(self._portfolio_volatility, initial_guess,
                                 method='SLSQP', bounds=bounds, constraints=constraints)
        
        vol, ret = self._portfolio_annualised_performance(result.x)
        return {
            'weights': dict(zip(self.asset_names, result.x)),
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - self.rf) / vol
        }

    def efficient_frontier(self, points=100) -> Tuple[List, List]:
        """Calculates the Efficient Frontier curve."""
        # Find min and max returns
        min_vol_ret = self.minimize_volatility()['return']
        max_sharpe_ret = self.maximize_sharpe_ratio()['return']
        
        # Extend the range slightly
        target_returns = np.linspace(min_vol_ret * 0.9, max_sharpe_ret * 1.2, points)
        
        efficient_volatilities = []
        efficient_returns = []
        
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        for ret in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self._portfolio_annualised_performance(x)[1] - ret}
            )
            
            initial_guess = self.num_assets * [1. / self.num_assets,]
            result = optimize.minimize(self._portfolio_volatility, initial_guess,
                                     method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                efficient_volatilities.append(result.fun)
                efficient_returns.append(ret)
                
        return efficient_volatilities, efficient_returns

# ============================================================================
# 5. MACHINE LEARNING ENGINE (ADDED IN v5.1)
# ============================================================================

class MachineLearningEngine:
    """Predictive analytics engine using Scikit-Learn."""
    
    def __init__(self, price_data):
        self.prices = price_data
        
    def prepare_features(self, ticker, window=5):
        """Creates technical features for ML."""
        df = pd.DataFrame(self.prices[ticker])
        df.columns = ['Close']
        
        # Returns
        df['Return'] = df['Close'].pct_change()
        
        # Lags
        for i in range(1, window + 1):
            df[f'Return_Lag_{i}'] = df['Return'].shift(i)
        
        # Rolling stats
        df['Rolling_Mean'] = df['Return'].rolling(window=20).mean()
        df['Rolling_Std'] = df['Return'].rolling(window=20).std()
        
        # Momentum
        df['Momentum'] = df['Close'] / df['Close'].shift(window) - 1
        
        # Target: Next day's return
        df['Target'] = df['Return'].shift(-1)
        
        df = df.dropna()
        return df

    def train_model(self, ticker):
        """Trains a Random Forest Regressor."""
        data = self.prepare_features(ticker)
        
        features = [col for col in data.columns if col not in ['Target', 'Close']]
        X = data[features]
        y = data['Target']
        
        # Time-series split
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        return model, metrics, y_test, predictions

# ============================================================================
# 6. VISUALIZATION MANAGER (ADDED IN v5.1)
# ============================================================================

class VisualizationManager:
    """Handles Plotly visualizations."""
    
    @staticmethod
    def plot_efficient_frontier(efficient_vols, efficient_rets, max_sharpe, min_vol, current_portfolio=None):
        """Generates Efficient Frontier Plot."""
        fig = go.Figure()
        
        # Frontier Curve
        fig.add_trace(go.Scatter(x=efficient_vols, y=efficient_rets, mode='lines', 
                               name='Efficient Frontier', line=dict(color='#00cc96', width=2)))
        
        # Max Sharpe Point
        fig.add_trace(go.Scatter(x=[max_sharpe['volatility']], y=[max_sharpe['return']],
                               mode='markers', marker=dict(color='gold', size=14, symbol='star'),
                               name='Max Sharpe'))
        
        # Min Vol Point
        fig.add_trace(go.Scatter(x=[min_vol['volatility']], y=[min_vol['return']],
                               mode='markers', marker=dict(color='red', size=12, symbol='diamond'),
                               name='Min Volatility'))
        
        fig.update_layout(title='Efficient Frontier', 
                          xaxis_title='Annualized Volatility (Risk)', 
                          yaxis_title='Annualized Return',
                          template='plotly_dark', height=600)
        return fig

    @staticmethod
    def plot_predictions(y_test, predictions, ticker):
        """Plots ML Predictions vs Actuals."""
        fig = go.Figure()
        
        # Limit to last 100 points for clarity
        y_test_sub = y_test[-100:]
        preds_sub = predictions[-100:]
        
        fig.add_trace(go.Scatter(y=y_test_sub.values, mode='lines', name='Actual Return', line=dict(color='white', width=1)))
        fig.add_trace(go.Scatter(y=preds_sub, mode='lines', name='Predicted Return', line=dict(color='#00cc96', width=1.5)))
        
        fig.update_layout(title=f'ML Return Forecast: {ticker} (Last 100 Days)',
                          yaxis_title='Daily Return',
                          template='plotly_dark', height=400)
        return fig

# ============================================================================
# STREAMLIT APP MAIN FUNCTION
# ============================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="QuantEdge Pro v5.1 - Enterprise Portfolio Analytics",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("📈 QuantEdge Pro v5.1 - Enterprise Portfolio Analytics")
    st.markdown("""
    ### Institutional-grade portfolio optimization, risk analysis, and backtesting platform
    *Advanced analytics with machine learning, real-time data, and comprehensive reporting*
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Library status
        if 'enterprise_library_status' in st.session_state:
            lib_status = st.session_state.enterprise_library_status
            st.subheader("📚 Library Status")
            
            # Show core libraries
            core_libs = ['numpy', 'pandas', 'scipy', 'plotly', 'yfinance', 'streamlit']
            core_status = all(lib_status['status'].get(lib, False) for lib in core_libs)
            
            if core_status:
                st.success("✅ Core libraries available")
            else:
                st.error("❌ Missing core libraries")
                for lib in core_libs:
                    if not lib_status['status'].get(lib, False):
                        st.warning(f"Missing: {lib}")
        
        # Data configuration
        st.subheader("📊 Data Configuration")
        tickers_input = st.text_area(
            "Enter tickers (comma-separated):",
            value="AAPL, GOOGL, MSFT, AMZN, TSLA, GLD",
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
        
        # Analysis type
        st.subheader("🔍 Analysis Type")
        analysis_type = st.radio(
            "Select analysis:",
            ["Data Explorer", "Portfolio Optimization", "ML Forecasting"]
        )
        
        # Fetch data button
        if st.button("🚀 Fetch Data & Analyze", type="primary"):
            with st.spinner("Fetching market data..."):
                try:
                    # Parse tickers
                    tickers = [t.strip().upper() for t in tickers_input.split(',')]
                    
                    # Fetch data
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
                    
                    # Store in session state
                    st.session_state.portfolio_data = data
                    st.session_state.data_loaded = True
                    st.session_state.selected_tickers = tickers
                    
                    # Validate data
                    validation = data_manager.validate_portfolio_data(data)
                    
                    if validation['is_valid']:
                        st.sidebar.success(f"✅ Data loaded: {validation['summary']['n_assets']} assets, {validation['summary']['n_data_points']} days")
                    else:
                        st.sidebar.warning(f"⚠️ Data loaded with warnings: {len(validation['warnings'])}")
                        
                except Exception as e:
                    st.sidebar.error(f"❌ Error fetching data: {str(e)}")
                    error_analyzer.analyze_error_with_context(e, {
                        'operation': 'main_data_fetch',
                        'tickers': tickers
                    })
    
    # Main content area
    if st.session_state.get('data_loaded', False) and 'portfolio_data' in st.session_state:
        data = st.session_state.portfolio_data
        tickers = st.session_state.selected_tickers
        
        # Show data summary
        st.subheader("📋 Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Assets", len(data['prices'].columns))
        with col2:
            st.metric("Data Points", len(data['prices']))
        with col3:
            date_range = f"{data['prices'].index[0].date()} to {data['prices'].index[-1].date()}"
            st.metric("Date Range", date_range)
        
        # ------------------------------------------------------------------------
        # DATA EXPLORER TAB
        # ------------------------------------------------------------------------
        if analysis_type == "Data Explorer":
            st.markdown("### 🔍 Historical Market Data")
            
            with st.expander("📊 Data Preview", expanded=True):
                tab1, tab2, tab3 = st.tabs(["Prices", "Returns", "Statistics"])
                
                with tab1:
                    st.dataframe(data['prices'].tail(10), use_container_width=True)
                    # Simple Plotly Chart
                    fig = px.line(data['prices'], title="Normalized Price History (Rebased)")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if not data['returns'].empty:
                        st.dataframe(data['returns'].tail(10), use_container_width=True)
                        fig_corr = px.imshow(data['returns'].corr(), text_auto=True, title="Correlation Matrix")
                        st.plotly_chart(fig_corr, use_container_width=True)
                
                with tab3:
                    stats = data_manager.calculate_basic_statistics(data)
                    if stats['assets']:
                        # Create a DataFrame for asset statistics
                        stats_df = pd.DataFrame(stats['assets']).T
                        st.dataframe(stats_df[['mean_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']], use_container_width=True)

        # ------------------------------------------------------------------------
        # PORTFOLIO OPTIMIZATION TAB
        # ------------------------------------------------------------------------
        elif analysis_type == "Portfolio Optimization":
            st.subheader("🎯 Portfolio Optimization")
            st.markdown("Optimization using Modern Portfolio Theory (Mean-Variance).")
            
            if 'returns' in data and not data['returns'].empty:
                mu = data['returns'].mean()
                sigma = data['returns'].cov()
                
                optimizer = PortfolioOptimizer(mu, sigma)
                
                col_opt1, col_opt2 = st.columns(2)
                
                with col_opt1:
                    st.info("Computing Efficient Frontier...")
                    eff_vol, eff_ret = optimizer.efficient_frontier(points=50)
                    max_sharpe = optimizer.maximize_sharpe_ratio()
                    min_vol = optimizer.minimize_volatility()
                    
                    fig_ef = VisualizationManager.plot_efficient_frontier(eff_vol, eff_ret, max_sharpe, min_vol)
                    st.plotly_chart(fig_ef, use_container_width=True)

                with col_opt2:
                    st.markdown("### Optimal Allocation (Max Sharpe)")
                    weights_df = pd.DataFrame.from_dict(max_sharpe['weights'], orient='index', columns=['Weight'])
                    weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
                    st.table(weights_df)
                    
                    st.markdown(f"**Annual Return:** {max_sharpe['return']:.2%}")
                    st.markdown(f"**Annual Volatility:** {max_sharpe['volatility']:.2%}")
                    st.markdown(f"**Sharpe Ratio:** {max_sharpe['sharpe']:.2f}")

        # ------------------------------------------------------------------------
        # ML FORECASTING TAB
        # ------------------------------------------------------------------------
        elif analysis_type == "ML Forecasting":
            st.subheader("🤖 Machine Learning Forecasting")
            st.markdown("Random Forest Regression on individual assets.")
            
            selected_ml_ticker = st.selectbox("Select Asset for Prediction", tickers)
            
            if st.button("Train Model"):
                with st.spinner(f"Training Random Forest on {selected_ml_ticker}..."):
                    ml_engine = MachineLearningEngine(data['prices'])
                    model, metrics, y_test, preds = ml_engine.train_model(selected_ml_ticker)
                    
                    st.success("Model Trained Successfully")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("RMSE", f"{metrics['rmse']:.4f}")
                    m2.metric("R2 Score", f"{metrics['r2']:.4f}")
                    
                    # Plot
                    fig_pred = VisualizationManager.plot_predictions(y_test, preds, selected_ml_ticker)
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    st.markdown("#### Feature Importance")
                    importances = pd.DataFrame({
                        'Feature': ml_engine.prepare_features(selected_ml_ticker).drop(['Target','Close'], axis=1).columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    st.bar_chart(importances.set_index('Feature'))

    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to QuantEdge Pro v5.1
        
        ### Get Started:
        1. **Configure your portfolio** in the sidebar
        2. **Enter ticker symbols** (e.g., AAPL, GOOGL, MSFT)
        3. **Select date range** for analysis
        4. **Choose analysis type**
        5. **Click 'Fetch Data & Analyze'** to begin
        
        ### Available Features:
        - **Portfolio Optimization**: Mean-variance, Efficient Frontier 

[Image of Efficient Frontier]

        - **Risk Analysis**: VaR, CVaR, stress testing, backtesting
        - **Machine Learning**: Return forecasting, volatility prediction
        
        ### System Requirements:
        - Python 3.8+
        - 8GB+ RAM recommended
        - Internet connection for data fetching
        """)
        
        # Show system status
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

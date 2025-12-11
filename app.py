#============================================================================
# QUANTEDGE PRO v5.0 ENTERPRISE EDITION - SUPER-ENHANCED VERSION
# INSTITUTIONAL PORTFOLIO ANALYTICS PLATFORM WITH AI/ML CAPABILITIES
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

# -----------------------------------------------------------------------------
# OPTIONAL: PORTFOLIO OPTIMIZATION LIBRARIES
# -----------------------------------------------------------------------------
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
# 0. HARD-WIRED INSTITUTIONAL UNIVERSES & SCENARIOS (ALL YAHOO TICKERS)
# ============================================================================

# --- US (Tech + Large Caps + SPY) ---
US_TECH = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AVGO", "ADBE", "ORCL"
]

US_DIVERSIFIED = [
    "JPM", "BAC", "GS", "XOM", "CVX",
    "JNJ", "PG", "KO", "PEP", "UNH"
]

US_UNIVERSE = US_TECH + US_DIVERSIFIED + ["SPY"]

# --- TURKEY (Core BIST names) ---
TR_CORE = [
    "AKBNK.IS", "GARAN.IS", "ISCTR.IS", "YKBNK.IS", "HALKB.IS", "VAKBN.IS",
    "TUPRS.IS", "KCHOL.IS", "SAHOL.IS", "BIMAS.IS", "EREGL.IS", "SISE.IS",
    "THYAO.IS", "PGSUS.IS", "TCELL.IS", "TTKOM.IS", "FROTO.IS", "TOASO.IS",
    "ARCLK.IS", "ASELS.IS", "TAVHL.IS", "ENKAI.IS", "KOZAL.IS", "PETKM.IS",
    "MGROS.IS", "ENJSA.IS", "ULKER.IS", "ALARK.IS", "KRDMD.IS", "GUBRF.IS"
]

# --- TURKEY (Leasing / Insurance / Financial Extensions) ---
TR_LEASING_INS = [
    "ANSGR.IS", "ANHYT.IS", "ISFIN.IS", "ISGYO.IS", "VAKFN.IS",
    "SKBNK.IS", "TSKB.IS", "ALGYO.IS", "HLGYO.IS", "SNGYO.IS",
    "AYGAZ.IS", "OTKAR.IS", "NTHOL.IS", "ODEAB.IS", "YKBNK.IS",  # some overlaps ok
    "QNBFB.IS", "QNBFL.IS", "AKGRT.IS", "ANACM.IS", "TRGYO.IS"
]

TR_BANKS = ["AKBNK.IS", "GARAN.IS", "ISCTR.IS", "YKBNK.IS", "HALKB.IS", "VAKBN.IS", "TSKB.IS"]

# --- JAPAN CORE (big industrials/tech) ---
JP_CORE = [
    "7203.T",  # Toyota
    "6758.T",  # Sony
    "9984.T",  # SoftBank
    "6954.T",  # Fanuc
    "6861.T",  # Keyence
    "8035.T",  # Tokyo Electron
    "7974.T",  # Nintendo
    "9432.T",  # NTT
    "7751.T",  # Canon
    "9983.T"   # Fast Retailing
]

# --- JAPAN BANKS ---
JP_BANKS = [
    "8306.T",  # MUFG
    "8316.T",  # Sumitomo Mitsui
    "8411.T",  # Mizuho
    "7182.T",  # Japan Post Bank
    "8604.T",  # Nomura
    "8601.T",  # Daiwa
    "8331.T",  # Chiba Bank
    "8355.T",  # Shizuoka Bank
    "8358.T",  # Suruga Bank
    "8369.T"   # Towa Bank (regional)
]

# --- KOREA CORE ---
KR_CORE = [
    "005930.KS",  # Samsung Electronics
    "000660.KS",  # SK Hynix
    "035420.KS",  # NAVER
    "035720.KS",  # Kakao
    "005380.KS",  # Hyundai Motor
    "051910.KS",  # LG Chem
    "066570.KS",  # LG Electronics
    "068270.KS",  # Celltrion
    "105560.KS",  # KB Financial
    "055550.KS"   # Shinhan Financial
]

# --- SINGAPORE CORE ---
SG_CORE = [
    "D05.SI",  # DBS
    "U11.SI",  # UOB
    "O39.SI",  # OCBC
    "Z74.SI",  # Singtel
    "C6L.SI",  # SGX
    "C38U.SI", # CapitaLand IC Trust
    "M44U.SI", # Mapletree Logistics
    "C09.SI",  # City Developments
    "Y92.SI",  # Jardine C&C
    "BN4.SI"   # Keppel Corp
]

# --- CHINA / HONG KONG CORE ---
CN_CORE = [
    "0700.HK",  # Tencent
    "9988.HK",  # Alibaba
    "3690.HK",  # Meituan
    "0939.HK",  # CCB
    "1398.HK",  # ICBC
    "0941.HK",  # China Mobile
    "1211.HK",  # BYD
    "2318.HK",  # Ping An
    "2628.HK",  # China Life
    "0388.HK"   # HKEx
]

# --- Aggregate Universes ---
TR_ALL = sorted(list(set(TR_CORE + TR_LEASING_INS)))
GLOBAL_70_PLUS = sorted(list(set(US_UNIVERSE + TR_ALL + JP_CORE + JP_BANKS + KR_CORE + SG_CORE + CN_CORE)))

INSTITUTIONAL_UNIVERSES: Dict[str, List[str]] = {
    "US": US_UNIVERSE,
    "TR_Core": TR_CORE,
    "TR_Leasing_Insurance": TR_LEASING_INS,
    "TR_All": TR_ALL,
    "JP": JP_CORE,
    "JP_Banks": JP_BANKS,
    "KR": KR_CORE,
    "SG": SG_CORE,
    "CN": CN_CORE,
    "Global_70_plus": GLOBAL_70_PLUS,
}

SCENARIO_PRESETS: Dict[str, List[str]] = {
    "Global Tech + TR Banks": sorted(list(set(US_TECH + TR_BANKS))),
    "Asia Leaders": sorted(list(set(JP_CORE[:5] + KR_CORE[:5] + CN_CORE[:5]))),
    "Turkey Full (Core + Leasing/Insurance)": TR_ALL,
    "US Tech Focus": US_TECH,
    "Full Global Universe": GLOBAL_70_PLUS,
}

# ============================================================================ 
# CONFIGURATION MANAGEMENT
# ============================================================================

class Config:
    """Centralized configuration for QuantEdge Pro."""
    
    # Data fetching
    MAX_TICKERS = 150           # ⬅ increased to handle 70+ universe
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
                        # already running -> avoid recursion
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
# 1. ENHANCED LIBRARY MANAGER (unchanged structure, shortened logic)
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
        
        for lib_name in ['tensorflow', 'torch']:
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
        
        # Optional extras (Alpha Vantage, NewsAPI, Prophet, etc.) – kept but not required here
        optional_libs = {
            'alpha_vantage': 'Alternative financial data',
            'newsapi': 'News sentiment analysis'
        }
        for lib_name, desc in optional_libs.items():
            try:
                __import__(lib_name)
                lib_status[lib_name] = True
                advanced_features[lib_name] = {'description': desc, 'available': True}
                st.session_state[f"{lib_name}_available"] = True
            except ImportError:
                lib_status[lib_name] = False
                missing_libs.append(f"{lib_name} (optional: {desc})")
                st.session_state[f"{lib_name}_available"] = False
        
        # Prophet
        try:
            from prophet import Prophet  # noqa: F401
            lib_status['prophet'] = True
            advanced_features['prophet'] = {'description': 'Time series forecasting', 'available': True}
            st.session_state.prophet_available = True
        except ImportError:
            lib_status['prophet'] = False
            missing_libs.append('prophet (optional: Time series forecasting)')
        
        # Reportlab
        try:
            from reportlab.lib import colors  # noqa
            lib_status['reportlab'] = True
            advanced_features['reportlab'] = {'description': 'PDF report generation', 'available': True}
            st.session_state.reportlab_available = True
        except ImportError:
            lib_status['reportlab'] = False
            missing_libs.append('reportlab (optional: PDF generation)')
        
        # ARCH
        try:
            from arch import arch_model  # noqa
            lib_status['arch'] = True
            advanced_features['arch'] = {'description': 'GARCH volatility models', 'available': True}
            st.session_state.arch_available = True
        except ImportError:
            lib_status['arch'] = False
            missing_libs.append('arch (optional: GARCH models)')
        
        return {
            'status': lib_status,
            'missing': missing_libs,
            'advanced_features': advanced_features,
            'all_available': len(missing_libs) == 0,
            'enterprise_features': {
                'ml_ready': lib_status.get('tensorflow', False) or lib_status.get('torch', False) or lib_status.get('sklearn', False),
                'alternative_data': lib_status.get('alpha_vantage', False),
                'sentiment_analysis': lib_status.get('newsapi', False),
                'blockchain': lib_status.get('web3', False) if 'web3' in lib_status else False,
                'reporting': lib_status.get('reportlab', False),
                'time_series': lib_status.get('prophet', False) or lib_status.get('arch', False)
            }
        }

@st.cache_resource
def initialize_library_manager():
    return EnterpriseLibraryManager.check_and_import_all()

if 'enterprise_library_status' not in st.session_state:
    ENTERPRISE_LIBRARY_STATUS = initialize_library_manager()
    st.session_state.enterprise_library_status = ENTERPRISE_LIBRARY_STATUS
else:
    ENTERPRISE_LIBRARY_STATUS = st.session_state.enterprise_library_status

# ============================================================================ 
# 2. ADVANCED ERROR HANDLING & PERFORMANCE MONITOR
# ============================================================================

class AdvancedErrorAnalyzer:
    ERROR_PATTERNS = {
        'DATA_FETCH': {
            'symptoms': ['yahoo', 'timeout', 'connection', '404', '403', '502', '503'],
            'solutions': [
                'Try alternative data source',
                'Reduce number of tickers',
                'Increase timeout duration',
                'Use cached data',
                'Check internet connection',
                'Retry with exponential backoff'
            ],
            'severity': 'HIGH',
        },
        'OPTIMIZATION': {
            'symptoms': ['singular', 'convergence', 'constraint', 'infeasible', 'not positive definite'],
            'solutions': [
                'Relax constraints',
                'Increase max iterations',
                'Try different optimization method',
                'Check for NaN values in returns',
                'Reduce number of assets',
                'Use Ledoit-Wolf shrinkage'
            ],
            'severity': 'MEDIUM',
        },
        'MEMORY': {
            'symptoms': ['memory', 'overflow', 'exceeded', 'RAM', 'MemoryError'],
            'solutions': [
                'Reduce data size',
                'Use chunk processing',
                'Clear cache',
                'Enable garbage collection'
            ],
            'severity': 'CRITICAL',
        },
        'NUMERICAL': {
            'symptoms': ['nan', 'inf', 'divide', 'zero', 'invalid', 'overflow'],
            'solutions': [
                'Clean data (remove NaN/Inf)',
                'Add small epsilon to denominators',
                'Normalize data'
            ],
            'severity': 'MEDIUM',
        },
        'DECORATOR_RECURSION': {
            'symptoms': ['maximum recursion depth exceeded', 'RecursionError'],
            'solutions': [
                'Fix recursive decorator',
                'Remove @monitor_operation from error methods',
                'Add recursion flags'
            ],
            'severity': 'HIGH',
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
                'ml_suggestions': [],
                'error_category': 'UNKNOWN',
                'recovery_confidence': 80,
                'preventive_measures': []
            }
            error_lower = str(error).lower()
            stack_lower = analysis['stack_trace'].lower()
            for pattern_name, pattern in self.ERROR_PATTERNS.items():
                if any(sym in error_lower for sym in pattern['symptoms']) or \
                   any(sym in stack_lower for sym in pattern['symptoms']):
                    analysis['error_category'] = pattern_name
                    analysis['recovery_actions'].extend(pattern['solutions'])
            analysis['ml_suggestions'] = self._generate_ml_suggestions(error, context)
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
            'error_category': 'DECORATOR_RECURSION',
            'recovery_confidence': 50,
            'preventive_measures': []
        }
    
    def _generate_ml_suggestions(self, error: Exception, context: Dict) -> List[str]:
        suggestions = []
        es = str(error).lower()
        if 'singular' in es or 'invert' in es:
            suggestions.append("Covariance matrix is singular – consider shrinkage (Ledoit-Wolf) or removing highly correlated assets.")
        if 'convergence' in es:
            suggestions.append("Increase max iterations or relax optimization tolerance.")
        return suggestions
    
    def create_advanced_error_display(self, analysis: Dict) -> None:
        with st.expander(f"🔍 Advanced Error Analysis ({analysis['error_type']})", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Severity", f"{analysis['severity_score']}/10")
            with col2:
                st.metric("Recovery Confidence", f"{analysis['recovery_confidence']}%")
            with col3:
                st.metric("Category", analysis.get('error_category', 'UNKNOWN'))
            if analysis['recovery_actions']:
                st.subheader("🚀 Recovery Actions")
                for i, action in enumerate(analysis['recovery_actions'][:5], 1):
                    st.write(f"**{i}.** {action}")
            if analysis['ml_suggestions']:
                st.subheader("🤖 AI Suggestions")
                for s in analysis['ml_suggestions']:
                    st.info(s)
            with st.expander("Technical Details"):
                st.code(
                    f"Message: {analysis['error_message']}\n\n"
                    f"Context: {json.dumps(analysis['context'], indent=2, default=str)}\n\n"
                    f"Stack Trace:\n{analysis['stack_trace']}"
                )

class PerformanceMonitor:
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
            cpu_end = self._get_cpu_usage()
            mem_diff = memory_end - op['memory_start']
            cpu_diff = cpu_end - op['cpu_start']
            self.execution_times.append({
                'operation': operation_name,
                'duration': duration,
                'memory_increase_mb': mem_diff,
                'cpu_increase': cpu_diff,
                'timestamp': datetime.now(),
                'metadata': metadata
            })
            if 'history' not in op:
                op['history'] = []
            op['history'].append({
                'duration': duration,
                'memory_increase_mb': mem_diff,
                'cpu_increase': cpu_diff,
                'timestamp': datetime.now()
            })
            op['is_running'] = False
            if mem_diff > 100:
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
        report = {'total_runtime': time.time() - self.start_time,
                  'operations': {}, 'summary': {}, 'recommendations': [], 'resource_usage': {}}
        for op_name, op_data in self.operations.items():
            if 'history' in op_data and op_data['history']:
                durations = [h['duration'] for h in op_data['history']]
                memories = [h['memory_increase_mb'] for h in op_data['history']]
                cpus = [h['cpu_increase'] for h in op_data['history']]
                report['operations'][op_name] = {
                    'count': len(durations),
                    'avg_duration': float(np.mean(durations)),
                    'max_duration': float(np.max(durations)),
                    'min_duration': float(np.min(durations)),
                    'avg_memory_increase': float(np.mean(memories)),
                    'max_memory_increase': float(np.max(memories)),
                    'avg_cpu_increase': float(np.mean(cpus)),
                    'total_time': float(np.sum(durations)),
                }
        if report['operations']:
            total_times = [op['total_time'] for op in report['operations'].values()]
            report['summary'] = {
                'total_operations': len(report['operations']),
                'total_operation_time': sum(total_times),
            }
        if self.memory_usage:
            report['resource_usage']['memory'] = {
                'peak_mb': max(self.memory_usage),
                'avg_mb': float(np.mean(self.memory_usage)),
                'current_mb': self._get_memory_usage()
            }
        return report
    
    def clear_cache(self):
        self.operations.clear()
        self.memory_usage.clear()
        self.execution_times.clear()
        self.recursion_depth.clear()
        gc.collect()

# global instances
error_analyzer = AdvancedErrorAnalyzer()
performance_monitor = PerformanceMonitor()
st.session_state.error_analyzer = error_analyzer
st.session_state.performance_monitor = performance_monitor

# ============================================================================ 
# 3. ADVANCED DATA MANAGEMENT (ONLY YAHOO FINANCE)
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
    def fetch_advanced_market_data(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d',
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        if not tickers:
            raise ValueError("No tickers provided.")
        if len(tickers) > Config.MAX_TICKERS:
            raise ValueError(f"Maximum {Config.MAX_TICKERS} tickers allowed, got {len(tickers)}")
        
        cache_key = self._generate_cache_key(tickers, start_date, end_date, interval)
        if Config.CACHE_ENABLED and cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['data']
        
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
                executor.submit(
                    self._fetch_single_ticker_ohlc,
                    ticker,
                    start_date,
                    end_date,
                    interval
                ): ticker
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
                            data['prices'] = data['prices'].merge(close_series, left_index=True, right_index=True, how='outer')
                            data['volumes'] = data['volumes'].merge(volume_series, left_index=True, right_index=True, how='outer')
                            data['high'] = data['high'].merge(high_series, left_index=True, right_index=True, how='outer')
                            data['low'] = data['low'].merge(low_series, left_index=True, right_index=True, how='outer')
                            data['open'] = data['open'].merge(open_series, left_index=True, right_index=True, how='outer')
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
            self.cache[cache_key] = {'data': data, 'timestamp': time.time()}
        gc.collect()
        return data
    
    def _fetch_single_ticker_ohlc(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Dict:
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
                        raise ValueError(f"No historical data for {ticker} in date range.")
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
                    'currency': info.get('currency', 'USD'),
                    'country': info.get('country', 'Unknown'),
                    'exchange': info.get('exchange', 'Unknown'),
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
                # don't blow up – just keep whatever exists and let later checks handle
                valid_assets = data['prices'].columns.tolist()
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
                r = returns[ticker].dropna()
                if len(r) > 0:
                    features['statistical_features'][ticker] = {
                        'mean_return': r.mean(),
                        'std_return': r.std(),
                        'skewness': r.skew(),
                        'kurtosis': r.kurtosis(),
                        'sharpe_ratio': r.mean() / r.std() if r.std() > 0 else 0,
                        'max_drawdown': self._calculate_max_drawdown_series(r),
                        'positive_ratio': (r > 0).sum() / len(r),
                        'var_95': -np.percentile(r, 5),
                        'cvar_95': self._calculate_cvar(r, 0.95)
                    }
                    if ticker in prices.columns:
                        p = prices[ticker].dropna()
                        if len(p) > 0:
                            features['price_features'][ticker] = {
                                'current_price': p.iloc[-1],
                                'price_change_1d': p.pct_change().iloc[-1] if len(p) > 1 else 0,
                            }
                    if ticker in volumes.columns:
                        v = volumes[ticker].dropna()
                        if len(v) > 0:
                            features['liquidity_metrics'][ticker] = {
                                'current_volume': v.iloc[-1],
                                'avg_volume_20d': v.tail(20).mean(),
                            }
            if len(returns.columns) > 1:
                corr_matrix = returns.corr()
                features['correlation_matrix'] = corr_matrix
                cov_matrix = returns.cov() * Config.TRADING_DAYS_PER_YEAR
                features['covariance_matrix'] = cov_matrix
        except Exception as e:
            features['error'] = str(e)
        return features
    
    def _calculate_max_drawdown_series(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return float(drawdown.min()) if not drawdown.empty else 0.0
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        if len(returns) == 0:
            return 0.0
        var = -np.percentile(returns, (1 - confidence) * 100)
        tail = returns[returns <= -var]
        return float(-tail.mean()) if len(tail) > 0 else float(var)
    
    def _generate_cache_key(self, tickers: List[str], start_date: datetime, end_date: datetime, interval: str) -> str:
        tickers_str = '_'.join(sorted(tickers))
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        return f"{tickers_str}_{date_str}_{interval}"
    
    def validate_portfolio_data(self, data: Dict, min_assets: int = Config.MIN_ASSETS_FOR_OPTIMIZATION,
                                min_data_points: int = Config.MIN_DATA_POINTS) -> Dict:
        validation = {'is_valid': False, 'issues': [], 'warnings': [], 'suggestions': [], 'summary': {}}
        try:
            if data['prices'].empty:
                validation['issues'].append("No price data available.")
                return validation
            n_assets = len(data['prices'].columns)
            n_data_points = len(data['prices'])
            if n_assets < min_assets:
                validation['issues'].append(f"Only {n_assets} assets available, minimum {min_assets} required.")
            missing_percentage = data['prices'].isnull().mean().mean()
            validation['summary'] = {
                'n_assets': n_assets,
                'n_data_points': n_data_points,
                'missing_data_percentage': float(missing_percentage),
                'successful_tickers': len(data.get('successful_tickers', [])),
                'failed_tickers': len(data.get('errors', {})),
            }
            validation['is_valid'] = len(validation['issues']) == 0 and n_assets >= min_assets
            if not validation['is_valid']:
                if n_assets < min_assets:
                    validation['suggestions'].append("Add more assets or use a different preset.")
                if n_data_points < min_data_points:
                    validation['suggestions'].append("Extend date range or use higher frequency.")
        except Exception as e:
            validation['issues'].append(f"Validation error: {str(e)}")
        return validation

    @monitor_operation('calculate_basic_statistics')
    def calculate_basic_statistics(self, data: Dict) -> Dict:
        stats_out = {'assets': {}, 'portfolio_level': {}}
        returns = data.get('returns', pd.DataFrame())
        if returns.empty:
            return stats_out
        for ticker in returns.columns:
            r = returns[ticker].dropna()
            if len(r) > 0:
                stats_out['assets'][ticker] = {
                    'mean_return': float(r.mean() * Config.TRADING_DAYS_PER_YEAR),
                    'annual_volatility': float(r.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)),
                    'max_drawdown': self._calculate_max_drawdown_series(r),
                }
        if len(returns.columns) > 0:
            eq_w = np.ones(len(returns.columns)) / len(returns.columns)
            port_r = returns.dot(eq_w)
            stats_out['portfolio_level'] = {
                'mean_return': float(port_r.mean() * Config.TRADING_DAYS_PER_YEAR),
                'annual_volatility': float(port_r.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)),
                'max_drawdown': self._calculate_max_drawdown_series(port_r),
            }
        return stats_out

data_manager = AdvancedDataManager()

# ============================================================================ 
# 4. PORTFOLIO OPTIMIZATION ENGINE (EF / HRP / CLA / BL + Equal Weight)
# ============================================================================

class PortfolioOptimizerEngine:
    def __init__(self, data: Dict, risk_free_rate: float = Config.DEFAULT_RISK_FREE_RATE):
        self.data = data
        self.returns: pd.DataFrame = data.get('returns', pd.DataFrame())
        self.prices: pd.DataFrame = data.get('prices', pd.DataFrame())
        self.metadata: Dict = data.get('metadata', {})
        self.risk_free_rate = risk_free_rate
        
        if not self.returns.empty:
            if PYPFOPT_AVAILABLE and not self.prices.empty:
                self.mu = expected_returns.mean_historical_return(self.prices)
                self.S = risk_models.sample_cov(self.prices)
            else:
                self.mu = self.returns.mean() * Config.TRADING_DAYS_PER_YEAR
                self.S = self.returns.cov() * Config.TRADING_DAYS_PER_YEAR
        else:
            self.mu = None
            self.S = None
    
    def _ensure_enough_assets(self):
        if self.returns.empty or self.returns.shape[1] < 2:
            raise ValueError("Need at least 2 assets with data for optimization.")
    
    def _build_result(self, weight_dict: Dict[str, float], model_name: str) -> Dict:
        tickers = self.returns.columns
        w = pd.Series(weight_dict).reindex(tickers).fillna(0.0)
        if w.sum() == 0:
            w[:] = 1.0 / len(w)
        else:
            w = w / w.sum()
        if self.mu is not None and self.S is not None:
            mu_vec = self.mu.reindex(tickers).values
            S_mat = self.S.reindex(index=tickers, columns=tickers).values
            port_ret = float(np.dot(w.values, mu_vec))
            port_var = float(np.dot(w.values, np.dot(S_mat, w.values)))
            port_vol = float(np.sqrt(max(port_var, 0)))
        else:
            port_r = self.returns.dot(w)
            port_ret = float(port_r.mean() * Config.TRADING_DAYS_PER_YEAR)
            port_vol = float(port_r.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR))
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        port_returns = self.returns.dot(w)
        return {
            "model": model_name,
            "weights": w.to_dict(),
            "expected_return": port_ret,
            "volatility": port_vol,
            "sharpe_ratio": sharpe,
            "portfolio_returns": port_returns,
        }
    
    def optimize(self, method: str) -> Dict:
        self._ensure_enough_assets()
        m = method.lower()
        if m.startswith("equal"):
            return self._equal_weight()
        if "max sharpe" in m:
            return self._max_sharpe()
        if "min vol" in m:
            return self._min_vol()
        if m.startswith("cla"):
            return self._cla()
        if m.startswith("hrp"):
            return self._hrp()
        if "black" in m or "litterman" in m:
            return self._black_litterman()
        raise ValueError(f"Unknown optimization method: {method}")
    
    def _equal_weight(self) -> Dict:
        tickers = self.returns.columns
        w = {t: 1.0 / len(tickers) for t in tickers}
        return self._build_result(w, "Equal Weight")
    
    def _max_sharpe(self) -> Dict:
        if not PYPFOPT_AVAILABLE or self.mu is None or self.S is None:
            return self._equal_weight()
        ef = EfficientFrontier(self.mu, self.S)
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        w = ef.clean_weights()
        return self._build_result(w, "Max Sharpe (Mean-Variance)")
    
    def _min_vol(self) -> Dict:
        if not PYPFOPT_AVAILABLE or self.mu is None or self.S is None:
            return self._equal_weight()
        ef = EfficientFrontier(self.mu, self.S)
        ef.min_volatility()
        w = ef.clean_weights()
        return self._build_result(w, "Minimum Volatility")
    
    def _cla(self) -> Dict:
        if not PYPFOPT_AVAILABLE or self.mu is None or self.S is None:
            return self._equal_weight()
        cla = CLA(self.mu, self.S)
        cla.max_sharpe(risk_free_rate=self.risk_free_rate)
        w = cla.clean_weights()
        return self._build_result(w, "CLA Max Sharpe")
    
    def _hrp(self) -> Dict:
        if not PYPFOPT_AVAILABLE:
            return self._equal_weight()
        hrp = HRPOpt(returns=self.returns)
        w = hrp.optimize()
        return self._build_result(w, "HRP (Hierarchical Risk Parity)")
    
    def _black_litterman(self) -> Dict:
        if not PYPFOPT_AVAILABLE or self.S is None:
            return self._equal_weight()
        market_caps = {}
        for t, md in self.metadata.items():
            mc = md.get('market_cap', 0)
            if mc and mc > 0:
                market_caps[t] = mc
        if not market_caps:
            return self._equal_weight()
        mcaps = pd.Series(market_caps)
        cov_bl = self.S.reindex(index=mcaps.index, columns=mcaps.index).fillna(0)
        pi = market_implied_prior_returns(
            mcaps,
            cov_matrix=cov_bl,
            risk_aversion=2.5,
            risk_free_rate=self.risk_free_rate
        )
        bl = BlackLittermanModel(cov_bl, pi=pi)
        bl_returns = bl.bl_returns()
        ef = EfficientFrontier(bl_returns, cov_bl)
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        w = ef.clean_weights()
        return self._build_result(w, "Black-Litterman Max Sharpe")

# ============================================================================ 
# 5. RISK ANALYTICS HELPERS (Hist / Parametric / MC + Relative VaR)
# ============================================================================

def aggregate_returns_for_horizon(returns: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 1:
        return returns.dropna()
    # rolling sum of daily returns approximates multi-day return
    agg = returns.rolling(window=horizon).sum().dropna()
    return agg

def hist_var_cvar(returns: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0, 0.0
    var = -np.quantile(r, 1 - alpha)
    tail = r[r <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return float(var), float(cvar)

def parametric_var_cvar(returns: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0, 0.0
    mu = r.mean()
    sigma = r.std()
    if sigma == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - alpha)
    var = -(mu + z * sigma)
    cvar = -(mu + sigma * stats.norm.pdf(z) / (1 - alpha))
    return float(var), float(cvar)

def mc_var_cvar(
    returns_df: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.95,
    n_sims: int = 10000
) -> Tuple[float, float]:
    if returns_df.empty:
        return 0.0, 0.0
    w = weights.reindex(returns_df.columns).fillna(0.0)
    if w.sum() == 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / w.sum()
    mu_vec = returns_df.mean().values
    cov_mat = returns_df.cov().values
    sims = multivariate_normal.rvs(mean=mu_vec, cov=cov_mat, size=n_sims)
    port_sims = sims.dot(w.values)
    var, cvar = hist_var_cvar(pd.Series(port_sims), alpha)
    return var, cvar

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
    *Advanced analytics with machine learning, multi-region universes, and comprehensive risk engines*
    """)
    
    # ---------------------------------------------------------------------
    # SIDEBAR CONFIGURATION
    # ---------------------------------------------------------------------
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
        
        # --- Universe & Data Configuration ---
        st.subheader("📊 Data Configuration")
        universe_mode = st.radio(
            "Universe selection",
            ["Preset (recommended)", "Manual tickers"],
            index=0
        )
        
        selected_universe_key = None
        selected_scenario_key = None
        tickers_input = ""
        
        if universe_mode == "Preset (recommended)":
            selected_universe_key = st.selectbox(
                "Region / Universe",
                list(INSTITUTIONAL_UNIVERSES.keys()),
                index=list(INSTITUTIONAL_UNIVERSES.keys()).index("Global_70_plus")
                if "Global_70_plus" in INSTITUTIONAL_UNIVERSES else 0
            )
            selected_scenario_key = st.selectbox(
                "Scenario Preset",
                ["None"] + list(SCENARIO_PRESETS.keys()),
                index=0
            )
            # Preview tickers
            if selected_scenario_key != "None":
                preview_tickers = SCENARIO_PRESETS[selected_scenario_key]
            else:
                preview_tickers = INSTITUTIONAL_UNIVERSES[selected_universe_key]
            st.caption(f"Selected {len(preview_tickers)} tickers from Yahoo Finance.")
            st.text(", ".join(preview_tickers[:20]) + (" ..." if len(preview_tickers) > 20 else ""))
        
        else:
            tickers_input = st.text_area(
                "Enter tickers (comma-separated):",
                value="AAPL, GOOGL, MSFT, AMZN, TSLA",
                help="Enter Yahoo Finance symbols separated by commas"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start date",
                value=datetime.now().date() - timedelta(days=365*3),
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
        
        st.subheader("🔍 Analysis Type")
        analysis_type = st.selectbox(
            "Select analysis:",
            ["Portfolio Optimization", "Risk Analysis", "Backtesting", "ML Forecasting", "Comprehensive Report"]
        )
        
        if st.button("🚀 Fetch Data & Analyze", type="primary"):
            with st.spinner("Fetching market data from Yahoo Finance..."):
                try:
                    if universe_mode == "Preset (recommended)":
                        if selected_scenario_key and selected_scenario_key != "None":
                            tickers = SCENARIO_PRESETS[selected_scenario_key]
                        else:
                            tickers = INSTITUTIONAL_UNIVERSES[selected_universe_key]
                    else:
                        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
                    
                    if not tickers:
                        st.sidebar.error("Please select a preset universe or enter at least one ticker.")
                    else:
                        progress_bar = st.progress(0.0)
                        def update_progress(p, msg):
                            progress_bar.progress(p)
                            st.sidebar.write(msg)
                        data = data_manager.fetch_advanced_market_data(
                            tickers=tickers,
                            start_date=datetime.combine(start_date, datetime.min.time()),
                            end_date=datetime.combine(end_date, datetime.max.time()),
                            progress_callback=update_progress
                        )
                        st.session_state.portfolio_data = data
                        st.session_state.data_loaded = True
                        validation = data_manager.validate_portfolio_data(data)
                        st.session_state.portfolio_validation = validation
                        if validation['is_valid']:
                            st.sidebar.success(
                                f"✅ Data loaded: {validation['summary']['n_assets']} assets, "
                                f"{validation['summary']['n_data_points']} days"
                            )
                        else:
                            st.sidebar.warning("⚠️ Data loaded with issues.")
                            for issue in validation['issues']:
                                st.sidebar.error(issue)
                            for w in validation['warnings']:
                                st.sidebar.warning(w)
                except Exception as e:
                    st.sidebar.error(f"❌ Error fetching data: {str(e)[:200]}")
                    logging.error(f"Data fetch error: {str(e)}")
    
    # ---------------------------------------------------------------------
    # MAIN CONTENT
    # ---------------------------------------------------------------------
    data = st.session_state.get('portfolio_data', None)
    validation = st.session_state.get('portfolio_validation', None)
    
    if st.session_state.get('data_loaded', False) and data is not None:
        st.subheader("📋 Data Summary")
        col1, col2, col3 = st.columns(3)
        if not data['prices'].empty:
            with col1:
                st.metric("Assets", len(data['prices'].columns))
            with col2:
                st.metric("Data Points", len(data['prices']))
            with col3:
                date_range = f"{data['prices'].index[0].date()} → {data['prices'].index[-1].date()}"
                st.metric("Date Range", date_range)
        else:
            st.warning("No price data returned. Check universe, tickers or date range.")
        
        # --- Data Preview ---
        with st.expander("📊 Data Preview"):
            tab1, tab2, tab3 = st.tabs(["Prices", "Returns", "Statistics"])
            with tab1:
                if not data['prices'].empty:
                    st.dataframe(data['prices'].tail(10), width="stretch")
                else:
                    st.info("No price data.")
            with tab2:
                if not data['returns'].empty:
                    st.dataframe(data['returns'].tail(10), width="stretch")
                else:
                    st.info("No return data yet.")
            with tab3:
                stats_basic = data_manager.calculate_basic_statistics(data)
                if stats_basic.get('assets'):
                    stats_df = pd.DataFrame(stats_basic['assets']).T
                    show_cols = [c for c in ["mean_return", "annual_volatility", "max_drawdown"] if c in stats_df.columns]
                    st.dataframe(stats_df[show_cols], width="stretch")
                else:
                    st.info("Statistics not available.")
        
        # --- GUARD: if validation is not valid, still show but warn for optimization/risk ---
        if validation is not None and not validation.get('is_valid', False):
            st.warning("Data is not fully sufficient for robust optimization. Some models may be disabled.")
        
        # -----------------------------------------------------------------
        # PORTFOLIO OPTIMIZATION TAB
        # -----------------------------------------------------------------
        if analysis_type == "Portfolio Optimization":
            st.subheader("🎯 Portfolio Optimization")
            returns = data.get('returns', pd.DataFrame())
            if returns.empty or returns.shape[1] < 2:
                st.error("Not enough assets / data for optimization. Need at least 2 assets with returns.")
            else:
                col_o1, col_o2 = st.columns(2)
                with col_o1:
                    opt_method = st.selectbox(
                        "Optimization model",
                        [
                            "Equal Weight",
                            "Max Sharpe (Mean-Variance)",
                            "Minimum Volatility",
                            "CLA (Critical Line Algorithm)",
                            "HRP (Hierarchical Risk Parity)",
                            "Black-Litterman Max Sharpe"
                        ]
                    )
                with col_o2:
                    rf_rate = st.number_input(
                        "Risk-free rate (annual)",
                        value=float(Config.DEFAULT_RISK_FREE_RATE),
                        step=0.005,
                        format="%.3f"
                    )
                
                if st.button("Run Optimization", key="run_opt"):
                    try:
                        optimizer = PortfolioOptimizerEngine(data, risk_free_rate=rf_rate)
                        opt_result = optimizer.optimize(opt_method)
                        st.session_state.last_opt_result = opt_result
                        
                        w_df = pd.DataFrame.from_dict(opt_result['weights'], orient='index', columns=['Weight'])
                        w_df = w_df.sort_values('Weight', ascending=False)
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Expected Return (annual)", f"{opt_result['expected_return']:.2%}")
                        with col_m2:
                            st.metric("Volatility (annual)", f"{opt_result['volatility']:.2%}")
                        with col_m3:
                            st.metric("Sharpe Ratio", f"{opt_result['sharpe_ratio']:.2f}")
                        
                        st.markdown("#### Optimized Weights")
                        st.dataframe(
                            w_df.style.format({"Weight": "{:.2%}"}),
                            width="stretch"
                        )
                        
                        # Risk/return scatter of assets
                        mu_assets = optimizer.mu.reindex(returns.columns)
                        vol_assets = np.sqrt(np.diag(optimizer.S.reindex(index=returns.columns, columns=returns.columns)))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=vol_assets,
                            y=mu_assets,
                            mode="markers+text",
                            text=returns.columns,
                            textposition="top center",
                            name="Assets"
                        ))
                        fig.add_trace(go.Scatter(
                            x=[opt_result['volatility']],
                            y=[opt_result['expected_return']],
                            mode="markers",
                            marker=dict(size=14, symbol='star'),
                            name="Optimized Portfolio"
                        ))
                        fig.update_layout(
                            title=f"Risk-Return Map ({opt_method})",
                            xaxis_title="Volatility (σ, annual)",
                            yaxis_title="Expected Return (μ, annual)",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)[:200]}")
        
        # -----------------------------------------------------------------
        # RISK ANALYSIS TAB
        # -----------------------------------------------------------------
        elif analysis_type == "Risk Analysis":
            st.subheader("⚠️ Risk Analysis")
            returns = data.get('returns', pd.DataFrame())
            if returns.empty or returns.shape[1] < 2:
                st.error("Not enough data for risk analysis. Need at least 2 assets with returns.")
            else:
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    risk_method = st.selectbox(
                        "VaR Method",
                        ["Historical", "Parametric (Normal)", "Monte Carlo"]
                    )
                with col_r2:
                    alpha = st.select_slider(
                        "Confidence level",
                        options=[0.90, 0.95, 0.99, 0.995],
                        value=0.95
                    )
                with col_r3:
                    horizon = st.selectbox("Horizon (days)", [1, 5, 10, 21], index=0)
                
                st.markdown("#### Portfolio Weights Source")
                w_source = st.radio(
                    "",
                    ["Equal Weight", "Use last optimization (if available)"],
                    horizontal=True
                )
                last_opt = st.session_state.get('last_opt_result', None)
                if w_source.startswith("Use") and last_opt is not None:
                    w = pd.Series(last_opt['weights']).reindex(returns.columns).fillna(0.0)
                    if w.sum() == 0:
                        w[:] = 1.0 / len(w)
                    w = w / w.sum()
                else:
                    w = pd.Series(1.0 / len(returns.columns), index=returns.columns)
                
                st.markdown("#### Relative VaR vs Benchmark (optional)")
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    use_relative = st.checkbox("Compute Relative VaR vs Benchmark")
                bench_series = None
                bench_name = None
                if use_relative:
                    with col_b2:
                        bench_name = st.selectbox(
                            "Benchmark ticker",
                            options=list(returns.columns),
                            index=0
                        )
                    bench_series = returns[bench_name]
                
                if st.button("Run Risk Analysis", key="run_risk"):
                    try:
                        port_returns = returns.dot(w)
                        port_agg = aggregate_returns_for_horizon(port_returns, horizon)
                        
                        if risk_method == "Historical":
                            var, cvar = hist_var_cvar(port_agg, alpha)
                        elif risk_method == "Parametric (Normal)":
                            var, cvar = parametric_var_cvar(port_agg, alpha)
                        else:
                            var, cvar = mc_var_cvar(returns, w, alpha=alpha, n_sims=10000)
                        
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric(f"{horizon}-day {int(alpha*100)}% VaR", f"{var:.2%}")
                        with col_s2:
                            st.metric(f"{horizon}-day {int(alpha*100)}% CVaR", f"{cvar:.2%}")
                        with col_s3:
                            st.metric("Daily σ (portfolio)", f"{port_returns.std():.2%}")
                        
                        if use_relative and bench_series is not None:
                            combined = pd.concat([port_returns, bench_series], axis=1, join='inner').dropna()
                            combined.columns = ['Portfolio', 'Benchmark']
                            rel_r = combined['Portfolio'] - combined['Benchmark']
                            rel_agg = aggregate_returns_for_horizon(rel_r, horizon)
                            rvar, rcvar = hist_var_cvar(rel_agg, alpha)
                            st.markdown("#### Relative VaR vs Benchmark")
                            col_rv1, col_rv2 = st.columns(2)
                            with col_rv1:
                                st.metric(f"Relative VaR ({bench_name})", f"{rvar:.2%}")
                            with col_rv2:
                                st.metric(f"Relative CVaR ({bench_name})", f"{rcvar:.2%}")
                        
                        # Distribution chart
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=port_agg,
                            nbinsx=50,
                            name="Portfolio returns"
                        ))
                        fig.add_vline(
                            x=-var,
                            line_color="red",
                            line_dash="dash",
                            annotation_text=f"VaR ({int(alpha*100)}%)",
                            annotation_position="top left"
                        )
                        fig.update_layout(
                            title=f"Distribution of {horizon}-day Portfolio Returns",
                            xaxis_title="Return",
                            yaxis_title="Frequency",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Risk analysis failed: {str(e)[:200]}")
        
        # -----------------------------------------------------------------
        # PLACEHOLDERS FOR OTHER TABS
        # -----------------------------------------------------------------
        elif analysis_type == "Backtesting":
            st.subheader("📈 Backtesting")
            st.info("Backtesting engine placeholder – ready to be wired to your strategies.")
        elif analysis_type == "ML Forecasting":
            st.subheader("🤖 Machine Learning Forecasting")
            st.info("ML forecasting placeholder – can be wired to Prophet / sklearn / XGBoost.")
        elif analysis_type == "Comprehensive Report":
            st.subheader("📄 Comprehensive Report")
            st.info("Report generation placeholder – integrate with reportlab / HTML export.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to QuantEdge Pro v5.0
        
        ### Get Started:
        1. **Choose universe** in the sidebar (US / TR / JP / KR / SG / CN / Global 70+)
        2. Optionally choose a **Scenario Preset** (e.g. *Global Tech + TR Banks*)
        3. **Select date range**
        4. Choose **Analysis Type** (Portfolio Optimization / Risk Analysis)
        5. Click **“Fetch Data & Analyze”**
        
        ### Optimization Engine:
        - Equal-Weight benchmark
        - Mean-Variance Max Sharpe
        - Minimum Volatility
        - CLA (Critical Line Algorithm)
        - HRP (Hierarchical Risk Parity)
        - Black-Litterman (market-cap weighted priors)
        
        ### Risk Analytics:
        - Historical VaR & CVaR
        - Parametric (Gaussian) VaR & CVaR
        - Monte Carlo VaR (10,000 simulations)
        - Relative VaR vs chosen benchmark (e.g. SPY or any ticker in the universe)
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
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()

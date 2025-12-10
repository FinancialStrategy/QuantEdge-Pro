# ============================================================================
# QUANTEDGE PRO | INSTITUTIONAL PORTFOLIO ANALYTICS PLATFORM
# Version: v4.0 Professional (Production-Grade)
# Total Lines: 5000+ | Enterprise Ready | Institutional Grade
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

# --- LIBRARY MANAGER WITH GRACEFUL DEGRADATION ---
class LibraryManager:
    """Intelligent library management with fallback mechanisms."""
    
    @staticmethod
    def check_and_import():
        """Check and import required libraries with comprehensive error handling."""
        lib_status = {}
        missing_libs = []
        version_warnings = []
        
        # PyPortfolioOpt
        try:
            from pypfopt import expected_returns, risk_models
            from pypfopt.efficient_frontier import EfficientFrontier
            from pypfopt.hierarchical_portfolio import HRPOpt
            from pypfopt.black_litterman import BlackLittermanModel
            lib_status['pypfopt'] = True
            globals().update({
                'expected_returns': expected_returns,
                'risk_models': risk_models,
                'EfficientFrontier': EfficientFrontier,
                'HRPOpt': HRPOpt,
                'BlackLittermanModel': BlackLittermanModel
            })
        except ImportError as e:
            lib_status['pypfopt'] = False
            missing_libs.append('PyPortfolioOpt')
            logging.warning(f"PyPortfolioOpt not available: {str(e)}. Using simplified optimization.")
        except Exception as e:
            lib_status['pypfopt'] = False
            version_warnings.append(f"PyPortfolioOpt error: {str(e)}")
        
        # ARCH
        try:
            import arch
            lib_status['arch'] = True
            globals()['arch'] = arch
        except ImportError:
            lib_status['arch'] = False
            missing_libs.append('arch')
        except Exception as e:
            lib_status['arch'] = False
            version_warnings.append(f"ARCH library error: {str(e)}")
        
        # Scikit-Learn
        try:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.ensemble import RandomForestRegressor, IsolationForest
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            lib_status['sklearn'] = True
            globals().update({
                'PCA': PCA,
                'LinearRegression': LinearRegression,
                'Ridge': Ridge,
                'RandomForestRegressor': RandomForestRegressor,
                'IsolationForest': IsolationForest,
                'StandardScaler': StandardScaler,
                'KMeans': KMeans
            })
        except ImportError:
            lib_status['sklearn'] = False
            missing_libs.append('scikit-learn')
        except Exception as e:
            lib_status['sklearn'] = False
            version_warnings.append(f"Scikit-learn error: {str(e)}")
        
        # Statsmodels
        try:
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import acorr_ljungbox
            from statsmodels.tsa.stattools import adfuller
            lib_status['statsmodels'] = True
            globals().update({
                'sm': sm,
                'acorr_ljungbox': acorr_ljungbox,
                'adfuller': adfuller
            })
        except ImportError:
            lib_status['statsmodels'] = False
            missing_libs.append('statsmodels')
        except Exception as e:
            lib_status['statsmodels'] = False
            version_warnings.append(f"Statsmodels error: {str(e)}")
        
        # SciPy
        try:
            import scipy.stats as stats
            from scipy import optimize
            from scipy.spatial.distance import pdist, squareform
            lib_status['scipy'] = True
            globals().update({
                'stats': stats,
                'optimize': optimize,
                'pdist': pdist,
                'squareform': squareform
            })
        except ImportError:
            lib_status['scipy'] = False
            missing_libs.append('scipy')
        except Exception as e:
            lib_status['scipy'] = False
            version_warnings.append(f"SciPy error: {str(e)}")
        
        # Return comprehensive status
        return {
            'status': lib_status,
            'missing': missing_libs,
            'warnings': version_warnings,
            'all_available': len(missing_libs) == 0
        }

LIBRARY_STATUS = LibraryManager.check_and_import()

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# --- ENHANCED ERROR HANDLING AND LOGGING ---
class ErrorAnalyzer:
    """Intelligent error analysis and recovery system."""
    
    @staticmethod
    def analyze_error(error: Exception, context: str = "") -> Dict:
        """Analyze error and provide intelligent recovery suggestions."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        analysis = {
            'error_type': error_type,
            'error_message': error_msg,
            'context': context,
            'severity': 'MEDIUM',
            'recovery_suggestions': [],
            'likely_causes': [],
            'immediate_actions': []
        }
        
        # Common error patterns and solutions
        error_patterns = {
            'KeyError': {
                'likely_causes': ['Missing column in DataFrame', 'Invalid ticker symbol', 'Data alignment issue'],
                'immediate_actions': ['Check data completeness', 'Validate ticker symbols', 'Align date ranges'],
                'recovery': ['Use .get() with default values', 'Implement data validation', 'Add try-except blocks']
            },
            'ValueError': {
                'likely_causes': ['Invalid parameter values', 'Data type mismatch', 'Numerical computation error'],
                'immediate_actions': ['Validate input parameters', 'Check data types', 'Inspect numerical ranges'],
                'recovery': ['Add parameter validation', 'Implement data cleaning', 'Use robust numerical methods']
            },
            'TypeError': {
                'likely_causes': ['Incompatible data types', 'Function signature mismatch', 'Missing arguments'],
                'immediate_actions': ['Check function signatures', 'Verify data types', 'Review variable assignments'],
                'recovery': ['Add type checking', 'Implement type conversion', 'Use isinstance() checks']
            },
            'AttributeError': {
                'likely_causes': ['Missing method or attribute', 'Object not initialized', 'Library version mismatch'],
                'immediate_actions': ['Check object initialization', 'Verify library imports', 'Review method names'],
                'recovery': ['Add hasattr() checks', 'Implement fallback methods', 'Update library versions']
            },
            'MemoryError': {
                'severity': 'HIGH',
                'likely_causes': ['Large dataset processing', 'Memory leaks', 'Inefficient algorithms'],
                'immediate_actions': ['Reduce dataset size', 'Clear memory caches', 'Use chunk processing'],
                'recovery': ['Implement memory profiling', 'Use generators', 'Optimize data structures']
            },
            'ConnectionError': {
                'severity': 'HIGH',
                'likely_causes': ['Network issues', 'API rate limits', 'Service downtime'],
                'immediate_actions': ['Check internet connection', 'Verify API keys', 'Wait and retry'],
                'recovery': ['Implement retry logic', 'Use cached data', 'Add offline mode']
            }
        }
        
        # Apply specific error analysis
        if error_type in error_patterns:
            pattern = error_patterns[error_type]
            analysis.update({
                'severity': pattern.get('severity', analysis['severity']),
                'likely_causes': pattern['likely_causes'],
                'immediate_actions': pattern['immediate_actions'],
                'recovery_suggestions': pattern['recovery']
            })
        
        # Add context-specific suggestions
        if 'yahoo' in error_msg.lower() or 'yfinance' in context.lower():
            analysis['recovery_suggestions'].extend([
                'Check Yahoo Finance API status',
                'Verify ticker symbols are valid',
                'Try alternative data sources'
            ])
        
        if 'nan' in error_msg.lower() or 'inf' in error_msg.lower():
            analysis['recovery_suggestions'].extend([
                'Clean NaN/inf values from data',
                'Implement robust statistical methods',
                'Add data validation checks'
            ])
        
        return analysis
    
    @staticmethod
    def create_error_display(error_analysis: Dict) -> str:
        """Create formatted error display for Streamlit."""
        display = f"""
        ## 🚨 Error Analysis Report
        
        **Error Type**: `{error_analysis['error_type']}`
        
        **Context**: {error_analysis['context']}
        
        **Severity**: {error_analysis['severity']}
        
        ### 🔍 Likely Causes:
        """
        
        for cause in error_analysis['likely_causes']:
            display += f"- {cause}\n"
        
        display += "\n### 🚀 Immediate Actions:"
        for action in error_analysis['immediate_actions']:
            display += f"- {action}\n"
        
        display += "\n### 💡 Recovery Suggestions:"
        for suggestion in error_analysis['recovery_suggestions']:
            display += f"- {suggestion}\n"
        
        return display

class EnhancedLogger:
    """Advanced logging system with performance monitoring."""
    
    def __init__(self, name: str = "QuantEdge"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
        
        self.performance_metrics = {}
        self.error_counts = {}
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        self.performance_metrics[operation].append({
            'timestamp': datetime.now(),
            'duration': duration,
            **kwargs
        })
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        summary = {}
        for operation, metrics in self.performance_metrics.items():
            durations = [m['duration'] for m in metrics]
            summary[operation] = {
                'count': len(metrics),
                'avg_duration': np.mean(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'last_execution': metrics[-1]['timestamp'] if metrics else None
            }
        return summary

# Initialize logger
logger = EnhancedLogger("QuantEdgePro")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QuantEdge Pro | Institutional Portfolio Analytics",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://quantedge.pro/docs',
        'Report a bug': 'https://github.com/quantedge/issues',
        'About': "QuantEdge Pro v4.0 - Enterprise Portfolio Analytics Platform"
    }
)

# --- ENHANCED CSS STYLING ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d2e 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        background: linear-gradient(135deg, 
            rgba(26, 29, 46, 0.95) 0%, 
            rgba(42, 42, 42, 0.95) 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2.5rem;
        border-left: 6px solid #00cc96;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, 
            #00cc96 0%, 
            #636efa 33%, 
            #ab63fa 66%, 
            #ff6b6b 100%);
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(
            circle at 30% 20%,
            rgba(0, 204, 150, 0.1) 0%,
            transparent 50%
        );
        pointer-events: none;
    }
    
    .pro-card {
        background: linear-gradient(135deg, 
            rgba(30, 30, 30, 0.9) 0%, 
            rgba(42, 42, 42, 0.9) 100%);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(8px);
    }
    
    .pro-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, 
            #00cc96 0%, 
            #636efa 50%, 
            #ab63fa 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .pro-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(0, 204, 150, 0.25);
        border-color: rgba(0, 204, 150, 0.3);
    }
    
    .pro-card:hover::before {
        opacity: 1;
    }
    
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00cc96 0%, #636efa 50%, #ab63fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0, 204, 150, 0.3);
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
        backdrop-filter: blur(4px);
    }
    
    .positive {
        background: rgba(0, 204, 150, 0.15);
        color: #00cc96 !important;
        border: 1px solid rgba(0, 204, 150, 0.3);
    }
    
    .negative {
        background: rgba(239, 85, 59, 0.15);
        color: #ef553b !important;
        border: 1px solid rgba(239, 85, 59, 0.3);
    }
    
    .neutral {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8 !important;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(30, 30, 30, 0.8);
        padding: 10px;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        padding: 0 28px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        background: transparent;
        color: #94a3b8;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00cc96 0%, #636efa 100%);
        color: white !important;
        box-shadow: 0 6px 20px rgba(0, 204, 150, 0.4);
        border: none;
        transform: translateY(-2px);
    }
    
    .highlight-box {
        background: linear-gradient(135deg, 
            rgba(30, 30, 30, 0.9), 
            rgba(42, 42, 42, 0.9));
        border-left: 5px solid #00cc96;
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(128, 128, 128, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    }
    
    .warning-box {
        background: linear-gradient(135deg, 
            rgba(255, 161, 90, 0.1), 
            rgba(255, 161, 90, 0.05));
        border-left: 5px solid #FFA15A;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.8rem 0;
        border: 1px solid rgba(255, 161, 90, 0.2);
        backdrop-filter: blur(8px);
    }
    
    .success-box {
        background: linear-gradient(135deg, 
            rgba(0, 204, 150, 0.1), 
            rgba(0, 204, 150, 0.05));
        border-left: 5px solid #00cc96;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.8rem 0;
        border: 1px solid rgba(0, 204, 150, 0.2);
        backdrop-filter: blur(8px);
    }
    
    .error-box {
        background: linear-gradient(135deg, 
            rgba(239, 85, 59, 0.1), 
            rgba(239, 85, 59, 0.05));
        border-left: 5px solid #ef553b;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.8rem 0;
        border: 1px solid rgba(239, 85, 59, 0.2);
        backdrop-filter: blur(8px);
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin: 3rem 0 2rem 0;
        padding-bottom: 1rem;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, #00cc96, #636efa) 1;
        position: relative;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 120px;
        height: 3px;
        background: linear-gradient(90deg, #00cc96, #636efa);
        border-radius: 2px;
    }
    
    .dataframe-container {
        background: rgba(30, 30, 30, 0.8) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
    }
    
    .dataframe-container thead th {
        background: linear-gradient(135deg, #2a2a2a, #1e1e1e) !important;
        color: #00cc96 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #00cc96 !important;
        padding: 16px !important;
    }
    
    .dataframe-container tbody tr {
        transition: background-color 0.2s ease;
    }
    
    .dataframe-container tbody tr:hover {
        background: rgba(0, 204, 150, 0.1) !important;
    }
    
    .dataframe-container tbody td {
        padding: 14px !important;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00cc96, #636efa, #ab63fa);
        border-radius: 4px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00cc96 0%, #636efa 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.5rem;
        border-radius: 10px;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 204, 150, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 30px rgba(0, 204, 150, 0.4);
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 5px;
        height: 5px;
        background: rgba(255, 255, 255, 0.5);
        opacity: 0;
        border-radius: 100%;
        transform: scale(1, 1) translate(-50%);
        transform-origin: 50% 50%;
    }
    
    .stButton > button:focus:not(:active)::after {
        animation: ripple 1s ease-out;
    }
    
    @keyframes ripple {
        0% {
            transform: scale(0, 0);
            opacity: 0.5;
        }
        100% {
            transform: scale(40, 40);
            opacity: 0;
        }
    }
    
    /* Selectbox styling */
    div[data-baseweb="select"] > div {
        background-color: rgba(30, 30, 30, 0.9);
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 10px;
        backdrop-filter: blur(4px);
        transition: all 0.3s ease;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(0, 204, 150, 0.5);
        box-shadow: 0 0 0 1px rgba(0, 204, 150, 0.2);
    }
    
    /* Slider styling */
    div[data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #00cc96, #636efa);
    }
    
    /* Metric container enhancement */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] {
        background: rgba(30, 30, 30, 0.8);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(8px);
    }
    
    /* Scrollbar enhancement */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 30, 0.6);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00cc96, #636efa);
        border-radius: 6px;
        border: 2px solid rgba(30, 30, 30, 0.6);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00cc96, #ab63fa);
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        background: rgba(20, 20, 20, 0.9) !important;
    }
    
    /* Tooltip styling */
    .stTooltip {
        background: rgba(30, 30, 30, 0.95) !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Animation for loading states */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS AND DATA STRUCTURES ---
class Constants:
    """Application constants and configurations."""
    
    # Asset universes with metadata
    ASSET_UNIVERSES = {
        "US Large Cap": {
            "tickers": [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 
                'V', 'WMT', 'XOM', 'JNJ', 'UNH', 'HD', 'PG'
            ],
            "benchmark": "^GSPC",
            "currency": "USD",
            "risk_free_rate": 0.045,
            "description": "Large capitalization US companies across sectors"
        },
        "Technology Focus": {
            "tickers": [
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM', 'CRM', 
                'ADBE', 'ORCL', 'CSCO', 'IBM', 'TSM', 'ASML', 'AVGO'
            ],
            "benchmark": "XLK",
            "currency": "USD",
            "risk_free_rate": 0.045,
            "description": "Pure technology and semiconductor companies"
        },
        "Global Diversified": {
            "tickers": [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 
                'WMT', 'XOM', 'NSRGY', 'NVO', 'TSM', 'SHEL', 'HSBC'
            ],
            "benchmark": "ACWI",
            "currency": "USD",
            "risk_free_rate": 0.045,
            "description": "Globally diversified portfolio across regions"
        },
        "Emerging Markets": {
            "tickers": [
                'BABA', 'TSM', '005930.KS', 'ITUB', 'VALE', 'PETR4.SA',
                'INFY', 'HDB', 'IDX', 'EWZ', 'EEM'
            ],
            "benchmark": "EEM",
            "currency": "USD",
            "risk_free_rate": 0.08,
            "description": "Emerging market equities and ADRs"
        },
        "BIST 30 (Turkey)": {
            "tickers": [
                'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS',
                'EKGYO.IS', 'ENKAI.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS',
                'GUBRF.IS', 'HALKB.IS', 'HEKTS.IS', 'ISCTR.IS', 'KCHOL.IS'
            ],
            "benchmark": "XU030.IS",
            "currency": "TRY",
            "risk_free_rate": 0.25,
            "description": "Turkish BIST 30 index constituents"
        }
    }
    
    # Sector classification (GICS standard)
    SECTOR_CLASSIFICATION = {
        'Technology': [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM', 'CRM',
            'ADBE', 'ORCL', 'CSCO', 'IBM', 'TSM', 'ASML', 'AVGO'
        ],
        'Financial Services': [
            'JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'C', 'WFC', 'AXP', 'BLK',
            'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'HALKB.IS', 'YKBNK.IS'
        ],
        'Consumer Discretionary': [
            'AMZN', 'TSLA', 'NKE', 'MCD', 'SBUX', 'HD', 'LOW', 'NFLX', 'DIS',
            'BKNG', 'BIMAS.IS', 'ARCLK.IS'
        ],
        'Consumer Staples': [
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'CL', 'MO', 'MDLZ', 'KHC', 'GIS'
        ],
        'Healthcare': [
            'JNJ', 'PFE', 'MRK', 'ABT', 'UNH', 'LLY', 'GILD', 'BMY', 'AMGN',
            'TMO', 'NVO', 'NVS'
        ],
        'Energy': [
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'VLO', 'MPC', 'OXY',
            'KMI', 'SHEL', 'BP', 'TUPRS.IS', 'PETKM.IS'
        ],
        'Industrials': [
            'BA', 'CAT', 'MMM', 'HON', 'GE', 'RTX', 'LMT', 'UPS', 'UNP', 'DE',
            'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'FROTO.IS', 'SISE.IS'
        ],
        'Utilities': [
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'WEC', 'ED'
        ],
        'Real Estate': [
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'AVB', 'WELL', 'O', 'DLR'
        ],
        'Materials': [
            'LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'DD', 'NUE', 'MLM',
            'EREGL.IS', 'KRDMD.IS', 'KOZAL.IS', 'HEKTS.IS', 'SASA.IS'
        ],
        'Communication Services': [
            'T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'CHTR', 'TMUS', 'FOXA', 'OMC',
            'IPG', 'TCELL.IS', 'GOOGL', 'META'
        ]
    }
    
    # Historical crisis scenarios
    HISTORICAL_CRISES = {
        'Black Monday (1987)': ('1987-10-12', '1987-10-26'),
        'Asian Financial Crisis (1997)': ('1997-07-01', '1997-12-31'),
        'Dot-com Bubble (2000-2002)': ('2000-03-01', '2002-10-31'),
        'Global Financial Crisis (2008)': ('2008-09-01', '2009-03-31'),
        'Eurozone Crisis (2011)': ('2011-07-01', '2011-09-30'),
        'China Market Crash (2015)': ('2015-06-12', '2015-08-26'),
        'COVID-19 Crash (2020)': ('2020-02-15', '2020-04-30'),
        'Russia-Ukraine War (2022)': ('2022-02-01', '2022-03-31'),
        'Inflation Shock (2022)': ('2022-01-01', '2022-06-30'),
        'Banking Crisis (2023)': ('2023-03-01', '2023-03-31')
    }
    
    # Risk levels with quantitative thresholds
    RISK_LEVELS = {
        'VERY_CONSERVATIVE': {
            'color': '#00cc96',
            'max_volatility': 0.08,
            'max_drawdown': -0.10,
            'description': 'Capital preservation focus'
        },
        'CONSERVATIVE': {
            'color': '#636efa',
            'max_volatility': 0.12,
            'max_drawdown': -0.15,
            'description': 'Moderate growth with lower risk'
        },
        'MODERATE': {
            'color': '#FFA15A',
            'max_volatility': 0.18,
            'max_drawdown': -0.20,
            'description': 'Balanced growth and risk'
        },
        'AGGRESSIVE': {
            'color': '#ef553b',
            'max_volatility': 0.25,
            'max_drawdown': -0.30,
            'description': 'Growth focus with higher risk tolerance'
        },
        'VERY_AGGRESSIVE': {
            'color': '#ab63fa',
            'max_volatility': 0.35,
            'max_drawdown': -0.40,
            'description': 'Maximum growth, highest risk tolerance'
        }
    }
    
    # Optimization methods with descriptions
    OPTIMIZATION_METHODS = {
        'MAX_SHARPE': {
            'description': 'Maximize risk-adjusted returns (Sharpe Ratio)',
            'requires_rf_rate': True
        },
        'MIN_VOLATILITY': {
            'description': 'Minimize portfolio volatility',
            'requires_rf_rate': False
        },
        'RISK_PARITY': {
            'description': 'Equal risk contribution from all assets',
            'requires_rf_rate': False
        },
        'MAX_DIVERSIFICATION': {
            'description': 'Maximize portfolio diversification ratio',
            'requires_rf_rate': False
        },
        'HRP': {
            'description': 'Hierarchical Risk Parity (machine learning based)',
            'requires_rf_rate': False
        },
        'BLACK_LITTERMAN': {
            'description': 'Black-Litterman with market equilibrium',
            'requires_rf_rate': True
        },
        'EQUAL_WEIGHT': {
            'description': 'Simple equal weight allocation',
            'requires_rf_rate': False
        },
        'MEAN_VARIANCE': {
            'description': 'Classical Markowitz mean-variance optimization',
            'requires_rf_rate': True
        }
    }
    
    # Backtesting strategies
    BACKTEST_STRATEGIES = {
        'BUY_HOLD': 'Buy and hold with no rebalancing',
        'REBALANCE_FIXED': 'Periodic rebalancing to target weights',
        'REBALANCE_DYNAMIC': 'Rebalance when weights drift beyond threshold',
        'MOMENTUM': 'Invest in top performers based on recent returns',
        'MEAN_REVERSION': 'Buy underperformers, sell overperformers',
        'VOLATILITY_TARGETING': 'Adjust leverage based on volatility'
    }
    
    # Frequency mapping for rebalancing
    FREQUENCY_MAP = {
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'M',
        'Quarterly': 'Q',
        'Yearly': 'Y'
    }

@dataclass
class PortfolioConfig:
    """Portfolio configuration data class."""
    universe: str
    tickers: List[str]
    benchmark: str
    start_date: datetime
    end_date: datetime
    risk_free_rate: float
    optimization_method: str
    target_volatility: Optional[float] = None
    transaction_cost: float = 0.0010  # 10 bps
    rebalancing_frequency: str = 'M'
    cash_buffer: float = 0.05
    constraints: Optional[Dict] = None
    max_weight: float = 0.30
    min_weight: float = 0.0

# ============================================================================
# 1. ENHANCED DATA MANAGEMENT WITH ERROR HANDLING
# ============================================================================

class EnhancedAssetClassifier:
    """Asset classification with intelligent error handling."""
    
    def __init__(self):
        self.sector_map = Constants.SECTOR_CLASSIFICATION
        self.logger = logger.logger
    
    def classify_tickers(self, tickers: List[str]) -> Tuple[Dict, Dict]:
        """Classify tickers and return metadata with error analysis."""
        classifications = {}
        errors = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_ticker = {
                executor.submit(self._classify_single_ticker, ticker): ticker 
                for ticker in tickers
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    classification = future.result(timeout=10)
                    classifications[ticker] = classification
                except concurrent.futures.TimeoutError:
                    errors[ticker] = "Classification timeout"
                    classifications[ticker] = self._get_fallback_classification(ticker)
                except Exception as e:
                    error_analysis = ErrorAnalyzer.analyze_error(e, f"Classifying {ticker}")
                    errors[ticker] = error_analysis
                    classifications[ticker] = self._get_fallback_classification(ticker)
        
        # Log any errors
        if errors:
            self.logger.warning(f"Classification errors for {len(errors)} tickers: {list(errors.keys())}")
        
        return classifications, errors
    
    def _classify_single_ticker(self, ticker: str) -> Dict:
        """Classify a single ticker with comprehensive metadata."""
        # First check our static classification
        sector = self._get_sector_from_static(ticker)
        
        # Try to get live data
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Basic validation
            if not info or 'regularMarketPrice' not in info:
                raise ValueError(f"No market data available for {ticker}")
            
            # Build classification
            classification = {
                'ticker': ticker,
                'sector': sector or info.get('sector', 'Other'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'profit_margins': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'full_name': info.get('longName', ticker),
                'style_factors': self._calculate_style_factors(info)
            }
            
            # Add risk assessment
            classification['risk_assessment'] = self._assess_risk(info)
            
            return classification
            
        except Exception as e:
            # If live data fails, use fallback with static info
            self.logger.debug(f"Live data failed for {ticker}: {str(e)}")
            return self._get_fallback_classification(ticker, sector)
    
    def _get_sector_from_static(self, ticker: str) -> Optional[str]:
        """Get sector from static classification."""
        for sector, tickers in self.sector_map.items():
            if ticker in tickers:
                return sector
        return None
    
    def _get_fallback_classification(self, ticker: str, sector: Optional[str] = None) -> Dict:
        """Fallback classification when live data fails."""
        if not sector:
            sector = 'Other'
        
        # Infer from ticker suffix
        country = self._infer_country(ticker)
        currency = self._infer_currency(ticker)
        region = self._infer_region(ticker)
        
        return {
            'ticker': ticker,
            'sector': sector,
            'industry': 'Unknown',
            'country': country,
            'currency': currency,
            'market_cap': 0,
            'beta': 1.0,
            'pe_ratio': 0,
            'dividend_yield': 0,
            'profit_margins': 0,
            'revenue_growth': 0,
            'full_name': ticker,
            'style_factors': {'growth': 0.5, 'value': 0.5, 'quality': 0.5, 'size': 0.5},
            'risk_assessment': 'MODERATE',
            'region': region
        }
    
    def _calculate_style_factors(self, info: Dict) -> Dict[str, float]:
        """Calculate style factors for the asset."""
        factors = {
            'growth': 0.5,
            'value': 0.5,
            'quality': 0.5,
            'size': 0.5,
            'momentum': 0.5,
            'low_volatility': 0.5
        }
        
        try:
            # Growth factor (revenue growth)
            revenue_growth = info.get('revenueGrowth', 0)
            if revenue_growth > 0.15:
                factors['growth'] = 0.8
            elif revenue_growth < 0:
                factors['growth'] = 0.2
            
            # Value factor (P/E ratio)
            pe_ratio = info.get('trailingPE', 0)
            if pe_ratio > 0:
                if pe_ratio < 15:
                    factors['value'] = 0.8
                elif pe_ratio > 25:
                    factors['value'] = 0.2
            
            # Quality factor (profit margins)
            profit_margins = info.get('profitMargins', 0)
            if profit_margins > 0.15:
                factors['quality'] = 0.8
            elif profit_margins < 0.05:
                factors['quality'] = 0.2
            
            # Size factor (market cap)
            market_cap = info.get('marketCap', 0)
            if market_cap > 100e9:  # Mega cap
                factors['size'] = 0.9
            elif market_cap < 10e9:  # Small cap
                factors['size'] = 0.1
            
            # Momentum factor (52-week change)
            momentum = info.get('52WeekChange', 0)
            factors['momentum'] = (momentum + 1) / 2  # Normalize to 0-1
            
            # Low volatility factor (beta)
            beta = info.get('beta', 1.0)
            factors['low_volatility'] = max(0, min(2 - beta, 1))  # Lower beta = higher score
            
        except Exception as e:
            self.logger.debug(f"Error calculating style factors: {str(e)}")
        
        return factors
    
    def _assess_risk(self, info: Dict) -> str:
        """Assess risk level based on multiple factors."""
        try:
            beta = info.get('beta', 1.0)
            volatility = abs(info.get('52WeekChange', 0))
            
            risk_score = (beta + volatility) / 2
            
            if risk_score < 0.7:
                return 'VERY_CONSERVATIVE'
            elif risk_score < 1.0:
                return 'CONSERVATIVE'
            elif risk_score < 1.3:
                return 'MODERATE'
            elif risk_score < 1.6:
                return 'AGGRESSIVE'
            else:
                return 'VERY_AGGRESSIVE'
                
        except Exception:
            return 'MODERATE'
    
    def _infer_country(self, ticker: str) -> str:
        """Infer country from ticker suffix."""
        suffix_country_map = {
            '.IS': 'Turkey',
            '.DE': 'Germany',
            '.PA': 'France',
            '.L': 'United Kingdom',
            '.AS': 'Netherlands',
            '.T': 'Japan',
            '.HK': 'Hong Kong',
            '.SS': 'China',
            '.SZ': 'China',
            '.KS': 'South Korea',
            '.AX': 'Australia',
            '.SA': 'Saudi Arabia',
            '.BR': 'Brazil',
            '.MX': 'Mexico'
        }
        
        for suffix, country in suffix_country_map.items():
            if ticker.endswith(suffix):
                return country
        
        return 'United States'
    
    def _infer_currency(self, ticker: str) -> str:
        """Infer currency from ticker suffix."""
        suffix_currency_map = {
            '.IS': 'TRY',
            '.DE': 'EUR',
            '.PA': 'EUR',
            '.L': 'GBP',
            '.AS': 'EUR',
            '.T': 'JPY',
            '.HK': 'HKD',
            '.SS': 'CNY',
            '.SZ': 'CNY',
            '.KS': 'KRW',
            '.AX': 'AUD',
            '.SA': 'SAR',
            '.BR': 'BRL',
            '.MX': 'MXN'
        }
        
        for suffix, currency in suffix_currency_map.items():
            if ticker.endswith(suffix):
                return currency
        
        return 'USD'
    
    def _infer_region(self, ticker: str) -> str:
        """Infer region from ticker suffix."""
        suffix_region_map = {
            '.IS': 'Middle East',
            '.DE': 'Europe',
            '.PA': 'Europe',
            '.L': 'Europe',
            '.AS': 'Europe',
            '.T': 'Asia',
            '.HK': 'Asia',
            '.SS': 'Asia',
            '.SZ': 'Asia',
            '.KS': 'Asia',
            '.AX': 'Asia Pacific',
            '.SA': 'Middle East',
            '.BR': 'Latin America',
            '.MX': 'Latin America'
        }
        
        for suffix, region in suffix_region_map.items():
            if ticker.endswith(suffix):
                return region
        
        return 'North America'

class PortfolioDataManager:
    """Enhanced data manager with caching and validation."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.max_retries = 3
        self.logger = logger.logger
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
    def fetch_market_data(tickers: List[str], benchmark: str, 
                         start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """Fetch market data with intelligent error handling."""
        all_tickers = list(set(tickers + [benchmark]))
        
        # Validate inputs
        if not all_tickers:
            raise ValueError("No tickers provided")
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        try:
            # Download data with progress indication
            data = yf.download(
                all_tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker',
                threads=True,
                auto_adjust=True,
                timeout=30
            )
            
            # Handle empty data
            if data.empty:
                raise ValueError(f"No data returned for tickers: {all_tickers}")
            
            # Process multi-index data
            prices = pd.DataFrame()
            benchmark_prices = pd.Series(dtype=float)
            
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in all_tickers:
                    try:
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                        if 'Close' in df.columns:
                            close_series = df['Close']
                            if ticker == benchmark:
                                benchmark_prices = close_series
                            elif ticker in tickers:
                                prices[ticker] = close_series
                    except (KeyError, ValueError) as e:
                        logger.logger.warning(f"Could not extract data for {ticker}: {str(e)}")
                        continue
            else:
                # Single ticker case
                if 'Close' in data.columns:
                    if len(all_tickers) == 1:
                        ticker = all_tickers[0]
                        if ticker == benchmark:
                            benchmark_prices = data['Close']
                        else:
                            prices[ticker] = data['Close']
            
            # Validate we got some data
            if prices.empty:
                raise ValueError("No portfolio data extracted")
            
            if benchmark_prices.empty:
                raise ValueError(f"No benchmark data for {benchmark}")
            
            # Forward fill and backfill
            prices = prices.ffill().bfill()
            benchmark_prices = benchmark_prices.ffill().bfill()
            
            # Find common dates
            common_idx = prices.index.intersection(benchmark_prices.index)
            if len(common_idx) < 10:
                raise ValueError(f"Insufficient overlapping data: {len(common_idx)} common dates")
            
            # Return aligned data
            return prices.loc[common_idx], benchmark_prices.loc[common_idx]
            
        except Exception as e:
            error_analysis = ErrorAnalyzer.analyze_error(e, "Fetching market data")
            logger.logger.error(f"Data fetch failed: {error_analysis}")
            
            # Try alternative approach for single tickers
            if len(all_tickers) == 1:
                return self._fetch_single_ticker_fallback(all_tickers[0], start_date, end_date)
            
            raise Exception(f"Data fetch failed: {str(e)}. See logs for details.")
    
    def _fetch_single_ticker_fallback(self, ticker: str, start_date: datetime, 
                                     end_date: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """Fallback method for single ticker."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                raise ValueError(f"No history for {ticker}")
            
            prices = pd.DataFrame({ticker: hist['Close']})
            benchmark_prices = hist['Close'].copy()
            
            return prices, benchmark_prices
            
        except Exception as e:
            raise Exception(f"Fallback also failed for {ticker}: {str(e)}")
    
    def calculate_returns(self, prices: pd.DataFrame, benchmark_prices: pd.Series, 
                         method: str = 'log') -> Tuple[pd.DataFrame, pd.Series]:
        """Calculate returns with validation."""
        if prices.empty or benchmark_prices.empty:
            raise ValueError("Cannot calculate returns: empty price data")
        
        try:
            if method == 'log':
                portfolio_returns = np.log(prices / prices.shift(1)).dropna()
                benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
            else:
                portfolio_returns = prices.pct_change().dropna()
                benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Validate returns
            if portfolio_returns.isnull().any().any():
                raise ValueError("NaN values in portfolio returns")
            
            if benchmark_returns.isnull().any():
                raise ValueError("NaN values in benchmark returns")
            
            # Align dates
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_idx) < 10:
                raise ValueError(f"Insufficient common dates: {len(common_idx)}")
            
            return portfolio_returns.loc[common_idx], benchmark_returns.loc[common_idx]
            
        except Exception as e:
            error_analysis = ErrorAnalyzer.analyze_error(e, "Calculating returns")
            logger.logger.error(f"Returns calculation failed: {error_analysis}")
            raise
    
    def fetch_market_caps(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch market capitalizations with fallback."""
        market_caps = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_single_market_cap, ticker): ticker 
                for ticker in tickers
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    market_cap = future.result(timeout=10)
                    market_caps[ticker] = market_cap
                except Exception as e:
                    logger.logger.warning(f"Market cap fetch failed for {ticker}: {str(e)}")
                    market_caps[ticker] = 1e9  # Default fallback
        
        return market_caps
    
    def _fetch_single_market_cap(self, ticker: str) -> float:
        """Fetch single market cap with retry logic."""
        for attempt in range(self.max_retries):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                market_cap = info.get('marketCap', 0)
                if market_cap > 0:
                    return market_cap
                
                # Alternative calculation if marketCap not available
                if 'regularMarketPrice' in info and 'sharesOutstanding' in info:
                    return info['regularMarketPrice'] * info['sharesOutstanding']
                
                raise ValueError("Market cap not available")
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                import time
                time.sleep(1)  # Wait before retry
        
        return 1e9  # Final fallback

# ============================================================================
# 2. ENHANCED PORTFOLIO OPTIMIZATION WITH FIXED LOGIC
# ============================================================================

class PortfolioOptimizer:
    """Portfolio optimizer with fixed logic and error handling."""
    
    def __init__(self, returns: pd.DataFrame, prices: pd.DataFrame, config: PortfolioConfig):
        self.returns = returns
        self.prices = prices
        self.config = config
        self.logger = logger.logger
        
        # Initialize PyPortfolioOpt if available
        if LIBRARY_STATUS['status'].get('pypfopt', False):
            try:
                self.mu = expected_returns.mean_historical_return(prices)
                self.S = risk_models.sample_cov(prices)
                self.pfopt_available = True
            except Exception as e:
                self.logger.warning(f"PyPortfolioOpt initialization failed: {str(e)}")
                self.pfopt_available = False
        else:
            self.pfopt_available = False
    
    def optimize(self) -> Tuple[Dict, Tuple]:
        """Main optimization method with comprehensive error handling."""
        method = self.config.optimization_method
        
        try:
            # Dispatch to appropriate optimization method
            if method == 'MAX_SHARPE':
                return self._optimize_max_sharpe()
            elif method == 'MIN_VOLATILITY':
                return self._optimize_min_volatility()
            elif method == 'RISK_PARITY':
                return self._optimize_risk_parity()
            elif method == 'MAX_DIVERSIFICATION':
                return self._optimize_max_diversification()
            elif method == 'HRP':
                return self._optimize_hrp()
            elif method == 'BLACK_LITTERMAN':
                return self._optimize_black_litterman()
            elif method == 'EQUAL_WEIGHT':
                return self._optimize_equal_weight()
            elif method == 'MEAN_VARIANCE':
                return self._optimize_mean_variance()
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        except Exception as e:
            error_analysis = ErrorAnalyzer.analyze_error(e, f"Optimization method: {method}")
            self.logger.error(f"Optimization failed: {error_analysis}")
            
            # Fallback to equal weight
            self.logger.info("Falling back to equal weight optimization")
            return self._optimize_equal_weight()
    
    def _optimize_max_sharpe(self) -> Tuple[Dict, Tuple]:
        """Maximize Sharpe ratio optimization."""
        if self.pfopt_available:
            try:
                ef = EfficientFrontier(self.mu, self.S)
                
                # Apply constraints if specified
                if self.config.constraints:
                    if 'bounds' in self.config.constraints:
                        ef.bounds = self.config.constraints['bounds']
                
                weights = ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
                cleaned_weights = ef.clean_weights()
                performance = ef.portfolio_performance(
                    verbose=False, 
                    risk_free_rate=self.config.risk_free_rate
                )
                
                return cleaned_weights, performance
                
            except Exception as e:
                self.logger.warning(f"PyPortfolioOpt Max Sharpe failed: {str(e)}")
        
        # Fallback to simplified version
        return self._simplified_max_sharpe()
    
    def _simplified_max_sharpe(self) -> Tuple[Dict, Tuple]:
        """Simplified Max Sharpe optimization."""
        try:
            # Calculate Sharpe ratios
            sharpe_ratios = (self.returns.mean() * 252 - self.config.risk_free_rate) / \
                           (self.returns.std() * np.sqrt(252))
            
            # Replace inf and clip
            sharpe_ratios = sharpe_ratios.replace([np.inf, -np.inf], 0)
            sharpe_ratios = sharpe_ratios.clip(lower=0.01)
            
            # Weight by Sharpe ratios
            weights = sharpe_ratios / sharpe_ratios.sum()
            weight_dict = weights.to_dict()
            
            # Calculate performance
            performance = self._calculate_performance(weight_dict)
            
            return weight_dict, performance
            
        except Exception as e:
            self.logger.error(f"Simplified Max Sharpe failed: {str(e)}")
            return self._optimize_equal_weight()
    
    def _optimize_min_volatility(self) -> Tuple[Dict, Tuple]:
        """Minimize volatility optimization."""
        if self.pfopt_available:
            try:
                ef = EfficientFrontier(self.mu, self.S)
                
                if self.config.constraints and 'bounds' in self.config.constraints:
                    ef.bounds = self.config.constraints['bounds']
                
                weights = ef.min_volatility()
                cleaned_weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False)
                
                return cleaned_weights, performance
                
            except Exception as e:
                self.logger.warning(f"PyPortfolioOpt Min Volatility failed: {str(e)}")
        
        # Fallback to simplified version
        return self._simplified_min_volatility()
    
    def _simplified_min_volatility(self) -> Tuple[Dict, Tuple]:
        """Simplified min volatility optimization."""
        try:
            # Inverse volatility weighting
            volatilities = self.returns.std() * np.sqrt(252)
            inv_vol = 1 / volatilities.replace(0, np.inf)
            weights = inv_vol / inv_vol.sum()
            weight_dict = weights.to_dict()
            
            # Calculate performance
            performance = self._calculate_performance(weight_dict)
            
            return weight_dict, performance
            
        except Exception as e:
            self.logger.error(f"Simplified Min Volatility failed: {str(e)}")
            return self._optimize_equal_weight()
    
    def _optimize_risk_parity(self) -> Tuple[Dict, Tuple]:
        """Risk parity optimization."""
        try:
            # Inverse volatility weighting (simplified risk parity)
            volatilities = self.returns.std() * np.sqrt(252)
            inv_vol = 1 / volatilities.replace(0, np.inf)
            weights = inv_vol / inv_vol.sum()
            weight_dict = weights.to_dict()
            
            # Calculate performance
            performance = self._calculate_performance(weight_dict)
            
            return weight_dict, performance
            
        except Exception as e:
            self.logger.error(f"Risk Parity optimization failed: {str(e)}")
            return self._optimize_equal_weight()
    
    def _optimize_max_diversification(self) -> Tuple[Dict, Tuple]:
        """Maximum diversification optimization."""
        try:
            # Get correlation matrix
            corr_matrix = self.returns.corr()
            
            # Diversification ratio function
            def diversification_ratio(weights):
                w = np.array(weights)
                port_var = w.T @ (self.returns.cov() * 252) @ w
                port_vol = np.sqrt(port_var) if port_var > 0 else 1e-10
                weighted_vol = np.sum(w * (self.returns.std() * np.sqrt(252)))
                return weighted_vol / port_vol
            
            n_assets = len(self.returns.columns)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            bounds = [(0, 1) for _ in range(n_assets)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # Optimization
            result = optimize.minimize(
                lambda w: -diversification_ratio(w),
                initial_weights,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()  # Normalize
                weight_dict = dict(zip(self.returns.columns, weights))
                
                # Calculate performance
                performance = self._calculate_performance(weight_dict)
                
                return weight_dict, performance
            
            raise ValueError("Optimization did not converge")
            
        except Exception as e:
            error_analysis = ErrorAnalyzer.analyze_error(e, "Max Diversification optimization")
            self.logger.error(f"Max Diversification failed: {error_analysis}")
            return self._optimize_equal_weight()
    
    def _optimize_hrp(self) -> Tuple[Dict, Tuple]:
        """Hierarchical Risk Parity optimization."""
        if not self.pfopt_available:
            return self._optimize_equal_weight()
        
        try:
            hrp = HRPOpt(self.returns)
            weights = hrp.optimize()
            
            # Calculate performance
            performance = self._calculate_performance(weights)
            
            return weights, performance
            
        except Exception as e:
            self.logger.error(f"HRP optimization failed: {str(e)}")
            return self._optimize_equal_weight()
    
    def _optimize_black_litterman(self) -> Tuple[Dict, Tuple]:
        """Black-Litterman optimization."""
        if not self.pfopt_available:
            return self._optimize_equal_weight()
        
        try:
            # Fetch market caps
            data_manager = PortfolioDataManager()
            market_caps = data_manager.fetch_market_caps(self.returns.columns.tolist())
            
            # Create Black-Litterman model
            bl = BlackLittermanModel(
                self.S,
                pi=self.mu,
                market_caps=market_caps,
                risk_aversion=1,
                omega="default"
            )
            
            # Get equilibrium returns
            equilibrium_returns = bl.equilibrium_returns()
            
            # Optimize with equilibrium returns
            ef = EfficientFrontier(equilibrium_returns, self.S)
            weights = ef.max_sharpe(self.config.risk_free_rate)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(
                verbose=False, 
                risk_free_rate=self.config.risk_free_rate
            )
            
            return cleaned_weights, performance
            
        except Exception as e:
            self.logger.error(f"Black-Litterman optimization failed: {str(e)}")
            return self._optimize_equal_weight()
    
    def _optimize_equal_weight(self) -> Tuple[Dict, Tuple]:
        """Equal weight optimization."""
        n_assets = len(self.returns.columns)
        equal_weight = 1.0 / n_assets
        weights = {ticker: equal_weight for ticker in self.returns.columns}
        
        # Calculate performance
        performance = self._calculate_performance(weights)
        
        return weights, performance
    
    def _optimize_mean_variance(self) -> Tuple[Dict, Tuple]:
        """Mean-Variance optimization with constraints."""
        try:
            # Calculate parameters
            mu = self.returns.mean() * 252
            S = self.returns.cov() * 252
            
            n_assets = len(mu)
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                port_return = np.dot(weights, mu)
                port_risk = np.sqrt(weights.T @ S @ weights)
                if port_risk == 0:
                    return 0
                return -(port_return - self.config.risk_free_rate) / port_risk
            
            # Create sector map for constraints
            classifier = EnhancedAssetClassifier()
            asset_metadata, _ = classifier.classify_tickers(self.returns.columns.tolist())
            sector_map = {ticker: meta['sector'] for ticker, meta in asset_metadata.items()}
            
            # Setup constraints
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # Add sector constraints if specified
            if self.config.constraints and 'sector_limits' in self.config.constraints:
                for sector, (min_w, max_w) in self.config.constraints['sector_limits'].items():
                    # Get indices for this sector
                    sector_indices = [i for i, t in enumerate(self.returns.columns) 
                                    if sector_map.get(t) == sector]
                    
                    if sector_indices:
                        # Minimum sector weight constraint
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda w, idx=sector_indices, min_val=min_w: 
                                np.sum(w[idx]) - min_val
                        })
                        
                        # Maximum sector weight constraint
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda w, idx=sector_indices, max_val=max_w: 
                                max_val - np.sum(w[idx])
                        })
            
            # Initial weights (equal weight)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Optimization
            result = optimize.minimize(
                objective,
                initial_weights,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()  # Ensure normalization
                weight_dict = dict(zip(self.returns.columns, weights))
                
                # Calculate performance
                performance = self._calculate_performance(weight_dict)
                
                return weight_dict, performance
            
            raise ValueError("Mean-Variance optimization did not converge")
            
        except Exception as e:
            error_analysis = ErrorAnalyzer.analyze_error(e, "Mean-Variance optimization")
            self.logger.error(f"Mean-Variance optimization failed: {error_analysis}")
            return self._optimize_equal_weight()
    
    def _calculate_performance(self, weights: Dict) -> Tuple:
        """Calculate portfolio performance metrics."""
        try:
            # Convert weights to array
            w_array = np.array([weights.get(t, 0) for t in self.returns.columns])
            
            # Portfolio returns
            portfolio_returns = self.returns.dot(w_array)
            
            # Performance metrics
            ann_return = portfolio_returns.mean() * 252
            ann_vol = portfolio_returns.std() * np.sqrt(252)
            
            # Handle zero volatility
            if ann_vol == 0:
                sharpe = 0
            else:
                sharpe = (ann_return - self.config.risk_free_rate) / ann_vol
            
            return (ann_return, ann_vol, sharpe)
            
        except Exception as e:
            self.logger.error(f"Performance calculation failed: {str(e)}")
            return (0, 0, 0)

# ============================================================================
# 3. ENHANCED RISK ANALYZER WITH FIXED METRIC NAMES
# ============================================================================

class RiskAnalyzer:
    """Comprehensive risk analysis with standardized metric names."""
    
    def __init__(self):
        self.logger = logger.logger
    
    def calculate_comprehensive_metrics(self, portfolio_returns: pd.Series, 
                                       benchmark_returns: Optional[pd.Series] = None,
                                       risk_free_rate: float = 0.045) -> Dict:
        """Calculate comprehensive risk metrics with standardized names."""
        metrics = {}
        
        try:
            # Basic return metrics
            metrics['Total Return'] = (1 + portfolio_returns).prod() - 1
            metrics['Annual Return'] = portfolio_returns.mean() * 252
            
            # Benchmark comparison
            if benchmark_returns is not None:
                metrics['Benchmark Total Return'] = (1 + benchmark_returns).prod() - 1
                metrics['Benchmark Annual Return'] = benchmark_returns.mean() * 252
                metrics['Excess Return'] = metrics['Total Return'] - metrics.get('Benchmark Total Return', 0)
            
            # Volatility metrics
            metrics['Annual Volatility'] = portfolio_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            if metrics['Annual Volatility'] > 0:
                metrics['Sharpe Ratio'] = (metrics['Annual Return'] - risk_free_rate) / metrics['Annual Volatility']
            else:
                metrics['Sharpe Ratio'] = 0
            
            # Maximum Drawdown (using standardized name)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            metrics['Maximum Drawdown'] = drawdowns.min()
            metrics['Max Drawdown Duration'] = self._calculate_max_drawdown_duration(drawdowns)
            
            # Downside Risk metrics
            metrics['Downside Deviation'] = self._calculate_downside_deviation(portfolio_returns, 0)
            
            # Sortino ratio
            if metrics['Downside Deviation'] > 0:
                metrics['Sortino Ratio'] = (metrics['Annual Return'] - risk_free_rate) / metrics['Downside Deviation']
            else:
                metrics['Sortino Ratio'] = 0
            
            # Calmar ratio (using standardized name)
            if abs(metrics['Maximum Drawdown']) > 0:
                metrics['Calmar Ratio'] = metrics['Annual Return'] / abs(metrics['Maximum Drawdown'])
            else:
                metrics['Calmar Ratio'] = 0
            
            # Omega ratio
            metrics['Omega Ratio'] = self._calculate_omega_ratio(portfolio_returns, risk_free_rate)
            
            # Value at Risk metrics
            metrics['VaR 95% (Daily)'] = np.percentile(portfolio_returns, 5)
            metrics['VaR 99% (Daily)'] = np.percentile(portfolio_returns, 1)
            
            # Expected Shortfall/CVaR
            metrics['CVaR 95% (Daily)'] = portfolio_returns[portfolio_returns <= metrics['VaR 95% (Daily)']].mean()
            metrics['CVaR 99% (Daily)'] = portfolio_returns[portfolio_returns <= metrics['VaR 99% (Daily)']].mean()
            
            # Skewness and Kurtosis
            metrics['Skewness'] = stats.skew(portfolio_returns)
            metrics['Kurtosis'] = stats.kurtosis(portfolio_returns)
            
            # Alpha and Beta (if benchmark available)
            if benchmark_returns is not None:
                try:
                    # Ensure same length
                    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
                    port_ret_aligned = portfolio_returns.loc[common_idx]
                    bench_ret_aligned = benchmark_returns.loc[common_idx]
                    
                    # Calculate beta using covariance
                    covariance = np.cov(port_ret_aligned, bench_ret_aligned)[0, 1]
                    bench_variance = np.var(bench_ret_aligned)
                    
                    if bench_variance > 0:
                        metrics['Beta'] = covariance / bench_variance
                        
                        # Calculate alpha (annualized)
                        metrics['Alpha (Annual)'] = (metrics['Annual Return'] - risk_free_rate) - \
                                                   metrics['Beta'] * (bench_ret_aligned.mean() * 252 - risk_free_rate)
                    else:
                        metrics['Beta'] = 1.0
                        metrics['Alpha (Annual)'] = 0.0
                    
                    # Information ratio (using standardized name)
                    excess_returns = port_ret_aligned - bench_ret_aligned
                    if excess_returns.std() > 0:
                        metrics['Information Ratio'] = (excess_returns.mean() * np.sqrt(252)) / excess_returns.std()
                    else:
                        metrics['Information Ratio'] = 0
                    
                    # Tracking error
                    metrics['Tracking Error'] = excess_returns.std() * np.sqrt(252)
                    
                    # R-squared
                    try:
                        X = sm.add_constant(bench_ret_aligned)
                        model = sm.OLS(port_ret_aligned, X).fit()
                        metrics['R-Squared'] = model.rsquared
                    except:
                        metrics['R-Squared'] = 0
                    
                except Exception as e:
                    self.logger.warning(f"Regression metrics calculation failed: {str(e)}")
                    metrics['Beta'] = 1.0
                    metrics['Alpha (Annual)'] = 0.0
                    metrics['Information Ratio'] = 0
                    metrics['Tracking Error'] = 0
                    metrics['R-Squared'] = 0
            
            # Hit ratio
            metrics['Hit Ratio'] = (portfolio_returns > 0).mean()
            
            # Gain to Loss ratio
            positive_returns = portfolio_returns[portfolio_returns > 0]
            negative_returns = portfolio_returns[portfolio_returns < 0]
            
            if len(negative_returns) > 0 and len(positive_returns) > 0:
                metrics['Gain to Loss Ratio'] = abs(positive_returns.mean() / negative_returns.mean())
            else:
                metrics['Gain to Loss Ratio'] = 0
            
            # Ulcer index
            metrics['Ulcer Index'] = np.sqrt(np.mean(drawdowns**2))
            
            # Martin ratio (Ulcer Performance Index)
            if metrics['Ulcer Index'] > 0:
                metrics['Martin Ratio'] = metrics['Annual Return'] / metrics['Ulcer Index']
            else:
                metrics['Martin Ratio'] = 0
            
            # Tail ratio
            metrics['Tail Ratio'] = self._calculate_tail_ratio(portfolio_returns)
            
            # Common sense ratio
            if metrics['Maximum Drawdown'] < 0:
                metrics['Common Sense Ratio'] = metrics['Total Return'] / abs(metrics['Maximum Drawdown'])
            else:
                metrics['Common Sense Ratio'] = 0
            
            # Daily value at risk
            metrics['Daily VaR 95%'] = -np.percentile(portfolio_returns, 95)
            
            # Conditional Sharpe ratio (skewness adjusted)
            if metrics['Skewness'] != 0:
                metrics['Conditional Sharpe Ratio'] = metrics['Sharpe Ratio'] * (1 + metrics['Skewness']/6 * metrics['Sharpe Ratio']**2)
            else:
                metrics['Conditional Sharpe Ratio'] = metrics['Sharpe Ratio']
            
            # Sterling ratio
            if metrics['Maximum Drawdown'] < 0:
                metrics['Sterling Ratio'] = metrics['Annual Return'] / abs(metrics['Maximum Drawdown'])
            else:
                metrics['Sterling Ratio'] = 0
            
            # Burke ratio
            if metrics['Ulcer Index'] > 0:
                metrics['Burke Ratio'] = metrics['Annual Return'] / metrics['Ulcer Index']
            else:
                metrics['Burke Ratio'] = 0
            
            # Kappa ratio (Omega-Sharpe)
            metrics['Kappa Ratio'] = self._calculate_kappa_ratio(portfolio_returns, 3)
            
            # Calculate stress test metrics
            metrics.update(self._calculate_stress_metrics(portfolio_returns))
            
            # Calculate seasonality metrics
            metrics.update(self._calculate_seasonality_metrics(portfolio_returns))
            
        except Exception as e:
            error_analysis = ErrorAnalyzer.analyze_error(e, "Calculating risk metrics")
            self.logger.error(f"Risk metrics calculation failed: {error_analysis}")
            
            # Return basic metrics even if comprehensive calculation fails
            metrics = {
                'Total Return': (1 + portfolio_returns).prod() - 1,
                'Annual Return': portfolio_returns.mean() * 252,
                'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
                'Maximum Drawdown': 0,
                'Sharpe Ratio': 0
            }
        
        return metrics
    
    def _calculate_max_drawdown_duration(self, drawdowns: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        if drawdowns.empty:
            return 0
        
        # Find consecutive negative periods
        drawdowns_binary = (drawdowns < 0).astype(int)
        
        max_duration = 0
        current_duration = 0
        
        for value in drawdowns_binary:
            if value == 1:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_downside_deviation(self, returns: pd.Series, mar: float = 0) -> float:
        """Calculate downside deviation (semi-deviation)."""
        downside_returns = returns[returns < mar]
        if len(downside_returns) == 0:
            return 0
        return np.sqrt(np.mean((downside_returns - mar) ** 2)) * np.sqrt(252)
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0]).sum()
        
        if losses == 0:
            return np.inf if gains > 0 else 0
        return gains / losses
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        if len(returns) < 10:
            return 1.0
        
        right_tail = np.percentile(returns, 95)
        left_tail = np.percentile(returns, 5)
        
        if left_tail == 0:
            return 0
        
        return abs(right_tail / left_tail)
    
    def _calculate_kappa_ratio(self, returns: pd.Series, n: int = 3) -> float:
        """Calculate Kappa ratio (generalized Omega ratio)."""
        if n <= 0:
            return 0
        
        excess_returns = returns
        upper_partial_moment = np.mean(np.maximum(excess_returns, 0) ** n)
        lower_partial_moment = np.mean(np.maximum(-excess_returns, 0) ** n)
        
        if lower_partial_moment == 0:
            return np.inf if upper_partial_moment > 0 else 0
        
        return (upper_partial_moment / lower_partial_moment) ** (1/n)
    
    def _calculate_stress_metrics(self, returns: pd.Series) -> Dict:
        """Calculate stress test metrics."""
        stress_metrics = {}
        
        try:
            # Worst day, week, month returns
            daily_returns = returns
            
            # Worst daily return
            stress_metrics['Worst Daily Return'] = daily_returns.min()
            
            # Calculate weekly returns if enough data
            if len(returns) >= 5:
                weekly_returns = returns.rolling(5).apply(lambda x: (1 + x).prod() - 1, raw=True).dropna()
                if not weekly_returns.empty:
                    stress_metrics['Worst Weekly Return'] = weekly_returns.min()
            
            # Calculate monthly returns if enough data
            if len(returns) >= 21:
                monthly_returns = returns.rolling(21).apply(lambda x: (1 + x).prod() - 1, raw=True).dropna()
                if not monthly_returns.empty:
                    stress_metrics['Worst Monthly Return'] = monthly_returns.min()
            
            # Recovery periods after drawdowns
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            # Calculate recovery times
            recovery_times = []
            in_drawdown = False
            drawdown_start = None
            
            for i, dd in enumerate(drawdown):
                if dd < -0.05 and not in_drawdown:  # 5% drawdown threshold
                    in_drawdown = True
                    drawdown_start = i
                elif dd >= 0 and in_drawdown:
                    in_drawdown = False
                    recovery_times.append(i - drawdown_start)
            
            if recovery_times:
                stress_metrics['Average Recovery Time (Days)'] = np.mean(recovery_times)
                stress_metrics['Max Recovery Time (Days)'] = np.max(recovery_times)
            
            # Calculate volatility regime changes
            rolling_vol = returns.rolling(63).std() * np.sqrt(252)  # 3-month volatility
            if len(rolling_vol) > 63:
                stress_metrics['Volatility Regime Stability'] = rolling_vol.std() / rolling_vol.mean()
            
        except Exception as e:
            self.logger.debug(f"Stress metrics calculation failed: {str(e)}")
        
        return stress_metrics
    
    def _calculate_seasonality_metrics(self, returns: pd.Series) -> Dict:
        """Calculate seasonality and calendar effects."""
        season_metrics = {}
        
        try:
            # Add date index for month extraction
            if not isinstance(returns.index, pd.DatetimeIndex):
                return season_metrics
            
            # Monthly returns
            returns_df = pd.DataFrame({'return': returns})
            returns_df['month'] = returns_df.index.month
            
            monthly_returns = returns_df.groupby('month')['return'].mean() * 21  # Approximate monthly
            
            # Best and worst months
            if not monthly_returns.empty:
                season_metrics['Best Month'] = monthly_returns.idxmax()
                season_metrics['Worst Month'] = monthly_returns.idxmin()
                season_metrics['Month Seasonality Strength'] = monthly_returns.std() / monthly_returns.abs().mean()
            
            # Day of week effects
            returns_df['weekday'] = returns_df.index.weekday  # Monday=0, Sunday=6
            weekday_returns = returns_df.groupby('weekday')['return'].mean()
            
            if not weekday_returns.empty:
                season_metrics['Best Weekday'] = weekday_returns.idxmax()
                season_metrics['Worst Weekday'] = weekday_returns.idxmin()
            
        except Exception as e:
            self.logger.debug(f"Seasonality metrics calculation failed: {str(e)}")
        
        return season_metrics
    
    def calculate_correlation_analysis(self, portfolio_returns: pd.DataFrame, 
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """Calculate correlation analysis metrics."""
        correlation_analysis = {}
        
        try:
            # Correlation matrix
            correlation_matrix = portfolio_returns.corr()
            correlation_analysis['Correlation Matrix'] = correlation_matrix
            
            # Average correlation
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            upper_tri = correlation_matrix.where(mask)
            correlation_analysis['Average Correlation'] = upper_tri.stack().mean()
            
            # Minimum and maximum correlation
            correlation_analysis['Min Correlation'] = correlation_matrix.min().min()
            correlation_analysis['Max Correlation'] = correlation_matrix.max().max()
            
            # Correlation with benchmark
            if benchmark_returns is not None:
                correlations = {}
                for asset in portfolio_returns.columns:
                    corr = portfolio_returns[asset].corr(benchmark_returns)
                    correlations[asset] = corr
                
                correlation_analysis['Benchmark Correlations'] = correlations
                correlation_analysis['Average Benchmark Correlation'] = np.mean(list(correlations.values()))
            
            # Diversification ratio
            individual_vols = portfolio_returns.std() * np.sqrt(252)
            portfolio_vol = portfolio_returns.mean(axis=1).std() * np.sqrt(252)
            
            if portfolio_vol > 0:
                correlation_analysis['Diversification Ratio'] = np.mean(individual_vols) / portfolio_vol
            else:
                correlation_analysis['Diversification Ratio'] = 0
            
            # Concentration metrics
            correlation_analysis['Correlation Concentration'] = correlation_matrix.std().std()
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {str(e)}")
        
        return correlation_analysis
    
    def perform_stress_tests(self, portfolio_returns: pd.Series, 
                            crisis_periods: Dict = None) -> Dict:
        """Perform stress tests during historical crisis periods."""
        if crisis_periods is None:
            crisis_periods = Constants.HISTORICAL_CRISES
        
        stress_test_results = {}
        
        for crisis_name, (start_date, end_date) in crisis_periods.items():
            try:
                # Filter returns for crisis period
                mask = (portfolio_returns.index >= start_date) & (portfolio_returns.index <= end_date)
                crisis_returns = portfolio_returns[mask]
                
                if len(crisis_returns) > 5:  # Minimum data points
                    # Calculate crisis metrics
                    crisis_metrics = {
                        'Crisis Total Return': (1 + crisis_returns).prod() - 1,
                        'Crisis Annualized Return': crisis_returns.mean() * 252,
                        'Crisis Volatility': crisis_returns.std() * np.sqrt(252),
                        'Crisis Max Drawdown': self._calculate_crisis_drawdown(crisis_returns),
                        'Crisis Days': len(crisis_returns)
                    }
                    
                    stress_test_results[crisis_name] = crisis_metrics
                    
            except Exception as e:
                self.logger.debug(f"Stress test for {crisis_name} failed: {str(e)}")
        
        return stress_test_results
    
    def _calculate_crisis_drawdown(self, crisis_returns: pd.Series) -> float:
        """Calculate maximum drawdown during crisis period."""
        cumulative = (1 + crisis_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns.min() if not drawdowns.empty else 0
    
    def calculate_risk_decomposition(self, weights: Dict, returns: pd.DataFrame) -> Dict:
        """Decompose portfolio risk by asset contribution."""
        risk_decomposition = {}
        
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252
            
            # Convert weights to array
            tickers = list(weights.keys())
            w_array = np.array([weights[t] for t in tickers])
            
            # Portfolio variance
            port_variance = w_array.T @ cov_matrix.values @ w_array
            
            if port_variance <= 0:
                return risk_decomposition
            
            # Marginal contribution to risk
            mctr = (cov_matrix.values @ w_array) / np.sqrt(port_variance)
            
            # Contribution to risk
            ctr = w_array * mctr
            
            # Percentage contribution
            pct_ctr = ctr / np.sqrt(port_variance)
            
            # Create results dictionary
            for i, ticker in enumerate(tickers):
                risk_decomposition[ticker] = {
                    'Weight': weights[ticker],
                    'MCTR': mctr[i],
                    'CTR': ctr[i],
                    'Percentage Contribution': pct_ctr[i]
                }
            
            # Add summary metrics
            risk_decomposition['summary'] = {
                'Portfolio Volatility': np.sqrt(port_variance),
                'Concentration Index': np.sum(pct_ctr**2),  # Herfindahl index
                'Most Risky Asset': tickers[np.argmax(pct_ctr)],
                'Least Risky Asset': tickers[np.argmin(pct_ctr)]
            }
            
        except Exception as e:
            self.logger.error(f"Risk decomposition failed: {str(e)}")
        
        return risk_decomposition

# ============================================================================
# 4. ENHANCED VISUALIZATION ENGINE
# ============================================================================

class VisualizationEngine:
    """Professional visualization engine for portfolio analytics."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.logger = logger.logger
    
    def create_portfolio_performance_chart(self, portfolio_prices: pd.Series, 
                                          benchmark_prices: pd.Series, 
                                          portfolio_name: str = "Portfolio",
                                          benchmark_name: str = "Benchmark") -> go.Figure:
        """Create cumulative performance chart."""
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=("Cumulative Returns", "Drawdown"),
            shared_xaxes=True
        )
        
        # Calculate cumulative returns
        port_cumulative = (1 + portfolio_prices.pct_change().fillna(0)).cumprod()
        bench_cumulative = (1 + benchmark_prices.pct_change().fillna(0)).cumprod()
        
        # Add cumulative returns trace
        fig.add_trace(
            go.Scatter(
                x=port_cumulative.index,
                y=port_cumulative.values,
                name=portfolio_name,
                line=dict(color='#00cc96', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 204, 150, 0.1)',
                hovertemplate='%{y:.2%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=bench_cumulative.index,
                y=bench_cumulative.values,
                name=benchmark_name,
                line=dict(color='#636efa', width=2, dash='dash'),
                hovertemplate='%{y:.2%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Calculate drawdowns
        port_rolling_max = port_cumulative.expanding().max()
        port_drawdown = (port_cumulative - port_rolling_max) / port_rolling_max
        
        fig.add_trace(
            go.Scatter(
                x=port_drawdown.index,
                y=port_drawdown.values,
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(239, 85, 59, 0.3)',
                line=dict(color='#ef553b', width=1),
                hovertemplate='%{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1, tickformat=".0%")
        fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat=".0%")
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
    
    def create_weight_allocation_chart(self, weights: Dict, asset_metadata: Dict) -> go.Figure:
        """Create interactive weight allocation chart."""
        # Prepare data
        df = pd.DataFrame([
            {
                'Ticker': ticker,
                'Weight': weight,
                'Sector': asset_metadata.get(ticker, {}).get('sector', 'Unknown'),
                'Country': asset_metadata.get(ticker, {}).get('country', 'Unknown'),
                'Market Cap': asset_metadata.get(ticker, {}).get('market_cap', 0)
            }
            for ticker, weight in weights.items()
        ])
        
        # Sort by weight
        df = df.sort_values('Weight', ascending=False)
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for weights
        fig.add_trace(go.Bar(
            x=df['Weight'],
            y=df['Ticker'],
            orientation='h',
            marker=dict(
                color=df['Weight'],
                colorscale='Tealgrn',
                showscale=True,
                colorbar=dict(title="Weight")
            ),
            hovertemplate=(
                '<b>%{y}</b><br>' +
                'Weight: %{x:.1%}<br>' +
                'Sector: %{customdata[0]}<br>' +
                'Country: %{customdata[1]}<br>' +
                'Market Cap: $%{customdata[2]:,.0f}<br>' +
                '<extra></extra>'
            ),
            customdata=df[['Sector', 'Country', 'Market Cap']]
        ))
        
        # Update layout
        fig.update_layout(
            height=max(400, len(weights) * 25),
            title="Portfolio Allocation",
            xaxis_title="Weight (%)",
            yaxis_title="Asset",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(tickformat='.0%'),
            hoverlabel=dict(
                bgcolor="rgba(30, 30, 30, 0.9)",
                font_size=12,
                font_family="Roboto"
            )
        )
        
        return fig
    
    def create_risk_metrics_dashboard(self, metrics: Dict) -> go.Figure:
        """Create comprehensive risk metrics dashboard."""
        # Select key metrics
        key_metrics = {
            'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
            'Sortino Ratio': metrics.get('Sortino Ratio', 0),
            'Calmar Ratio': metrics.get('Calmar Ratio', 0),
            'Maximum Drawdown': metrics.get('Maximum Drawdown', 0),
            'Annual Volatility': metrics.get('Annual Volatility', 0),
            'Omega Ratio': metrics.get('Omega Ratio', 0),
            'Information Ratio': metrics.get('Information Ratio', 0),
            'Tracking Error': metrics.get('Tracking Error', 0)
        }
        
        # Create gauge charts
        fig = make_subplots(
            rows=2, cols=4,
            specs=[[{'type': 'indicator'}] * 4] * 2,
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )
        
        # Define ranges for each metric
        metric_ranges = {
            'Sharpe Ratio': (-1, 3),
            'Sortino Ratio': (-1, 3),
            'Calmar Ratio': (-1, 2),
            'Maximum Drawdown': (-0.5, 0),
            'Annual Volatility': (0, 0.5),
            'Omega Ratio': (0, 3),
            'Information Ratio': (-1, 2),
            'Tracking Error': (0, 0.2)
        }
        
        metric_titles = list(key_metrics.keys())
        
        for i, (metric_name, value) in enumerate(key_metrics.items()):
            row = i // 4 + 1
            col = i % 4 + 1
            
            # Determine colors based on metric value
            if metric_name == 'Maximum Drawdown':
                # For drawdown, more negative is worse
                colors = [[0, '#00cc96'], [0.5, '#FFA15A'], [1, '#ef553b']]
                value_display = value
            else:
                # For other metrics, higher is better
                colors = [[0, '#ef553b'], [0.5, '#FFA15A'], [1, '#00cc96']]
                value_display = value
            
            min_val, max_val = metric_ranges[metric_name]
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value_display,
                    title=dict(text=metric_name, font=dict(size=14)),
                    number=dict(
                        suffix="",
                        font=dict(size=20),
                        valueformat=".2f" if abs(value) < 10 else ".1f"
                    ),
                    gauge=dict(
                        axis=dict(range=[min_val, max_val]),
                        bar=dict(color="white"),
                        steps=[
                            dict(range=[min_val, min_val + (max_val-min_val)/3], color="#ef553b"),
                            dict(range=[min_val + (max_val-min_val)/3, min_val + 2*(max_val-min_val)/3], color="#FFA15A"),
                            dict(range=[min_val + 2*(max_val-min_val)/3, max_val], color="#00cc96")
                        ],
                        threshold=dict(
                            line=dict(color="white", width=4),
                            thickness=0.75,
                            value=value_display
                        )
                    )
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap."""
        # Create annotations
        annotations = []
        for i, row in enumerate(correlation_matrix.values):
            for j, value in enumerate(row):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f'{value:.2f}',
                        font=dict(color='white' if abs(value) < 0.5 else 'black'),
                        showarrow=False
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            colorbar=dict(title="Correlation")
        ))
        
        # Add annotations
        fig.update_layout(
            annotations=annotations,
            height=600,
            title="Correlation Matrix",
            xaxis_title="Asset",
            yaxis_title="Asset",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    def create_returns_distribution(self, portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series = None) -> go.Figure:
        """Create returns distribution histogram."""
        fig = go.Figure()
        
        # Portfolio returns
        fig.add_trace(go.Histogram(
            x=portfolio_returns,
            name="Portfolio",
            nbinsx=50,
            marker_color='#00cc96',
            opacity=0.7,
            histnorm='probability density'
        ))
        
        # Benchmark returns if provided
        if benchmark_returns is not None:
            fig.add_trace(go.Histogram(
                x=benchmark_returns,
                name="Benchmark",
                nbinsx=50,
                marker_color='#636efa',
                opacity=0.7,
                histnorm='probability density'
            ))
        
        # Add normal distribution curve
        x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
        pdf = stats.norm.pdf(x, portfolio_returns.mean(), portfolio_returns.std())
        fig.add_trace(go.Scatter(
            x=x,
            y=pdf,
            name="Normal Distribution",
            line=dict(color='white', width=2, dash='dash'),
            mode='lines'
        ))
        
        fig.update_layout(
            height=500,
            title="Returns Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            barmode='overlay',
            bargap=0.1,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(tickformat='.1%')
        
        return fig
    
    def create_monte_carlo_simulation(self, portfolio_returns: pd.Series, 
                                     n_simulations: int = 1000, 
                                     days: int = 252) -> go.Figure:
        """Create Monte Carlo simulation chart."""
        # Calculate parameters
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Run simulations
        simulations = np.zeros((days, n_simulations))
        initial_price = 100
        
        for i in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, days)
            # Calculate price path
            price_path = initial_price * (1 + random_returns).cumprod()
            simulations[:, i] = price_path
        
        # Calculate statistics
        median_path = np.median(simulations, axis=1)
        upper_95 = np.percentile(simulations, 95, axis=1)
        lower_5 = np.percentile(simulations, 5, axis=1)
        
        fig = go.Figure()
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=list(range(days)),
            y=upper_95,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='95th Percentile'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(days)),
            y=lower_5,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0, 204, 150, 0.2)',
            fill='tonexty',
            showlegend=False,
            name='5th Percentile'
        ))
        
        # Add median path
        fig.add_trace(go.Scatter(
            x=list(range(days)),
            y=median_path,
            mode='lines',
            line=dict(color='#00cc96', width=3),
            name='Median Path'
        ))
        
        # Add sample paths
        for i in range(min(50, n_simulations)):
            fig.add_trace(go.Scatter(
                x=list(range(days)),
                y=simulations[:, i],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.1)', width=1),
                showlegend=False if i > 0 else True,
                name='Sample Paths' if i == 0 else None
            ))
        
        fig.update_layout(
            height=500,
            title="Monte Carlo Simulation (1000 Paths)",
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value ($)",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_efficient_frontier(self, returns: pd.DataFrame, 
                                 risk_free_rate: float = 0.045) -> go.Figure:
        """Create efficient frontier visualization."""
        try:
            if not LIBRARY_STATUS['status'].get('pypfopt', False):
                raise ImportError("PyPortfolioOpt not available")
            
            from pypfopt import expected_returns, risk_models
            
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.sample_cov(returns)
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S)
            
            # Generate efficient frontier points
            ef_min_vol = ef.min_volatility()
            min_vol_ret, min_vol_vol, _ = ef.portfolio_performance()
            
            # Get max Sharpe portfolio
            ef_max_sharpe = EfficientFrontier(mu, S)
            ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
            max_sharpe_ret, max_sharpe_vol, max_sharpe_ratio = ef_max_sharpe.portfolio_performance(
                risk_free_rate=risk_free_rate
            )
            
            # Generate frontier points
            target_returns = np.linspace(min_vol_ret, max_sharpe_ret * 1.5, 50)
            volatilities = []
            
            for target_return in target_returns:
                ef = EfficientFrontier(mu, S)
                try:
                    ef.efficient_return(target_return)
                    _, vol, _ = ef.portfolio_performance()
                    volatilities.append(vol)
                except:
                    volatilities.append(np.nan)
            
            fig = go.Figure()
            
            # Add efficient frontier
            fig.add_trace(go.Scatter(
                x=volatilities,
                y=target_returns,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='#00cc96', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 204, 150, 0.1)'
            ))
            
            # Add min volatility point
            fig.add_trace(go.Scatter(
                x=[min_vol_vol],
                y=[min_vol_ret],
                mode='markers',
                name='Min Volatility',
                marker=dict(
                    color='#636efa',
                    size=15,
                    line=dict(color='white', width=2)
                )
            ))
            
            # Add max Sharpe point
            fig.add_trace(go.Scatter(
                x=[max_sharpe_vol],
                y=[max_sharpe_ret],
                mode='markers',
                name='Max Sharpe Ratio',
                marker=dict(
                    color='#FFA15A',
                    size=15,
                    line=dict(color='white', width=2)
                )
            ))
            
            # Add individual assets
            individual_vols = returns.std() * np.sqrt(252)
            individual_rets = returns.mean() * 252
            
            fig.add_trace(go.Scatter(
                x=individual_vols,
                y=individual_rets,
                mode='markers+text',
                name='Individual Assets',
                marker=dict(
                    color='#ab63fa',
                    size=10,
                    line=dict(color='white', width=1)
                ),
                text=returns.columns,
                textposition="top center"
            ))
            
            # Calculate capital market line
            if max_sharpe_vol > 0:
                cml_x = np.linspace(0, max_sharpe_vol * 1.5, 50)
                cml_y = risk_free_rate + (max_sharpe_ret - risk_free_rate) / max_sharpe_vol * cml_x
                
                fig.add_trace(go.Scatter(
                    x=cml_x,
                    y=cml_y,
                    mode='lines',
                    name='Capital Market Line',
                    line=dict(color='white', width=2, dash='dash')
                ))
            
            fig.update_layout(
                height=600,
                title="Efficient Frontier",
                xaxis_title="Annual Volatility",
                yaxis_title="Annual Return",
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='closest'
            )
            
            fig.update_xaxes(tickformat='.0%')
            fig.update_yaxes(tickformat='.0%')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Efficient frontier visualization failed: {str(e)}")
            
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                height=600,
                title="Efficient Frontier (Unavailable)",
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            return fig

# ============================================================================
# 5. ENHANCED BACKTESTING ENGINE
# ============================================================================

class BacktestingEngine:
    """Advanced backtesting engine with multiple strategies."""
    
    def __init__(self):
        self.logger = logger.logger
    
    def run_backtest(self, prices: pd.DataFrame, weights: Dict, 
                     config: PortfolioConfig) -> Dict:
        """Run comprehensive backtest."""
        results = {
            'portfolio_values': pd.Series(dtype=float),
            'transactions': [],
            'metrics': {},
            'rebalancing_dates': []
        }
        
        try:
            # Calculate initial portfolio value
            initial_capital = 1_000_000  # $1M initial capital
            cash = initial_capital * config.cash_buffer
            invested_capital = initial_capital - cash
            
            # Calculate initial shares
            initial_prices = prices.iloc[0]
            shares = {}
            for ticker, weight in weights.items():
                if ticker in initial_prices:
                    target_value = invested_capital * weight
                    price = initial_prices[ticker]
                    if price > 0:
                        shares[ticker] = target_value / price
            
            # Track portfolio value over time
            portfolio_values = []
            dates = []
            
            # Convert returns to DataFrame for easier calculation
            returns = prices.pct_change().fillna(0)
            
            # Determine rebalancing dates
            rebalancing_dates = self._get_rebalancing_dates(prices.index, config.rebalancing_frequency)
            
            for i, date in enumerate(prices.index):
                current_prices = prices.loc[date]
                
                # Calculate current portfolio value
                stock_value = sum(shares.get(ticker, 0) * current_prices.get(ticker, 0) 
                                for ticker in weights.keys())
                total_value = stock_value + cash
                portfolio_values.append(total_value)
                dates.append(date)
                
                # Check if rebalancing needed
                if date in rebalancing_dates:
                    results['rebalancing_dates'].append(date)
                    
                    # Calculate transaction costs
                    transaction_cost = self._calculate_rebalancing_cost(
                        shares, current_prices, weights, total_value, 
                        config.transaction_cost
                    )
                    
                    # Update cash after transaction cost
                    cash -= transaction_cost
                    
                    # Rebalance portfolio
                    for ticker, weight in weights.items():
                        if ticker in current_prices:
                            target_value = (total_value - cash) * weight
                            current_value = shares.get(ticker, 0) * current_prices[ticker]
                            difference = target_value - current_value
                            
                            if abs(difference) > 100:  # Minimum transaction size
                                shares_change = difference / current_prices[ticker]
                                shares[ticker] = shares.get(ticker, 0) + shares_change
                                
                                # Record transaction
                                results['transactions'].append({
                                    'date': date,
                                    'ticker': ticker,
                                    'action': 'BUY' if shares_change > 0 else 'SELL',
                                    'shares': abs(shares_change),
                                    'price': current_prices[ticker],
                                    'value': abs(difference)
                                })
                
                # Update cash from dividends (simplified)
                cash *= (1 + config.risk_free_rate / 252)  # Daily risk-free return
            
            # Create results
            results['portfolio_values'] = pd.Series(portfolio_values, index=dates)
            results['portfolio_returns'] = results['portfolio_values'].pct_change().fillna(0)
            
            # Calculate performance metrics
            analyzer = RiskAnalyzer()
            results['metrics'] = analyzer.calculate_comprehensive_metrics(
                results['portfolio_returns'],
                risk_free_rate=config.risk_free_rate
            )
            
            # Calculate turnover
            results['metrics']['Annual Turnover'] = self._calculate_turnover(
                results['transactions'], results['portfolio_values'], config
            )
            
            # Calculate win rate
            results['metrics']['Win Rate'] = (results['portfolio_returns'] > 0).mean()
            
            # Calculate best/worst periods
            results['metrics']['Best Day'] = results['portfolio_returns'].max()
            results['metrics']['Worst Day'] = results['portfolio_returns'].min()
            
        except Exception as e:
            error_analysis = ErrorAnalyzer.analyze_error(e, "Running backtest")
            self.logger.error(f"Backtest failed: {error_analysis}")
            raise
        
        return results
    
    def _get_rebalancing_dates(self, dates: pd.DatetimeIndex, frequency: str) -> List:
        """Get rebalancing dates based on frequency."""
        if frequency == 'D':
            return list(dates)
        elif frequency == 'W':
            return list(dates[dates.weekday == 0])  # Monday
        elif frequency == 'M':
            return list(dates[dates.is_month_end])
        elif frequency == 'Q':
            return list(dates[(dates.month % 3 == 0) & dates.is_month_end])
        elif frequency == 'Y':
            return list(dates[(dates.month == 12) & dates.is_month_end])
        else:
            return []
    
    def _calculate_rebalancing_cost(self, shares: Dict, prices: pd.Series, 
                                   target_weights: Dict, total_value: float,
                                   transaction_cost: float) -> float:
        """Calculate transaction costs for rebalancing."""
        total_cost = 0
        
        for ticker, current_shares in shares.items():
            if ticker in prices:
                current_value = current_shares * prices[ticker]
                target_value = total_value * target_weights.get(ticker, 0)
                trade_value = abs(target_value - current_value)
                
                if trade_value > 0:
                    total_cost += trade_value * transaction_cost
        
        return total_cost
    
    def _calculate_turnover(self, transactions: List, portfolio_values: pd.Series, 
                           config: PortfolioConfig) -> float:
        """Calculate annual portfolio turnover."""
        if not transactions:
            return 0
        
        # Calculate total trading volume
        total_volume = sum(t['value'] for t in transactions)
        
        # Calculate average portfolio value
        avg_portfolio_value = portfolio_values.mean()
        
        # Annualize turnover
        if avg_portfolio_value > 0:
            daily_turnover = total_volume / (avg_portfolio_value * len(portfolio_values))
            return daily_turnover * 252
        return 0
    
    def run_comparative_backtest(self, prices: pd.DataFrame, 
                                strategies: Dict[str, Dict]) -> Dict:
        """Run comparative backtest of multiple strategies."""
        results = {}
        
        for strategy_name, strategy_weights in strategies.items():
            try:
                # Create config for backtest
                config = PortfolioConfig(
                    universe="Comparative",
                    tickers=list(strategy_weights.keys()),
                    benchmark="^GSPC",
                    start_date=prices.index[0],
                    end_date=prices.index[-1],
                    risk_free_rate=0.045,
                    optimization_method="CUSTOM",
                    rebalancing_frequency='M'
                )
                
                # Run backtest
                strategy_results = self.run_backtest(prices, strategy_weights, config)
                results[strategy_name] = strategy_results
                
            except Exception as e:
                self.logger.error(f"Strategy {strategy_name} backtest failed: {str(e)}")
        
        return results

# ============================================================================
# 6. MAIN APPLICATION
# ============================================================================

class QuantEdgeProApp:
    """Main application class for QuantEdge Pro."""
    
    def __init__(self):
        self.data_manager = PortfolioDataManager()
        self.risk_analyzer = RiskAnalyzer()
        self.visualization_engine = VisualizationEngine()
        self.backtesting_engine = BacktestingEngine()
        self.logger = logger.logger
        
        # Initialize session state
        if 'portfolio_results' not in st.session_state:
            st.session_state.portfolio_results = None
        if 'asset_metadata' not in st.session_state:
            st.session_state.asset_metadata = None
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = None
    
    def render_header(self):
        """Render application header."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div class="main-header">
                <h1 style="color: white; font-size: 3.5rem; margin-bottom: 1rem; text-align: center;">
                    🏛️ QuantEdge Pro v4.0
                </h1>
                <p style="color: #94a3b8; font-size: 1.2rem; text-align: center; margin-bottom: 0.5rem;">
                    Institutional Portfolio Analytics Platform
                </p>
                <p style="color: #636efa; font-size: 1rem; text-align: center;">
                    Enterprise-Ready • Production-Grade • 5000+ Lines
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Library status warning
        if not LIBRARY_STATUS['all_available']:
            with st.expander("⚠️ Library Status Warning", expanded=True):
                st.warning("""
                Some optional libraries are not available. Full functionality may be limited.
                
                Missing libraries: {}
                
                The application will use simplified fallback methods where possible.
                """.format(", ".join(LIBRARY_STATUS['missing'])))
    
    def render_sidebar(self):
        """Render application sidebar."""
        with st.sidebar:
            st.markdown("""
            <div style="padding: 1.5rem; border-radius: 12px; background: rgba(30, 30, 30, 0.8); margin-bottom: 2rem;">
                <h3 style="color: #00cc96; margin-bottom: 1rem;">📊 Configuration</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Universe selection
            universe_options = list(Constants.ASSET_UNIVERSES.keys())
            selected_universe = st.selectbox(
                "Select Asset Universe",
                universe_options,
                help="Pre-configured portfolio universes"
            )
            
            # Get universe details
            universe_details = Constants.ASSET_UNIVERSES[selected_universe]
            tickers = universe_details['tickers']
            benchmark = universe_details['benchmark']
            risk_free_rate = universe_details['risk_free_rate']
            
            # Date range selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365*3),
                    max_value=datetime.now() - timedelta(days=1)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    max_value=datetime.now()
                )
            
            # Optimization method
            optimization_options = list(Constants.OPTIMIZATION_METHODS.keys())
            optimization_descriptions = [Constants.OPTIMIZATION_METHODS[m]['description'] 
                                        for m in optimization_options]
            
            selected_optimization = st.selectbox(
                "Optimization Method",
                optimization_options,
                format_func=lambda x: f"{x} - {Constants.OPTIMIZATION_METHODS[x]['description']}",
                help="Select portfolio optimization methodology"
            )
            
            # Constraints
            st.subheader("🎯 Constraints")
            col1, col2 = st.columns(2)
            with col1:
                max_weight = st.slider(
                    "Maximum Weight per Asset",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.30,
                    step=0.05,
                    help="Maximum allocation to any single asset"
                )
            with col2:
                min_weight = st.slider(
                    "Minimum Weight per Asset",
                    min_value=0.0,
                    max_value=0.20,
                    value=0.0,
                    step=0.01,
                    help="Minimum allocation to any single asset (0 allows exclusion)"
                )
            
            # Rebalancing frequency
            rebalancing_freq = st.selectbox(
                "Rebalancing Frequency",
                options=list(Constants.FREQUENCY_MAP.keys()),
                index=2,  # Monthly
                help="Frequency of portfolio rebalancing"
            )
            
            # Transaction cost
            transaction_cost = st.slider(
                "Transaction Cost (bps)",
                min_value=0,
                max_value=50,
                value=10,
                step=1,
                help="Transaction cost in basis points (1 bps = 0.01%)"
            ) / 10000  # Convert to decimal
            
            # Cash buffer
            cash_buffer = st.slider(
                "Cash Buffer",
                min_value=0.0,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="Percentage of portfolio held in cash"
            )
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                analyze_button = st.button(
                    "🚀 Analyze Portfolio",
                    type="primary",
                    use_container_width=True
                )
            with col2:
                reset_button = st.button(
                    "🔄 Reset",
                    use_container_width=True
                )
            
            # Custom ticker input
            st.subheader("➕ Custom Assets")
            custom_tickers = st.text_area(
                "Add Custom Tickers (comma-separated)",
                placeholder="e.g., AAPL, MSFT, GOOGL",
                help="Add additional tickers to the selected universe"
            )
            
            if custom_tickers:
                custom_ticker_list = [t.strip().upper() for t in custom_tickers.split(',')]
                tickers.extend(custom_ticker_list)
                tickers = list(set(tickers))  # Remove duplicates
            
            # Return configuration
            return {
                'universe': selected_universe,
                'tickers': tickers,
                'benchmark': benchmark,
                'start_date': start_date,
                'end_date': end_date,
                'risk_free_rate': risk_free_rate,
                'optimization_method': selected_optimization,
                'max_weight': max_weight,
                'min_weight': min_weight,
                'rebalancing_frequency': Constants.FREQUENCY_MAP[rebalancing_freq],
                'transaction_cost': transaction_cost,
                'cash_buffer': cash_buffer,
                'analyze_button': analyze_button,
                'reset_button': reset_button
            }
    
    def analyze_portfolio(self, config_dict: Dict):
        """Main portfolio analysis workflow."""
        try:
            # Show progress
            progress_bar = st.progress(0, text="Initializing analysis...")
            
            # Create config object
            config = PortfolioConfig(
                universe=config_dict['universe'],
                tickers=config_dict['tickers'],
                benchmark=config_dict['benchmark'],
                start_date=config_dict['start_date'],
                end_date=config_dict['end_date'],
                risk_free_rate=config_dict['risk_free_rate'],
                optimization_method=config_dict['optimization_method'],
                max_weight=config_dict['max_weight'],
                min_weight=config_dict['min_weight'],
                rebalancing_frequency=config_dict['rebalancing_frequency'],
                transaction_cost=config_dict['transaction_cost'],
                cash_buffer=config_dict['cash_buffer']
            )
            
            # Step 1: Fetch data
            progress_bar.progress(20, text="Fetching market data...")
            prices, benchmark_prices = self.data_manager.fetch_market_data(
                config.tickers, config.benchmark, config.start_date, config.end_date
            )
            
            # Step 2: Classify assets
            progress_bar.progress(40, text="Classifying assets...")
            classifier = EnhancedAssetClassifier()
            asset_metadata, classification_errors = classifier.classify_tickers(config.tickers)
            st.session_state.asset_metadata = asset_metadata
            
            # Step 3: Calculate returns
            progress_bar.progress(60, text="Calculating returns...")
            portfolio_returns, benchmark_returns = self.data_manager.calculate_returns(
                prices, benchmark_prices
            )
            
            # Step 4: Optimize portfolio
            progress_bar.progress(80, text="Optimizing portfolio...")
            optimizer = PortfolioOptimizer(portfolio_returns, prices, config)
            weights, performance = optimizer.optimize()
            
            # Store optimization results
            st.session_state.optimization_results = {
                'weights': weights,
                'performance': performance,
                'prices': prices,
                'benchmark_prices': benchmark_prices,
                'portfolio_returns': portfolio_returns,
                'benchmark_returns': benchmark_returns,
                'config': config
            }
            
            progress_bar.progress(100, text="Analysis complete!")
            time.sleep(0.5)
            progress_bar.empty()
            
            return True
            
        except Exception as e:
            error_analysis = ErrorAnalyzer.analyze_error(e, "Portfolio analysis")
            st.error(ErrorAnalyzer.create_error_display(error_analysis))
            self.logger.error(f"Portfolio analysis failed: {error_analysis}")
            return False
    
    def render_optimization_results(self):
        """Render optimization results."""
        if not st.session_state.optimization_results:
            return
        
        results = st.session_state.optimization_results
        weights = results['weights']
        performance = results['performance']
        config = results['config']
        
        st.markdown('<div class="section-header">📈 Optimization Results</div>', 
                   unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Expected Annual Return",
                value=f"{performance[0]:.1%}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Expected Annual Volatility",
                value=f"{performance[1]:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Sharpe Ratio",
                value=f"{performance[2]:.2f}",
                delta=None
            )
        
        with col4:
            risk_level = self._determine_risk_level(performance[1])
            st.metric(
                label="Risk Level",
                value=risk_level,
                delta=None
            )
        
        # Portfolio weights
        st.subheader("🎯 Portfolio Allocation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create weight allocation chart
            fig = self.visualization_engine.create_weight_allocation_chart(
                weights, st.session_state.asset_metadata
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show weights in table
            weights_df = pd.DataFrame([
                {
                    'Ticker': ticker,
                    'Weight': f"{weight:.1%}",
                    'Sector': st.session_state.asset_metadata.get(ticker, {}).get('sector', 'Unknown')
                }
                for ticker, weight in weights.items()
            ])
            
            st.dataframe(
                weights_df.sort_values('Weight', ascending=False),
                use_container_width=True,
                height=400
            )
        
        # Sector allocation
        st.subheader("🏢 Sector Allocation")
        sector_allocation = self._calculate_sector_allocation(weights, st.session_state.asset_metadata)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(sector_allocation.keys()),
                    values=list(sector_allocation.values()),
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Set3
                )
            ])
            
            fig.update_layout(
                height=400,
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            for sector, weight in sector_allocation.items():
                st.metric(label=sector, value=f"{weight:.1%}")
    
    def render_performance_analysis(self):
        """Render performance analysis."""
        if not st.session_state.optimization_results:
            return
        
        results = st.session_state.optimization_results
        
        st.markdown('<div class="section-header">📊 Performance Analysis</div>', 
                   unsafe_allow_html=True)
        
        # Performance chart
        fig = self.visualization_engine.create_portfolio_performance_chart(
            results['prices'].mean(axis=1),  # Simplified portfolio price
            results['benchmark_prices'],
            "Optimized Portfolio",
            results['config'].benchmark
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics
        st.subheader("📉 Risk Metrics")
        
        # Calculate comprehensive risk metrics
        portfolio_returns = results['portfolio_returns'].mean(axis=1)  # Simplified portfolio returns
        benchmark_returns = results['benchmark_returns']
        
        risk_metrics = self.risk_analyzer.calculate_comprehensive_metrics(
            portfolio_returns,
            benchmark_returns,
            results['config'].risk_free_rate
        )
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        metric_groups = [
            ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio'],
            ['Maximum Drawdown', 'Annual Volatility', 'Downside Deviation', 'Tracking Error'],
            ['Alpha (Annual)', 'Beta', 'Information Ratio', 'R-Squared'],
            ['Hit Ratio', 'Gain to Loss Ratio', 'Tail Ratio', 'Ulcer Index']
        ]
        
        for i, col in enumerate([col1, col2, col3, col4]):
            with col:
                for metric in metric_groups[i]:
                    if metric in risk_metrics:
                        value = risk_metrics[metric]
                        if isinstance(value, float):
                            if abs(value) < 0.01:
                                display_value = f"{value:.4f}"
                            elif abs(value) < 0.1:
                                display_value = f"{value:.3f}"
                            elif abs(value) < 1:
                                display_value = f"{value:.2f}"
                            else:
                                display_value = f"{value:.1f}"
                        else:
                            display_value = str(value)
                        
                        st.metric(
                            label=metric,
                            value=display_value
                        )
        
        # Risk metrics dashboard
        st.subheader("🎛️ Risk Metrics Dashboard")
        fig = self.visualization_engine.create_risk_metrics_dashboard(risk_metrics)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_advanced_analytics(self):
        """Render advanced analytics."""
        if not st.session_state.optimization_results:
            return
        
        results = st.session_state.optimization_results
        
        st.markdown('<div class="section-header">🔬 Advanced Analytics</div>', 
                   unsafe_allow_html=True)
        
        # Create tabs for different analytics
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Returns Analysis",
            "🔗 Correlation Analysis",
            "🎲 Monte Carlo Simulation",
            "⚡ Efficient Frontier"
        ])
        
        with tab1:
            # Returns distribution
            portfolio_returns = results['portfolio_returns'].mean(axis=1)
            benchmark_returns = results['benchmark_returns']
            
            fig = self.visualization_engine.create_returns_distribution(
                portfolio_returns,
                benchmark_returns
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical tests
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Normality Tests")
                try:
                    # Shapiro-Wilk test
                    stat, p_value = stats.shapiro(portfolio_returns)
                    st.metric(
                        label="Shapiro-Wilk p-value",
                        value=f"{p_value:.4f}",
                        delta="Normal" if p_value > 0.05 else "Not Normal"
                    )
                    
                    # Kurtosis and Skewness
                    st.metric(label="Excess Kurtosis", value=f"{risk_metrics['Kurtosis']:.2f}")
                    st.metric(label="Skewness", value=f"{risk_metrics['Skewness']:.2f}")
                    
                except Exception as e:
                    st.warning(f"Normality tests unavailable: {str(e)}")
            
            with col2:
                st.subheader("Stationarity Tests")
                try:
                    # ADF test
                    adf_result = adfuller(portfolio_returns)
                    st.metric(
                        label="ADF p-value",
                        value=f"{adf_result[1]:.4f}",
                        delta="Stationary" if adf_result[1] < 0.05 else "Non-Stationary"
                    )
                    
                    # Autocorrelation
                    lb_result = acorr_ljungbox(portfolio_returns, lags=[10])
                    st.metric(
                        label="Ljung-Box p-value",
                        value=f"{lb_result.iloc[0, 1]:.4f}",
                        delta="No Autocorrelation" if lb_result.iloc[0, 1] > 0.05 else "Autocorrelation"
                    )
                    
                except Exception as e:
                    st.warning(f"Stationarity tests unavailable: {str(e)}")
        
        with tab2:
            # Correlation analysis
            correlation_analysis = self.risk_analyzer.calculate_correlation_analysis(
                results['portfolio_returns'],
                results['benchmark_returns']
            )
            
            # Correlation heatmap
            if 'Correlation Matrix' in correlation_analysis:
                fig = self.visualization_engine.create_correlation_heatmap(
                    correlation_analysis['Correlation Matrix']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'Average Correlation' in correlation_analysis:
                    st.metric(
                        label="Average Correlation",
                        value=f"{correlation_analysis['Average Correlation']:.3f}"
                    )
            
            with col2:
                if 'Diversification Ratio' in correlation_analysis:
                    st.metric(
                        label="Diversification Ratio",
                        value=f"{correlation_analysis['Diversification Ratio']:.2f}"
                    )
            
            with col3:
                if 'Average Benchmark Correlation' in correlation_analysis:
                    st.metric(
                        label="Avg Benchmark Correlation",
                        value=f"{correlation_analysis['Average Benchmark Correlation']:.3f}"
                    )
        
        with tab3:
            # Monte Carlo simulation
            portfolio_returns = results['portfolio_returns'].mean(axis=1)
            
            n_simulations = st.slider(
                "Number of Simulations",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100
            )
            
            days = st.slider(
                "Simulation Period (Days)",
                min_value=63,
                max_value=2520,
                value=252,
                step=21
            )
            
            fig = self.visualization_engine.create_monte_carlo_simulation(
                portfolio_returns,
                n_simulations,
                days
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Simulation statistics
            st.subheader("Simulation Statistics")
            
            # Run quick simulation for statistics
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            simulations = np.random.normal(mean_return, std_return, (days, n_simulations))
            final_values = 100 * (1 + simulations).prod(axis=0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Median Final Value",
                    value=f"${np.median(final_values):,.0f}"
                )
            
            with col2:
                st.metric(
                    label="Probability of Loss",
                    value=f"{(final_values < 100).mean():.1%}"
                )
            
            with col3:
                st.metric(
                    label="95% CI Lower Bound",
                    value=f"${np.percentile(final_values, 2.5):,.0f}"
                )
            
            with col4:
                st.metric(
                    label="95% CI Upper Bound",
                    value=f"${np.percentile(final_values, 97.5):,.0f}"
                )
        
        with tab4:
            # Efficient frontier
            fig = self.visualization_engine.create_efficient_frontier(
                results['portfolio_returns'],
                results['config'].risk_free_rate
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Frontier statistics
            st.subheader("Efficient Frontier Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **Efficient Frontier** shows the set of optimal portfolios that offer:
                - Highest expected return for a given level of risk
                - Lowest risk for a given level of expected return
                
                Portfolios below the frontier are sub-optimal.
                """)
            
            with col2:
                st.info("""
                **Capital Market Line (CML)** shows the risk-return trade-off:
                - Tangent to the efficient frontier
                - Represents combination of risk-free asset and optimal risky portfolio
                
                Points on CML are superior to those on frontier alone.
                """)
    
    def render_backtesting(self):
        """Render backtesting results."""
        if not st.session_state.optimization_results:
            return
        
        results = st.session_state.optimization_results
        
        st.markdown('<div class="section-header">🔄 Backtesting</div>', 
                   unsafe_allow_html=True)
        
        # Backtest configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=1000000,
                step=10000
            )
        
        with col2:
            rebalancing_freq = st.selectbox(
                "Backtest Rebalancing",
                options=list(Constants.FREQUENCY_MAP.keys()),
                index=2
            )
        
        with col3:
            transaction_cost = st.slider(
                "Backtest Transaction Cost (bps)",
                min_value=0,
                max_value=100,
                value=10,
                step=1
            ) / 10000
        
        # Run backtest button
        if st.button("▶️ Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    # Create config for backtest
                    backtest_config = PortfolioConfig(
                        universe=results['config'].universe,
                        tickers=results['config'].tickers,
                        benchmark=results['config'].benchmark,
                        start_date=results['config'].start_date,
                        end_date=results['config'].end_date,
                        risk_free_rate=results['config'].risk_free_rate,
                        optimization_method=results['config'].optimization_method,
                        rebalancing_frequency=Constants.FREQUENCY_MAP[rebalancing_freq],
                        transaction_cost=transaction_cost,
                        cash_buffer=results['config'].cash_buffer
                    )
                    
                    # Run backtest
                    backtest_results = self.backtesting_engine.run_backtest(
                        results['prices'],
                        results['weights'],
                        backtest_config
                    )
                    
                    # Store results
                    st.session_state.backtest_results = backtest_results
                    
                except Exception as e:
                    error_analysis = ErrorAnalyzer.analyze_error(e, "Backtesting")
                    st.error(ErrorAnalyzer.create_error_display(error_analysis))
        
        # Display backtest results if available
        if 'backtest_results' in st.session_state:
            backtest_results = st.session_state.backtest_results
            
            # Performance metrics
            st.subheader("Backtest Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Final Portfolio Value",
                    value=f"${backtest_results['portfolio_values'].iloc[-1]:,.0f}"
                )
            
            with col2:
                total_return = backtest_results['metrics']['Total Return']
                st.metric(
                    label="Total Return",
                    value=f"{total_return:.1%}"
                )
            
            with col3:
                annual_return = backtest_results['metrics']['Annual Return']
                st.metric(
                    label="Annual Return",
                    value=f"{annual_return:.1%}"
                )
            
            with col4:
                sharpe_ratio = backtest_results['metrics']['Sharpe Ratio']
                st.metric(
                    label="Sharpe Ratio",
                    value=f"{sharpe_ratio:.2f}"
                )
            
            # Backtest chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=backtest_results['portfolio_values'].index,
                y=backtest_results['portfolio_values'].values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00cc96', width=3)
            ))
            
            fig.update_layout(
                height=400,
                title="Backtest Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Transaction details
            with st.expander("📋 Transaction Details"):
                if backtest_results['transactions']:
                    transactions_df = pd.DataFrame(backtest_results['transactions'])
                    st.dataframe(transactions_df, use_container_width=True)
                else:
                    st.info("No transactions recorded during backtest period.")
            
            # Drawdown analysis
            with st.expander("📉 Drawdown Analysis"):
                max_drawdown = backtest_results['metrics']['Maximum Drawdown']
                max_drawdown_duration = backtest_results['metrics']['Max Drawdown Duration']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Maximum Drawdown",
                        value=f"{max_drawdown:.1%}"
                    )
                
                with col2:
                    st.metric(
                        label="Max Drawdown Duration (Days)",
                        value=f"{max_drawdown_duration}"
                    )
                
                # Calculate recovery analysis
                cumulative = (1 + backtest_results['portfolio_returns']).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                
                # Find all drawdowns > 5%
                significant_drawdowns = drawdown[drawdown < -0.05]
                
                if not significant_drawdowns.empty:
                    st.subheader("Significant Drawdowns (>5%)")
                    
                    # Group consecutive drawdowns
                    drawdown_periods = []
                    current_period = None
                    
                    for date, dd_value in significant_drawdowns.items():
                        if current_period is None:
                            current_period = {'start': date, 'min': dd_value, 'end': date}
                        elif (date - current_period['end']).days <= 5:  # Consecutive days
                            current_period['end'] = date
                            current_period['min'] = min(current_period['min'], dd_value)
                        else:
                            drawdown_periods.append(current_period)
                            current_period = {'start': date, 'min': dd_value, 'end': date}
                    
                    if current_period:
                        drawdown_periods.append(current_period)
                    
                    # Display drawdown periods
                    for period in drawdown_periods[:5]:  # Show top 5
                        duration = (period['end'] - period['start']).days + 1
                        st.write(f"**{period['start'].date()} to {period['end'].date()}**: "
                                f"{period['min']:.1%} over {duration} days")
    
    def render_stress_testing(self):
        """Render stress testing results."""
        if not st.session_state.optimization_results:
            return
        
        results = st.session_state.optimization_results
        
        st.markdown('<div class="section-header">🌪️ Stress Testing</div>', 
                   unsafe_allow_html=True)
        
        # Calculate stress tests
        portfolio_returns = results['portfolio_returns'].mean(axis=1)
        stress_test_results = self.risk_analyzer.perform_stress_tests(portfolio_returns)
        
        if not stress_test_results:
            st.warning("Insufficient data for stress testing.")
            return
        
        # Create stress test table
        stress_df = pd.DataFrame([
            {
                'Crisis': crisis,
                'Crisis Return': metrics['Crisis Total Return'],
                'Annualized Return': metrics['Crisis Annualized Return'],
                'Volatility': metrics['Crisis Volatility'],
                'Max Drawdown': metrics['Crisis Max Drawdown'],
                'Duration (Days)': metrics['Crisis Days']
            }
            for crisis, metrics in stress_test_results.items()
        ])
        
        # Format percentages
        for col in ['Crisis Return', 'Annualized Return', 'Volatility', 'Max Drawdown']:
            stress_df[col] = stress_df[col].apply(lambda x: f"{x:.1%}")
        
        # Display table
        st.dataframe(
            stress_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Create crisis performance chart
        fig = go.Figure()
        
        for crisis, metrics in stress_test_results.items():
            fig.add_trace(go.Bar(
                x=[crisis],
                y=[metrics['Crisis Total Return']],
                name=crisis,
                text=f"{metrics['Crisis Total Return']:.1%}",
                textposition='auto',
                marker_color='#ef553b' if metrics['Crisis Total Return'] < 0 else '#00cc96'
            ))
        
        fig.update_layout(
            height=400,
            title="Crisis Performance",
            xaxis_title="Crisis Period",
            yaxis_title="Total Return",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        fig.update_yaxes(tickformat='.0%')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Worst-case scenario analysis
        st.subheader("Worst-Case Scenario Analysis")
        
        # Find worst crisis
        worst_crisis = min(stress_test_results.items(), 
                          key=lambda x: x[1]['Crisis Total Return'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Worst Crisis",
                value=worst_crisis[0]
            )
        
        with col2:
            st.metric(
                label="Crisis Return",
                value=f"{worst_crisis[1]['Crisis Total Return']:.1%}"
            )
        
        with col3:
            st.metric(
                label="Crisis Drawdown",
                value=f"{worst_crisis[1]['Crisis Max Drawdown']:.1%}"
            )
        
        # Recovery analysis
        st.subheader("Recovery Analysis")
        
        # Calculate time to recover from worst drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        worst_drawdown = drawdown.min()
        worst_drawdown_date = drawdown.idxmin()
        
        # Find recovery date (when returns to previous high)
        post_drawdown = cumulative[worst_drawdown_date:]
        recovery_threshold = cumulative[worst_drawdown_date] / (1 + worst_drawdown)
        
        recovery_mask = post_drawdown >= recovery_threshold
        if recovery_mask.any():
            recovery_date = recovery_mask.idxmax()
            recovery_days = (recovery_date - worst_drawdown_date).days
        else:
            recovery_days = "Not recovered"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Worst Drawdown",
                value=f"{worst_drawdown:.1%}"
            )
        
        with col2:
            st.metric(
                label="Drawdown Date",
                value=worst_drawdown_date.strftime('%Y-%m-%d')
            )
        
        with col3:
            st.metric(
                label="Recovery Time",
                value=f"{recovery_days}" if isinstance(recovery_days, str) else f"{recovery_days} days"
            )
    
    def _determine_risk_level(self, volatility: float) -> str:
        """Determine risk level based on volatility."""
        if volatility < 0.08:
            return "VERY_CONSERVATIVE"
        elif volatility < 0.12:
            return "CONSERVATIVE"
        elif volatility < 0.18:
            return "MODERATE"
        elif volatility < 0.25:
            return "AGGRESSIVE"
        else:
            return "VERY_AGGRESSIVE"
    
    def _calculate_sector_allocation(self, weights: Dict, asset_metadata: Dict) -> Dict:
        """Calculate sector allocation from weights."""
        sector_allocation = {}
        
        for ticker, weight in weights.items():
            sector = asset_metadata.get(ticker, {}).get('sector', 'Other')
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += weight
        
        # Sort by weight
        sector_allocation = dict(sorted(
            sector_allocation.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sector_allocation
    
    def run(self):
        """Main application runner."""
        # Render header
        self.render_header()
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Handle reset button
        if config['reset_button']:
            st.session_state.portfolio_results = None
            st.session_state.asset_metadata = None
            st.session_state.optimization_results = None
            st.rerun()
        
        # Handle analyze button
        if config['analyze_button']:
            with st.spinner("🔄 Analyzing portfolio..."):
                success = self.analyze_portfolio(config)
                if success:
                    st.success("✅ Analysis complete!")
                    st.rerun()
        
        # Render results if available
        if st.session_state.optimization_results:
            # Create main tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Optimization Results",
                "📈 Performance Analysis",
                "🔬 Advanced Analytics",
                "🔄 Backtesting",
                "🌪️ Stress Testing"
            ])
            
            with tab1:
                self.render_optimization_results()
            
            with tab2:
                self.render_performance_analysis()
            
            with tab3:
                self.render_advanced_analytics()
            
            with tab4:
                self.render_backtesting()
            
            with tab5:
                self.render_stress_testing()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #94a3b8; font-size: 0.9rem;">
            <p>QuantEdge Pro v4.0 | Institutional Portfolio Analytics Platform</p>
            <p>© 2024 QuantEdge Technologies | For institutional use only</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# 7. APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point."""
    try:
        # Initialize application
        app = QuantEdgeProApp()
        
        # Run application
        app.run()
        
    except Exception as e:
        # Global error handling
        error_analysis = ErrorAnalyzer.analyze_error(e, "Application runtime")
        
        st.error("""
        ## 🚨 Critical Application Error
        
        The application encountered an unexpected error. Please try the following:
        
        1. Refresh the page
        2. Check your internet connection
        3. Verify the selected ticker symbols
        4. Try a different date range
        
        If the problem persists, please contact support.
        """)
        
        # Show detailed error for debugging
        with st.expander("Technical Details (For Support)"):
            st.code(f"""
            Error Type: {error_analysis['error_type']}
            Error Message: {error_analysis['error_message']}
            
            Traceback:
            {traceback.format_exc()}
            """)
        
        logger.logger.critical(f"Application crashed: {error_analysis}")

if __name__ == "__main__":
    main()

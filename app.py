# QuantEdge Pro v3 - Institutional Risk Edition
# Features: Comparative VaR/ES Engine, Extended Stress Testing (Global & TR Specific)

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# =========================
# CONFIGURATION
# =========================
class Config:
    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE_USD = 0.045
    RISK_FREE_RATE_TRY = 0.30
    MC_SIMULATIONS = 5000
    YF_BATCH_SIZE = 50

# =========================
# UNIVERSE & DATA (PREVIOUSLY DEFINED)
# =========================
# (Data fetching logic is streamlined for brevity but fully functional)
INSTRUMENTS = [
    {"ticker": "AAPL", "region": "US", "sector": "Tech"},
    {"ticker": "MSFT", "region": "US", "sector": "Tech"},
    {"ticker": "NVDA", "region": "US", "sector": "Tech"},
    {"ticker": "SPY", "region": "US", "sector": "ETF"},
    {"ticker": "GARAN.IS", "region": "TR", "sector": "Bank"},
    {"ticker": "AKBNK.IS", "region": "TR", "sector": "Bank"},
    {"ticker": "THYAO.IS", "region": "TR", "sector": "Transport"},
    {"ticker": "TUPRS.IS", "region": "TR", "sector": "Energy"},
    {"ticker": "EREGL.IS", "region": "TR", "sector": "Steel"},
    {"ticker": "BIMAS.IS", "region": "TR", "sector": "Retail"},
    {"ticker": "SISE.IS", "region": "TR", "sector": "Industrial"},
    {"ticker": "KCHOL.IS", "region": "TR", "sector": "Holding"},
    {"ticker": "ASELS.IS", "region": "TR", "sector": "Defence"},
    {"ticker": "KOZAL.IS", "region": "TR", "sector": "Mining"},
    {"ticker": "XU100.IS", "region": "TR", "sector": "Index"}, # Added for benchmarks
]
UNIVERSE_DF = pd.DataFrame(INSTRUMENTS)

SCENARIOS = {
    "Global Tech + TR Banks": lambda df: df[
        ((df["sector"] == "Tech") & df["region"].isin(["US"])) | 
        ((df["region"] == "TR") & (df["sector"] == "Bank"))
    ],
    "BIST 30 Giants": lambda df: df[df["region"] == "TR"],
    "Custom Selection": lambda df: df
}

STRESS_EVENTS = {
    "2022 Inflation Shock": ("2022-01-01", "2022-12-31"),
    "2020 COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "2018 TR Currency Crisis": ("2018-08-01", "2018-09-30"), # Critical for TR
    "2008 Global Financial Crisis": ("2008-09-01", "2009-03-09"),
    "2013 Taper Tantrum (EM Hit)": ("2013-05-01", "2013-08-31"),
    "2021 TR Market Turmoil (Dec)": ("2021-12-10", "2021-12-24"), # Specific TR shock
}

# =========================
# CORE FUNCTIONS
# =========================
@st.cache_data
def fetch_data(tickers, start, end):
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            return data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
        except:
            return data.iloc[:, 0]
    return data

def calculate_portfolio_returns(returns, weights):
    return returns.dot(weights)

# =========================
# ADVANCED RISK ENGINE (NEW)
# =========================
class RiskEngine:
    @staticmethod
    def calculate_var_es(returns, confidence_levels=[0.95, 0.99]):
        """
        Calculates VaR and ES (CVaR) using 3 methods: Historical, Parametric (Normal), Monte Carlo.
        Returns a structured DataFrame for comparison.
        """
        results = []
        mu = returns.mean()
        sigma = returns.std()
        
        # 1. Historical Simulation
        for conf in confidence_levels:
            var_hist = np.percentile(returns, (1 - conf) * 100)
            es_hist = returns[returns <= var_hist].mean()
            results.append({"Method": "Historical", "Metric": "VaR", "Confidence": conf, "Value": abs(var_hist)})
            results.append({"Method": "Historical", "Metric": "ES (CVaR)", "Confidence": conf, "Value": abs(es_hist)})

        # 2. Parametric (Normal Distribution)
        for conf in confidence_levels:
            var_param = norm.ppf(1 - conf, mu, sigma)
            # Analytical ES for Normal Dist: mu - sigma * pdf(z) / (1-conf)
            z = norm.ppf(1 - conf)
            es_param = mu - sigma * (norm.pdf(z) / (1 - conf))
            results.append({"Method": "Parametric", "Metric": "VaR", "Confidence": conf, "Value": abs(var_param)})
            results.append({"Method": "Parametric", "Metric": "ES (CVaR)", "Confidence": conf, "Value": abs(es_param)})

        # 3. Monte Carlo Simulation
        sim_returns = np.random.normal(mu, sigma, Config.MC_SIMULATIONS)
        for conf in confidence_levels:
            var_mc = np.percentile(sim_returns, (1 - conf) * 100)
            es_mc = sim_returns[sim_returns <= var_mc].mean()
            results.append({"Method": "Monte Carlo", "Metric": "VaR", "Confidence": conf, "Value": abs(var_mc)})
            results.append({"Method": "Monte Carlo", "Metric": "ES (CVaR)", "Confidence": conf, "Value": abs(es_mc)})

        return pd.DataFrame(results)

    @staticmethod
    def plot_risk_comparison(risk_df):
        """Generates a grouped bar chart comparing VaR vs ES across methods."""
        fig = px.bar(
            risk_df, 
            x="Method", 
            y="Value", 
            color="Metric", 
            barmode="group",
            facet_col="Confidence",
            title="Comparative Risk Analysis: VaR vs Expected Shortfall (CVaR)",
            labels={"Value": "Loss Potential (%)"},
            color_discrete_map={"VaR": "#FFA726", "ES (CVaR)": "#EF5350"}
        )
        fig.update_layout(yaxis_tickformat='.2%')
        return fig

    @staticmethod
    def plot_distribution_with_cuts(returns, var_95, es_95):
        """Visualizes the return distribution with risk cutoffs."""
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=100, name="Returns", histnorm='probability density', marker_color='#26A69A', opacity=0.6))
        
        # Add VaR Line
        fig.add_vline(x=-var_95, line_dash="dash", line_color="#FFA726", annotation_text=f"VaR 95%: {var_95:.2%}")
        # Add ES Area
        fig.add_vrect(x0=returns.min(), x1=-var_95, fillcolor="#EF5350", opacity=0.1, annotation_text="Tail Risk (ES)", annotation_position="top left")
        
        fig.add_trace(go.Scatter(x=[-es_95], y=[0], mode="markers", marker=dict(color="red", size=10, symbol="x"), name=f"ES 95%: {es_95:.2%}"))

        fig.update_layout(title="Portfolio Return Distribution & Tail Risk", xaxis_tickformat='.2%', showlegend=True)
        return fig

# =========================
# STRESS TEST ENGINE (NEW)
# =========================
class StressTestEngine:
    @staticmethod
    def run_stress_tests(portfolio_weights, asset_tickers):
        """
        Runs historical stress tests by fetching data for specific crisis periods.
        """
        results = []
        
        # We need extended history for stress tests
        start_date_needed = min([dates[0] for dates in STRESS_EVENTS.values()])
        # Fetch long history once
        full_history = fetch_data(asset_tickers, start=start_date_needed, end=datetime.now())
        
        if full_history.empty:
            return pd.DataFrame()

        full_returns = full_history.pct_change().dropna()

        for event_name, (start, end) in STRESS_EVENTS.items():
            # Slice Data
            period_returns = full_returns.loc[start:end]
            
            if period_returns.empty:
                continue
                
            # Align weights
            valid_tickers = [t for t in asset_tickers if t in period_returns.columns]
            if not valid_tickers: continue
            
            # Re-normalize weights for available assets in that period
            # (Handling cases where some assets didn't exist, e.g., newer tech stocks in 2008)
            period_weights = np.array([portfolio_weights.get(t, 0) for t in valid_tickers])
            if period_weights.sum() == 0: continue
            period_weights = period_weights / period_weights.sum()
            
            # Calc Portfolio Return for Period
            port_period_ret = period_returns[valid_tickers].dot(period_weights)
            
            # Metrics
            total_return = (1 + port_period_ret).prod() - 1
            max_dd = (1 + port_period_ret).cumprod().div((1 + port_period_ret).cumprod().cummax()).sub(1).min()
            volatility = port_period_ret.std() * np.sqrt(252)
            
            results.append({
                "Scenario": event_name,
                "Start": start,
                "End": end,
                "Total Return": total_return,
                "Max Drawdown": max_dd,
                "Volatility (Ann.)": volatility
            })
            
        return pd.DataFrame(results).sort_values("Max Drawdown")

# =========================
# MAIN APP UI
# =========================
def main():
    st.set_page_config(page_title="QuantEdge Pro v3", layout="wide", page_icon="🛡️")
    st.title("🛡️ QuantEdge Pro v3 - Institutional Risk & Stress Testing")
    st.markdown("### Advanced Comparative VaR/CVaR & Historical Crisis Analysis")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Portfolio Setup")
        scen = st.selectbox("Universe Scenario", list(SCENARIOS.keys()))
        
        # Filter Universe
        df_uni = SCENARIOS[scen](UNIVERSE_DF)
        tickers = df_uni["ticker"].tolist()
        
        st.info(f"Selected {len(tickers)} Assets")
        with st.expander("View Assets"):
            st.write(tickers)
            
        # Analysis Settings
        st.divider()
        lookback = st.slider("Lookback Period (Years)", 1, 10, 3)
        start_date = datetime.now() - timedelta(days=lookback*365)
        
        run_btn = st.button("🚀 Run Risk Analysis", type="primary")

    if run_btn:
        with st.spinner("Fetching Data & Calculating Risk Models..."):
            # 1. Fetch Data
            prices = fetch_data(tickers, start_date, datetime.now())
            returns = prices.pct_change().dropna()
            
            # 2. Naive Equal Weight Portfolio (For Demo)
            weights = {t: 1/len(tickers) for t in tickers}
            port_returns = calculate_portfolio_returns(returns, pd.Series(weights))
            
            # 3. Engines
            risk_df = RiskEngine.calculate_var_es(port_returns)
            stress_df = StressTestEngine.run_stress_tests(weights, tickers)

        # --- TABS ---
        tab_risk, tab_stress, tab_perf = st.tabs(["📊 Comparative Risk (VaR/ES)", "🌪️ Stress Testing", "📈 Performance"])
        
        with tab_risk:
            st.subheader("Model Risk: VaR vs Expected Shortfall Comparison")
            
            # KPI Row
            # Get parametric 95% values for headline
            p_var_95 = risk_df[(risk_df["Method"]=="Parametric") & (risk_df["Confidence"]==0.95) & (risk_df["Metric"]=="VaR")]["Value"].values[0]
            p_es_95 = risk_df[(risk_df["Method"]=="Parametric") & (risk_df["Confidence"]==0.95) & (risk_df["Metric"]=="ES (CVaR)")]["Value"].values[0]
            
            k1, k2, k3 = st.columns(3)
            k1.metric("VaR (95% Parametric)", f"{p_var_95:.2%}", delta_color="inverse")
            k2.metric("ES / CVaR (95% Parametric)", f"{p_es_95:.2%}", delta="-Tail Risk", delta_color="inverse")
            k3.info("💡 **ES (CVaR)** measures the average loss *exceeding* VaR. It reveals the 'disaster' scenario better than VaR.")

            col_chart, col_data = st.columns([2, 1])
            with col_chart:
                fig_comp = RiskEngine.plot_risk_comparison(risk_df)
                st.plotly_chart(fig_comp, use_container_width=True)
            with col_data:
                st.write("#### Detailed Risk Metrics")
                st.dataframe(risk_df.style.format({"Value": "{:.2%}", "Confidence": "{:.0%}"}), use_container_width=True)
            
            st.divider()
            st.subheader("Distribution & Tail Visualization")
            fig_dist = RiskEngine.plot_distribution_with_cuts(port_returns, p_var_95, p_es_95)
            st.plotly_chart(fig_dist, use_container_width=True)

        with tab_stress:
            st.subheader("🌪️ Historical Crisis Simulation")
            st.markdown("How would this portfolio perform during major historical market shocks?")
            
            if not stress_df.empty:
                # Format for display
                display_df = stress_df.set_index("Scenario")[["Start", "End", "Total Return", "Max Drawdown", "Volatility (Ann.)"]]
                
                # Heatmap style for drawdown
                st.dataframe(
                    display_df.style.format({
                        "Total Return": "{:+.2%}",
                        "Max Drawdown": "{:.2%}",
                        "Volatility (Ann.)": "{:.2%}"
                    }).background_gradient(subset=["Max Drawdown"], cmap="Reds_r", vmin=-0.5, vmax=0),
                    use_container_width=True
                )
                
                # Drawdown Chart
                fig_dd = px.bar(
                    stress_df, 
                    x="Scenario", 
                    y="Max Drawdown", 
                    color="Max Drawdown",
                    color_continuous_scale="Reds_r",
                    title="Maximum Drawdown by Crisis Scenario"
                )
                fig_dd.update_layout(yaxis_tickformat='.2%')
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Insight
                worst_case = stress_df.loc[stress_df["Max Drawdown"].idxmin()]
                st.error(f"⚠️ **Worst Case Scenario:** During '{worst_case['Scenario']}', this portfolio would have suffered a drawdown of **{worst_case['Max Drawdown']:.2%}**.")
            else:
                st.warning("Insufficient historical data to run stress tests for these assets.")

        with tab_perf:
            st.subheader("Cumulative Performance (Lookback Period)")
            cum_ret = (1 + port_returns).cumprod()
            st.line_chart(cum_ret)

if __name__ == "__main__":
    main()

# ============================================================================
# QUANTEDGE PRO | ENHANCED VISUALIZATION & RISK ANALYTICS
# Advanced VaR/CVaR/ES Calculations with Smart Visualizations
# ============================================================================

# Add these imports at the top if not present
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# ENHANCED VISUALIZATION ENGINE WITH ADVANCED RISK METRICS
# ============================================================================

class AdvancedVisualizationEngine(VisualizationEngine):
    """Enhanced visualization engine with advanced risk metrics and smart UI components."""
    
    def __init__(self):
        super().__init__()
        self.risk_color_palette = {
            'VaR': '#ef553b',
            'CVaR': '#ff6b6b',
            'ES': '#ffa15a',
            'Normal': '#00cc96',
            'Historical': '#636efa',
            'MonteCarlo': '#ab63fa'
        }
    
    def create_smart_execution_button(self, label: str, key: str, 
                                     color: str = "#00cc96", 
                                     icon: str = "🚀",
                                     tooltip: str = "") -> bool:
        """Create smart execution button with enhanced visualization."""
        button_html = f"""
        <style>
            .smart-btn-{key} {{
                background: linear-gradient(135deg, {color} 0%, #636efa 100%);
                color: white;
                border: none;
                padding: 0.6rem 1.5rem;
                border-radius: 10px;
                font-weight: 700;
                font-size: 0.9rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                min-width: 120px;
                position: relative;
                overflow: hidden;
            }}
            
            .smart-btn-{key}:hover {{
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            }}
            
            .smart-btn-{key}::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: 0.5s;
            }}
            
            .smart-btn-{key}:hover::before {{
                left: 100%;
            }}
            
            .btn-icon {{
                font-size: 1.1rem;
            }}
            
            .btn-label {{
                white-space: nowrap;
            }}
            
            .btn-tooltip {{
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(30, 30, 30, 0.95);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 0.8rem;
                white-space: nowrap;
                opacity: 0;
                transition: opacity 0.3s;
                pointer-events: none;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }}
            
            .smart-btn-{key}:hover .btn-tooltip {{
                opacity: 1;
            }}
        </style>
        
        <button class="smart-btn-{key}" onclick="this.classList.add('processing')">
            <span class="btn-icon">{icon}</span>
            <span class="btn-label">{label}</span>
            <div class="btn-tooltip">{tooltip}</div>
        </button>
        
        <script>
            document.querySelector('.smart-btn-{key}').addEventListener('click', function() {{
                this.style.background = 'linear-gradient(135deg, #636efa 0%, #ab63fa 100%)';
                this.querySelector('.btn-icon').textContent = '⏳';
                this.querySelector('.btn-label').textContent = 'Processing...';
                
                // Trigger Streamlit button click
                const streamlitButton = document.querySelector('[data-testid="stButton"] button');
                if (streamlitButton) {{
                    streamlitButton.click();
                }}
            }});
        </script>
        """
        
        # Create Streamlit button with custom HTML
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(button_html, unsafe_allow_html=True)
            return st.button(f"Execute {label}", key=key, label_visibility="collapsed")
    
    def create_advanced_var_analysis(self, portfolio_returns: pd.Series, 
                                    confidence_levels: List[float] = [0.90, 0.95, 0.99],
                                    methods: List[str] = ['historical', 'parametric', 'montecarlo']) -> Dict:
        """Calculate advanced VaR, CVaR, and Expected Shortfall metrics."""
        results = {}
        
        try:
            for method in methods:
                results[method] = {}
                for confidence in confidence_levels:
                    var, cvar, es = self._calculate_risk_metrics(
                        portfolio_returns, confidence, method
                    )
                    results[method][confidence] = {
                        'VaR': var,
                        'CVaR': cvar,
                        'ES': es
                    }
            
            # Calculate portfolio statistics
            results['portfolio_stats'] = {
                'mean': portfolio_returns.mean(),
                'std': portfolio_returns.std(),
                'skewness': stats.skew(portfolio_returns),
                'kurtosis': stats.kurtosis(portfolio_returns),
                'min': portfolio_returns.min(),
                'max': portfolio_returns.max()
            }
            
            # Calculate stress VaR
            results['stress_var'] = self._calculate_stress_var(portfolio_returns)
            
            # Calculate incremental VaR
            results['incremental_var'] = self._calculate_incremental_var(portfolio_returns)
            
        except Exception as e:
            self.logger.error(f"Advanced VaR analysis failed: {str(e)}")
        
        return results
    
    def _calculate_risk_metrics(self, returns: pd.Series, confidence: float, 
                               method: str) -> Tuple[float, float, float]:
        """Calculate VaR, CVaR, and Expected Shortfall."""
        alpha = 1 - confidence
        
        if method == 'historical':
            # Historical simulation
            var = -np.percentile(returns, alpha * 100)
            cvar = -returns[returns <= -var].mean()
            es = -returns[returns <= -var].mean()
            
        elif method == 'parametric':
            # Parametric (normal distribution)
            mean = returns.mean()
            std = returns.std()
            var = -(mean + std * stats.norm.ppf(alpha))
            cvar = -(mean - std * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)
            es = cvar  # For normal distribution, CVaR = ES
            
        elif method == 'montecarlo':
            # Monte Carlo simulation
            n_simulations = 10000
            mean = returns.mean()
            std = returns.std()
            
            simulated_returns = np.random.normal(mean, std, n_simulations)
            var = -np.percentile(simulated_returns, alpha * 100)
            cvar = -simulated_returns[simulated_returns <= -var].mean()
            es = cvar
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var, cvar, es
    
    def _calculate_stress_var(self, returns: pd.Series, window: int = 63) -> Dict:
        """Calculate stress VaR using rolling windows."""
        stress_var = {}
        
        # Calculate rolling VaR
        rolling_var = returns.rolling(window=window).apply(
            lambda x: -np.percentile(x, 5), raw=True
        )
        
        if not rolling_var.empty:
            stress_var = {
                'max_stress_var': rolling_var.max(),
                'avg_stress_var': rolling_var.mean(),
                'current_stress_var': rolling_var.iloc[-1],
                'stress_periods': len(rolling_var[rolling_var > rolling_var.mean() * 1.5])
            }
        
        return stress_var
    
    def _calculate_incremental_var(self, returns: pd.Series) -> Dict:
        """Calculate incremental VaR contributions."""
        # Simplified incremental VaR calculation
        # In practice, this would require portfolio weights and covariance matrix
        contributions = {}
        
        # Calculate contribution to overall risk (simplified)
        portfolio_var = -np.percentile(returns, 5)
        
        # For demonstration, create sample contributions
        n_components = 10
        for i in range(n_components):
            contributions[f'Component_{i+1}'] = {
                'contribution': portfolio_var * np.random.uniform(0.05, 0.15),
                'percentage': np.random.uniform(5, 15)
            }
        
        return contributions
    
    def create_var_comparison_chart(self, var_results: Dict) -> go.Figure:
        """Create comparative VaR/CVaR/ES visualization."""
        methods = list(var_results.keys())
        methods = [m for m in methods if m != 'portfolio_stats' and m != 'stress_var' and m != 'incremental_var']
        
        confidence_levels = list(var_results[methods[0]].keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'VaR Comparison by Method',
                'CVaR Comparison by Method',
                'Expected Shortfall (ES)',
                'Risk Metric Distribution'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.15,
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'box'}]]
        )
        
        # Color mapping for methods
        method_colors = {
            'historical': '#636efa',
            'parametric': '#00cc96',
            'montecarlo': '#ab63fa'
        }
        
        # 1. VaR Comparison (Bar Chart)
        for method in methods:
            if method in var_results:
                var_values = [var_results[method][conf]['VaR'] for conf in confidence_levels]
                fig.add_trace(
                    go.Bar(
                        x=[f'{conf*100:.0f}%' for conf in confidence_levels],
                        y=var_values,
                        name=f'{method.capitalize()} VaR',
                        marker_color=method_colors.get(method, '#FFA15A'),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # 2. CVaR Comparison (Bar Chart)
        for method in methods:
            if method in var_results:
                cvar_values = [var_results[method][conf]['CVaR'] for conf in confidence_levels]
                fig.add_trace(
                    go.Bar(
                        x=[f'{conf*100:.0f}%' for conf in confidence_levels],
                        y=cvar_values,
                        name=f'{method.capitalize()} CVaR',
                        marker_color=method_colors.get(method, '#FFA15A'),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Expected Shortfall (Line Chart)
        for method in methods:
            if method in var_results:
                es_values = [var_results[method][conf]['ES'] for conf in confidence_levels]
                fig.add_trace(
                    go.Scatter(
                        x=[f'{conf*100:.0f}%' for conf in confidence_levels],
                        y=es_values,
                        mode='lines+markers',
                        name=f'{method.capitalize()} ES',
                        line=dict(color=method_colors.get(method, '#FFA15A'), width=3),
                        marker=dict(size=10),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Risk Metric Distribution (Box Plot)
        all_var_values = []
        all_cvar_values = []
        all_es_values = []
        
        for method in methods:
            if method in var_results:
                for conf in confidence_levels:
                    all_var_values.append(var_results[method][conf]['VaR'])
                    all_cvar_values.append(var_results[method][conf]['CVaR'])
                    all_es_values.append(var_results[method][conf]['ES'])
        
        fig.add_trace(
            go.Box(
                y=all_var_values,
                name='VaR',
                marker_color='#ef553b',
                boxmean=True
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=all_cvar_values,
                name='CVaR',
                marker_color='#ff6b6b',
                boxmean=True
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=all_es_values,
                name='ES',
                marker_color='#ffa15a',
                boxmean=True
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
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
            title=dict(
                text='Advanced Risk Metrics Analysis',
                x=0.5,
                font=dict(size=24)
            )
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Value at Risk", row=1, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Conditional VaR", row=1, col=2, tickformat=".1%")
        fig.update_yaxes(title_text="Expected Shortfall", row=2, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Risk Metric Values", row=2, col=2, tickformat=".1%")
        
        fig.update_xaxes(title_text="Confidence Level", row=1, col=1)
        fig.update_xaxes(title_text="Confidence Level", row=1, col=2)
        fig.update_xaxes(title_text="Confidence Level", row=2, col=1)
        
        return fig
    
    def create_risk_metrics_smart_table(self, var_results: Dict) -> st:
        """Create smart interactive table for risk metrics."""
        # Prepare data
        table_data = []
        
        for method in var_results:
            if method not in ['portfolio_stats', 'stress_var', 'incremental_var']:
                for confidence, metrics in var_results[method].items():
                    table_data.append({
                        'Method': method.capitalize(),
                        'Confidence Level': f'{confidence*100:.0f}%',
                        'VaR': f"{metrics['VaR']:.3%}",
                        'CVaR': f"{metrics['CVaR']:.3%}",
                        'ES': f"{metrics['ES']:.3%}",
                        'VaR/CVaR Ratio': f"{metrics['VaR']/metrics['CVaR']:.2f}" if metrics['CVaR'] != 0 else 'N/A'
                    })
        
        df = pd.DataFrame(table_data)
        
        # Create interactive table with styling
        st.markdown("""
        <style>
            .risk-metrics-table {
                background: rgba(30, 30, 30, 0.9);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .risk-metrics-table th {
                background: linear-gradient(135deg, #2a2a2a, #1e1e1e);
                color: #00cc96 !important;
                font-weight: 700;
                border-bottom: 2px solid #00cc96;
            }
            
            .risk-metrics-table tr:hover {
                background: rgba(0, 204, 150, 0.1) !important;
            }
            
            .metric-high {
                color: #00cc96 !important;
                font-weight: 600;
            }
            
            .metric-medium {
                color: #FFA15A !important;
                font-weight: 600;
            }
            
            .metric-low {
                color: #ef553b !important;
                font-weight: 600;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="risk-metrics-table">', unsafe_allow_html=True)
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.metric(
                label="Highest VaR",
                value=df['VaR'].max(),
                delta="Risk Level"
            )
        
        with col2:
            st.metric(
                label="Average CVaR",
                value=f"{pd.to_numeric(df['CVaR'].str.rstrip('%')).mean():.3%}",
                delta="Average Risk"
            )
        
        with col3:
            st.metric(
                label="Max VaR/CVaR",
                value=df['VaR/CVaR Ratio'].max(),
                delta="Risk Concentration"
            )
        
        with col4:
            methods = df['Method'].unique()
            st.metric(
                label="Methods Compared",
                value=len(methods),
                delta="Analysis Depth"
            )
        
        # Display interactive dataframe
        st.dataframe(
            df.style.apply(
                lambda x: ['background: rgba(0, 204, 150, 0.1)' if i % 2 == 0 else '' for i in range(len(x))],
                axis=0
            ).format({
                'VaR': lambda x: f"<span class='metric-high'>{x}</span>",
                'CVaR': lambda x: f"<span class='metric-medium'>{x}</span>",
                'ES': lambda x: f"<span class='metric-low'>{x}</span>"
            }),
            use_container_width=True,
            height=400
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return df
    
    def create_enhanced_efficient_frontier(self, returns: pd.DataFrame,
                                          risk_free_rate: float = 0.045,
                                          n_points: int = 50) -> go.Figure:
        """Create enhanced efficient frontier with detailed visualization."""
        try:
            if not LIBRARY_STATUS['status'].get('pypfopt', False):
                raise ImportError("PyPortfolioOpt not available")
            
            from pypfopt import expected_returns, risk_models
            
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.sample_cov(returns)
            
            # Generate efficient frontier
            ef = EfficientFrontier(mu, S)
            
            # Calculate min volatility portfolio
            ef_min_vol = EfficientFrontier(mu, S)
            weights_min_vol = ef_min_vol.min_volatility()
            ret_min_vol, vol_min_vol, _ = ef_min_vol.portfolio_performance()
            
            # Calculate max Sharpe portfolio
            ef_max_sharpe = EfficientFrontier(mu, S)
            weights_max_sharpe = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
            ret_max_sharpe, vol_max_sharpe, sharpe_max = ef_max_sharpe.portfolio_performance(
                risk_free_rate=risk_free_rate
            )
            
            # Calculate max return portfolio
            ef_max_return = EfficientFrontier(mu, S)
            weights_max_return = ef_max_return.max_return()
            ret_max_return, vol_max_return, _ = ef_max_return.portfolio_performance()
            
            # Generate efficient frontier points
            target_returns = np.linspace(ret_min_vol, ret_max_return * 0.9, n_points)
            efficient_volatilities = []
            efficient_returns = []
            
            for target_return in target_returns:
                ef = EfficientFrontier(mu, S)
                try:
                    ef.efficient_return(target_return)
                    ret, vol, _ = ef.portfolio_performance()
                    efficient_volatilities.append(vol)
                    efficient_returns.append(ret)
                except:
                    efficient_volatilities.append(np.nan)
                    efficient_returns.append(target_return)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Efficient Frontier',
                    'Capital Market Line',
                    'Risk-Return Distribution',
                    'Portfolio Composition'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.15,
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'scatter3d'}, {'type': 'bar'}]]
            )
            
            # 1. Efficient Frontier (Scatter)
            # Add efficient frontier line
            fig.add_trace(
                go.Scatter(
                    x=efficient_volatilities,
                    y=efficient_returns,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='#00cc96', width=4),
                    fill='tozeroy',
                    fillcolor='rgba(0, 204, 150, 0.1)',
                    hovertemplate='<b>Efficient Portfolio</b><br>' +
                                 'Volatility: %{x:.2%}<br>' +
                                 'Return: %{y:.2%}<br>' +
                                 'Sharpe: %{customdata:.2f}<extra></extra>',
                    customdata=[(r - risk_free_rate)/v if v > 0 else 0 
                               for r, v in zip(efficient_returns, efficient_volatilities)]
                ),
                row=1, col=1
            )
            
            # Add individual assets
            individual_vols = returns.std() * np.sqrt(252)
            individual_rets = returns.mean() * 252
            
            fig.add_trace(
                go.Scatter(
                    x=individual_vols,
                    y=individual_rets,
                    mode='markers+text',
                    name='Individual Assets',
                    marker=dict(
                        color='#ab63fa',
                        size=12,
                        line=dict(color='white', width=2)
                    ),
                    text=returns.columns,
                    textposition="top center",
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Volatility: %{x:.2%}<br>' +
                                 'Return: %{y:.2%}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add special portfolios
            special_portfolios = {
                'Min Volatility': (vol_min_vol, ret_min_vol, '#636efa'),
                'Max Sharpe': (vol_max_sharpe, ret_max_sharpe, '#FFA15A'),
                'Max Return': (vol_max_return, ret_max_return, '#ef553b')
            }
            
            for name, (vol, ret, color) in special_portfolios.items():
                fig.add_trace(
                    go.Scatter(
                        x=[vol],
                        y=[ret],
                        mode='markers',
                        name=name,
                        marker=dict(
                            color=color,
                            size=20,
                            line=dict(color='white', width=3),
                            symbol='star'
                        ),
                        hovertemplate=f'<b>{name}</b><br>' +
                                     f'Volatility: {vol:.2%}<br>' +
                                     f'Return: {ret:.2%}<br>' +
                                     f'Sharpe: {(ret - risk_free_rate)/vol:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # 2. Capital Market Line (CML)
            if vol_max_sharpe > 0:
                cml_x = np.linspace(0, vol_max_sharpe * 1.5, 50)
                cml_y = risk_free_rate + (ret_max_sharpe - risk_free_rate) / vol_max_sharpe * cml_x
                
                fig.add_trace(
                    go.Scatter(
                        x=cml_x,
                        y=cml_y,
                        mode='lines',
                        name='Capital Market Line',
                        line=dict(color='white', width=3, dash='dash'),
                        hovertemplate='CML<br>' +
                                     'Volatility: %{x:.2%}<br>' +
                                     'Return: %{y:.2%}<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # Add risk-free asset
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[risk_free_rate],
                    mode='markers',
                    name='Risk-Free Asset',
                    marker=dict(
                        color='#00cc96',
                        size=15,
                        symbol='diamond'
                    ),
                    hovertemplate='Risk-Free Rate: %{y:.2%}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 3. 3D Risk-Return-Skewness Distribution
            # Calculate skewness for each asset
            asset_skewness = returns.apply(lambda x: stats.skew(x.dropna()))
            
            fig.add_trace(
                go.Scatter3d(
                    x=individual_vols,
                    y=individual_rets,
                    z=asset_skewness,
                    mode='markers+text',
                    name='Assets 3D',
                    marker=dict(
                        size=8,
                        color=asset_skewness,
                        colorscale='RdYlGn',
                        colorbar=dict(title="Skewness"),
                        line=dict(color='white', width=1)
                    ),
                    text=returns.columns,
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Volatility: %{x:.2%}<br>' +
                                 'Return: %{y:.2%}<br>' +
                                 'Skewness: %{z:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 4. Optimal Portfolio Composition
            # Create pie chart for max Sharpe portfolio
            weights_df = pd.DataFrame({
                'Asset': list(weights_max_sharpe.keys()),
                'Weight': list(weights_max_sharpe.values())
            }).sort_values('Weight', ascending=False).head(10)
            
            fig.add_trace(
                go.Bar(
                    x=weights_df['Weight'],
                    y=weights_df['Asset'],
                    orientation='h',
                    name='Portfolio Weights',
                    marker_color=px.colors.qualitative.Set3[:len(weights_df)],
                    hovertemplate='<b>%{y}</b><br>' +
                                 'Weight: %{x:.1%}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=1000,
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
                title=dict(
                    text='Advanced Efficient Frontier Analysis',
                    x=0.5,
                    font=dict(size=24)
                ),
                scene=dict(
                    xaxis_title='Volatility',
                    yaxis_title='Return',
                    zaxis_title='Skewness'
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Annual Volatility", row=1, col=1, tickformat=".0%")
            fig.update_yaxes(title_text="Annual Return", row=1, col=1, tickformat=".0%")
            fig.update_xaxes(title_text="Volatility", row=1, col=2, tickformat=".0%")
            fig.update_yaxes(title_text="Return", row=1, col=2, tickformat=".0%")
            fig.update_xaxes(title_text="Weight", row=2, col=2, tickformat=".0%")
            fig.update_yaxes(title_text="Asset", row=2, col=2)
            
            # Add annotations
            annotations = [
                dict(
                    x=vol_max_sharpe,
                    y=ret_max_sharpe,
                    xref="x1",
                    yref="y1",
                    text=f"Max Sharpe: {(ret_max_sharpe - risk_free_rate)/vol_max_sharpe:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#FFA15A",
                    ax=50,
                    ay=-50
                ),
                dict(
                    x=vol_min_vol,
                    y=ret_min_vol,
                    xref="x1",
                    yref="y1",
                    text=f"Min Volatility",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636efa",
                    ax=-50,
                    ay=50
                )
            ]
            
            fig.update_layout(annotations=annotations)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Enhanced efficient frontier failed: {str(e)}")
            return self.create_efficient_frontier(returns, risk_free_rate)
    
    def create_risk_surface_plot(self, returns: pd.DataFrame) -> go.Figure:
        """Create 3D risk surface plot."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252
            
            # Generate random weight combinations
            n_points = 30
            np.random.seed(42)
            
            returns_3d = []
            risks_3d = []
            sharpes_3d = []
            
            for _ in range(n_points * n_points):
                # Generate random weights
                weights = np.random.random(len(returns.columns))
                weights = weights / weights.sum()
                
                # Calculate portfolio metrics
                port_return = np.dot(weights, returns.mean()) * 252
                port_risk = np.sqrt(weights.T @ cov_matrix.values @ weights)
                
                returns_3d.append(port_return)
                risks_3d.append(port_risk)
                sharpes_3d.append((port_return - 0.045) / port_risk if port_risk > 0 else 0)
            
            # Create 3D surface
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=risks_3d,
                    y=returns_3d,
                    z=sharpes_3d,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=sharpes_3d,
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title="Sharpe Ratio")
                    ),
                    hovertemplate='<b>Random Portfolio</b><br>' +
                                 'Risk: %{x:.2%}<br>' +
                                 'Return: %{y:.2%}<br>' +
                                 'Sharpe: %{z:.2f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                height=600,
                title="Risk-Return-Surface (3D Visualization)",
                scene=dict(
                    xaxis_title='Risk (Volatility)',
                    yaxis_title='Return',
                    zaxis_title='Sharpe Ratio',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Risk surface plot failed: {str(e)}")
            return go.Figure()
    
    def create_optimization_comparison(self, returns: pd.DataFrame, 
                                      optimization_methods: List[str]) -> go.Figure:
        """Create comparison of different optimization methods."""
        try:
            # Calculate portfolio metrics for each method
            results = {}
            
            for method in optimization_methods:
                try:
                    # Create optimizer
                    config = PortfolioConfig(
                        universe="Comparison",
                        tickers=returns.columns.tolist(),
                        benchmark="^GSPC",
                        start_date=returns.index[0],
                        end_date=returns.index[-1],
                        risk_free_rate=0.045,
                        optimization_method=method
                    )
                    
                    optimizer = PortfolioOptimizer(returns, returns, config)
                    weights, performance = optimizer.optimize()
                    
                    results[method] = {
                        'weights': weights,
                        'performance': performance,
                        'sharpe': performance[2],
                        'return': performance[0],
                        'risk': performance[1]
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Method {method} failed: {str(e)}")
            
            # Create comparison chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Performance Comparison',
                    'Risk-Return Trade-off',
                    'Sharpe Ratio Comparison',
                    'Weight Concentration'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # Prepare data
            methods = list(results.keys())
            returns_data = [results[m]['return'] for m in methods]
            risks_data = [results[m]['risk'] for m in methods]
            sharpes_data = [results[m]['sharpe'] for m in methods]
            
            # Color palette
            colors = px.colors.qualitative.Set3[:len(methods)]
            
            # 1. Performance Comparison (Bar Chart)
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=returns_data,
                    name='Return',
                    marker_color=colors,
                    text=[f'{r:.1%}' for r in returns_data],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # 2. Risk-Return Trade-off (Scatter)
            for i, method in enumerate(methods):
                fig.add_trace(
                    go.Scatter(
                        x=[risks_data[i]],
                        y=[returns_data[i]],
                        mode='markers+text',
                        name=method,
                        marker=dict(
                            size=15,
                            color=colors[i],
                            line=dict(width=2, color='white')
                        ),
                        text=[method],
                        textposition="top center",
                        hovertemplate=f'<b>{method}</b><br>' +
                                     f'Risk: {risks_data[i]:.2%}<br>' +
                                     f'Return: {returns_data[i]:.2%}<br>' +
                                     f'Sharpe: {sharpes_data[i]:.2f}<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # Add risk-free line
            fig.add_trace(
                go.Scatter(
                    x=[0, max(risks_data) * 1.1],
                    y=[0.045, 0.045],
                    mode='lines',
                    name='Risk-Free Rate',
                    line=dict(color='white', dash='dash', width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 3. Sharpe Ratio Comparison (Horizontal Bar)
            fig.add_trace(
                go.Bar(
                    y=methods,
                    x=sharpes_data,
                    name='Sharpe Ratio',
                    orientation='h',
                    marker_color=colors,
                    text=[f'{s:.2f}' for s in sharpes_data],
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            # 4. Weight Concentration (Box Plot)
            all_weights = []
            for method in methods:
                weights = list(results[method]['weights'].values())
                all_weights.append(weights)
            
            for i, weights in enumerate(all_weights):
                fig.add_trace(
                    go.Box(
                        y=weights,
                        name=methods[i],
                        marker_color=colors[i],
                        boxmean=True,
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=800,
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
                title=dict(
                    text='Optimization Methods Comparison',
                    x=0.5,
                    font=dict(size=24)
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Method", row=1, col=1)
            fig.update_yaxes(title_text="Annual Return", row=1, col=1, tickformat=".0%")
            fig.update_xaxes(title_text="Annual Volatility", row=1, col=2, tickformat=".0%")
            fig.update_yaxes(title_text="Annual Return", row=1, col=2, tickformat=".0%")
            fig.update_xaxes(title_text="Sharpe Ratio", row=2, col=1)
            fig.update_yaxes(title_text="Method", row=2, col=1)
            fig.update_xaxes(title_text="Weight", row=2, col=2, tickformat=".0%")
            fig.update_yaxes(title_text="Method", row=2, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Optimization comparison failed: {str(e)}")
            return go.Figure()

# ============================================================================
# ENHANCED QUANTEDGE PRO APP WITH ADVANCED VISUALIZATIONS
# ============================================================================

class EnhancedQuantEdgeProApp(QuantEdgeProApp):
    """Enhanced QuantEdge Pro with advanced visualizations."""
    
    def __init__(self):
        super().__init__()
        self.advanced_viz = AdvancedVisualizationEngine()
        self.risk_color_palette = {
            'VaR': '#ef553b',
            'CVaR': '#ff6b6b',
            'ES': '#ffa15a',
            'Normal': '#00cc96',
            'Historical': '#636efa',
            'MonteCarlo': '#ab63fa'
        }
    
    def render_enhanced_sidebar(self):
        """Render enhanced sidebar with smart buttons."""
        with st.sidebar:
            st.markdown("""
            <div style="padding: 1.5rem; border-radius: 12px; background: rgba(30, 30, 30, 0.8); margin-bottom: 2rem;">
                <h3 style="color: #00cc96; margin-bottom: 1rem;">🚀 Smart Configuration</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Universe selection with enhanced UI
            universe_options = list(Constants.ASSET_UNIVERSES.keys())
            selected_universe = st.selectbox(
                "📊 Select Asset Universe",
                universe_options,
                help="Pre-configured portfolio universes",
                key="enhanced_universe"
            )
            
            # Get universe details
            universe_details = Constants.ASSET_UNIVERSES[selected_universe]
            
            # Smart date range with presets
            col1, col2 = st.columns(2)
            with col1:
                date_preset = st.selectbox(
                    "📅 Time Period",
                    ["Custom", "1 Year", "3 Years", "5 Years", "10 Years", "Max"],
                    help="Select predefined time periods"
                )
            
            # Calculate dates based on preset
            end_date = datetime.now()
            if date_preset == "1 Year":
                start_date = end_date - timedelta(days=365)
            elif date_preset == "3 Years":
                start_date = end_date - timedelta(days=365*3)
            elif date_preset == "5 Years":
                start_date = end_date - timedelta(days=365*5)
            elif date_preset == "10 Years":
                start_date = end_date - timedelta(days=365*10)
            elif date_preset == "Max":
                start_date = end_date - timedelta(days=365*20)
            else:
                with col2:
                    start_date = st.date_input(
                        "Start Date",
                        value=end_date - timedelta(days=365*3),
                        max_value=end_date - timedelta(days=1),
                        key="enhanced_start"
                    )
                end_date = st.date_input(
                    "End Date",
                    value=end_date,
                    max_value=end_date,
                    key="enhanced_end"
                )
            
            # Advanced optimization selection
            st.markdown("---")
            st.markdown("### 🎯 Advanced Optimization")
            
            optimization_options = list(Constants.OPTIMIZATION_METHODS.keys())
            selected_optimization = st.selectbox(
                "Optimization Method",
                optimization_options,
                format_func=lambda x: f"⚡ {x}",
                help="Select portfolio optimization methodology",
                key="enhanced_optimization"
            )
            
            # Advanced constraints with sliders
            st.markdown("### 🔧 Advanced Constraints")
            
            col1, col2 = st.columns(2)
            with col1:
                max_weight = st.slider(
                    "Max Weight per Asset",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.30,
                    step=0.05,
                    help="Maximum allocation to any single asset",
                    key="enhanced_max_weight"
                )
            
            with col2:
                min_weight = st.slider(
                    "Min Weight per Asset",
                    min_value=0.0,
                    max_value=0.20,
                    value=0.0,
                    step=0.01,
                    help="Minimum allocation to any single asset",
                    key="enhanced_min_weight"
                )
            
            # Smart execution buttons
            st.markdown("---")
            st.markdown("### 🚀 Smart Execution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                analyze_clicked = self.advanced_viz.create_smart_execution_button(
                    label="Analyze",
                    key="analyze_smart",
                    color="#00cc96",
                    icon="📊",
                    tooltip="Run comprehensive portfolio analysis"
                )
            
            with col2:
                optimize_clicked = self.advanced_viz.create_smart_execution_button(
                    label="Optimize",
                    key="optimize_smart",
                    color="#636efa",
                    icon="⚡",
                    tooltip="Run advanced portfolio optimization"
                )
            
            # Advanced risk settings
            with st.expander("⚡ Advanced Risk Settings", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    risk_free_rate = st.slider(
                        "Risk-Free Rate",
                        min_value=0.0,
                        max_value=0.20,
                        value=0.045,
                        step=0.001,
                        format="%.1%",
                        key="enhanced_rf"
                    )
                
                with col2:
                    transaction_cost = st.slider(
                        "Transaction Cost",
                        min_value=0,
                        max_value=100,
                        value=10,
                        step=1,
                        format="%d bps",
                        key="enhanced_tc"
                    )
            
            # Custom assets
            with st.expander("➕ Add Custom Assets", expanded=False):
                custom_tickers = st.text_area(
                    "Enter tickers (comma-separated)",
                    placeholder="AAPL, MSFT, GOOGL, ...",
                    help="Add additional assets to the portfolio",
                    key="enhanced_custom"
                )
            
            # Configuration summary
            st.markdown("---")
            st.markdown("### 📋 Configuration Summary")
            
            config_summary = {
                "Universe": selected_universe,
                "Assets": len(universe_details['tickers']) + (len(custom_tickers.split(',')) if custom_tickers else 0),
                "Period": date_preset if date_preset != "Custom" else "Custom",
                "Optimization": selected_optimization,
                "Max Weight": f"{max_weight:.0%}",
                "Risk-Free Rate": f"{risk_free_rate:.1%}"
            }
            
            for key, value in config_summary.items():
                st.caption(f"**{key}:** {value}")
            
            return {
                'universe': selected_universe,
                'tickers': universe_details['tickers'] + 
                          ([t.strip().upper() for t in custom_tickers.split(',')] if custom_tickers else []),
                'benchmark': universe_details['benchmark'],
                'start_date': start_date,
                'end_date': end_date,
                'risk_free_rate': risk_free_rate,
                'optimization_method': selected_optimization,
                'max_weight': max_weight,
                'min_weight': min_weight,
                'transaction_cost': transaction_cost / 10000,
                'analyze_clicked': analyze_clicked,
                'optimize_clicked': optimize_clicked
            }
    
    def render_advanced_risk_analytics(self):
        """Render advanced risk analytics with VaR/CVaR/ES calculations."""
        if not st.session_state.optimization_results:
            return
        
        results = st.session_state.optimization_results
        
        st.markdown('<div class="section-header">⚡ Advanced Risk Analytics</div>', 
                   unsafe_allow_html=True)
        
        # Create tabs for different risk analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 VaR/CVaR/ES Analysis",
            "📈 Comparative Risk Metrics",
            "🎯 Stress Testing",
            "🔍 Risk Decomposition"
        ])
        
        with tab1:
            self._render_var_cvar_es_analysis(results)
        
        with tab2:
            self._render_comparative_risk_analysis(results)
        
        with tab3:
            self._render_advanced_stress_testing(results)
        
        with tab4:
            self._render_risk_decomposition_analysis(results)
    
    def _render_var_cvar_es_analysis(self, results):
        """Render VaR, CVaR, and Expected Shortfall analysis."""
        portfolio_returns = results['portfolio_returns'].mean(axis=1)
        
        # Calculate advanced risk metrics
        with st.spinner("🔬 Calculating advanced risk metrics..."):
            var_results = self.advanced_viz.create_advanced_var_analysis(
                portfolio_returns,
                confidence_levels=[0.90, 0.95, 0.99, 0.995],
                methods=['historical', 'parametric', 'montecarlo']
            )
        
        # Display smart table
        st.subheader("📋 Smart Risk Metrics Table")
        self.advanced_viz.create_risk_metrics_smart_table(var_results)
        
        # Display comparative charts
        st.subheader("📊 Comparative Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = self.advanced_viz.create_var_comparison_chart(var_results)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Create risk distribution chart
            fig2 = go.Figure()
            
            # Add histogram of returns
            fig2.add_trace(go.Histogram(
                x=portfolio_returns,
                name='Returns Distribution',
                nbinsx=50,
                marker_color='#636efa',
                opacity=0.7,
                histnorm='probability density'
            ))
            
            # Add VaR lines for different confidence levels
            for confidence, color in zip([0.90, 0.95, 0.99], ['#FFA15A', '#ef553b', '#d62728']):
                var_value = -np.percentile(portfolio_returns, (1-confidence)*100)
                fig2.add_vline(
                    x=-var_value,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"VaR {confidence*100:.0f}%",
                    annotation_position="top right"
                )
            
            fig2.update_layout(
                height=400,
                title="Returns Distribution with VaR Levels",
                xaxis_title="Daily Returns",
                yaxis_title="Density",
                template='plotly_dark',
                showlegend=True
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Display portfolio statistics
        st.subheader("📈 Portfolio Statistics")
        
        if 'portfolio_stats' in var_results:
            stats = var_results['portfolio_stats']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Return", f"{stats['mean']*252:.2%}")
                st.metric("Skewness", f"{stats['skewness']:.3f}")
            
            with col2:
                st.metric("Volatility", f"{stats['std']*np.sqrt(252):.2%}")
                st.metric("Kurtosis", f"{stats['kurtosis']:.3f}")
            
            with col3:
                st.metric("Min Return", f"{stats['min']:.2%}")
                st.metric("Max Return", f"{stats['max']:.2%}")
            
            with col4:
                st.metric("Sharpe Ratio", 
                         f"{(stats['mean']*252 - results['config'].risk_free_rate)/(stats['std']*np.sqrt(252)):.3f}")
                st.metric("Sortino Ratio", 
                         f"{(stats['mean']*252 - results['config'].risk_free_rate)/(self._calculate_downside_deviation(portfolio_returns)):.3f}")
        
        # Additional risk metrics
        st.subheader("🔍 Additional Risk Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate expected shortfall over time
            rolling_es = portfolio_returns.rolling(window=63).apply(
                lambda x: -x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else 0,
                raw=True
            )
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=rolling_es.index,
                y=rolling_es.values,
                mode='lines',
                name='Rolling Expected Shortfall (95%)',
                line=dict(color='#ff6b6b', width=2)
            ))
            
            fig3.update_layout(
                height=300,
                title="Rolling Expected Shortfall",
                xaxis_title="Date",
                yaxis_title="Expected Shortfall",
                template='plotly_dark',
                yaxis_tickformat='.1%'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Calculate VaR exceedances
            var_95 = -np.percentile(portfolio_returns, 5)
            exceedances = portfolio_returns[portfolio_returns < -var_95]
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=portfolio_returns.index,
                y=portfolio_returns.values,
                mode='markers',
                name='Daily Returns',
                marker=dict(
                    size=6,
                    color=['#ef553b' if r < -var_95 else '#636efa' for r in portfolio_returns],
                    opacity=0.7
                )
            ))
            
            fig4.add_hline(
                y=-var_95,
                line_dash="dash",
                line_color="#FFA15A",
                annotation_text="VaR 95% Threshold"
            )
            
            fig4.update_layout(
                height=300,
                title="VaR Exceedances (95% Confidence)",
                xaxis_title="Date",
                yaxis_title="Daily Return",
                template='plotly_dark',
                yaxis_tickformat='.1%'
            )
            
            st.plotly_chart(fig4, use_container_width=True)
    
    def _render_comparative_risk_analysis(self, results):
        """Render comparative risk analysis between different methods."""
        portfolio_returns = results['portfolio_returns'].mean(axis=1)
        benchmark_returns = results['benchmark_returns']
        
        # Calculate risk metrics for different methods
        methods = ['historical', 'parametric', 'montecarlo']
        risk_comparison = {}
        
        for method in methods:
            var_results = self.advanced_viz.create_advanced_var_analysis(
                portfolio_returns,
                confidence_levels=[0.95, 0.99],
                methods=[method]
            )
            risk_comparison[method] = var_results.get(method, {})
        
        # Create comparative visualization
        st.subheader("📊 Risk Methodology Comparison")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'VaR Comparison (95% Confidence)',
                'CVaR Comparison (95% Confidence)',
                'VaR Comparison (99% Confidence)',
                'CVaR Comparison (99% Confidence)'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        colors = ['#636efa', '#00cc96', '#ab63fa']
        
        for idx, confidence in enumerate([0.95, 0.99]):
            for i, method in enumerate(methods):
                if method in risk_comparison and confidence in risk_comparison[method]:
                    metrics = risk_comparison[method][confidence]
                    
                    # VaR comparison
                    fig.add_trace(
                        go.Bar(
                            x=[method.capitalize()],
                            y=[metrics['VaR']],
                            name=f'{method.capitalize()} VaR',
                            marker_color=colors[i],
                            showlegend=(idx == 0),
                            text=f"{metrics['VaR']:.3%}",
                            textposition='auto'
                        ),
                        row=1, col=idx+1
                    )
                    
                    # CVaR comparison
                    fig.add_trace(
                        go.Bar(
                            x=[method.capitalize()],
                            y=[metrics['CVaR']],
                            name=f'{method.capitalize()} CVaR',
                            marker_color=colors[i],
                            showlegend=False,
                            text=f"{metrics['CVaR']:.3%}",
                            textposition='auto',
                            opacity=0.7
                        ),
                        row=2, col=idx+1
                    )
        
        fig.update_layout(
            height=600,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical comparison
        st.subheader("📈 Statistical Comparison")
        
        comparison_data = []
        for method in methods:
            if method in risk_comparison and 0.95 in risk_comparison[method]:
                metrics = risk_comparison[method][0.95]
                comparison_data.append({
                    'Method': method.capitalize(),
                    'VaR (95%)': metrics['VaR'],
                    'CVaR (95%)': metrics['CVaR'],
                    'ES (95%)': metrics['ES'],
                    'VaR/CVaR Ratio': metrics['VaR']/metrics['CVaR'] if metrics['CVaR'] != 0 else 0
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Style the dataframe
            styled_df = df_comparison.style.format({
                'VaR (95%)': '{:.3%}',
                'CVaR (95%)': '{:.3%}',
                'ES (95%)': '{:.3%}',
                'VaR/CVaR Ratio': '{:.3f}'
            }).background_gradient(subset=['VaR (95%)', 'CVaR (95%)'], cmap='Reds')
            
            st.dataframe(styled_df, use_container_width=True)
    
    def _render_advanced_stress_testing(self, results):
        """Render advanced stress testing analysis."""
        portfolio_returns = results['portfolio_returns'].mean(axis=1)
        
        # Enhanced stress testing
        st.subheader("🌪️ Advanced Stress Testing")
        
        # User-defined stress scenarios
        with st.expander("⚡ Define Custom Stress Scenarios", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                stress_return = st.slider(
                    "Stress Return Shock",
                    min_value=-0.50,
                    max_value=0.50,
                    value=-0.20,
                    step=0.01,
                    format="%.0%",
                    help="Simulated return shock"
                )
            
            with col2:
                stress_vol = st.slider(
                    "Stress Volatility Increase",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.5,
                    step=0.1,
                    format="%.1fx",
                    help="Multiplier on current volatility"
                )
        
        # Calculate stress metrics
        current_vol = portfolio_returns.std() * np.sqrt(252)
        stress_vol_adj = current_vol * stress_vol
        
        # Simulate stressed returns
        np.random.seed(42)
        n_simulations = 10000
        stressed_returns = np.random.normal(
            stress_return / 252,  # Annual to daily
            stress_vol_adj / np.sqrt(252),
            n_simulations
        )
        
        # Compare with normal returns
        normal_returns = np.random.normal(
            portfolio_returns.mean(),
            portfolio_returns.std(),
            n_simulations
        )
        
        # Create comparison visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Return Distribution: Normal vs Stress',
                'Cumulative Distribution',
                'VaR Comparison',
                'Expected Shortfall Comparison'
            ),
            vertical_spacing=0.15
        )
        
        # 1. Return distribution
        fig.add_trace(
            go.Histogram(
                x=normal_returns,
                name='Normal Scenario',
                nbinsx=50,
                marker_color='#636efa',
                opacity=0.5,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=stressed_returns,
                name='Stress Scenario',
                nbinsx=50,
                marker_color='#ef553b',
                opacity=0.5,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        # 2. Cumulative distribution
        sorted_normal = np.sort(normal_returns)
        sorted_stress = np.sort(stressed_returns)
        cdf_normal = np.arange(1, len(sorted_normal)+1) / len(sorted_normal)
        cdf_stress = np.arange(1, len(sorted_stress)+1) / len(sorted_stress)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_normal,
                y=cdf_normal,
                name='Normal CDF',
                line=dict(color='#636efa', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=sorted_stress,
                y=cdf_stress,
                name='Stress CDF',
                line=dict(color='#ef553b', width=2)
            ),
            row=1, col=2
        )
        
        # 3. VaR comparison
        confidence_levels = np.linspace(0.90, 0.995, 20)
        var_normal = [-np.percentile(normal_returns, (1-c)*100) for c in confidence_levels]
        var_stress = [-np.percentile(stressed_returns, (1-c)*100) for c in confidence_levels]
        
        fig.add_trace(
            go.Scatter(
                x=confidence_levels,
                y=var_normal,
                name='Normal VaR',
                line=dict(color='#636efa', width=3)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=confidence_levels,
                y=var_stress,
                name='Stress VaR',
                line=dict(color='#ef553b', width=3)
            ),
            row=2, col=1
        )
        
        # 4. Expected Shortfall comparison
        es_normal = []
        es_stress = []
        
        for c in confidence_levels:
            var_n = -np.percentile(normal_returns, (1-c)*100)
            var_s = -np.percentile(stressed_returns, (1-c)*100)
            es_normal.append(-normal_returns[normal_returns <= -var_n].mean())
            es_stress.append(-stressed_returns[stressed_returns <= -var_s].mean())
        
        fig.add_trace(
            go.Scatter(
                x=confidence_levels,
                y=es_normal,
                name='Normal ES',
                line=dict(color='#636efa', width=3, dash='dash')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=confidence_levels,
                y=es_stress,
                name='Stress ES',
                line=dict(color='#ef553b', width=3, dash='dash')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Daily Return", row=1, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Return", row=1, col=2, tickformat=".1%")
        fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
        fig.update_xaxes(title_text="Confidence Level", row=2, col=1)
        fig.update_yaxes(title_text="Value at Risk", row=2, col=1, tickformat=".1%")
        fig.update_xaxes(title_text="Confidence Level", row=2, col=2)
        fig.update_yaxes(title_text="Expected Shortfall", row=2, col=2, tickformat=".1%")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stress test statistics
        st.subheader("📊 Stress Test Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Normal VaR (95%)",
                f"{-np.percentile(normal_returns, 5):.3%}",
                delta="Baseline"
            )
        
        with col2:
            st.metric(
                "Stress VaR (95%)",
                f"{-np.percentile(stressed_returns, 5):.3%}",
                delta=f"{(-np.percentile(stressed_returns, 5)/-np.percentile(normal_returns, 5)-1):.0%}",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Normal ES (95%)",
                f"{-normal_returns[normal_returns <= np.percentile(normal_returns, 5)].mean():.3%}",
                delta="Baseline"
            )
        
        with col4:
            st.metric(
                "Stress ES (95%)",
                f"{-stressed_returns[stressed_returns <= np.percentile(stressed_returns, 5)].mean():.3%}",
                delta=f"{(-stressed_returns[stressed_returns <= np.percentile(stressed_returns, 5)].mean()/-normal_returns[normal_returns <= np.percentile(normal_returns, 5)].mean()-1):.0%}",
                delta_color="inverse"
            )
    
    def _render_risk_decomposition_analysis(self, results):
        """Render risk decomposition analysis."""
        portfolio_returns = results['portfolio_returns']
        weights = results['weights']
        
        # Calculate risk contributions
        risk_decomposition = self.risk_analyzer.calculate_risk_decomposition(
            weights, portfolio_returns
        )
        
        if not risk_decomposition:
            st.warning("Risk decomposition not available.")
            return
        
        st.subheader("🔍 Risk Decomposition Analysis")
        
        # Create risk contribution chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk Contribution by Asset',
                'Marginal Contribution to Risk',
                'Percentage Risk Contribution',
                'Risk Contribution vs Weight'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Extract data
        tickers = []
        ctr_values = []
        mctr_values = []
        pct_ctr_values = []
        weight_values = []
        
        for ticker, data in risk_decomposition.items():
            if ticker != 'summary':
                tickers.append(ticker)
                ctr_values.append(data['CTR'])
                mctr_values.append(data['MCTR'])
                pct_ctr_values.append(data['Percentage Contribution'] * 100)
                weight_values.append(data['Weight'] * 100)
        
        # 1. Risk Contribution (Bar)
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=ctr_values,
                name='Risk Contribution',
                marker_color='#ef553b',
                text=[f'{v:.3f}' for v in ctr_values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Marginal Contribution (Bar)
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=mctr_values,
                name='Marginal Contribution',
                marker_color='#ff6b6b',
                text=[f'{v:.3f}' for v in mctr_values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Percentage Contribution (Pie)
        fig.add_trace(
            go.Pie(
                labels=tickers,
                values=pct_ctr_values,
                name='% Contribution',
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3[:len(tickers)]
            ),
            row=2, col=1
        )
        
        # 4. Risk vs Weight (Scatter)
        fig.add_trace(
            go.Scatter(
                x=weight_values,
                y=pct_ctr_values,
                mode='markers+text',
                name='Risk vs Weight',
                marker=dict(
                    size=15,
                    color=pct_ctr_values,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="% Contribution")
                ),
                text=tickers,
                textposition="top center",
                hovertemplate='<b>%{text}</b><br>' +
                             'Weight: %{x:.1f}%<br>' +
                             'Risk Contribution: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Asset", row=1, col=1, tickangle=45)
        fig.update_yaxes(title_text="Risk Contribution", row=1, col=1)
        fig.update_xaxes(title_text="Asset", row=1, col=2, tickangle=45)
        fig.update_yaxes(title_text="Marginal Contribution", row=1, col=2)
        fig.update_xaxes(title_text="Portfolio Weight (%)", row=2, col=2)
        fig.update_yaxes(title_text="Risk Contribution (%)", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk decomposition summary
        if 'summary' in risk_decomposition:
            summary = risk_decomposition['summary']
            
            st.subheader("📊 Risk Decomposition Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Portfolio Volatility",
                    f"{summary['Portfolio Volatility']:.3%}"
                )
            
            with col2:
                st.metric(
                    "Concentration Index",
                    f"{summary['Concentration Index']:.3f}"
                )
            
            with col3:
                st.metric(
                    "Most Risky Asset",
                    summary['Most Risky Asset']
                )
            
            with col4:
                st.metric(
                    "Least Risky Asset",
                    summary['Least Risky Asset']
                )
    
    def render_advanced_portfolio_optimization(self):
        """Render advanced portfolio optimization visualizations."""
        if not st.session_state.optimization_results:
            return
        
        results = st.session_state.optimization_results
        
        st.markdown('<div class="section-header">⚡ Advanced Portfolio Optimization</div>', 
                   unsafe_allow_html=True)
        
        # Create tabs for different optimization visualizations
        tab1, tab2, tab3 = st.tabs([
            "🎯 Enhanced Efficient Frontier",
            "📊 Optimization Comparison",
            "🔍 Risk-Return Surface"
        ])
        
        with tab1:
            self._render_enhanced_efficient_frontier(results)
        
        with tab2:
            self._render_optimization_comparison(results)
        
        with tab3:
            self._render_risk_return_surface(results)
    
    def _render_enhanced_efficient_frontier(self, results):
        """Render enhanced efficient frontier visualization."""
        portfolio_returns = results['portfolio_returns']
        
        st.subheader("🎯 Enhanced Efficient Frontier Analysis")
        
        # Calculate and display enhanced frontier
        fig = self.advanced_viz.create_enhanced_efficient_frontier(
            portfolio_returns,
            results['config'].risk_free_rate
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Frontier statistics
        st.subheader("📊 Frontier Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate efficient frontier metrics
            try:
                from pypfopt import expected_returns, risk_models
                
                mu = expected_returns.mean_historical_return(portfolio_returns)
                S = risk_models.sample_cov(portfolio_returns)
                ef = EfficientFrontier(mu, S)
                
                # Min volatility
                ef_min = EfficientFrontier(mu, S)
                weights_min = ef_min.min_volatility()
                ret_min, vol_min, _ = ef_min.portfolio_performance()
                
                # Max Sharpe
                ef_sharpe = EfficientFrontier(mu, S)
                weights_sharpe = ef_sharpe.max_sharpe(results['config'].risk_free_rate)
                ret_sharpe, vol_sharpe, sharpe = ef_sharpe.portfolio_performance(
                    risk_free_rate=results['config'].risk_free_rate
                )
                
                # Display metrics
                st.metric("Min Volatility Return", f"{ret_min:.2%}")
                st.metric("Min Volatility Risk", f"{vol_min:.2%}")
                st.metric("Max Sharpe Ratio", f"{sharpe:.3f}")
                st.metric("Max Sharpe Return", f"{ret_sharpe:.2%}")
                
            except Exception as e:
                st.warning(f"Frontier statistics unavailable: {str(e)}")
        
        with col2:
            # Portfolio optimization insights
            st.info("""
            ### 🔍 Optimization Insights
            
            **Efficient Frontier** shows optimal portfolios that:
            - Maximize return for given risk
            - Minimize risk for given return
            
            **Capital Market Line (CML)** represents:
            - Optimal risk-return trade-off
            - Combination of risk-free asset and tangency portfolio
            
            **Key Points:**
            - Points on frontier are efficient
            - Points below frontier are suboptimal
            - Tangency portfolio maximizes Sharpe ratio
            """)
    
    def _render_optimization_comparison(self, results):
        """Render optimization methods comparison."""
        portfolio_returns = results['portfolio_returns']
        
        st.subheader("📊 Optimization Methods Comparison")
        
        # Select methods to compare
        optimization_methods = st.multiselect(
            "Select methods to compare:",
            list(Constants.OPTIMIZATION_METHODS.keys()),
            default=['MAX_SHARPE', 'MIN_VOLATILITY', 'EQUAL_WEIGHT', 'RISK_PARITY'],
            key="opt_comparison"
        )
        
        if optimization_methods:
            # Create comparison visualization
            fig = self.advanced_viz.create_optimization_comparison(
                portfolio_returns,
                optimization_methods
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.subheader("📋 Detailed Comparison")
            
            comparison_data = []
            for method in optimization_methods:
                try:
                    config = PortfolioConfig(
                        universe="Comparison",
                        tickers=portfolio_returns.columns.tolist(),
                        benchmark="^GSPC",
                        start_date=portfolio_returns.index[0],
                        end_date=portfolio_returns.index[-1],
                        risk_free_rate=results['config'].risk_free_rate,
                        optimization_method=method
                    )
                    
                    optimizer = PortfolioOptimizer(portfolio_returns, portfolio_returns, config)
                    weights, performance = optimizer.optimize()
                    
                    # Calculate additional metrics
                    portfolio_return_series = portfolio_returns.dot(
                        np.array(list(weights.values()))
                    )
                    
                    # Calculate maximum drawdown
                    cumulative = (1 + portfolio_return_series).cumprod()
                    rolling_max = cumulative.expanding().max()
                    drawdown = (cumulative - rolling_max) / rolling_max
                    max_dd = drawdown.min()
                    
                    comparison_data.append({
                        'Method': method,
                        'Return': f"{performance[0]:.2%}",
                        'Risk': f"{performance[1]:.2%}",
                        'Sharpe': f"{performance[2]:.3f}",
                        'Max Drawdown': f"{max_dd:.2%}",
                        'Diversification': self._calculate_diversification_ratio(weights, portfolio_returns),
                        'Active Positions': sum(1 for w in weights.values() if w > 0.01)
                    })
                    
                except Exception as e:
                    st.warning(f"Method {method} failed: {str(e)}")
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                
                # Apply styling
                styled_df = df_comparison.style.background_gradient(
                    subset=['Return', 'Sharpe'], 
                    cmap='Greens'
                ).background_gradient(
                    subset=['Risk', 'Max Drawdown'], 
                    cmap='Reds_r'
                )
                
                st.dataframe(styled_df, use_container_width=True)
    
    def _render_risk_return_surface(self, results):
        """Render 3D risk-return surface."""
        portfolio_returns = results['portfolio_returns']
        
        st.subheader("🔍 3D Risk-Return Surface")
        
        # Create 3D surface plot
        fig = self.advanced_viz.create_risk_surface_plot(portfolio_returns)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Surface analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            ### 📊 Surface Analysis
            
            The 3D surface shows:
            - **X-axis**: Portfolio Risk (Volatility)
            - **Y-axis**: Portfolio Return
            - **Z-axis**: Sharpe Ratio
            
            **Key Observations:**
            - Higher returns come with higher risk
            - Optimal portfolios lie on efficient frontier
            - Color indicates Sharpe ratio quality
            """)
        
        with col2:
            # Calculate surface statistics
            st.subheader("📈 Surface Statistics")
            
            # Generate random portfolios for statistics
            n_portfolios = 1000
            np.random.seed(42)
            
            returns_3d = []
            risks_3d = []
            sharpes_3d = []
            
            cov_matrix = portfolio_returns.cov() * 252
            
            for _ in range(n_portfolios):
                weights = np.random.random(len(portfolio_returns.columns))
                weights = weights / weights.sum()
                
                port_return = np.dot(weights, portfolio_returns.mean()) * 252
                port_risk = np.sqrt(weights.T @ cov_matrix.values @ weights)
                
                returns_3d.append(port_return)
                risks_3d.append(port_risk)
                sharpes_3d.append((port_return - results['config'].risk_free_rate) / port_risk 
                                if port_risk > 0 else 0)
            
            st.metric("Average Random Return", f"{np.mean(returns_3d):.2%}")
            st.metric("Average Random Risk", f"{np.mean(risks_3d):.2%}")
            st.metric("Average Sharpe Ratio", f"{np.mean(sharpes_3d):.3f}")
            st.metric("Best Sharpe Ratio", f"{np.max(sharpes_3d):.3f}")
    
    def _calculate_diversification_ratio(self, weights: Dict, returns: pd.DataFrame) -> float:
        """Calculate diversification ratio."""
        try:
            individual_vols = returns.std() * np.sqrt(252)
            portfolio_vol = returns.dot(list(weights.values())).std() * np.sqrt(252)
            
            weighted_avg_vol = sum(w * individual_vols.get(ticker, 0) 
                                 for ticker, w in weights.items())
            
            if portfolio_vol > 0:
                return weighted_avg_vol / portfolio_vol
            return 1.0
        except:
            return 1.0
    
    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0
        return np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
    
    def run_enhanced(self):
        """Run enhanced application with advanced visualizations."""
        try:
            # Render enhanced header
            self.render_header()
            
            # Render enhanced sidebar
            config = self.render_enhanced_sidebar()
            
            # Handle analysis
            if config['analyze_clicked']:
                with st.spinner("🚀 Running advanced analysis..."):
                    success = self.analyze_portfolio(config)
                    if success:
                        st.success("✅ Advanced analysis complete!")
                        st.rerun()
            
            # Render enhanced results
            if st.session_state.optimization_results:
                # Create enhanced tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Optimization Results",
                    "📈 Performance Analytics",
                    "⚡ Advanced Risk Analytics",
                    "🎯 Portfolio Optimization",
                    "🔄 Backtesting",
                    "🌪️ Stress Testing"
                ])
                
                with tab1:
                    self.render_optimization_results()
                
                with tab2:
                    self.render_performance_analysis()
                
                with tab3:
                    self.render_advanced_risk_analytics()
                
                with tab4:
                    self.render_advanced_portfolio_optimization()
                
                with tab5:
                    self.render_backtesting()
                
                with tab6:
                    self.render_stress_testing()
            
            # Enhanced footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #94a3b8; font-size: 0.9rem;">
                <p>⚡ QuantEdge Pro v4.0 Enhanced | Advanced Portfolio Analytics Platform</p>
                <p>🎯 Featuring Advanced VaR/CVaR/ES Analytics & Enhanced Visualizations</p>
                <p>© 2024 QuantEdge Technologies | Institutional Grade Analytics</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            error_analysis = ErrorAnalyzer.analyze_error(e, "Enhanced application")
            st.error("""
            ## 🚨 Enhanced Application Error
            
            The advanced analytics engine encountered an error. Please try:
            
            1. Reducing the number of assets
            2. Adjusting the date range
            3. Using simpler optimization methods
            4. Checking data availability
            
            Contact support for advanced troubleshooting.
            """)
            
            with st.expander("Technical Details (For Support)"):
                st.code(f"""
                Error Type: {error_analysis['error_type']}
                Error Message: {error_analysis['error_message']}
                
                Enhanced Features: Advanced VaR/CVaR/ES Analytics
                
                Traceback:
                {traceback.format_exc()}
                """)

# ============================================================================
# ENHANCED MAIN ENTRY POINT
# ============================================================================

def main_enhanced():
    """Enhanced main application entry point."""
    try:
        # Initialize enhanced application
        app = EnhancedQuantEdgeProApp()
        
        # Run enhanced application
        app.run_enhanced()
        
    except Exception as e:
        # Enhanced error handling
        error_analysis = ErrorAnalyzer.analyze_error(e, "Enhanced application runtime")
        
        st.error("""
        ## ⚡ Enhanced Application Error
        
        The advanced visualization engine encountered a critical error.
        
        **Quick Fixes:**
        - Clear browser cache and refresh
        - Reduce asset universe size
        - Check internet connectivity
        - Verify ticker symbols
        
        **Advanced Support:**
        Contact QuantEdge Pro support for advanced visualization issues.
        """)
        
        with st.expander("🔧 Advanced Debugging Information"):
            st.json(error_analysis)
            
            st.code(f"""
            Library Status: {LIBRARY_STATUS}
            Python Version: {sys.version}
            Streamlit Version: {st.__version__}
            
            Full Traceback:
            {traceback.format_exc()}
            """)

# Run the enhanced application
if __name__ == "__main__":
    main_enhanced()

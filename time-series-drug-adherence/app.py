import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Drug Adherence Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stAlert {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">💊 Drug Refill Adherence Forecasting System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Load data
@st.cache_data
def load_data():
    """Load the monthly adherence data"""
    df = pd.read_csv("data/monthly_overall.csv")
    df['month_start'] = pd.to_datetime(df['month_start'])
    df_valid = df[df['refill_adherence'].notna()].copy()
    return df_valid

try:
    data = load_data()
    st.sidebar.success(f"✓ Data loaded: {len(data)} months")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar parameters
st.sidebar.subheader("Model Parameters")
train_size = st.sidebar.slider(
    "Training Data Size (%)", 
    min_value=60, 
    max_value=90, 
    value=80, 
    step=5,
    help="Percentage of data to use for training"
)

auto_params = st.sidebar.checkbox(
    "Auto-select ARIMA parameters", 
    value=True,
    help="Automatically find best (p,d,q) parameters"
)

if not auto_params:
    st.sidebar.markdown("**Manual ARIMA Parameters:**")
    p = st.sidebar.number_input("p (AR order)", min_value=0, max_value=5, value=1)
    d = st.sidebar.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
    q = st.sidebar.number_input("q (MA order)", min_value=0, max_value=5, value=1)
    manual_order = (p, d, q)

forecast_periods = st.sidebar.slider(
    "Forecast Periods (months)", 
    min_value=1, 
    max_value=6, 
    value=1,
    help="Number of months to forecast ahead"
)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "📈 Exploratory Analysis", 
    "🤖 Model Training", 
    "🔮 Forecast", 
    "📉 Performance"
])

# Prepare data
data_ts = data.set_index('month_start').sort_index()
target_col = 'refill_adherence'
n_train = int(len(data_ts) * train_size / 100)
train_data = data_ts.iloc[:n_train]
test_data = data_ts.iloc[n_train:]

y_train = train_data[target_col]
y_test = test_data[target_col] if len(test_data) > 0 else None

# TAB 1: Data Overview
with tab1:
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Months", 
            len(data_ts),
            help="Total months with valid adherence data"
        )
    
    with col2:
        st.metric(
            "Training Months", 
            len(train_data),
            help="Months used for model training"
        )
    
    with col3:
        st.metric(
            "Test Months", 
            len(test_data),
            help="Months used for validation"
        )
    
    with col4:
        st.metric(
            "Avg Adherence", 
            f"{data_ts[target_col].mean():.2f}%",
            help="Average refill adherence across all months"
        )
    
    st.markdown("---")
    
    # Time series plot
    st.subheader("Refill Adherence Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(train_data.index, train_data[target_col], 'o-', 
            linewidth=2, markersize=6, color='steelblue', label='Training Data')
    
    if len(test_data) > 0:
        ax.plot(test_data.index, test_data[target_col], 'o-', 
                linewidth=2, markersize=6, color='orange', label='Test Data')
        ax.axvline(x=test_data.index[0], color='red', linestyle='--', 
                   linewidth=2, alpha=0.5, label='Train/Test Split')
    
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Refill Adherence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Drug Refill Adherence Time Series', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Data table
    st.subheader("Recent Data")
    display_cols = ['refill_adherence', 'total_rx', 'filled_rx', 'unique_patients', 'status_coverage']
    st.dataframe(
        data_ts[display_cols].tail(10).style.format({
            'refill_adherence': '{:.2f}%',
            'status_coverage': '{:.2f}%',
            'total_rx': '{:.0f}',
            'filled_rx': '{:.0f}',
            'unique_patients': '{:.0f}'
        }),
        use_container_width=True
    )

# TAB 2: Exploratory Analysis
with tab2:
    st.header("📈 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stationarity Test")
        adf_result = adfuller(y_train, autolag='AIC')
        
        st.write("**Augmented Dickey-Fuller Test:**")
        st.write(f"- ADF Statistic: `{adf_result[0]:.4f}`")
        st.write(f"- p-value: `{adf_result[1]:.4f}`")
        
        if adf_result[1] < 0.05:
            st.success("✓ Series is **STATIONARY** (p-value < 0.05)")
            st.info("Can use ARIMA models without differencing")
        else:
            st.warning("⚠ Series is **NON-STATIONARY** (p-value >= 0.05)")
            st.info("May need differencing for ARIMA models")
        
        st.write("**Descriptive Statistics:**")
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Range'],
            'Value': [
                f"{y_train.mean():.2f}%",
                f"{y_train.std():.2f}%",
                f"{y_train.min():.2f}%",
                f"{y_train.max():.2f}%",
                f"{y_train.max() - y_train.min():.2f}%"
            ]
        })
        st.table(stats_df)
    
    with col2:
        st.subheader("Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y_train, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(y_train.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y_train.mean():.2f}%')
        ax.set_xlabel('Refill Adherence (%)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title('Adherence Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
    
    # ACF and PACF
    st.subheader("Autocorrelation Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    plot_acf(y_train, lags=min(10, len(y_train)//2-1), ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=11, fontweight='bold')
    
    plot_pacf(y_train, lags=min(10, len(y_train)//2-1), ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("📌 **ACF** shows correlation with past values (useful for MA order). **PACF** shows direct correlation (useful for AR order).")

# TAB 3: Model Training
with tab3:
    st.header("🤖 ARIMA Model Training")
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training ARIMA model..."):
            
            if auto_params:
                # Grid search for best parameters
                st.subheader("Searching for Best Parameters...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                p_values = range(0, 4)
                d_values = range(0, 2)
                q_values = range(0, 4)
                
                best_aic = np.inf
                best_order = None
                best_model = None
                
                total_combinations = len(p_values) * len(d_values) * len(q_values)
                current = 0
                
                results = []
                
                for p, d, q in product(p_values, d_values, q_values):
                    current += 1
                    progress_bar.progress(current / total_combinations)
                    status_text.text(f"Testing ARIMA({p},{d},{q})...")
                    
                    try:
                        model = ARIMA(y_train, order=(p, d, q))
                        fitted = model.fit()
                        
                        results.append({
                            'p': p, 'd': d, 'q': q,
                            'AIC': fitted.aic,
                            'BIC': fitted.bic
                        })
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_model = fitted
                    except:
                        continue
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✓ Best ARIMA model: **ARIMA{best_order}**")
                st.info(f"AIC: `{best_aic:.2f}`")
                
                # Show top 5 models
                results_df = pd.DataFrame(results).sort_values('AIC').head(5)
                st.subheader("Top 5 Models by AIC")
                st.dataframe(
                    results_df.style.format({'AIC': '{:.2f}', 'BIC': '{:.2f}'}),
                    use_container_width=True
                )
                
            else:
                # Use manual parameters
                st.info(f"Training ARIMA{manual_order}...")
                model = ARIMA(y_train, order=manual_order)
                best_model = model.fit()
                best_order = manual_order
                st.success(f"✓ Model trained: **ARIMA{best_order}**")
                st.info(f"AIC: `{best_model.aic:.2f}`")
            
            # Store in session state
            st.session_state['arima_model'] = best_model
            st.session_state['arima_order'] = best_order
            
            # Model summary
            st.subheader("Model Summary")
            st.text(str(best_model.summary()))
            
            # Generate predictions on test set if available
            if len(test_data) > 0:
                forecast = best_model.get_forecast(steps=len(y_test))
                predictions = forecast.predicted_mean.values
                conf_int = forecast.conf_int().values
                
                # Calculate metrics
                mae = mean_absolute_error(y_test.values, predictions)
                rmse = np.sqrt(mean_squared_error(y_test.values, predictions))
                mape = mean_absolute_percentage_error(y_test.values, predictions) * 100
                
                st.session_state['test_predictions'] = predictions
                st.session_state['test_conf_int'] = conf_int
                st.session_state['test_metrics'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                
                st.success("✓ Model validated on test set")
    
    else:
        st.info("👆 Click 'Train Model' to start training the ARIMA model")

# TAB 4: Forecast
with tab4:
    st.header("🔮 Future Forecast")
    
    if 'arima_model' in st.session_state:
        model = st.session_state['arima_model']
        order = st.session_state['arima_order']
        
        st.info(f"Using trained model: **ARIMA{order}**")
        
        # Generate forecast
        forecast = model.get_forecast(steps=forecast_periods)
        forecast_mean = forecast.predicted_mean.values
        forecast_conf_int = forecast.conf_int().values
        
        # Create future dates
        last_date = y_train.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=forecast_periods, 
            freq='MS'
        )
        
        # Display forecast table
        st.subheader("Forecast Results")
        forecast_df = pd.DataFrame({
            'Month': future_dates.strftime('%Y-%m'),
            'Predicted Adherence (%)': forecast_mean,
            'Lower 95% CI': forecast_conf_int[:, 0],
            'Upper 95% CI': forecast_conf_int[:, 1]
        })
        
        st.dataframe(
            forecast_df.style.format({
                'Predicted Adherence (%)': '{:.2f}',
                'Lower 95% CI': '{:.2f}',
                'Upper 95% CI': '{:.2f}'
            }).background_gradient(subset=['Predicted Adherence (%)'], cmap='RdYlGn', vmin=70, vmax=100),
            use_container_width=True
        )
        
        # Visualization
        st.subheader("Forecast Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Historical data
        ax.plot(y_train.index, y_train.values, 'o-', linewidth=2, markersize=6,
                color='steelblue', label='Historical Data')
        
        # Test data if available
        if len(test_data) > 0 and 'test_predictions' in st.session_state:
            ax.plot(y_test.index, y_test.values, 'o-', linewidth=2, markersize=6,
                    color='orange', label='Actual (Test)', zorder=10)
            ax.plot(y_test.index, st.session_state['test_predictions'], 's--', 
                    linewidth=2, markersize=7, color='green', label='Predicted (Test)', alpha=0.7)
        
        # Forecast
        ax.plot(future_dates, forecast_mean, '^-', linewidth=2.5, markersize=8,
                color='red', label='Forecast', zorder=10)
        ax.fill_between(future_dates, forecast_conf_int[:, 0], forecast_conf_int[:, 1],
                        alpha=0.3, color='red', label='95% Confidence Interval')
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Refill Adherence (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'ARIMA{order} Forecast - Next {forecast_periods} Month(s)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Adherence alerts
        st.subheader("📊 Adherence Alerts")
        for i, (date, pred) in enumerate(zip(future_dates, forecast_mean)):
            if pred < 80:
                st.error(f"⚠️ **{date.strftime('%B %Y')}**: Predicted adherence ({pred:.2f}%) is **below 80%** threshold")
            elif pred > 95:
                st.success(f"✓ **{date.strftime('%B %Y')}**: Excellent adherence ({pred:.2f}%) - **above 95%**")
            else:
                st.info(f"✓ **{date.strftime('%B %Y')}**: Good adherence ({pred:.2f}%) - within 80-95% range")
        
        # Download forecast
        st.subheader("💾 Download Results")
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name=f"adherence_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("⚠️ Please train the model first in the 'Model Training' tab")

# TAB 5: Performance
with tab5:
    st.header("📉 Model Performance")
    
    if 'test_metrics' in st.session_state and len(test_data) > 0:
        metrics = st.session_state['test_metrics']
        predictions = st.session_state['test_predictions']
        conf_int = st.session_state['test_conf_int']
        
        # Metrics display
        st.subheader("Performance Metrics on Test Set")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE (Mean Absolute Error)", f"{metrics['MAE']:.4f}")
        with col2:
            st.metric("RMSE (Root Mean Squared Error)", f"{metrics['RMSE']:.4f}")
        with col3:
            st.metric("MAPE (Mean Absolute % Error)", f"{metrics['MAPE']:.2f}%")
        
        st.markdown("---")
        
        # Predictions vs Actual
        st.subheader("Predictions vs Actual Values")
        comparison_df = pd.DataFrame({
            'Month': y_test.index.strftime('%Y-%m'),
            'Actual': y_test.values,
            'Predicted': predictions,
            'Error': y_test.values - predictions,
            'Abs Error': np.abs(y_test.values - predictions),
            'Lower 95% CI': conf_int[:, 0],
            'Upper 95% CI': conf_int[:, 1]
        })
        
        st.dataframe(
            comparison_df.style.format({
                'Actual': '{:.2f}%',
                'Predicted': '{:.2f}%',
                'Error': '{:.2f}%',
                'Abs Error': '{:.2f}%',
                'Lower 95% CI': '{:.2f}%',
                'Upper 95% CI': '{:.2f}%'
            }).background_gradient(subset=['Abs Error'], cmap='Reds'),
            use_container_width=True
        )
        
        # Residual analysis
        st.subheader("Residual Analysis")
        residuals = y_test.values - predictions
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(y_test.index, residuals, 'o-', color='purple', linewidth=2, markersize=7)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
            ax.fill_between(y_test.index, residuals, 0, alpha=0.3, color='purple')
            ax.set_xlabel('Month', fontsize=10, fontweight='bold')
            ax.set_ylabel('Residual', fontsize=10, fontweight='bold')
            ax.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(residuals, bins=10, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(residuals), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(residuals):.2f}')
            ax.set_xlabel('Residual', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Residual statistics
        st.subheader("Residual Statistics")
        resid_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Std Dev', 'Min', 'Max'],
            'Value': [
                f"{np.mean(residuals):.4f}",
                f"{np.std(residuals):.4f}",
                f"{np.min(residuals):.4f}",
                f"{np.max(residuals):.4f}"
            ]
        })
        st.table(resid_stats)
        
    else:
        st.warning("⚠️ No test data available or model not trained yet")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Drug Refill Adherence Forecasting System | Built with Streamlit & ARIMA</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    unsafe_allow_html=True
)


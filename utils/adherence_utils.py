"""
Helper functions for Drug Adherence Time-Series dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from itertools import product


def load_adherence_data() -> pd.DataFrame:
    """
    Load the monthly adherence data
    
    Returns:
        DataFrame with adherence data
    """
    data_path = Path("time-series-drug-adherence/data/monthly_overall.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    df['month_start'] = pd.to_datetime(df['month_start'])
    df_valid = df[df['refill_adherence'].notna()].copy()
    
    return df_valid


def create_ts_features(df: pd.DataFrame, target_col: str, n_lags: int = 3) -> pd.DataFrame:
    """
    Create time series features for XGBoost
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        n_lags: Number of lag features to create
    
    Returns:
        DataFrame with additional features
    """
    df_features = df.copy()
    
    # Create lag features
    for lag in range(1, n_lags + 1):
        df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Rolling statistics
    df_features['rolling_mean_3'] = df_features[target_col].shift(1).rolling(window=3).mean()
    df_features['rolling_std_3'] = df_features[target_col].shift(1).rolling(window=3).std()
    
    # Time-based features
    if isinstance(df_features.index, pd.DatetimeIndex):
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
    else:
        df_features['month'] = pd.to_datetime(df_features.index).month
        df_features['quarter'] = pd.to_datetime(df_features.index).quarter
    
    return df_features


def train_xgboost_model(train_data: pd.DataFrame, target_col: str, 
                        n_lags: int = 3) -> Tuple[xgb.XGBRegressor, list]:
    """
    Train XGBoost model for adherence forecasting
    
    Args:
        train_data: Training data
        target_col: Target column name
        n_lags: Number of lag features
    
    Returns:
        Tuple of (trained model, feature names)
    """
    df_features = create_ts_features(train_data, target_col, n_lags)
    
    lag_features = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'month', 'quarter']
    xgb_features = lag_features + ['total_rx', 'status_coverage', 'unique_patients']
    
    # Remove rows with NaN
    df_clean = df_features.dropna()
    
    X_train = df_clean[xgb_features]
    y_train = df_clean[target_col]
    
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    return model, xgb_features


def train_arima_model(train_data: pd.Series, max_p: int = 2, 
                      max_d: int = 1, max_q: int = 2) -> Tuple[ARIMA, tuple]:
    """
    Train ARIMA model with automatic parameter selection
    
    Args:
        train_data: Training time series
        max_p: Maximum p parameter
        max_d: Maximum d parameter
        max_q: Maximum q parameter
    
    Returns:
        Tuple of (fitted model, best order)
    """
    best_aic = np.inf
    best_order = None
    best_model = None
    
    for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        try:
            model = ARIMA(train_data, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
                best_model = fitted
        except:
            continue
    
    if best_model is None:
        raise ValueError("Could not fit ARIMA model with given parameters")
    
    return best_model, best_order


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with metrics
    """
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }


def generate_forecast(model: xgb.XGBRegressor, features: list, 
                     data_ts: pd.DataFrame, target_col: str,
                     forecast_periods: int, period_type: str = "monthly") -> Tuple[list, list, list, list]:
    """
    Generate multi-step forecast using XGBoost
    
    Args:
        model: Trained XGBoost model
        features: List of feature names
        data_ts: Historical time series data
        target_col: Target column name
        forecast_periods: Number of periods to forecast
        period_type: "monthly" or "bi-weekly"
    
    Returns:
        Tuple of (forecast_dates, forecasts, lower_ci, upper_ci)
    """
    forecasts = []
    forecast_dates = []
    
    # Create working copy
    data_working = data_ts.copy()
    
    # Determine date offset based on period type
    if period_type == "bi-weekly":
        date_offset = pd.DateOffset(weeks=2)
    else:
        date_offset = pd.DateOffset(months=1)
    
    for i in range(forecast_periods):
        # Create features for next period
        df_forecast = create_ts_features(data_working, target_col, n_lags=3)
        
        # Get next date
        if isinstance(data_working.index, pd.DatetimeIndex):
            next_date = data_working.index[-1] + date_offset
        else:
            next_date = pd.to_datetime(data_working.index[-1]) + date_offset
        
        forecast_dates.append(next_date)
        
        # Use last available features
        last_features = df_forecast.iloc[-1][features].values.reshape(1, -1)
        
        # Handle NaN in features
        last_features = np.nan_to_num(last_features, nan=data_working[target_col].mean())
        
        # Predict
        pred = model.predict(last_features)[0]
        forecasts.append(pred)
        
        # Update data_working with prediction for next iteration
        new_row = pd.DataFrame({target_col: [pred]}, index=[next_date])
        for col in data_working.columns:
            if col not in new_row.columns:
                new_row[col] = data_working[col].iloc[-1]
        data_working = pd.concat([data_working, new_row])
    
    # Calculate confidence intervals
    hist_std = data_ts[target_col].std()
    lower_ci = [f - 1.96 * hist_std for f in forecasts]
    upper_ci = [f + 1.96 * hist_std for f in forecasts]
    
    return forecast_dates, forecasts, lower_ci, upper_ci


def get_adherence_status(value: float) -> str:
    """
    Get status label for adherence value
    
    Args:
        value: Adherence percentage
    
    Returns:
        Status string with emoji
    """
    if value >= 95:
        return "🟢 Excellent"
    elif value >= 85:
        return "🟡 Good"
    elif value >= 80:
        return "🟠 Acceptable"
    else:
        return "🔴 Action Required"


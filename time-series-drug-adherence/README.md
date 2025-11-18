# Drug Refill Adherence Forecasting System 💊

An interactive Streamlit application for forecasting drug refill adherence using ARIMA time series models.

## Features

- 📊 **Data Overview**: Visualize historical adherence trends
- 📈 **Exploratory Analysis**: Stationarity tests, ACF/PACF plots
- 🤖 **Model Training**: Automatic or manual ARIMA parameter selection
- 🔮 **Forecasting**: Generate future adherence predictions with confidence intervals
- 📉 **Performance Metrics**: Evaluate model accuracy on test data

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/edonisalijaj/Downloads/tsfda
   ```

2. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install streamlit
   ```

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage Guide

### 1. Data Overview Tab
- View historical adherence trends
- See train/test split
- Review recent data points

### 2. Exploratory Analysis Tab
- Check stationarity with ADF test
- Analyze autocorrelation (ACF/PACF)
- View adherence distribution

### 3. Model Training Tab
- **Auto-select parameters**: Let the app find the best ARIMA(p,d,q)
- **Manual parameters**: Specify your own p, d, q values
- Click "Train Model" to start training
- View model summary and validation metrics

### 4. Forecast Tab
- Generate future adherence predictions
- View confidence intervals
- See adherence alerts (below 80%, above 95%)
- Download forecast results as CSV

### 5. Performance Tab
- View MAE, RMSE, MAPE metrics
- Compare predictions vs actual values
- Analyze residuals

## Configuration Options (Sidebar)

- **Training Data Size**: Adjust train/test split (60-90%)
- **Auto-select ARIMA parameters**: Toggle automatic parameter search
- **Manual Parameters**: Set p, d, q values manually
- **Forecast Periods**: Choose how many months to forecast (1-6)

## Data Requirements

The app expects a CSV file at `data/monthly_overall.csv` with at least these columns:
- `month_start`: Date column (YYYY-MM-DD format)
- `refill_adherence`: Target variable (percentage)
- `total_rx`: Total prescriptions
- `filled_rx`: Filled prescriptions
- `unique_patients`: Number of unique patients
- `status_coverage`: Status coverage percentage

## Tips

- Start with **80% training data** for balanced results
- Use **auto-select parameters** for best model selection
- Check the **stationarity test** in EDA tab to understand your data
- Review **performance metrics** before making decisions
- Download forecasts for reporting purposes

## Technical Details

- **Model**: ARIMA (AutoRegressive Integrated Moving Average)
- **Parameter Selection**: Grid search over p∈[0,3], d∈[0,1], q∈[0,3]
- **Selection Criterion**: AIC (Akaike Information Criterion)
- **Confidence Intervals**: 95% prediction intervals

## Troubleshooting

**App won't start:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that `data/monthly_overall.csv` exists

**Model training fails:**
- Reduce the parameter search space
- Try manual parameters instead
- Check for missing data in the adherence column

**Poor forecast accuracy:**
- Increase training data size
- Try different ARIMA parameters
- Check if data has strong seasonality (consider SARIMA)

## Project Structure

```
tsfda/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── monthly_overall.csv         # Input data
│   └── monthly_atc3.csv           # Alternative data
└── notebooks/
    ├── data_preprocessing.ipynb    # Data preparation
    └── model_selection.ipynb       # Model experiments
```

## License

Educational and research purposes.


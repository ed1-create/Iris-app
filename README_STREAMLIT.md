# Healthcare Analytics Dashboard

Professional Streamlit application combining Patient Segmentation and Drug Adherence Forecasting analytics.

## Features

### Tab 1: Patient Segmentation
- **Cohort Selection**: Switch between I10 (Hypertension) and Z01 (Preventive Care) cohorts
- **Executive Summary**: Key metrics dashboard with cluster statistics
- **Cluster Overview**: Interactive visualizations of cluster sizes and distributions
- **Cluster Explorer**: Detailed exploration of individual clusters with:
  - Cluster profiles and characteristics
  - Clinical metrics (SBP, BMI, Age)
  - Comorbidity prevalence
  - Utilization patterns
- **Visualizations Gallery**: Embedded images including:
  - Silhouette score comparisons
  - Bootstrap stability analysis
  - Evaluation metrics dashboard
  - Clinical profile heatmaps
  - PCA visualizations
- **Comparative Analysis**: Side-by-side cluster comparisons
- **Patient Lookup**: Search for individual patients and view cluster assignments

### Tab 2: Drug Adherence Forecasting
- **Executive Dashboard**: Real-time adherence metrics and trends
- **Model Performance**: Train and compare XGBoost and ARIMA models
- **Multi-Period Forecasting**: Generate 1-12 month forecasts with confidence intervals
- **Interactive Visualizations**: Historical trends and forecast projections
- **Export Capabilities**: Download forecasts as CSV

## Installation

1. **Navigate to project directory:**
   ```bash
   cd /Users/edonisalijaj/Downloads/patient-segmentation
   ```

2. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Data Requirements

The application expects the following data files:

### Patient Segmentation Data:
- `notebooks/outputs/data/i10_cluster_profiles.csv`
- `notebooks/outputs/data/i10_cluster_assignments.csv`
- `notebooks/outputs/data/i10_clustering_evaluation.csv`
- `notebooks/outputs/data/z01_cluster_profiles.csv`
- `notebooks/outputs/data/z01_cluster_assignments.csv`
- `notebooks/outputs/data/z01_clustering_evaluation.csv`

### Visualizations:
- `notebooks/outputs/visualizations/i10_clustering/*.png`
- `notebooks/outputs/visualizations/z01_clustering/*.png`

### Drug Adherence Data:
- `time-series-drug-adherence/data/monthly_overall.csv`

## Usage Guide

### Patient Segmentation Tab

1. **Select Cohort**: Use the sidebar to choose between I10 (Hypertension) or Z01 (Preventive Care)
2. **View Summary**: Review key metrics in the Executive Summary section
3. **Explore Clusters**: Use the Cluster Explorer dropdown to view detailed cluster profiles
4. **View Visualizations**: Browse the Visualizations Gallery to see analysis charts
5. **Compare Clusters**: Use the Comparative Analysis section to compare cluster characteristics
6. **Lookup Patients**: Enter a patient ID in the Patient Lookup section to find cluster assignments

### Drug Adherence Tab

1. **Configure Settings**: Adjust forecast horizon and training data percentage in the sidebar
2. **Train Models**: Click "Train & Compare Models" to train XGBoost and/or ARIMA models
3. **View Performance**: Review model performance metrics and predictions
4. **Generate Forecast**: View multi-period forecasts with confidence intervals
5. **Export Results**: Download forecast results as CSV

## Technical Details

### Technologies Used
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive charts
- **XGBoost**: Machine learning model
- **Statsmodels**: Time series analysis (ARIMA)

### Performance Optimizations
- Data caching with `@st.cache_data` decorators
- Lazy loading of visualizations
- Efficient data filtering and processing

### Architecture
- **app.py**: Main application file
- **utils/segmentation_utils.py**: Patient segmentation helper functions
- **utils/adherence_utils.py**: Drug adherence helper functions

## Troubleshooting

**App won't start?**
```bash
# Kill any existing Streamlit processes
kill $(lsof -ti:8501)
# Restart the app
streamlit run app.py
```

**Missing data files?**
- Ensure all required CSV files exist in the specified directories
- Run the segmentation notebooks to generate cluster data if missing

**Visualizations not showing?**
- Check that PNG files exist in the visualization directories
- Verify file paths are correct

**Model training errors?**
- Ensure sufficient training data (at least 12 months recommended)
- Check that all required features are present in the data

## Support

For questions or issues, contact your analytics team.

## License

Internal use only.


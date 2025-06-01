import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def get_prophet_predict(data: pd.DataFrame, forecast_periods: int, regressors=None):
    # Ensure regressors is a list
    regressors = regressors or []
    # Check input 
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must have a 'Close' column")
    missing = [r for r in regressors if r not in data.columns]
    if missing:
        raise ValueError(f"Missing regressors: {missing}")

    #Reset index, ensure 'ds' is the date column, and select necessary columns
    df = (
        data.copy()
        .rename_axis('Time')
        .reset_index()[['Time', 'Close'] + regressors]
        .rename(columns={'Time': 'ds', 'Close': 'y'})
    )
    #Convert 'ds' column to datetime
    df['ds'] = pd.to_datetime(df['ds'])
    #Split into train/test sets (time series, no shuffle)
    train_df, test_df = train_test_split(df, shuffle=False, test_size=0.02)
    #Initialize and train the Prophet model
    model = Prophet()
    for reg in regressors:
        model.add_regressor(reg)
    model.fit(train_df)
    # Create future DataFrame (including test period and periods to forecast)
    future = model.make_future_dataframe(periods=len(test_df) + forecast_periods)
    # Merge additional regressor data if any
    if regressors:
        reg_df = df[['ds'] + regressors].copy()
        future = future.merge(reg_df, on='ds', how='left')
        future[regressors] = future[regressors].fillna(method='ffill')
    # Forecast
    forecast = model.predict(future)
    # Extract actual values and predictions
    train_len, test_len = len(train_df), len(test_df)
    true_values = test_df['y'].values
    validation_pred = forecast['yhat'].iloc[train_len:train_len+test_len].values
    future_pred = forecast['yhat'].iloc[train_len+test_len:train_len+test_len+forecast_periods].values

    return true_values, validation_pred, model, future_pred

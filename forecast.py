import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
import streamlit as st

@st.cache_data(show_spinner=False)
def arima_forecast(dates, series, horizon=7, smooth=True, window=3, z_thresh=3):
    """
    Optimized ARIMA forecast with caching and fallback to Linear Regression if ARIMA fails.
    """
    # Prepare data
    s = pd.Series(series.values, index=pd.to_datetime(dates)).asfreq('D')
    s = s.interpolate().fillna(method='bfill').fillna(method='ffill')

    # Outlier handling
    z = np.abs((s - s.mean()) / s.std())
    outliers = z > z_thresh
    if outliers.any():
        s[outliers] = np.nan
        s = s.interpolate()

    # Optional smoothing
    if smooth:
        s = s.rolling(window=window, min_periods=1, center=True).mean()

    # Step 1: Try Auto ARIMA
    try:
        stepwise = auto_arima(
            s,
            start_p=1, start_q=1, max_p=3, max_q=3,
            seasonal=False, m=1,
            error_action='ignore', suppress_warnings=True,
            stepwise=True, maxiter=25
        )
        order = stepwise.order
    except Exception as e:
        st.warning(f"⚠️ Auto ARIMA failed: {e}. Using fallback model.")
        order = None

    # Step 2: Try fitting SARIMAX if ARIMA worked
    if order:
        try:
            model = SARIMAX(s, order=order, enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=100)
            fc = res.get_forecast(steps=horizon)
            mean_fc = fc.predicted_mean
            index = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
            df_fc = pd.DataFrame({'time': index, 'forecast': mean_fc.values})
            summary = f"✅ ARIMA order={order}, AIC={res.aic:.2f}, Outliers handled={outliers.sum()}."
            return df_fc, summary
        except Exception as e:
            st.warning(f"⚠️ SARIMAX fit failed: {e}. Switching to fallback model.")

    # Step 3: Fallback Linear Regression model (simple trend)
    X = np.arange(len(s)).reshape(-1, 1)
    y = s.values
    model_lr = LinearRegression()
    model_lr.fit(X, y)

    future_X = np.arange(len(s), len(s) + horizon).reshape(-1, 1)
    preds = model_lr.predict(future_X)
    index = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
    df_fc = pd.DataFrame({'time': index, 'forecast': preds})

    summary = f"✅ Fallback Linear Regression used. Outliers handled={outliers.sum()}."
    return df_fc, summary

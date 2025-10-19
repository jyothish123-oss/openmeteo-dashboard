import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_weather(lat, lon, days=90):
    """
    Fetch daily historical weather data from Open-Meteo API.
    Automatically switches between archive and forecast endpoints.
    Returns a cleaned pandas DataFrame.
    """
    today = datetime.utcnow().date()
    end = today
    start = end - timedelta(days=days)

    # Use correct API endpoint
    BASE = (
        "https://api.open-meteo.com/v1/forecast"
        if start >= today
        else "https://archive-api.open-meteo.com/v1/archive"
    )

    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start.isoformat(),
        'end_date': end.isoformat(),
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
        'timezone': 'UTC'
    }

    try:
        r = requests.get(BASE, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

    # Extract data
    daily = j.get('daily', {})
    df = pd.DataFrame({
        'time': pd.to_datetime(daily.get('time', [])),
        'temperature_2m_max': daily.get('temperature_2m_max', []),
        'temperature_2m_min': daily.get('temperature_2m_min', []),
        'precipitation_sum': daily.get('precipitation_sum', [])
    })

    if not df.empty:
        df['temperature_2m'] = df[['temperature_2m_max', 'temperature_2m_min']].mean(axis=1)
        df = df.sort_values('time').reset_index(drop=True)

    return df

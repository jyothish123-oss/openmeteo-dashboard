import streamlit as st
from data import fetch_historical_weather
from forecast import arima_forecast
from utils import format_df_for_display, download_link
import plotly.express as px
from streamlit_folium import st_folium
import folium
import pandas as pd

# --- Page setup ---
st.set_page_config(page_title="OpenMeteo Dashboard", layout="wide")
st.title("🌤️ OpenMeteo Dashboard — Weather Analytics & Forecasting")

# --- Sidebar inputs ---
st.sidebar.header("Location & Forecast Options")
city = st.sidebar.text_input("City name (for label only)", value="Mumbai")
lat = st.sidebar.number_input("Latitude", value=19.0760, format="%.6f")
lon = st.sidebar.number_input("Longitude", value=72.8777, format="%.6f")
days = st.sidebar.slider("Historical days to fetch", min_value=7, max_value=365, value=90)
forecast_horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=7)
uploaded = st.sidebar.file_uploader("Or upload CSV (time,value) for forecasting", type=['csv'])

# --- Sidebar buttons ---
from streamlit.runtime.scriptrunner import rerun

if st.sidebar.button("Load & Forecast Data"):
    st.session_state["run_forecast"] = True

if st.sidebar.button("Clear Results"):
    st.session_state.clear()
    rerun()

# --- Main logic ---
if st.session_state.get("run_forecast", False):
    df = pd.DataFrame()

    # --- Uploaded CSV ---
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, parse_dates=[0])
            if df.shape[1] != 2:
                st.error("CSV must have exactly 2 columns: time,value")
            else:
                df.columns = ["time", "value"]
                df = df.sort_values("time").reset_index(drop=True)

                st.subheader("📈 Uploaded Time Series Data")
                st.dataframe(format_df_for_display(df))

                with st.spinner("Training ARIMA model and generating forecast..."):
                    fc_df, summary = arima_forecast(df['time'], df['value'], int(forecast_horizon))

                st.subheader("🔮 Forecast Results")
                st.write(summary)
                st.line_chart(fc_df.set_index('time')['forecast'])
                st.dataframe(fc_df)
                st.markdown(download_link(fc_df, f"forecast_{city}.csv"), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

    # --- Fetch data from Open-Meteo API ---
    else:
        with st.spinner("Fetching historical weather data from Open-Meteo..."):
            df = fetch_historical_weather(lat, lon, days)

        if df.empty:
            st.error("No data fetched. Please check your coordinates or try again later.")
        else:
            st.subheader(f"🌍 Historical Weather — {city} ({lat:.4f}, {lon:.4f})")
            st.dataframe(format_df_for_display(df))

            # Temperature plot
            if 'temperature_2m' in df.columns:
                fig = px.line(df, x='time', y='temperature_2m', title='Temperature (°C)')
                st.plotly_chart(fig, use_container_width=True)

                # Forecast
                with st.spinner("Training ARIMA Forecast..."):
                    try:
                        fc_df, summary = arima_forecast(df['time'], df['temperature_2m'], int(forecast_horizon))
                        st.subheader("🔮 Forecast (Temperature °C)")
                        st.write(summary)
                        st.line_chart(fc_df.set_index('time')['forecast'])
                        st.dataframe(fc_df)
                        st.markdown(download_link(fc_df, f"forecast_{city}.csv"), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during forecast: {e}")

            # Map visualization
            st.subheader("🗺️ Location Map")
            m = folium.Map(location=[lat, lon], zoom_start=6)
            folium.Marker([lat, lon], popup=city).add_to(m)
            st_folium(m, width=700)

else:
    st.info("Enter location details in the sidebar and click **'Load & Forecast Data'**, or upload a CSV file.")



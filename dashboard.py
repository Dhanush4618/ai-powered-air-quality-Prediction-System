import streamlit as st
import requests
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Real-Time Air Quality Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .good { color: #00cc96; }
    .moderate { color: #ffa15c; }
    .unhealthy { color: #ef553b; }
    .very-unhealthy { color: #ab63fa; }
    .hazardous { color: #ff6692; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌍 Real-Time Air Quality Prediction</h1>', unsafe_allow_html=True)
st.markdown("Live AQI monitoring using Random Forest model and Open-Meteo API")

# Sidebar for configuration
st.sidebar.title("Configuration")
api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 10, 60, 30)
use_sample_data = st.sidebar.checkbox("Use sample data (if API fails)")

# Sample data for testing
sample_data = {
    "AQI_Predicted": 45.5,
    "status": "Good 😊",
    "pollutants": {
        "pm2_5": 25.0, "pm10": 45.0, "no2": 12.0, 
        "so2": 5.0, "co": 0.5, "o3": 45.0, "nh3": 0.0
    },
    "location": "Delhi, India",
    "timestamp": "Sample Data"
}

# Initialize session state for storing historical data
if 'aqi_history' not in st.session_state:
    st.session_state.aqi_history = []
if 'pollutant_history' not in st.session_state:
    st.session_state.pollutant_history = []

def fetch_data():
    """Fetch data from API or use sample data"""
    if use_sample_data:
        return sample_data
    
    try:
        response = requests.get(f"{api_url}/predict", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return sample_data
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return sample_data

def get_status_color(status):
    """Get color based on AQI status"""
    if "Good" in status:
        return "green"
    elif "Moderate" in status:
        return "orange"
    elif "Unhealthy" in status:
        return "red"
    elif "Very Unhealthy" in status:
        return "purple"
    else:
        return "maroon"

# Main dashboard layout
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("📊 Live AQI Monitoring")
    
    # Fetch current data
    data = fetch_data()
    
    # Display AQI metric with color coding
    status_color = get_status_color(data["status"])
    
    st.metric(
        label="Current AQI",
        value=f"{data['AQI_Predicted']}",
        delta=data["status"],
        delta_color="off"
    )
    
    # AQI gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = data["AQI_Predicted"],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AQI Level"},
        gauge = {
            'axis': {'range': [None, 300]},
            'bar': {'color': status_color},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 150], 'color': "orange"},
                {'range': [150, 200], 'color': "red"},
                {'range': [200, 300], 'color': "darkred"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': data["AQI_Predicted"]
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.subheader("📈 AQI History")
    
    # Update history
    current_time = time.strftime("%H:%M:%S")
    st.session_state.aqi_history.append({
        "time": current_time,
        "aqi": data["AQI_Predicted"]
    })
    
    # Keep only last 20 readings
    if len(st.session_state.aqi_history) > 20:
        st.session_state.aqi_history.pop(0)
    
    # Create history chart
    if st.session_state.aqi_history:
        history_df = pd.DataFrame(st.session_state.aqi_history)
        st.line_chart(history_df.set_index("time"))

with col3:
    st.subheader("🌫️ Pollutant Levels")
    
    pollutants = data["pollutants"]
    
    # Display pollutant levels
    for poll_name, value in pollutants.items():
        st.progress(
            min(value / 100, 1.0), 
            text=f"{poll_name.upper()}: {value}"
        )

# Pollutant details section
st.subheader("🔍 Detailed Pollutant Analysis")

poll_cols = st.columns(4)
pollutants = data["pollutants"]

with poll_cols[0]:
    st.metric("PM2.5", f"{pollutants['pm2_5']} μg/m³")
with poll_cols[1]:
    st.metric("PM10", f"{pollutants['pm10']} μg/m³")
with poll_cols[2]:
    st.metric("NO₂", f"{pollutants['no2']} μg/m³")
with poll_cols[3]:
    st.metric("O₃", f"{pollutants['o3']} μg/m³")

# Location and timestamp info
st.info(f"📍 **Location**: {data.get('location', 'Unknown')} | ⏰ **Last Updated**: {current_time}")

# Auto-refresh
st.sidebar.info(f"Next update in {refresh_rate} seconds")
time.sleep(refresh_rate)
st.rerun()
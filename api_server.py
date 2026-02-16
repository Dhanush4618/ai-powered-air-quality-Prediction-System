from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import requests
import numpy as np
import os
from typing import Dict, Any

app = FastAPI(title="Air Quality Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model with error handling
try:
    # Use relative path
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def fetch_live_data(lat=28.6139, lon=77.2090) -> Dict[str, float]:
    """Fetch live air quality data from Open-Meteo API"""
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone&timezone=auto"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        hourly = data["hourly"]
        
        # Get the latest readings
        latest = {
            "pm2_5": hourly["pm2_5"][-1] if hourly["pm2_5"] else 0,
            "pm10": hourly["pm10"][-1] if hourly["pm10"] else 0,
            "no2": hourly["nitrogen_dioxide"][-1] if hourly["nitrogen_dioxide"] else 0,
            "so2": hourly["sulphur_dioxide"][-1] if hourly["sulphur_dioxide"] else 0,
            "co": hourly["carbon_monoxide"][-1] if hourly["carbon_monoxide"] else 0,
            "o3": hourly["ozone"][-1] if hourly["ozone"] else 0,
            "nh3": 0.0  # Open-Meteo doesn't provide NH3, setting to 0
        }
        
        print("✅ Fetched live data:", latest)
        return latest
        
    except Exception as e:
        print(f"❌ Error fetching live data: {e}")
        # Return sample data as fallback
        return {
            "pm2_5": 25.0, "pm10": 45.0, "no2": 12.0, 
            "so2": 5.0, "co": 0.5, "o3": 45.0, "nh3": 0.0
        }

def get_aqi_status(aqi: float) -> str:
    """Convert AQI value to status category"""
    if aqi <= 50:
        return "Good 😊"
    elif aqi <= 100:
        return "Moderate 😐"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups 😷"
    elif aqi <= 200:
        return "Unhealthy 😞"
    elif aqi <= 300:
        return "Very Unhealthy 😨"
    else:
        return "Hazardous ☠️"

@app.get("/")
def home():
    return {
        "message": "Air Quality Prediction API is Running!",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_custom": "/predict/custom?pm25=10&pm10=20&no2=15&so2=5&co=0.5&o3=40&nh3=0"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model else "model_not_loaded",
        "model_loaded": model is not None
    }

@app.get("/predict")
def predict():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Fetch live data
        data = fetch_live_data()
        
        # Prepare features in the exact order expected by the model
        # Based on your model: ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3']
        # Note: We don't have NO data, so we'll use 0 as placeholder
        X = np.array([[
            data["pm2_5"],      # PM2.5
            data["pm10"],       # PM10  
            0.0,               # NO (not available)
            data["no2"],       # NO2
            data["nh3"],       # NH3
            data["co"],        # CO
            data["so2"],       # SO2
            data["o3"]         # O3
        ]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return {
            "AQI_Predicted": round(prediction, 2),
            "status": get_aqi_status(prediction),
            "pollutants": data,
            "location": "Delhi, India",
            "timestamp": "Live data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/predict/custom")
def predict_custom(
    pm25: float, pm10: float, no2: float, so2: float, 
    co: float, o3: float, nh3: float = 0.0
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        X = np.array([[
            pm25, 0.0, no2, nh3, co, so2, o3, pm10  # Adjusted order based on model features
        ]])
        
        prediction = model.predict(X)[0]
        
        return {
            "AQI_Predicted": round(prediction, 2),
            "status": get_aqi_status(prediction),
            "input_parameters": {
                "PM2.5": pm25, "PM10": pm10, "NO2": no2, 
                "SO2": so2, "CO": co, "O3": o3, "NH3": nh3
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
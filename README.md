## AI-Powered Air Quality Prediction System

This project is an end-to-end **air quality prediction system** powered by a machine learning model. It exposes a **FastAPI-based backend** for predictions and a **dashboard UI** for interacting with the model and visualizing results.

### Features
- **Model training**: Scripts to train and persist a prediction model (`model_training.py`, `model.pkl`).
- **REST API**: FastAPI server in `api_server.py` exposing endpoints for health checks and predictions.
- **Health checks & tools**: Utility script `check_endpoints.py` to verify API availability.
- **Dashboard**: A simple dashboard in `dashboard.py` to interact with the model and view predictions.

### Requirements
All Python dependencies are listed in `requirements.txt`.

### Setup
```bash
cd "D:\aoq pro"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Running the API server
```bash
cd "D:\aoq pro"
.venv\Scripts\activate
uvicorn api_server:app --reload
```

The API will typically be available at `http://127.0.0.1:8000`.

### Checking API endpoints
```bash
cd "D:\aoq pro"
.venv\Scripts\activate
python check_endpoints.py
```

### Running the dashboard
If your dashboard is a Streamlit app:
```bash
cd "D:\aoq pro"
.venv\Scripts\activate
streamlit run dashboard.py
```

### Project structure
```text
api_server.py       # FastAPI server exposing prediction endpoints
dashboard.py        # Dashboard UI for interacting with the model
check_endpoints.py  # Script to verify API endpoints
model_training.py   # Model training script
model.pkl           # Trained model artifact
requirements.txt    # Python dependencies
```

### Repository
This code is hosted on GitHub at:
`https://github.com/Dhanush4618/ai-powered-air-quality-Prediction-System`


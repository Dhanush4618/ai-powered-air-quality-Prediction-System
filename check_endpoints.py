import requests

endpoints = [
    ("API root", "http://127.0.0.1:8000/"),
    ("API predict", "http://127.0.0.1:8000/predict"),
    ("Streamlit", "http://localhost:8501")
]

for name, url in endpoints:
    try:
        r = requests.get(url, timeout=5)
        print(f"{name}: {url} -> {r.status_code}")
    except Exception as e:
        print(f"{name}: {url} -> ERROR: {e}")

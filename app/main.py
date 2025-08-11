# app/main.py
from fastapi import FastAPI, Request
import logging
import sqlite3
from datetime import datetime
import pandas as pd

from src.inference import get_production_model, predict_price

# Prometheus glue from metrics.py
from app.metrics import (
    install_metrics,
    metrics_endpoint,
    PREDICTIONS_TOTAL,
    INFERENCE_SECONDS,
)

# ✅ Use the Pydantic schema with validations
from app.schemas import HousingFeatures

LOG_FILE_PATH = "logs/predictions.log"
DB_FILE_PATH = "database/prediction_logs.db"

app = FastAPI(
    title="California Housing Price Predictor",
    description="Predicts median housing price and logs inputs/outputs",
    version="1.1",
)

# Install Prometheus middleware
install_metrics(app)


logging.basicConfig(
    filename=LOG_FILE_PATH, level=logging.INFO, format="%(asctime)s - %(message)s"
)


def init_db():
    conn = sqlite3.connect(DB_FILE_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input TEXT,
            prediction REAL
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


def log_to_db(input_data: dict, prediction: float):
    conn = sqlite3.connect(DB_FILE_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO logs (timestamp, input, prediction) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), str(input_data), float(prediction)),
    )
    conn.commit()
    conn.close()


@app.get("/")
def read_root():
    return {"message": "California Housing Prediction API is live."}


@app.post("/predict")
async def predict(input_data: HousingFeatures, request: Request):
    model, preprocessor = get_production_model("CaliforniaHousingModel")
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])

    # Measure inference time + increment counter
    with INFERENCE_SECONDS.time():
        prediction = predict_price(model, preprocessor, input_df)
    PREDICTIONS_TOTAL.inc()

    logging.info(f"Input: {input_dict} => Prediction: {prediction}")
    log_to_db(input_dict, prediction)
    return {"predicted_price": float(prediction)}


# Prometheus scrape endpoint
@app.get("/metrics")
def metrics():
    return metrics_endpoint()


# Default lightweight stats (keep separate from /metrics)
@app.get("/stats")
def stats():
    conn = sqlite3.connect(DB_FILE_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM logs")
    total_requests = c.fetchone()[0]
    conn.close()
    return {"total_prediction_requests": total_requests}

from fastapi import FastAPI, Request
from pydantic import BaseModel
import logging
import sqlite3
from datetime import datetime
import pandas as pd

# Import from src.inference
from src.inference import get_production_model, predict_price

# Init FastAPI
app = FastAPI(
    title="California Housing Price Predictor",
    description="Predicts median housing price and logs inputs/outputs",
    version="1.1"
)

# Input schema
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    OceanProximity: str

# Setup logging to file
logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Load production model



# Initialize SQLite DB
def init_db():
    conn = sqlite3.connect("prediction_logs.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input TEXT,
            prediction REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Save log to SQLite
def log_to_db(input_data: dict, prediction: float):
    conn = sqlite3.connect("prediction_logs.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO logs (timestamp, input, prediction) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), str(input_data), prediction)
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
    prediction = predict_price(model, preprocessor, input_df)

    # Log to file and SQLite
    logging.info(f"Input: {input_dict} => Prediction: {prediction}")
    log_to_db(input_dict, prediction)

    return {"predicted_price": prediction}

@app.get("/metrics")
def metrics():
    conn = sqlite3.connect("prediction_logs.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM logs")
    total_requests = c.fetchone()[0]
    conn.close()
    return {"total_prediction_requests": total_requests}
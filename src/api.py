import time
from fastapi import FastAPI, UploadFile, File, Request
import os
import logging
from .inference import PneumoniaModel

app = FastAPI(title="Pneumonia Detection API")

# Setup generic basic logging for M5 assignment constraints
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add Latency middleware for M5 constraint
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request: {request.method} {request.url.path} - Latency: {process_time:.4f}s")
    return response

# Initialize inference model handler
pneumonia_model = PneumoniaModel()

@app.get("/health")
def health_check():
    """M2 Requirement: Health check endpoint"""
    return {"status": "ok", "model_ready": "model.pt" in str(os.listdir("models") if os.path.exists("models") else []) }

import os

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """M2 Requirement: Prediction endpoint returning class/probabilities"""
    logger.info(f"Received inference request for file: {file.filename}")
    image_data = await file.read()
    
    try:
        prediction_result = pneumonia_model.predict(image_data)
        logger.info(f"Prediction target achieved: {prediction_result['prediction']}")
        return prediction_result
    except Exception as e:
        logger.error(f"Inference API failed: {e}")
        return {"error": str(e)}

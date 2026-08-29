# Pneumonia Detection MLOps Pipeline

This repository contains the end-to-end Machine Learning Operations (MLOps) pipeline for detecting Pneumonia in Chest X-Rays, satisfying all requirements of the Assignment 2 specification.

## Project Structure

```
.
├── .github/workflows/ci.yml # GitHub Actions Continuous Integration (M3)
├── data/                    # Dataset directory (ignored in git, managed by DVC)
├── models/                  # Trained models (ignored in git, managed by DVC)
├── src/                     # Core application source code
│   ├── api.py               # FastAPI inference service (M2, M5 Monitoring)
│   ├── inference.py         # Prediction logic extraction
│   ├── data_prep.py         # Data splitting & augmentation (M1)
│   └── train.py             # Model training with PyTorch & MLflow (M1)
├── tests/                   # Pytest automated testing suite
├── docker-compose.yml       # Docker Deployment Orchestration (M4)
├── Dockerfile               # Container spec for inference API
├── dvc.yaml                 # DVC Pipeline configuration
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup & Execution Instructions

### 1. Environment Initialization
Ensure your virtual environment is active and dependencies are fully installed:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Data & Model Pipeline (DVC & MLflow)
To reproduce the data processing and training pipeline locally:
```bash
dvc repro
```
This will automatically execute `src/data_prep.py` followed by `src/train.py`, saving the final artifact to `models/model.pt` while logging hyperparameters to a local MLflow instance.

### 3. Running the API Locally Context
To start the FastAPI server with live reloading (useful for debugging):
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
- Health Check: `http://localhost:8000/health`
- Predict Endpoint: POST to `http://localhost:8000/predict` with a file element.

### 4. Container Deployment (Docker)
To build and deploy the application purely in Docker (as orchestrated by M4 specifications):
```bash
docker-compose up -d --build
```

### 5. Running Automated Tests
Run the pytest suite to ensure endpoint health and data thresholds are valid:
```bash
pytest tests/
```

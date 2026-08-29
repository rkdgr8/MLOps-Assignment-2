import pytest
from fastapi.testclient import TestClient
from src.api import app
from PIL import Image
import io

client = TestClient(app)

def test_health_check():
    """M3 requirement: test inference function endpoint health"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_fails_without_image():
    """Test standard fallback when predict is called improperly"""
    response = client.post("/predict")
    # Missing required field triggers 422 Unprocessable Entity
    assert response.status_code == 422

# Testing predict locally requires generating a dummy bytes stream representing a valid image format
def test_predict_dummy_image():
    # Generate 1x1 black pixel image safely
    img = Image.new('RGB', (100, 100), color = 'black')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    response = client.post(
        "/predict", 
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    
    # Provided a dummy model is loaded or real model, output should conform to standard schema
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["PNEUMONIA", "NORMAL"]

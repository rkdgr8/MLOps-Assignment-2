import time
import requests
import sys
import io
from PIL import Image

BASE_URL = "http://localhost:8000"

def create_dummy_image():
    """Create a minimal 1x1 PNG image in-memory for testing the endpoint."""
    img = Image.new('RGB', (224, 224), color = 'white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def run_tests():
    print("-----------------------------------------")
    print("Starting post-deploy Smoke Tests sequence")
    print("-----------------------------------------")

    print("\nPhase 1: Testing /health endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/health")
        if r.status_code == 200:
            print("â /health OK")
        else:
            print(f"â /health Failed with HTTP {r.status_code}: {r.text}")
            sys.exit(1)
    except Exception as e:
        print(f"â Critical Exception hitting /health: {e}")
        sys.exit(1)
        
    print("\nPhase 2: Testing /predict endpoint...")
    try:
        files = {'file': ('test.png', create_dummy_image(), 'image/png')}
        r = requests.post(f"{BASE_URL}/predict", files=files)
        
        if r.status_code == 200:
            resp = r.json()
            if "prediction" in resp:
                print(f"â /predict OK. Got sample prediction label: {resp['prediction']}")
            else:
                print(f"â /predict didn't return a standard prediction key: {resp}")
                sys.exit(1)
        else:
            print(f"â /predict Failed with HTTP {r.status_code}: {r.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"â Critical Exception hitting /predict: {e}")
        sys.exit(1)
        
    print("\n-----------------------------------------")
    print("All smoke tests passed successfully! â")
    print("-----------------------------------------")

if __name__ == "__main__":
    run_tests()

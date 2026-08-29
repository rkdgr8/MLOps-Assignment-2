FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    # Explicitly install python-multipart required by FastAPI forms
    pip install --no-cache-dir python-multipart

# Copy Application code
COPY src/ ./src/
# Pre-copy models directory if exists so container builds even without DVC pull in dev
COPY models/ ./models/

EXPOSE 8000

# Run Uvicorn standard
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

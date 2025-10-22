FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend requirements and install
# build context is the repository root, so reference backend/requirements.txt
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the image
COPY . .

# Expose backend port (FastAPI/uvicorn)
EXPOSE 8000

# Default command: run FastAPI app (backend.app.main:app)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

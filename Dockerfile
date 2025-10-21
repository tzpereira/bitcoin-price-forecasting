FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build deps (kept minimal). xgboost may require compilation on some platforms.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

EXPOSE 8501

ENV STREAMLIT_SERVER_ENABLECORS=false

# Run Streamlit using the project's main entry (which prepares data and shows the dashboard)
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

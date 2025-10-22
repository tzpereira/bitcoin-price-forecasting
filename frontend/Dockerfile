FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install frontend requirements
COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only frontend folder to keep image small
COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

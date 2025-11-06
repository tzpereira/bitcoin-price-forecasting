import os
from dotenv import load_dotenv
from fastapi import Request, HTTPException

load_dotenv()
API_TOKEN = os.environ.get("API_TOKEN")

def verify_token(request: Request):
    token = request.headers.get("X-API-Token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")

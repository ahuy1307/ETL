import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
from ETL import run_etl
from utils.logger import AppLog
from Schema import create_schema
# Configure AppLog
load_dotenv()

# Environment variables for configuration
BASE_URL = os.getenv("BASE_URL")
PAGE_SIZE = int(os.getenv("PAGE_SIZE"))
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = os.getenv("TABLE_ID")

async def job():
    AppLog.info("Running scheduled job")
    run_etl()

async def periodic_job():
    while True:
        await job()
        await asyncio.sleep(900)  # 15 minutes

# Lifespan Context Manager
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    await create_schema()
    clustering_task = asyncio.create_task(periodic_job())
    try:
        yield  
    finally:
        clustering_task.cancel()  
        try:
            await clustering_task
        except asyncio.CancelledError:
            AppLog.error("CLustering task is cancelled")
            pass

app = FastAPI(lifespan=app_lifespan)

@app.post("/etl/test")
async def test_etl():
    """API Endpoint to test the ETL process (without loading to BigQuery)."""
    await job()

    return {
        "status": "success",
        "message": "ETL process completed successfully"
    }

@app.get("/health")
def health_check():
    """API Endpoint for health check."""
    return {
        "status": "success",
        "message": "Service is running"
    }

@app.get("/")
def home():
    """Home page."""
    return {"message": "Hello, World!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
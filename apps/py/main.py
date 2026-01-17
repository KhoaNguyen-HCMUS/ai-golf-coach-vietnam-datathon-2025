"""
FastAPI Application cho Golf Swing Prediction Service
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path


PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

app = FastAPI(
    title="Golf Swing Prediction API",
    description="API để predict handicap band từ skeleton data",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """Response model cho health check."""

    status: str
    message: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Golf Swing Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", message="API is running")


if __name__ == "__main__":
    print(f"Starting server on {HOST}:{PORT}")
    print(f"API docs: http://{HOST}:{PORT}/docs")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True, log_level="info")

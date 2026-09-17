"""
Main FastAPI Application Entrypoint for SIH-26078.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.app.api.routes import router
from backend.app.db.database import init_db, SessionLocal
from backend.app.db import crud
from backend.app.pipeline.orchestrator import orchestrator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema
    init_db()
    # Check if baseline run exists; if not, generate default demo scenario
    db = SessionLocal()
    try:
        runs = crud.get_all_runs(db)
        if not runs:
            print("[INFO] Initializing default flagship scenario: run_demo_monsoon_depression...")
            orchestrator.run_full_pipeline(
                run_id="run_demo_monsoon_depression",
                scenario_type="monsoon_depression",
                seed=42,
                db=db
            )
            print("[INFO] Default flagship scenario successfully initialized.")
    except Exception as e:
        print(f"[WARN] Startup initialization notice: {e}")
    finally:
        db.close()
    yield

app = FastAPI(
    title="SIH-26078 Weather Anomaly Intelligence Platform",
    description="AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "SIH-26078 Extreme Weather Anomaly Intelligence Backend",
        "documentation": "/docs",
        "status": "operational"
    }

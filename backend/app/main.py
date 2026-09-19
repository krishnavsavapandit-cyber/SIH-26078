"""
Main FastAPI Application Entrypoint for SIH-26078.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os

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

# Resolve path to compiled frontend/dist directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "index.html").exists():
    # Mount assets directory if it exists
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Exclude /api, /docs, /openapi.json, /redoc from SPA catch-all
        if full_path.startswith("api") or full_path in ["docs", "openapi.json", "redoc"]:
            return None
        file_target = FRONTEND_DIST_DIR / full_path
        if full_path and file_target.is_file():
            return FileResponse(file_target)
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
else:
    @app.get("/")
    def root():
        return {
            "message": "SIH-26078 Extreme Weather Anomaly Intelligence Backend",
            "documentation": "/docs",
            "status": "operational"
        }


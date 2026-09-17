"""
Database Session and Engine Factory for SIH-26078.
Provides primary SQLite file database with PostgreSQL compatibility.
Scientific tensors reside in NetCDF storage while metadata, tracks, alerts, and verification
results are managed via SQLAlchemy.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.config import DB_PATH

# Allow environment override for PostgreSQL in production/cloud
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for API endpoints to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes all metadata database tables."""
    Base.metadata.create_all(bind=engine)

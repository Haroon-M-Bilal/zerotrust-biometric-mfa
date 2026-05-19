"""
SQLite database engine, session factory, and declarative Base.
Other modules import `get_db` (FastAPI dependency) or `SessionLocal` directly.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import get_settings

settings = get_settings()

# SQLite needs check_same_thread=False for FastAPI's threaded request handling
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session, closes it after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Called once on app startup."""
    # Import models so they register with Base.metadata before create_all
    from app.models import user, session as session_model, audit, account, transaction  # noqa: F401
    Base.metadata.create_all(bind=engine)
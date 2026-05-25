"""
FastAPI entry point for the Zero-Trust Biometric MFA system.
Wires together middleware, routes, and lifecycle hooks.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth_routes, transaction_routes
from app.db.database import init_db
from config.settings import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    # Startup: ensure DB tables exist
    init_db()
    yield
    # Shutdown


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(auth_routes.router)
app.include_router(transaction_routes.router)


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """Liveness probe. Returns app metadata and confirms the service is up."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/", tags=["system"])
async def root() -> dict:
    """Root endpoint. Points clients to the interactive docs."""
    return {
        "message": f"{settings.app_name} is running",
        "docs": "/docs",
        "health": "/health",
    }
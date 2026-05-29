"""
FastAPI entry point for the Zero-Trust Biometric MFA system.
Wires together middleware, routes, and lifecycle hooks.
Serves the frontend SPA at /.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth_routes, transaction_routes, admin_routes
from app.db.database import init_db
from config.settings import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


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
app.include_router(admin_routes.router)


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}


# ---- Static frontend ----
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
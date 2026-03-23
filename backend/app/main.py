import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.routes import health, chat, documents, search
from app.api.routes import auth, dashboard_routes
from app.services.embedder import embedder
from app.services.session_store import session_store
from app.database.session import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory rate limit state
_request_log: dict = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Chatbot API...")
    init_db()
    embedder.load()
    cleaned = session_store.cleanup_expired()
    logger.info(f"Startup complete. Cleaned {cleaned} expired sessions.")
    yield
    logger.info("Shutting down RAG Chatbot API.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Chatbot API",
        description=(
            "Enterprise-grade Retrieval-Augmented Generation chatbot. "
            "Upload PDFs and chat with your documents."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging ───────────────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    # ── Rate limiting (simple in-memory sliding window) ───────────────────────
    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        # Skip rate limiting for health checks and static files
        if request.url.path in ("/", "/docs", "/redoc", "/openapi.json", "/api/v1/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = datetime.now(timezone.utc)
        window = timedelta(seconds=settings.RATE_LIMIT_WINDOW)

        _request_log[client_ip] = [
            t for t in _request_log[client_ip] if now - t < window
        ]

        if len(_request_log[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: {settings.RATE_LIMIT_REQUESTS} requests "
                        f"per {settings.RATE_LIMIT_WINDOW}s. Please slow down."
                    )
                },
            )

        _request_log[client_ip].append(now)
        return await call_next(request)

    # ── Routes ────────────────────────────────────────────────────────────────
    PREFIX = "/api/v1"
    app.include_router(health.router, prefix=PREFIX, tags=["Health"])
    app.include_router(auth.router, prefix=PREFIX, tags=["Auth"])
    app.include_router(chat.router, prefix=PREFIX, tags=["Chat"])
    app.include_router(search.router, prefix=PREFIX, tags=["Search"])
    app.include_router(documents.router, prefix=PREFIX, tags=["Documents"])
    app.include_router(dashboard_routes.router, prefix=PREFIX, tags=["Dashboard"])

    # ── Static files (widget + frontend) ─────────────────────────────────────
    # Works both locally (repo root) and in Docker (/app)
    _here = Path(__file__).resolve().parent          # backend/app/
    BASE = _here.parent.parent                        # repo root (local) or /app (Docker)

    # Serve widget JS at /rag-widget.js
    widget_file = BASE / "widget" / "dist" / "rag-widget.iife.js"
    if widget_file.exists():
        @app.get("/rag-widget.js", include_in_schema=False)
        def serve_widget():
            return FileResponse(
                widget_file,
                media_type="application/javascript",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        logger.info("Widget file registered at /rag-widget.js")

    # Serve built React frontend (SPA) — must be last so it doesn't swallow /api routes
    frontend_dist = BASE / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(full_path: str = ""):
            # Let API routes pass through; only serve SPA for non-API paths
            index = frontend_dist / "index.html"
            return FileResponse(index)

        logger.info("Frontend SPA registered from %s", frontend_dist)
    else:
        @app.get("/", tags=["Root"], summary="API info")
        def root():
            return {
                "name": "RAG Chatbot API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/api/v1/health",
            }

    return app


app = create_app()

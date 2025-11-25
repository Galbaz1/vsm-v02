from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.core.config import get_settings
from api.endpoints import health, search, agentic

app = FastAPI(
    title="Manual Search API",
    version="0.1.0",
    description="Semantic search API over ADE-parsed manuals stored in Weaviate.",
)

settings = get_settings()
cors_origins = settings.cors_origins.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(health.router)
app.include_router(search.router)
app.include_router(agentic.router, tags=["agentic"])

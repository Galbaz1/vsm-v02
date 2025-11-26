import os
from functools import lru_cache
from pydantic import BaseModel

class Settings(BaseModel):
    api_base_url: str = "http://localhost:8001"
    pdf_base_url: str = "/static/manuals"
    preview_base_url: str = "/static/previews"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"

@lru_cache
def get_settings() -> Settings:
    return Settings(
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8001"),
        pdf_base_url=os.getenv("PDF_BASE_URL", "/static/manuals"),
        preview_base_url=os.getenv("PREVIEW_BASE_URL", "/static/previews"),
        cors_origins=os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"),
    )

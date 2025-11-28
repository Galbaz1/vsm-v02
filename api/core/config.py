import os
from functools import lru_cache
from typing import Literal
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings with mode-switchable provider configuration."""
    
    # === MODE SELECTION ===
    vsm_mode: Literal["local", "cloud"] = Field(
        default="local",
        description="Deployment mode: 'local' (Ollama+MLX+ColQwen) or 'cloud' (Gemini+Jina CLIP)"
    )
    
    # === API SETTINGS ===
    api_base_url: str = "http://localhost:8001"
    pdf_base_url: str = "/static/manuals"
    preview_base_url: str = "/static/previews"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
    
    # === LOCAL PROVIDERS ===
    # Ollama (LLM + Text Embeddings)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:120b"
    ollama_embed_model: str = "bge-m3"
    
    # MLX VLM
    mlx_vlm_base_url: str = "http://localhost:8000"
    
    # Weaviate Local (Docker)
    weaviate_local_url: str = "http://localhost:8080"
    
    # === CLOUD PROVIDERS ===
    # Gemini (LLM + VLM)
    gemini_api_key: str = Field(default="", description="Google AI Gemini API key")
    gemini_model: str = "gemini-2.5-flash"
    gemini_thinking_budget: int = Field(
        default=-1,
        description="Thinking budget: -1=dynamic, 0=off, 1-24576=fixed tokens"
    )
    
    # Jina (Text Embeddings + Visual Search via Weaviate integration)
    jina_api_key: str = Field(default="", description="Jina AI API key for embeddings and CLIP")
    
    # Weaviate Cloud (handles both text and visual search)
    weaviate_cloud_url: str = Field(default="", description="Weaviate Cloud cluster URL")
    weaviate_cloud_api_key: str = Field(default="", description="Weaviate Cloud API key")
    
    # OpenAI (fallback for Gemini failures)
    openai_api_key: str = Field(default="", description="OpenAI API key for GPT-5.1 fallback")
    openai_model: str = Field(default="gpt-5.1", description="OpenAI model (GPT-5.1 Responses API)")


@lru_cache
def get_settings() -> Settings:
    """
    Load settings from environment variables.
    
    Environment variables override defaults.
    """
    return Settings(
        # Mode
        vsm_mode=os.getenv("VSM_MODE", "local"),
        
        # API
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8001"),
        pdf_base_url=os.getenv("PDF_BASE_URL", "/static/manuals"),
        preview_base_url=os.getenv("PREVIEW_BASE_URL", "/static/previews"),
        cors_origins=os.getenv(
            "CORS_ALLOW_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
        ),
        
        # Local
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"),
        mlx_vlm_base_url=os.getenv("MLX_VLM_BASE_URL", "http://localhost:8000"),
        weaviate_local_url=os.getenv("WEAVIATE_LOCAL_URL", "http://localhost:8080"),
        
        # Cloud
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "-1")),
        jina_api_key=os.getenv("JINA_API_KEY", ""),
        weaviate_cloud_url=os.getenv("WEAVIATE_URL", ""),
        weaviate_cloud_api_key=os.getenv("WEAVIATE_API_KEY", ""),
        
        # OpenAI fallback
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
    )

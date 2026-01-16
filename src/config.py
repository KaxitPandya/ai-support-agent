"""
Configuration settings for the Knowledge Assistant.
Uses pydantic-settings for environment variable management.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # API Settings
    app_name: str = "Knowledge Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # LLM Provider (OpenAI only)
    llm_provider: str = "openai"
    
    # OpenAI Settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 1024
    
    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # RAG Settings
    top_k_results: int = 5
    similarity_threshold: float = 0.3
    
    # Vector Store Settings
    vector_store_path: str = "./data/vector_store"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

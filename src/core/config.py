"""
Configuration settings for the ChatKit RAG integration.
Uses Pydantic settings management.
"""
from pydantic_settings import BaseSettings
from typing import Optional

from .constants import (
    API_VERSION,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_COHERE_MODEL,
    DEFAULT_MIN_SIMILARITY_SCORE,
    DEFAULT_SEARCH_LIMIT,
    MAX_TOKENS,
    MAX_QUERY_LENGTH,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW
)


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = f"/api/{API_VERSION}"
    PROJECT_NAME: str = "ChatKit RAG Integration"
    VERSION: str = "1.0.0"

    # Gemini Settings
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = DEFAULT_GEMINI_MODEL

    # Cohere Settings
    COHERE_API_KEY: str
    COHERE_MODEL: str = DEFAULT_COHERE_MODEL

    # Qdrant Settings
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    TEXTBOOK_COLLECTION_NAME: str = "textbook_content"

    # Database Settings
    DATABASE_URL: str

    # Application Settings
    MIN_SIMILARITY_SCORE: float = DEFAULT_MIN_SIMILARITY_SCORE
    SEARCH_LIMIT: int = DEFAULT_SEARCH_LIMIT
    MAX_RESPONSE_TOKENS: int = MAX_TOKENS
    MAX_QUERY_LENGTH: int = MAX_QUERY_LENGTH
    RATE_LIMIT_REQUESTS: int = RATE_LIMIT_REQUESTS
    RATE_LIMIT_WINDOW: int = RATE_LIMIT_WINDOW  # in seconds

    class Config:
        env_file = ".env"


settings = Settings()
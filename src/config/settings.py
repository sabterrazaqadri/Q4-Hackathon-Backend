from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # OpenAI settings - Required for RAG Agent
    openai_api_key: str
    openai_model: str = "gpt-4-1106-preview"

    # Cohere settings
    cohere_api_key: str
    cohere_model: str = "embed-multilingual-v2.0"

    # Qdrant settings
    qdrant_url: str
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "textbook_chunks"

    # Book URLs to crawl
    book_urls: List[str] = []

    # Chunking settings
    chunk_size: int = 512  # Number of tokens per chunk
    chunk_overlap: int = 50  # Number of tokens to overlap between chunks

    # Crawler settings
    crawl_delay: float = 1.0  # Delay between requests in seconds
    max_retries: int = 3
    timeout: int = 30

    # Retrieval settings
    default_top_k: int = 5
    default_score_threshold: Optional[float] = None

    # API settings
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # Security settings
    allowed_origins: list = ["http://localhost:3000", "http://127.0.0.1:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create a single instance of settings
settings = Settings()
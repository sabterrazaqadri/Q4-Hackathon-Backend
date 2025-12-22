"""
Custom exceptions for the ChatKit RAG integration.
"""
import logging
from typing import Optional


# Set up logging
logger = logging.getLogger(__name__)


class ChatKitRAGException(Exception):
    """Base exception class for the ChatKit RAG integration."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
        # Log the exception
        logger.error(f"{self.__class__.__name__}: {message} (Code: {error_code})")


class RAGException(ChatKitRAGException):
    """Exception raised when there's an issue with the RAG system."""
    pass


class ContextRetrievalError(RAGException):
    """Exception raised when context retrieval fails."""
    pass


class ResponseGenerationError(RAGException):
    """Exception raised when response generation fails."""
    pass


class QueryValidationError(ChatKitRAGException):
    """Exception raised when query validation fails."""
    pass


class ConfigurationError(ChatKitRAGException):
    """Exception raised when there's a configuration issue."""
    pass


class RateLimitExceeded(ChatKitRAGException):
    """Exception raised when rate limit is exceeded."""
    pass


def setup_logging():
    """Set up logging configuration for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            # Add file handler if needed
            # logging.FileHandler("app.log")
        ]
    )
"""
Custom exceptions for the RAG Backend service
"""


class RagBackendError(Exception):
    """Base exception for RAG Backend service"""
    pass


class CrawlError(RagBackendError):
    """Exception raised during web crawling operations"""
    pass


class ChunkingError(RagBackendError):
    """Exception raised during content chunking operations"""
    pass


class EmbeddingError(RagBackendError):
    """Exception raised during embedding generation operations"""
    pass


class StorageError(RagBackendError):
    """Exception raised during vector storage operations"""
    pass


class ConfigurationError(RagBackendError):
    """Exception raised when configuration is invalid"""
    pass


class ValidationError(RagBackendError):
    """Exception raised when input validation fails"""
    pass
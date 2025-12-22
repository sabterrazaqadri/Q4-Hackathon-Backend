"""
Main application entry point for the ChatKit RAG integration.
Based on the implementation plan in plan.md.
"""
import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from .chat.endpoints import router as chat_router
from .chat.services import get_chat_service
from .rag.services import get_rag_service
from .core.config import settings
from .core.exceptions import ChatKitRAGException, setup_logging
from .core.rate_limiter import rate_limit_middleware

# Set up logging
setup_logging()

# Create the FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for integrating OpenAI ChatKit with the RAG system for the Physical AI & Humanoid Robotics textbook"
)

# Add rate limiting middleware first (so it's executed first)
app.middleware("http")(rate_limit_middleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose custom headers if needed
    # expose_headers=["Access-Control-Allow-Origin"]
)

# Include API routers
app.include_router(chat_router, prefix=settings.API_V1_STR)

# Add exception handlers
@app.exception_handler(ChatKitRAGException)
async def chatkit_rag_exception_handler(request, exc: ChatKitRAGException):
    return {
        "error": {
            "message": exc.message,
            "type": exc.__class__.__name__,
            "code": exc.error_code
        }
    }


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "ChatKit RAG Integration API", "version": settings.VERSION}


# Dependency injection for services
@app.get("/health")
async def health_check(
    chat_service = Depends(get_chat_service),
    rag_service = Depends(get_rag_service)
):
    # Test that services can be instantiated
    return {"status": "healthy", "message": "All services are running"}
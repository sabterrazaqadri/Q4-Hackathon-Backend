import logging
import time
import traceback
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from src.config.settings import settings
from src.api.routes import rag
from src.api.models import ErrorResponse
import secrets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Agent API",
    description="""
    API for the RAG-enabled agent that answers questions from Physical AI & Humanoid Robotics textbook content.

    ## Features

    - Query the RAG agent with textbook-based questions
    - Support for context with selected text
    - Structured responses with source citations
    - Comprehensive error handling
    - Request validation

    ## Endpoints

    - `/api/v1/rag/query`: Main endpoint for submitting questions to the RAG agent
    - `/api/v1/rag/query/validate`: Validate queries without processing them
    - `/`: Health check endpoint
    - `/health`: Health check endpoint

    ## Models

    The API uses standardized models for requests and responses:

    - QueryRequest: For submitting questions to the agent
    - AgentResponse: For receiving answers from the agent
    - ErrorResponse: For structured error responses
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware for logging and monitoring
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    start_time = time.time()

    # Get client info
    client_host = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else "unknown"

    # Log incoming request
    logger.info(f"Request: {request.method} {request.url.path} from {client_host}:{client_port}")

    # Process the request
    response = await call_next(request)

    # Calculate processing time
    processing_time = time.time() - start_time

    # Log the response
    logger.info(
        f"Response: {response.status_code} for {request.method} {request.url.path} "
        f"in {processing_time:.3f}s"
    )

    return response

# Add middleware for adding server timing headers
@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Response-Time"] = f"{process_time:.3f}s"

    # Log performance metrics
    logger.info(f"Performance: Request {request.method} {request.url.path} took {process_time:.3f}s")

    # Add performance metrics to the response headers if it's a RAG query
    if "/api/v1/rag/query" in str(request.url):
        response.headers["X-Performance-Category"] = "RAG-Processing"

    return response


# Performance monitoring middleware
@app.middleware("http")
async def performance_monitoring(request: Request, call_next: Callable):
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Performance metrics logging
    if process_time > 2.0:  # Log slow requests (>2s)
        logger.warning(
            f"SLOW REQUEST: {request.method} {request.url.path} "
            f"took {process_time:.3f}s from {request.client.host if request.client else 'unknown'}"
        )

    # Add performance metrics to response headers for monitoring
    response.headers["X-Server-Process-Time"] = f"{process_time:.3f}s"

    return response


@app.get("/metrics")
def get_metrics():
    """
    Performance metrics endpoint for monitoring
    """
    return {
        "status": "operational",
        "service": "RAG Agent API",
        "endpoint": "/metrics",
        "timestamp": time.time()
    }

# Import and add comprehensive security middleware
from src.api.middleware.security import SecurityMiddleware, RateLimitMiddleware, InputValidationMiddleware

# Add security middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests_per_minute=60)
app.add_middleware(InputValidationMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins if hasattr(settings, 'allowed_origins') else ["http://localhost:3000", "https://physical-ai-humanoid-textbook-mu.vercel.app/"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(rag.router, prefix="/api/v1")

@app.get("/")
def read_root():
    logger.info("Health check endpoint accessed")
    return {"message": "RAG Agent API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RAG Agent API"}


# Exception handlers for structured error responses
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP {exc.status_code} error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=exc.detail
        ).dict()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="VALIDATION_ERROR",
            message=f"Validation error: {exc.errors()[0]['msg'] if exc.errors() else 'Invalid input parameters'}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred"
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG Agent API server")
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )